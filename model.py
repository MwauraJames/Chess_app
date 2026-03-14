import numpy as np
import cv2
import tensorflow as tf
import io


CLASS_NAMES = [
    'blackbishop', 'blackking', 'blackknight', 'blackpawn', 'blackqueen', 'blackrook',
    'emptysquare',
    'whitebishop', 'whiteking', 'whiteknight', 'whitepawn', 'whitequeen', 'whiterook'
]

LABEL_TO_FEN = {
    'whitepawn': 'P', 'whiteknight': 'N', 'whitebishop': 'B',
    'whiterook': 'R', 'whitequeen': 'Q', 'whiteking': 'K',
    'blackpawn': 'p', 'blackknight': 'n', 'blackbishop': 'b',
    'blackrook': 'r', 'blackqueen': 'q', 'blackking': 'k',
    'emptysquare': '1'
}


def load_model(model_path: str):
    """
    Load a TFLite model and return its interpreter.
    TFLite uses an 'interpreter' instead of a Keras model object —
    it is much lighter on RAM than the full Keras model.
    """
    print(f"[model] Loading TFLite model from '{model_path}'...")
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # allocate_tensors() tells TFLite to reserve memory for inputs/outputs
    interpreter.allocate_tensors()

    print("[model] TFLite model ready.")
    return interpreter


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → OpenCV BGR image.
    Replaces cv2.imread() since we receive bytes from the API.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG or PNG.")

    return img


def extract_squares(img: np.ndarray) -> np.ndarray:
    """
    Slice the chessboard into 64 squares, resize to 128×128, convert BGR→RGB.
    Returns batch of shape (64, 128, 128, 3).
    """
    h, w, _ = img.shape
    sq_h, sq_w = h // 8, w // 8
    squares = []

    for r in range(8):
        for c in range(8):
            square = img[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w]
            square = cv2.resize(square, (128, 128))
            square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            squares.append(square)

    return np.array(squares, dtype=np.float32)   # TFLite requires float32


def predict_board(interpreter, image_bytes: bytes) -> list:
    """
    Full pipeline: bytes → squares → TFLite prediction → 8×8 label grid.

    TFLite works differently from Keras:
    - You set the input tensor manually
    - You call invoke() to run the model
    - You read the output tensor manually
    Instead of model.predict(batch), we feed one square at a time.
    """
    img    = preprocess_image(image_bytes)
    batch  = extract_squares(img)   # shape: (64, 128, 128, 3)

    # Get input and output tensor details from the interpreter
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predicted_labels = []

    for i in range(64):
        # Add batch dimension: (128, 128, 3) → (1, 128, 128, 3)
        square = np.expand_dims(batch[i], axis=0)

        # Feed the square into the model
        interpreter.set_tensor(input_details[0]['index'], square)

        # Run inference
        interpreter.invoke()

        # Read the output probabilities
        output = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 13)

        # Pick class with highest probability
        predicted_index = np.argmax(output[0])
        predicted_labels.append(CLASS_NAMES[predicted_index])

    return np.array(predicted_labels).reshape(8, 8).tolist()


def board_to_fen(predictions_grid: list, whos_turn: str = "w") -> str:
    """Convert 8×8 label grid → FEN string."""
    fen_rows = []

    for row in predictions_grid:
        raw_row = "".join([LABEL_TO_FEN.get(piece, '?') for piece in row])

        # compress consecutive empty squares e.g. '111' → '3'
        for i in range(8, 1, -1):
            raw_row = raw_row.replace('1' * i, str(i))

        fen_rows.append(raw_row)

    board_fen = "/".join(fen_rows)
    return f"{board_fen} {whos_turn} - - 0 1"


def draw_best_move(image_bytes: bytes, best_move: str) -> bytes:
    """
    Draw a green arrow on the chessboard showing the best move.
    Returns annotated image as PNG bytes.
    """
    img = preprocess_image(image_bytes)
    h, w, _ = img.shape
    sq_h, sq_w = h // 8, w // 8

    files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    ranks = {'8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7}

    start_sq = best_move[:2]
    end_sq   = best_move[2:4]

    col_start = files[start_sq[0]]
    row_start = ranks[start_sq[1]]
    col_end   = files[end_sq[0]]
    row_end   = ranks[end_sq[1]]

    start_center = (col_start * sq_w + sq_w // 2, row_start * sq_h + sq_h // 2)
    end_center   = (col_end   * sq_w + sq_w // 2, row_end   * sq_h + sq_h // 2)

    cv2.arrowedLine(img, start_center, end_center, (0, 255, 0), 6, tipLength=0.2)

    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode result image.")

    return buffer.tobytes()
