import numpy as np
import cv2
import tensorflow as tf
import io
from PIL import Image


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
    """Load the Keras chess model from disk."""
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into an OpenCV-compatible numpy array (BGR).
    Replaces cv2.imread() since we receive bytes from the API instead of a file path.
    """
    # decode bytes → numpy array → OpenCV image (BGR)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG or PNG.")

    return img


def extract_squares(img: np.ndarray) -> np.ndarray:
    """
    Slice the chessboard into 64 squares, resize each to 128×128,
    and convert BGR → RGB to match training-time preprocessing.
    Returns a batch of shape (64, 128, 128, 3).
    """
    h, w, _ = img.shape
    sq_h, sq_w = h // 8, w // 8
    squares = []

    for r in range(8):
        for c in range(8):
            square = img[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w]
            square = cv2.resize(square, (128, 128))
            square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)   # BGR → RGB
            squares.append(square)

    return np.array(squares)   # shape: (64, 128, 128, 3)


def predict_board(model, image_bytes: bytes) -> list:
    """
    Full pipeline: bytes → squares → model prediction → 8×8 label grid.
    Returns a list of lists, e.g. [['blackrook', 'blackknight', ...], ...].
    """
    img = preprocess_image(image_bytes)
    batch = extract_squares(img)

    predictions = model.predict(batch, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = [CLASS_NAMES[i] for i in predicted_indices]

    return np.array(predicted_labels).reshape(8, 8).tolist()


def board_to_fen(predictions_grid: list, whos_turn: str = "w") -> str:
    """Convert the 8×8 label grid into a valid FEN string."""
    fen_rows = []

    for row in predictions_grid:
        raw_row = "".join([LABEL_TO_FEN.get(piece, '?') for piece in row])

        # compress consecutive empty squares (e.g. '111' → '3')
        for i in range(8, 1, -1):
            raw_row = raw_row.replace('1' * i, str(i))

        fen_rows.append(raw_row)

    board_fen = "/".join(fen_rows)
    return f"{board_fen} {whos_turn} - - 0 1"


def draw_best_move(image_bytes: bytes, best_move: str) -> bytes:
    """
    Draw a green arrow on the original chessboard image showing the best move.
    Returns the annotated image as PNG bytes.
    """
    img = preprocess_image(image_bytes)
    h, w, _ = img.shape
    sq_h, sq_w = h // 8, w // 8

    files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    ranks = {'8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7}

    start_sq = best_move[:2]
    end_sq = best_move[2:4]

    col_start = files[start_sq[0]]
    row_start = ranks[start_sq[1]]
    col_end = files[end_sq[0]]
    row_end = ranks[end_sq[1]]

    start_center = (col_start * sq_w + sq_w // 2, row_start * sq_h + sq_h // 2)
    end_center = (col_end * sq_w + sq_w // 2, row_end * sq_h + sq_h // 2)

    # draw the arrow (green, thick)
    cv2.arrowedLine(img, start_center, end_center, (0, 255, 0), 6, tipLength=0.2)

    # encode annotated image back to PNG bytes
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode result image.")

    return buffer.tobytes()
