import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from stockfish import Stockfish

from model import load_model, predict_board, board_to_fen, draw_best_move
from schema import PredictionResponse

# ── Config (set these as environment variables on Render) ────────────────────
MODEL_PATH      = os.getenv("MODEL_PATH",      "my_chess_model.v2.keras")
STOCKFISH_PATH  = os.getenv("STOCKFISH_PATH",  "/usr/games/stockfish")   # Docker path
SEARCH_DEPTH    = int(os.getenv("SEARCH_DEPTH", 5))
PORT            = int(os.getenv("PORT", 8000))

ALLOWED_TYPES   = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE   = 10 * 1024 * 1024   # 10 MB

state = {}

# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Loading model from: {MODEL_PATH}")
    state["model"] = load_model(MODEL_PATH)
    print("[startup] Model ready.")
    yield
    state.clear()
    print("[shutdown] Model unloaded.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChessVision API",
    description="Upload a chessboard image → get back an annotated image with the best move.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the frontend (index.html + any static assets) from /frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the main UI page."""
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "healthy" if "model" in state else "unhealthy",
        "model_loaded": "model" in state,
    }


# ── Stockfish helper ──────────────────────────────────────────────────────────
def get_stockfish_move(fen: str, depth: int = SEARCH_DEPTH):
    sf = Stockfish(path=STOCKFISH_PATH)
    sf.update_engine_parameters({"Hash": 2048, "Threads": 2})
    sf.set_depth(depth)
    sf.set_fen_position(fen)

    best_move = sf.get_best_move()
    if not best_move:
        raise ValueError("Stockfish could not find a best move for this position.")

    start_sq   = best_move[:2]
    end_sq     = best_move[2:4]
    promotion  = best_move[4:]

    piece_enum = sf.get_what_is_on_square(start_sq)
    piece_name = str(piece_enum).split('_')[-1].capitalize() if piece_enum else "Piece"

    human_move = f"{piece_name} to {end_sq}"
    if promotion:
        promo_names = {'q': 'Queen', 'r': 'Rook', 'b': 'Bishop', 'n': 'Knight'}
        human_move += f" (promoting to {promo_names.get(promotion, promotion)})"

    evaluation = sf.get_evaluation()
    return best_move, human_move, evaluation


# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post(
    "/predict/image",
    summary="Upload chessboard → annotated image with best move arrow",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"image/png": {}}, "description": "Annotated PNG with green arrow"},
        400: {"description": "Bad request (wrong file type, too large, invalid position)"},
        500: {"description": "Server / model error"},
    },
)
async def predict_image(
    file:       UploadFile = File(..., description="Chessboard screenshot (JPG or PNG)"),
    whos_turn:  str        = Query(default="w",            description="'w' = white, 'b' = black"),
    depth:      int        = Query(default=SEARCH_DEPTH,   description="Stockfish search depth 1–20"),
):
    # 1. Validate type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only JPG and PNG are accepted."
        )

    # 2. Read bytes
    image_bytes = await file.read()

    # 3. Validate size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum is 10 MB.")

    try:
        # 4. Classify all 64 squares
        board_grid = predict_board(state["model"], image_bytes)

        # 5. Build FEN string
        fen = board_to_fen(board_grid, whos_turn=whos_turn)

        # 6. Ask Stockfish for the best move
        best_move, human_move, evaluation = get_stockfish_move(fen, depth=depth)

        # 7. Draw green arrow on original image
        annotated_bytes = draw_best_move(image_bytes, best_move)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # 8. Stream PNG back — metadata travels in custom headers
    return StreamingResponse(
        io.BytesIO(annotated_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition":  "inline; filename=best_move.png",
            "X-Best-Move":          best_move,
            "X-Human-Move":         human_move,
            "X-FEN":                fen,
            "X-Evaluation":         str(evaluation["value"]),
            "Access-Control-Expose-Headers":
                "X-Best-Move, X-Human-Move, X-FEN, X-Evaluation",
        },
    )


# ── JSON-only endpoint (optional) ────────────────────────────────────────────
@app.post("/predict/json", response_model=PredictionResponse)
async def predict_json(
    file:      UploadFile = File(...),
    whos_turn: str        = Query(default="w"),
    depth:     int        = Query(default=SEARCH_DEPTH),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file.content_type}'.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")

    try:
        board_grid = predict_board(state["model"], image_bytes)
        fen        = board_to_fen(board_grid, whos_turn=whos_turn)
        best_move, human_move, evaluation = get_stockfish_move(fen, depth=depth)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        fen_string=fen,
        best_move=best_move,
        human_readable_move=human_move,
        evaluation_type=evaluation["type"],
        evaluation_value=evaluation["value"],
        whos_turn=whos_turn,
    )
