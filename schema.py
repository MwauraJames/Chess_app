from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    fen_string: str
    best_move: str
    human_readable_move: str
    evaluation_type: str
    evaluation_value: int
    whos_turn: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
