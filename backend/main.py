import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dia.model import Dia  # <-- dia package

# -----------------------------------------------------------------------------
# Logging & dirs
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")

AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Model Manager
# -----------------------------------------------------------------------------
class ModelManager:
    def __init__(self, model_id: str = "nari-labs/Dia-1.6B-0626"):
        self.model_id = model_id
        self.model: Optional[Dia] = None

    def load_model(self):
        logger.info(f"Loading Dia model: {self.model_id}")
        self.model = Dia.from_pretrained(self.model_id)
        logger.info("Dia model loaded")

    def unload_model(self):
        self.model = None
        logger.info("Dia model unloaded")

    def get(self) -> Dia:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

model_manager = ModelManager()

# -----------------------------------------------------------------------------
# Request schemas
# -----------------------------------------------------------------------------
class AudioPrompt(BaseModel):
    sample_rate: int
    audio_data: List[float]  # raw mono PCM in [-1, 1]

class GenerateRequest(BaseModel):
    text_input: str
    audio_prompt: Optional[AudioPrompt] = None
    max_new_tokens: int = 1024
    cfg_scale: float = 3.0
    temperature: float = 1.3
    top_p: float = 0.95
    cfg_filter_top_k: int = 35
    speed_factor: float = 0.94  # ignored if dia doesn't support it

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def process_audio_prompt(sample_rate: int, audio_data: List[float], target_rate: int = 44100) -> np.ndarray:
    """
    Convert incoming audio to np.float32 mono at 44.1 kHz.
    """
    x = np.asarray(audio_data, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    x = np.clip(x, -1.0, 1.0)

    if sample_rate != target_rate:
        # Simple resample using numpy repeat/average (replace with librosa/torchaudio if available)
        import math
        from math import gcd
        g = gcd(sample_rate, target_rate)
        up = target_rate // g
        down = sample_rate // g
        x_up = np.repeat(x, up)
        if down > 1:
            n = (x_up.shape[0] // down) * down
            x_up = x_up[:n].reshape(-1, down).mean(axis=1).astype(np.float32)
        x = x_up
    return x.astype(np.float32).flatten()

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting up application...")
    model_manager.load_model()
    yield
    logger.info("Shutting down application...")
    model_manager.unload_model()

app = FastAPI(
    title="Dia Text-to-Speech API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/generate")
async def run_inference(request: GenerateRequest):
    text = (request.text_input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    out_path = AUDIO_DIR / f"{int(time.time())}.wav"
    try:
        model = model_manager.get()

        audio_np: Optional[np.ndarray] = None
        if request.audio_prompt is not None:
            audio_np = process_audio_prompt(
                request.audio_prompt.sample_rate,
                request.audio_prompt.audio_data,
                44100,
            )

        gen_kwargs = dict(
            max_new_tokens=request.max_new_tokens,
            guidance_scale=request.cfg_scale,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.cfg_filter_top_k,
        )
        if request.speed_factor is not None:
            gen_kwargs["speed_factor"] = request.speed_factor

        logger.info(f"Generating (prompt={'yes' if audio_np is not None else 'no'})...")
        if audio_np is not None:
            wav = model.generate(text=text, audio=audio_np, **gen_kwargs)
        else:
            wav = model.generate(text=text, **gen_kwargs)

        wav = np.asarray(wav, dtype=np.float32).flatten()
        sf.write(str(out_path), wav, 44100)
        logger.info(f"Saved {out_path}")

        return FileResponse(str(out_path), media_type="audio/wav", filename=out_path.name)

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))
