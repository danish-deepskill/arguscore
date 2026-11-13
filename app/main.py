import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.model import JENIS_ARGUMEN, PATH_MODEL, ArguScore

FOLDER_STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="ArguScore API")
_model = None


def get_model():
    global _model
    if _model is None:
        if not PATH_MODEL.exists():
            raise HTTPException(503, f"File model belum ada: {PATH_MODEL.name}. "
                                     "Download dari Colab lalu taruh di folder model/.")
        _model = ArguScore()
    return _model


class Potongan(BaseModel):
    jenis: str
    teks: str


class Permintaan(BaseModel):
    potongan: List[Potongan]


@app.get("/")
def halaman_utama():
    return FileResponse(FOLDER_STATIC / "index.html")


@app.get("/info")
def info():
    m = get_model()
    return {"model": m.nama, "w_ineffective": m.w, "label": m.label, "jenis": JENIS_ARGUMEN}


@app.post("/predict")
def predict(req: Permintaan):
    potongan = [p.model_dump() for p in req.potongan if p.teks.strip()]
    if not potongan:
        raise HTTPException(400, "Isi minimal satu potongan argumen.")
    for p in potongan:
        if p["jenis"] not in JENIS_ARGUMEN:
            raise HTTPException(400, f"Jenis argumen tidak dikenal: {p['jenis']}")

    m = get_model()
    mulai = time.perf_counter()
    hasil = m.prediksi(potongan)
    waktu_ms = (time.perf_counter() - mulai) * 1000
    return {"hasil": hasil, "waktu_ms": round(waktu_ms, 1)}
