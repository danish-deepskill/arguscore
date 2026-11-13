"""Memuat model ArguScore (.pth dari Colab) dan menyiapkan fitur yang sama
seperti di notebook Part 2."""

import pickle
import zipfile
from pathlib import Path

import pandas as pd

PATH_MODEL = Path(__file__).resolve().parent.parent / "model" / "arguscore_model.pth"

JENIS_ARGUMEN = [
    "Lead", "Position", "Claim", "Counterclaim",
    "Rebuttal", "Evidence", "Concluding Statement",
]


def load_pth(path):
    # File .pth dibuat dengan torch.save() di Colab.
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        # Kalau torch tidak terpasang, isi .pth (bentuknya zip) dibaca langsung.
        # Bisa karena isinya pipeline sklearn biasa, tidak ada tensor.
        with zipfile.ZipFile(path) as z:
            nama = [n for n in z.namelist() if n.endswith("data.pkl")][0]
            with z.open(nama) as f:
                return pickle.load(f)


def buat_fitur(potongan):
    """potongan = list of dict {"jenis": ..., "teks": ...}, urut sesuai esai.
    Semua potongan dianggap berasal dari satu esai yang sama."""
    df = pd.DataFrame(potongan)
    df["discourse_type"] = df["jenis"]
    df["text"] = df["teks"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["inp"] = "[" + df["discourse_type"].str.upper() + "] " + df["text"].str.lower()

    # fitur per potongan (sama dengan sel 4.2 di notebook)
    kata = df["text"].str.split()
    df["n_words"] = kata.str.len()
    df["n_chars"] = df["text"].str.len()
    df["n_sent"] = df["text"].str.count(r"[.!?]+") + 1
    df["avg_wlen"] = df["text"].str.replace(" ", "").str.len() / df["n_words"]
    df["uniq_ratio"] = kata.apply(lambda k: len({w.lower() for w in k}) / len(k))
    df["n_comma"] = df["text"].str.count(",")
    df["has_digit"] = df["text"].str.contains(r"\d").astype(int)
    df["has_quote"] = df["text"].str.contains('"', regex=False).astype(int)

    # fitur esai
    n = len(df)
    df["pos_in_essay"] = [i / n for i in range(n)]
    df["essay_n_disc"] = n
    df["essay_words"] = df["n_words"].sum()
    df["essay_mean_words"] = df["n_words"].mean()
    for t in ["Evidence", "Counterclaim", "Rebuttal"]:
        df[f"essay_n_{t.lower()}"] = int((df["discourse_type"] == t).sum())
    return df


class ArguScore:
    def __init__(self, path=PATH_MODEL):
        paket = load_pth(path)
        self.nama = paket.get("nama", "model")
        self.model = paket["model"]
        self.w = paket.get("w_ineffective") or 1.0
        self.label = list(paket.get("label", ["Ineffective", "Adequate", "Effective"]))
        self.fitur = paket["fitur"]
        self._cek_versi(paket.get("versi", {}))

    def _cek_versi(self, versi_colab):
        import lightgbm
        import sklearn
        lokal = {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}
        for lib, v in versi_colab.items():
            if lokal.get(lib) != v:
                print(f"[peringatan] {lib} di Colab {v}, di sini {lokal.get(lib)}. "
                      f"Kalau ada error, samakan versinya.")

    def prediksi(self, potongan):
        df = buat_fitur(potongan)
        prob = self.model.predict_proba(df[self.fitur])

        # penyesuaian bobot kelas Ineffective (w dari Part 2)
        prob_w = prob.copy()
        prob_w[:, 0] *= self.w
        idx = prob_w.argmax(axis=1)

        hasil = []
        for i, row in df.iterrows():
            hasil.append({
                "jenis": row["jenis"],
                "teks": row["text"],
                "label": self.label[idx[i]],
                "prob": {self.label[k]: round(float(prob[i, k]), 3) for k in range(len(self.label))},
            })
        return hasil
