# Sel tambahan di notebook Colab Part 2 untuk menyimpan model final sebagai .pth
# lalu mengunduhnya ke laptop.
#
# Jika runtime Colab masih aktif (variabel model_final, W_final, dll. masih ada),
# cukup jalankan BAGIAN B.
# Jika runtime sudah reset, jalankan dulu sel 1 sampai 4.4 di notebook
# (library, dataset, feature engineering, split, prep_tree), lalu BAGIAN A, lalu BAGIAN B.

# ---------- BAGIAN A: latih ulang model final (hanya jika runtime reset) ----------
# Parameter diambil dari hasil tuning "LightGBM (good fit)" di Part 2.
model_final = Pipeline([
    ("prep", prep_tree),
    ("clf", lgb.LGBMClassifier(
        class_weight="balanced", subsample_freq=1, random_state=SEED,
        n_jobs=-1, verbose=-1,
        n_estimators=200, learning_rate=0.1, num_leaves=15,
        min_child_samples=100, reg_lambda=1, subsample=0.9, colsample_bytree=0.5,
    )),
])
model_final.fit(X_train, y_train)
NAMA_FINAL = "LightGBM (good fit)"
W_final = 1.1

# ---------- BAGIAN B: simpan sebagai .pth dan download ----------
import torch, sklearn, lightgbm
from google.colab import files

paket = {
    "nama": NAMA_FINAL,
    "model": model_final,
    "w_ineffective": W_final,
    "label": NAMA_KELAS,
    "fitur": FITUR,
    "versi": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__},
}
torch.save(paket, "arguscore_model.pth")
print("Tersimpan:", paket["nama"], "| w =", W_final, "|", paket["versi"])
files.download("arguscore_model.pth")
