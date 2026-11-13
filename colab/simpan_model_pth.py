# Sel tambahan di akhir notebook Colab Part 2 (bagian 10) untuk menyimpan model final
# sebagai .pth lalu mengunduhnya. Sebelum sel ini, jalankan sel 1 sampai 4.4
# (library, dataset, feature engineering, split, prep_tree).

# 10. Simpan model final untuk Part 3 (format .pth)
# Model final = LightGBM (good fit) dari bagian 7.6, dilatih ulang dengan parameter yang sama
import torch, sklearn, lightgbm
from google.colab import files

model_final = Pipeline([
    ("prep", prep_tree),
    ("clf", lgb.LGBMClassifier(class_weight="balanced", subsample_freq=1, random_state=SEED,
                               n_jobs=-1, verbose=-1, n_estimators=200, learning_rate=0.1,
                               num_leaves=15, min_child_samples=100, reg_lambda=1,
                               subsample=0.9, colsample_bytree=0.5)),
])
model_final.fit(X_train, y_train)
NAMA_FINAL, W_final = "LightGBM (good fit)", 1.1

# cek ulang di data uji
p = model_final.predict_proba(X_test)
p[:, 0] *= W_final
pred = p.argmax(1)
print("F1 macro uji (w = 1.1)  :", round(f1_score(y_test, pred, average="macro"), 3))
print("Recall Ineffective uji  :", round(recall_score(y_test, pred, average=None)[0], 3))

paket = {"nama": NAMA_FINAL, "model": model_final, "w_ineffective": W_final,
         "label": NAMA_KELAS, "fitur": FITUR,
         "versi": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}}
torch.save(paket, "arguscore_model.pth")
print("Model disimpan: arguscore_model.pth |", paket["versi"])
files.download("arguscore_model.pth")
