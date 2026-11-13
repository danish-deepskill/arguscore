# ArguScore

Aplikasi web untuk menilai kualitas potongan argumen dalam esai berbahasa Inggris.
Setiap potongan (Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement)
diberi label **Effective**, **Adequate**, atau **Ineffective**.

Tugas Besar Machine Learning Part 3
Danish Rafie Al Rasyad, NIM 202343501486, kelas X7J

Studi kasusnya ilustrasi: penilai awal untuk AI Tutor critical thinking (DeepSkill).
Ini proyek tugas kuliah, bukan sistem resmi DeepSkill.

## Model

- Dataset: Kaggle, *Feedback Prize - Predicting Effective Arguments*
- Preprocessing: TF-IDF (unigram + bigram), TruncatedSVD 100 komponen, one-hot jenis argumen, 15 fitur angka
- Model final: LightGBM (good fit) dari Part 2, dengan bobot kelas Ineffective w = 1.1
- Training dan tuning di Google Colab, lalu model disimpan sebagai `model/arguscore_model.pth`
  (kode selnya ada di `colab/simpan_model_pth.py`)

## Struktur folder

```
app/main.py      backend FastAPI (endpoint /predict dan /info)
app/model.py     memuat .pth dan membuat fitur yang sama seperti di notebook
static/index.html  tampilan web (VueJS)
model/           file model .pth dari Colab
colab/           sel Colab untuk menyimpan dan mengunduh model
```

## Cara menjalankan

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Buka http://127.0.0.1:8000, klik **Isi contoh**, lalu **Nilai argumen**.

Catatan: versi scikit-learn dan lightgbm sebaiknya sama dengan versi di Colab
(versinya ikut tersimpan di file .pth dan akan muncul peringatan kalau berbeda).
Torch tidak wajib dipasang. Kalau ada, file .pth dimuat dengan `torch.load`.

## Contoh request API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"potongan": [{"jenis": "Position", "teks": "Students should be allowed to use phones in class."},
                    {"jenis": "Rebuttal", "teks": "But that is just not true."}]}'
```
