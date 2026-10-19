# An Intelligent Framework for Cuffless Blood Pressure Estimation (Phase 1)

Software-only AI prototype that accepts manually supplied PPG data, estimates
systolic (SBP) and diastolic (DBP) blood pressure using a trained
**ResNet-50 + Multi-Head Cross-Attention** regression model, visualises the
signals, and provides an animated **BP Care AI Assistant** for explaining
results and offering safe, general dietary and lifestyle guidance.

## Stack

* Frontend: HTML5 / CSS3 / JavaScript + Chart.js (responsive dashboard)
* Backend: Python + FastAPI (REST APIs)
* AI/ML: PyTorch, ResNet-50, Multi-Head Cross-Attention, NumPy, Pandas, SciPy, scikit-learn

## Dataset (public, research use)

The model is trained on the public **PPG-based-BP-assessment** dataset
(56 outpatient subjects, 2-min PPG recordings at 100 Hz, reference cuff-measured
SBP/DBP in mmHg, participant IDs).

Source: https://github.com/sanvsquezsz/PPG-based-BP-assessment
Citation: Argüello-Prada, E.J., Castaño Mosquera, C.D. (2025). "Exploring
supervised machine learning models to estimate blood pressure using non-fiducial
features of the photoplethysmogram (PPG) and its derivatives." Phys Eng Sci Med.

Dataset files are downloaded automatically by `backend/dataset_prepare.py` into
`backend/data/ppg_bp_dataset/raw/`.

## Quick start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare the dataset (downloads 56 subject recordings from GitHub)

```bash
cd backend
python dataset_prepare.py
```

This writes `backend/data/ppg_bp_dataset/metadata.csv` and `splits.json`.

### 3. Train the model (ResNet-50 backbone frozen; projection + MHCA + head trained)

```bash
cd backend
python train_model.py
```

This saves:
* `backend/models/resnet_mhca_bp.pth` — trained weights
* `backend/models/training_config.json` — hyperparameters + target stats
* `backend/models/evaluation_results.json` — held-out test metrics (MAE, RMSE, ME, std)

Training runs on CPU and takes a few minutes. Participant-level 70/15/15 split
is used (no participant leakage across train/val/test).

### 4. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend

Visit: http://127.0.0.1:8000/ui/

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Backend health + model mode (trained/demo) |
| `/api/predict` | POST (multipart) | Upload a CSV/TXT PPG file + estimate BP |
| `/api/predict/json` | POST (JSON) | Manual signal input: `{"signal":[...],"fs":100}` |
| `/api/demo` | GET | Run pipeline on a synthetic demo signal |
| `/api/evaluate` | GET | Held-out test-set evaluation metrics |
| `/api/evaluate/full` | GET | Full evaluation incl. per-sample predictions |
| `/api/assistant` | POST | BP Care AI Assistant chat (explanation + guidance) |
| `/api/dietary` | POST | General dietary guidance |

## How it works

1. User uploads a CSV/TXT PPG file or pastes signal values manually.
2. Input is validated (empty file, missing values, insufficient samples, invalid sampling frequency).
3. Signal quality is assessed (amplitude, SNR, pulse regularity).
4. Butterworth band-pass filtering (0.5–5 Hz), then PPG / vPPG / aPPG generation.
5. STFT spectrogram images are generated and stacked as a 3-channel ResNet-50 input.
6. ResNet-50 extracts 2048-d features per signal form; Multi-Head Cross-Attention fuses them (PPG form queries vPPG/aPPG context).
7. A fully-connected regression head predicts SBP and DBP.
8. Results, signal quality, and a reliability indicator are displayed.
9. The animated BP Care AI Assistant explains the result and offers dietary / lifestyle guidance.

## Model status

If `models/resnet_mhca_bp.pth` is present, the app runs in **trained** mode and uses the real neural network. Otherwise it falls back to a **demo** calibrated estimator and clearly labels output as non-AI.

Current trained-model evaluation (held-out test set, 108 windows from 9 unseen participants):

* SBP: MAE 14.75, RMSE 16.93, ME −2.13, std 16.79 mmHg
* DBP: MAE 11.88, RMSE 14.93, ME −5.39, std 13.92 mmHg

## Phase 1 restrictions (not implemented)

No Arduino / ESP32 / Raspberry Pi sensors, no live sensor streaming, no wearables, no clinical deployment, no medical diagnosis, no treatment recommendations.

## Testing

```bash
cd backend
python -m pytest tests/ -q
# or without pytest:
python tests/test_api.py
```

Tests cover valid/invalid CSV/TXT upload, missing column, empty file, invalid values, insufficient samples, invalid sampling frequency, poor-quality signal, missing model, inference failure, successful prediction, assistant response, dietary handling, and the full upload-to-result workflow.

## Important notice

This is a research/academic software prototype, **not a medical device**. Outputs are AI estimates, not confirmed blood-pressure measurements. Always confirm with a validated blood-pressure cuff and consult a qualified healthcare professional. No food or lifestyle change is presented as a treatment for high blood pressure.
