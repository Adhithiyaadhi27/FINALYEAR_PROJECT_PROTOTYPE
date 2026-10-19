# System Architecture

## An Intelligent Framework for Cuffless Blood Pressure Estimation (Phase 1)

### Overview

A software-only prototype that accepts manually supplied PPG data (CSV/TXT upload or pasted values), estimates systolic (SBP) and diastolic (DBP) blood pressure using a trained deep-learning model, visualises the signals, and provides an animated AI health assistant for explaining results and offering safe dietary and lifestyle guidance.

### High-level data flow

```
User uploads PPG CSV/TXT or pastes samples
        |
        v
Backend /api/predict (FastAPI)
        |
        v
Input validation (empty, missing values, insufficient samples, invalid fs)
        |
        v
Signal quality assessment (amplitude, SNR, pulse regularity)
        |
        v
Butterworth band-pass filter (0.5–5 Hz, order 4, zero-phase)
        |
        v
PPG / vPPG (1st derivative) / aPPG (2nd derivative)
        |
        v
STFT spectrogram images (224x224) -> stacked 3-channel tensor
        |
        v
ResNet-50 (ImageNet, frozen) -> 2048-d features per form
        |
        v
Multi-Head Cross-Attention (PPG queries attend over vPPG/aPPG context)
        |
        v
Fully-connected regression head (256 -> 64 -> 2) -> SBP, DBP
        |
        v
JSON response -> frontend dashboard + BP Care AI Assistant
```

### Backend (FastAPI)

| File | Responsibility |
|---|---|
| `backend/main.py` | FastAPI app, CORS, static frontend, all REST endpoints |
| `backend/app/pipeline.py` | End-to-end orchestration: validation, quality, preprocessing, model inference, result assembly |
| `backend/app/model.py` | ResNet-50 + Multi-Head Cross-Attention network definition, weight loading, inference |
| `backend/app/signal_processing.py` | Butterworth filter, derivatives, feature extraction, input validation, signal quality assessment, CSV/TXT demux |
| `backend/app/image_generation.py` | STFT spectrogram image generation, stacking for ResNet-50 |
| `backend/app/demo_model.py` | Calibrated demo estimator (used only when trained weights are absent) |
| `backend/app/sample_signal.py` | Synthetic PPG generator for demo/offline testing |
| `backend/dataset_prepare.py` | Downloads and verifies the public PPG+BP dataset |
| `backend/train_model.py` | Trains projection + MHCA + head (backbone frozen), saves weights + config + evaluation |
| `backend/tests/test_api.py` | Automated API tests |

### Frontend

| File | Responsibility |
|---|---|
| `frontend/index.html` | Responsive dashboard: upload, validation, stages, waveforms, images, results, evaluation, dietary, info, disclaimer |
| `frontend/styles.css` | White background, deep navy + cherry-red accents, rounded cards, reduced-motion support |
| `frontend/app.js` | Chart.js waveforms, API integration, validation, results rendering, animated BP Care AI Assistant |

### Model architecture

* Input: stacked spectrogram images, shape `(3 forms, 3 channels, 224, 224)`
* Backbone: ResNet-50 (ImageNet weights), frozen, stem up to global pool -> `(2048,)` per form
* Projection: `2048 -> D_MODEL (256)` per form -> tokens `(B, 3, 256)`
* Multi-Head Cross-Attention: Query = PPG-form token; Keys/Values = all three form tokens; 8 heads
* Regression head: `256 -> ReLU -> 64 -> 2` (SBP, DBP)
* Weights: `backend/models/resnet_mhca_bp.pth`

### Training

* Dataset: public PPG-based-BP-assessment (56 subjects, 100 Hz)
* Segmentation: non-overlapping 10 s windows (1000 samples), labelled with subject SBP/DBP
* Split: participant-level 70/15/15 (39 train / 8 val / 9 test) — no leakage
* Target standardisation using train-set mean/std
* Trainable: projection, MHCA, regression head (backbone frozen)
* Optimiser: Adam, lr 1e-3, batch 32, 20 epochs, MSE loss
* Best model selected by validation loss

### Evaluation (held-out test set, 108 windows / 9 unseen participants)

* SBP: MAE 14.75, RMSE 16.93, ME −2.13, std 16.79 mmHg
* DBP: MAE 11.88, RMSE 14.93, ME −5.39, std 13.92 mmHg
* Offline dataset evaluation only — not clinical validation.

### Signal quality assessment

Checks: amplitude (flatline/clipping), signal-to-noise ratio, pulse regularity (peak-to-peak interval CV), vPPG/aPPG morphology. Levels: `good` / `needs_attention` / `insufficient`.

### Reliability indicator

Plain-language indicator based on model mode (trained/demo) and signal quality. Never a fabricated percentage.

### BP Care AI Assistant

Animated avatar with blinking, typing indicator, expandable chat, suggested questions, minimize/close, responsive mobile layout, reduced-motion support. Reads SBP/DBP/signal quality from the real prediction response. Explains results, gives general dietary/food/lifestyle guidance, states limitations, avoids diagnosis and medication advice.

### Phase 1 restrictions

No hardware sensors, Arduino, ESP32, Raspberry Pi, live streaming, wearables, clinical deployment, diagnosis, or treatment recommendations.
