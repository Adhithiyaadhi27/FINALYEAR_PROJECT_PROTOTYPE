# PROJECT AUDIT & IMPLEMENTATION PLAN
## An Intelligent Framework for Cuffless Blood Pressure Estimation (Phase 1)

---

## 1. EXISTING REPOSITORY AUDIT

### Working components (preserve & reuse)
| Component | File | Status |
|---|---|---|
| FastAPI app + CORS + static frontend | `backend/main.py` | WORKING |
| Health endpoint | `GET /api/health` | WORKING |
| PPG upload + predict endpoint | `POST /api/predict` | WORKING (CSV/TXT parse) |
| Demo synthetic PPG generator | `backend/app/sample_signal.py` | WORKING |
| Butterworth band-pass filter (0.5–5 Hz) | `backend/app/signal_processing.py` | WORKING |
| PPG / vPPG / aPPG derivatives | `backend/app/signal_processing.py` | WORKING |
| STFT spectrogram image generation | `backend/app/image_generation.py` | WORKING |
| ResNet-50 backbone + projection | `backend/app/model.py` | WORKING (builds) |
| Multi-head attention (currently SELF-attention) | `backend/app/model.py` | WORKING but NOT cross-attention |
| Regression head (256→64→2) | `backend/app/model.py` | WORKING |
| Demo calibrated BP estimator | `backend/app/demo_model.py` | WORKING (rule-based, NOT AI) |
| Pipeline orchestration + stage tracking | `backend/app/pipeline.py` | WORKING |
| Frontend (HTML/CSS/JS, canvas waveforms) | `frontend/*.*` | WORKING (dark theme) |
| Dependencies (torch, torchvision, scipy, fastapi) | `backend/requirements.txt` | INSTALLED |

### Missing / incomplete components
| # | Missing feature | Severity |
|---|---|---|
| 1 | **No trained model weights** — `backend/models/` only has `.gitkeep`. App always runs in DEMO MODE (rule-based estimator, not AI). | CRITICAL |
| 2 | **No labelled PPG+BP dataset** — nothing to train on. | CRITICAL |
| 3 | Attention is self-attention (tokens attend to themselves), not Multi-Head CROSS-Attention as spec requires. | HIGH |
| 4 | No model training or evaluation script; no MAE/RMSE/ME/std metrics. | HIGH |
| 5 | Frontend is dark-themed; spec wants white + deep navy + cherry-red. Missing most spec sections. | HIGH |
| 6 | No animated AI health assistant (BP Care AI Assistant). | HIGH |
| 7 | No BP result explanation, dietary guidance, food portion, lifestyle guidance. | HIGH |
| 8 | No signal quality assessment (good/needs attention/insufficient). | MEDIUM |
| 9 | Upload lacks column selection, sampling-frequency input, signal preview, missing-value / insufficient-sample detection. | MEDIUM |
| 10 | No `/api/evaluate`, `/api/assistant`, `/api/dietary` endpoints. | MEDIUM |
| 11 | No automated tests, no README updates, no architecture docs, no screenshots. | MEDIUM |

### Key distinction: demo vs. real inference
`demo_model.py` `predict_demo_bp()` is a deterministic physiological formula (HR + augmentation index → SBP/DBP) with pseudo-random jitter. It is **not** the trained model. The pipeline only uses the trained network head when `models/resnet_mhca_bp.pth` exists AND `torch_info.trained_weights` is true. Currently that file is absent, so every prediction is demo output.

---

## 2. DATASET SELECTION (VERIFIED)

**Selected:** `sanvsquezsz/PPG-based-BP-assessment` (GitHub public research dataset)

| Requirement | Verified |
|---|---|
| PPG waveforms | Yes — 2-min (120 s) recordings |
| Reference SBP | Yes — "Systolic BP" column (mmHg) |
| Reference DBP | Yes — "Diastolic BP" column (mmHg) |
| Sampling frequency | Yes — 100 Hz |
| Participant IDs | Yes — `PPG_subj_##` (56 subjects) |
| Research usage | Yes — public GitHub, citation requested (Argüello-Prada et al., Phys Eng Sci Med 2025) |

**Format** (`PPG_csv_info/PPG_subj_wInfo_##.csv`):
`Record#, Age, Gender, Diagnosed, Treatment, SystolicBP, DiastolicBP, HR, JNC, AHA, <12000 signal samples>`

**Download:** ZIP (3.5 MB) from GitHub; individual CSVs via raw.githubusercontent.com.

**Split strategy (participant-level, no leakage):**
- 56 subjects → Train 39 / Val 8 / Test 9 (70/15/15)
- Each subject segmented into 10 s non-overlapping windows (1000 samples at 100 Hz) → 12 windows/subject → ~672 samples total
- Windows from the same subject stay in the same split

---

## 3. IMPLEMENTATION PLAN (ORDER OF WORK)

### Phase A — Data preparation
1. Download the 56 subject CSVs into `backend/data/ppg_bp_dataset/`
2. Write `backend/dataset_prepare.py` to parse, segment, label, and save a consolidated `dataset.csv` + `splits.json`

### Phase B — Preprocessing & signal quality (enhance existing)
3. Enhance `signal_processing.py`: header-aware parsing, signal-column selection, fs validation, missing-value detection, insufficient-sample detection, signal quality assessment (good/needs attention/insufficient)
4. Enhance `pipeline.py`: attach signal-quality status, model status, uncertainty indicator, clear demo-vs-trained labelling

### Phase C — Model: genuine cross-attention + training
5. Rewrite `model.py` attention as Multi-Head CROSS-Attention (PPG form queries vPPG/aPPG forms); keep dims (2048→256, 8 heads, head→2)
6. Write `backend/train_model.py`: freeze ResNet-50 backbone, train projection+MHCA+head on spectrogram images, save `models/resnet_mhca_bp.pth` + `models/training_config.json` (target stats, hyperparams)

### Phase D — Evaluation
7. Evaluate on held-out test split: MAE, RMSE, ME, std of errors per SBP and DBP; save results to `models/evaluation_results.json`

### Phase E — Backend API
8. `main.py`: add `/api/evaluate`, `/api/assistant` (explanation + dietary), `/api/dietary`, improve `/api/predict` validation, keep `/api/health`, `/api/sample`

### Phase F — Frontend redesign
9. Redesign `index.html` + `styles.css`: white background, deep navy + cherry-red accents, rounded cards, responsive
10. Rewrite `app.js`: Chart.js waveforms, upload with column selection + fs + signal preview, validation/errors, SBP/DBP cards, model evaluation section, animated BP Care AI assistant (explanation + dietary + food/lifestyle), disclaimer

### Phase G — Testing & docs
11. `backend/tests/`: automated tests (valid CSV/TXT, missing column, empty file, invalid values, insufficient samples, invalid fs, poor quality, missing model, inference failure, successful prediction, assistant, dietary, mobile, full workflow)
12. Run tests, report actual results
13. Update `README.md`; add `ARCHITECTURE.md`; known limitations & future work
14. Capture screenshots of functioning prototype

### Phase 1 restrictions (NOT implemented)
No hardware, Arduino, ESP32, Raspberry Pi, live streaming, wearables, clinical deployment, diagnosis, or treatment recommendations.

---

## 4. DELIVERABLES CHECKLIST
1. Working software prototype
2. PPG CSV/TXT upload ✓ (existing, to be enhanced)
3. Manual signal input
4. Signal preprocessing pipeline ✓ (existing, to be enhanced)
5. PPG/vPPG/aPPG generation ✓ (existing)
6. Signal visualization ✓ (existing, to be Chart.js)
7. ResNet-50 feature extraction ✓ (existing)
8. Multi-Head Cross-Attention (to be implemented — currently self-attention)
9. Trained SBP/DBP regression model (to be trained)
10. Model evaluation results (to be produced)
11. Responsive BP estimation dashboard (to be redesigned)
12. Animated AI health assistant (new)
13. BP result explanation (new)
14. Dietary recommendations (new)
15. Food portion guidance (new)
16. Lifestyle education (new)
17. Signal quality assessment (new)
18. Error handling & testing (new)
19. README & setup (update)
20. Screenshots
21. Architecture docs (new)
22. Known limitations & future work

---

## 5. RISK NOTE
Training ResNet-50-based model on CPU with 56 subjects is feasible for a prototype (frozen backbone, small head). If training proves too slow, the fallback is to train only the projection+MHCA+head on precomputed frozen backbone features and clearly document it. Under no circumstances will a rule-based/random output be presented as a trained AI prediction; if weights are unavailable, the app will state "model training pending / demo mode."
