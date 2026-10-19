
# Project Status and Next Steps

## Project Goal

Software-only prototype that accepts manually supplied PPG data, estimates
systolic/diastolic blood pressure using a trained ResNet-50 + Multi-Head
Cross-Attention model, visualises signals, and provides an animated BP Care
AI Assistant for explaining results and offering safe dietary/lifestyle guidance.

## Completed (Phase 1)

- FastAPI backend + HTML/CSS/JS frontend in place (existing components preserved)
- Public dataset selected, downloaded, verified: **PPG-based-BP-assessment** —
  56 subjects, 2-min PPG at 100 Hz, reference SBP/DBP mmHg, participant IDs
  (`backend/data/ppg_bp_dataset/`, `metadata.csv`, `splits.json`)
- Signal preprocessing pipeline: input validation, missing-value/insufficient-sample
  detection, invalid-fs detection, Butterworth band-pass (0.5–5 Hz), PPG/vPPG/aPPG
- Signal quality assessment: amplitude, SNR, pulse regularity, morphology
  (good / needs_attention / insufficient)
- Genuine **Multi-Head Cross-Attention** implemented (PPG queries attend over
  vPPG/aPPG context) — replaces previous self-attention
- ResNet-50 feature extraction (ImageNet backbone, frozen)
- Model trained (backbone frozen; projection + MHCA + head trained), weights saved
  to `backend/models/resnet_mhca_bp.pth`
- Held-out test evaluation: SBP MAE 14.75 / DBP MAE 11.88 mmHg
  (`backend/models/evaluation_results.json`)
- Backend endpoints: `/api/health`, `/api/predict` (multipart file),
  `/api/predict/json` (manual signal), `/api/demo`, `/api/evaluate`,
  `/api/evaluate/full`, `/api/assistant`, `/api/dietary`
- Frontend redesigned: white background, deep navy + cherry-red accents,
  responsive dashboard, Chart.js waveforms, SBP/DBP cards, model evaluation,
  dietary guidance, limitations/disclaimer
- Animated **BP Care AI Assistant**: avatar with blinking, typing indicator,
  expandable chat, suggested questions, minimize/close, mobile layout,
  reduced-motion support; auto-opens after a valid prediction; reads real SBP/DBP
- Automated tests (`backend/tests/test_api.py`): 19/19 passing
- README, ARCHITECTURE, PROJECT_AUDIT documentation

## Current prediction status

The app runs in **trained** mode with real weights. Predictions come from the
trained ResNet-50 + MHCA regression head and are clearly labelled as AI estimates
(not confirmed medical measurements). When weights are absent it falls back to a
clearly-labelled demo calibrated estimator.

## Remaining limitations (documented)

- Offline dataset evaluation only; not clinical validation
- Single public dataset; may not generalise to other populations/devices
- MAE ~14.8 (SBP) / ~11.9 (DBP) on held-out test data — prototype-grade, not diagnostic
- No hardware sensor integration (Phase 1 restriction)
- Assistant is rule-based/educational, not a medical professional

## Demonstration instructions

1. `cd backend && pip install -r requirements.txt`
2. `python dataset_prepare.py`  (downloads dataset)
3. `python train_model.py`      (trains + saves weights + evaluation)
4. `uvicorn main:app --reload --port 8000`
5. Open http://127.0.0.1:8000/ui/
6. Upload a PPG CSV/TXT or paste samples, set fs, run estimation
7. View SBP/DBP results, waveforms, quality, evaluation; BP Care AI Assistant opens automatically
