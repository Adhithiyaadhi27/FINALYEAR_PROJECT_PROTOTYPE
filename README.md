# Cuffless Blood Pressure Estimation (Prototype)

Backend: FastAPI + PyTorch (ResNet-50 + MHCA)  
Frontend: Plain HTML/CSS/JavaScript

## Run backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open:

- UI: http://127.0.0.1:8000/ui/
- API health: http://127.0.0.1:8000/api/health

## Run self-test

```bash
cd backend
python _selftest.py
```

## Notes

- If `models/resnet_mhca_bp.pth` is missing, the app runs in **DEMO MODE** and uses
  a calibrated PPG-feature estimator. The neural network still runs for feature
  extraction and attention visualisation.
- The API supports optional uploaded signal files (CSV/TXT) through `/api/predict`.
- This is a research/prototype application, not a medical device.
