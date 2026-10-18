
# Project Status and Next Steps

## Project Goal

This prototype processes a PPG signal and estimates systolic and diastolic blood pressure (SBP/DBP). It is a research prototype, not a medical device.

## Completed

- FastAPI backend and browser-based HTML/CSS/JavaScript frontend are in place.
- The backend health check and synthetic demo prediction route were tested successfully.
- The CSV upload route was tested with a generated numeric signal: 299 samples at 125 Hz (about 2.4 seconds).
- The upload response completed the signal-processing pipeline, including band-pass filtering, PPG/vPPG/aPPG generation, feature-image creation, model feature extraction/attention, and results generation.
- The frontend code includes controls for running the demo and uploading a signal file, and displays signal plots, processing stages, features, images, attention, and prediction results.
- Python dependencies from `backend/requirements.txt` are installed in the current environment.

## Current Limitations

- The CSV test used a generated signal, not data from a physical PPG sensor or a person.
- The app accepts uploaded files; direct MAX30102-to-app acquisition has not been implemented or tested.
- The test CSV was short. It verified the route and processing path, not the quality of a full-length recording.
- Trained ResNet-50 + MHCA blood-pressure weights are not present. The app therefore reports a calibrated demo estimate; this is not a validated BP measurement.
- The browser interface has not yet been verified end-to-end in a browser with an uploaded file.
- There is no collected, labeled dataset or reported model-accuracy validation yet.

## What the Student Should Do

1. Decide on and obtain the hardware: MAX30102 PPG module and an ESP32 or Arduino board. Check that the module provides raw optical samples, not only a displayed heart-rate value.
2. Wire the sensor to the board using the module's documented voltage and I2C connections.
3. Record raw PPG samples while keeping the finger still and sensor contact consistent. Aim for a 20-second recording and record the actual sampling rate.
4. Save a CSV with a header and one numeric PPG sample per row, for example:

   ```csv
   ppg
   12345
   12360
   12341
   ```

   The current parser treats every number in the file as a signal sample. Do not include numeric timestamps or BP labels in this signal CSV; keep those in a separate record for now.
5. If collecting human data, get informed consent, use anonymous participant IDs, store data securely, and follow the institution's ethics/supervisor requirements. Record a reference BP from a validated cuff at the same session, with appropriate supervision.
6. Share a sample sensor CSV and its actual sampling rate so the upload can be tested against real sensor data.

## Remaining Project Work

- Verify the browser upload workflow and charts using a CSV file.
- Implement or select a reliable way to save sensor readings from the chosen microcontroller as CSV.
- Check signal quality, missing samples, sample rate, and recording duration on real sensor recordings.
- Build a consented dataset pairing PPG recordings with reference cuff SBP/DBP labels, following supervisor and ethics guidance.
- Train the model using a documented training/validation/test split and report suitable error metrics against reference measurements.
- Load the trained weights into the app and verify that it switches from demo mode to trained mode.
- Clearly label all outputs as experimental and not for diagnosis or treatment.

## Recommended Order

1. Obtain the sensor and microcontroller, then collect one test recording.
2. Confirm the recording format and sampling rate; upload it and inspect the displayed waveforms and processing stages.
3. Only after the capture path is reliable, collect appropriately consented, cuff-labeled data.
4. Train and evaluate the model; integrate weights only after the evaluation is documented.

## Current Prediction Status

The upload route works for numeric CSV input, but a successful response only confirms that the software pipeline ran. Until trained weights and independent validation data are available, its BP numbers must be treated as demo output, not real or clinically reliable blood pressure.
