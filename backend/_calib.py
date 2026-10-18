from app.sample_signal import generate_ppg
from app import signal_processing as sp

for hr in (60, 72, 85, 100):
    raw = generate_ppg(duration=20.0, fs=125.0, heart_rate=hr, seed=7)
    filt, v, a = sp.compute_forms(raw, 125.0)
    f = sp.extract_features(filt, a, 125.0)
    print(f"HR set={hr}: "
          f"hr={f['heart_rate']:.1f} ai={f['augmentation_index']:.3f} "
          f"ri={f['reflection_index']:.3f} ppv={f['ppv']:.3f} "
          f"rise={f['pulse_rise_time']:.3f} peak={f['systolic_peak']:.3f}")
