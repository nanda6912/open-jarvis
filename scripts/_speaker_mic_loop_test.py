"""Temporary speaker->mic loopback test: play 'hello jarvis' fixture
through the default output while the production KWS stack listens.

Not part of the deliverable test suite; used for physical validation
when a human voice is unavailable.
"""
import sys, time
import numpy as np
import sounddevice as sd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.voice.kws_benchmark import load_wav_16k
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager
from src.voice.wake_word_detection_service import WakeWordDetectionService
from src.voice.wake_word_engine import create_engine

audio, _ = load_wav_16k(Path('tests/fixtures/commands/hello_jarvis.wav'))
audio = np.concatenate([audio, np.zeros(16000, dtype=np.float32)])

mgr = AudioDeviceManager()
cap = AudioCaptureService(device_manager=mgr)
eng = create_engine()
hits = []
det = WakeWordDetectionService(
    cap, eng, threshold=0.25, cooldown_seconds=1.0,
    on_detected=lambda kw: hits.append((time.time(), kw)))

cap.start()  # production auto resolution
det.start()
print('capture:', cap.device.name if cap.device else None,
      cap.sample_rate, 'Hz', cap.input_channels, 'ch')

try:
    for trial in range(4):
        time.sleep(2.0)
        print(f'--- playing trial {trial+1} ---', flush=True)
        sd.play(audio, samplerate=16000)
        sd.wait()
        time.sleep(2.5)
        st = det.stats()
        print(f"  det={st.get('detections')} proc={st.get('frames_processed')} "
              f"q={st.get('queue_depth')} kws_rms={st.get('kws_input_rms')}",
              flush=True)
finally:
    det.stop()
    cap.stop()
    eng.close()

print('HITS:', hits)
print('windows:', len(det.activity_windows))
for w in det.activity_windows:
    print(' ', {k: w.get(k) for k in ('keyword_detected', 'rescored',
          'detection_method', 'rescore_succeeded', 'audio_seconds',
          'rms_max', 'peak_max')})
print('frames_dropped:', cap.stats().get('frames_dropped'))
print('overflow:', cap.stats().get('overflow_callbacks'))
print('clipped:', cap.stats().get('clipped_frames'))
print('inference_ms_ema:', getattr(eng, 'inference_ms_ema', None))
print('engine inferences:', getattr(eng, 'inferences', None))
