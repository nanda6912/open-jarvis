"""Per-channel mic probe: records each input channel separately so we
can verify which channel carries intelligible speech on the Intel SST
4-channel array. Manual diagnostic only; nothing persisted unless
--save is passed."""
import argparse, sys, time
import numpy as np
import sounddevice as sd
import wave

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, default=2)
p.add_argument("--seconds", type=float, default=15.0)
p.add_argument("--rate", type=int, default=16000)
p.add_argument("--save", default=None)
a = p.parse_args()

print("Recording %.0fs on device %d — say 'Hello Jarvis' clearly 2-3 times." % (a.seconds, a.device))
chunks = []
def cb(indata, frames, t, status):
    chunks.append(indata.copy())

with sd.InputStream(device=a.device, samplerate=a.rate, channels=None,
                    dtype="float32", blocksize=1600, callback=cb):
    sd.sleep(int(a.seconds * 1000))

audio = np.concatenate(chunks, axis=0)  # (n, ch)
nch = audio.shape[1]
print("captured", audio.shape, "channels:", nch)
for c in range(nch):
    ch = audio[:, c]
    rms = float(np.sqrt(np.mean(ch**2)))
    peak = float(np.max(np.abs(ch)))
    # per-second rms profile
    prof = [float(np.sqrt(np.mean(s**2))) for s in
            np.array_split(ch, max(1, int(len(ch)/a.rate)))]
    print(f"ch{c}: rms={rms:.5f} peak={peak:.3f} max1s={max(prof):.4f}")

if a.save:
    for c in range(nch):
        path = a.save.replace(".wav", f"_ch{c}.wav")
        pcm = (np.clip(audio[:, c], -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(a.rate)
            w.writeframes(pcm.tobytes())
        print("saved", path)
