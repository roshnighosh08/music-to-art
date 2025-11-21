# music_to_art_generator.py

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance

# ---------- CONFIG ----------
AUDIO_PATH = "sample.wav"  # Input audio file
OUTPUT_DIR = "outputs"     # Directory to save images
DPI = 300

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1. LOAD AUDIO ----------
y, sr = librosa.load(AUDIO_PATH, sr=None)

# ---------- 2. SAVE WAVEFORM ----------
waveform_path = os.path.join(OUTPUT_DIR, "waveform.png")
plt.figure(figsize=(12, 4))
plt.plot(y, color='cyan', alpha=0.7)
plt.axis('off')
plt.tight_layout()
plt.savefig(waveform_path, dpi=DPI, bbox_inches='tight')
plt.close()

# ---------- 3. SAVE SPECTROGRAM ----------
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

spectrogram_path = os.path.join(OUTPUT_DIR, "spectrogram.png")
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
plt.axis('off')
plt.tight_layout()
plt.savefig(spectrogram_path, dpi=DPI, bbox_inches='tight')
plt.close()

# ---------- 4. COMBINE WAVEFORM + SPECTROGRAM ----------
# Open images
wave_img = Image.open(waveform_path).convert("RGBA")
spec_img = Image.open(spectrogram_path).convert("RGBA")

# Resize waveform to match spectrogram
wave_img = wave_img.resize(spec_img.size)

# Reduce waveform opacity
wave_alpha = wave_img.split()[3].point(lambda p: p * 0.5)  # 50% opacity
wave_img.putalpha(wave_alpha)

# Combine images
combined_img = Image.alpha_composite(spec_img, wave_img)
combined_path = os.path.join(OUTPUT_DIR, "combined_art.png")
combined_img.save(combined_path)

# ---------- 5. APPLY ARTISTIC FILTERS ----------
art_img = combined_img.convert("RGB")
art_img = ImageEnhance.Contrast(art_img).enhance(1.5)  # increase contrast
art_img = art_img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # edge enhancement

final_art_path = os.path.join(OUTPUT_DIR, "final_art.png")
art_img.save(final_art_path)

# ---------- DONE ----------
print("🎨 Music → Art generation complete!")
print(f"- Waveform saved at: {waveform_path}")
print(f"- Spectrogram saved at: {spectrogram_path}")
print(f"- Combined image saved at: {combined_path}")
print(f"- Stylized final art saved at: {final_art_path}")
