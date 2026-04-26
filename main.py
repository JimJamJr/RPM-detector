import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'test_audio/sample_set/5000-hold.wav'
y, sr = librosa.load(audio_path)

# controls the length of the window for the STFT. A larger n_fft provides better frequency resolution but worse time resolution.
window = 2048*4  # It's suggested to use a power of 2 for n_fft for computational efficiency.

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y, n_fft=window) # Increased n_fft for better frequency resolution.

# Convert the amplitude to decibels
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.ylim(0, 200)  # Limit the y-axis to 200 Hz
plt.tight_layout()

# get the maginitude of the STFT
magnitude = np.abs(D)

# Find the index of the maximum magnitude
freqs = np.argmax(magnitude, axis=0)

# Convert the index to frequency
frequencies = freqs * sr / (window)  # Convert index to frequency using the sampling rate and n_fft

# Plot the frequencies over time
print("Frequencies (Hz) over time:")
print(frequencies[:40])  # Print the first 40 frequencies
plt.figure(figsize=(10, 4))
plt.plot(frequencies)
plt.title('Dominant Frequency Over Time')
plt.xlabel('Time (frames)') 
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.tight_layout()
plt.show()
