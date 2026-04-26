import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'test_audio/sample_set/4500-hold.wav'
y, sr = librosa.load(audio_path)

# controls the length of the window for the STFT. A larger n_fft provides better frequency resolution but worse time resolution.
window = 2048*8  # It's suggested to use a power of 2 for n_fft for computational efficiency.

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y, n_fft=window) # Increased n_fft for better frequency resolution.

f = librosa.fft_frequencies(sr=sr, n_fft=window)  # Get the corresponding frequencies for the STFT bins

bottom_band = 10  # Define the lower frequency limit
top_band = 200    # Define the upper frequency limit

mask = (f >= bottom_band) & (f <= top_band)  # Define a mask for frequencies between 20 Hz and 200 Hz
D_filtered = D.copy()  # Copy the original STFT to apply the filter
D_filtered[~mask, :] = 0  # Set the values outside the desired frequency range to zero

# Convert the amplitude to decibels
S_db = librosa.amplitude_to_db(np.abs(D_filtered), ref=np.max)

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

# Now we will try to estimate the fundamental frequency (f0) using a weighted approach based on the peaks in the frequency spectrum.
from scipy.signal import find_peaks

mask = (f >= bottom_band) & (f <= 100)  # Define a mask for frequencies between 20 Hz and 100 Hz # We are interested in the lower frequency range for fundamental frequency estimation
D_filtered = D.copy()  # Copy the original STFT to apply the filter
D_filtered[~mask, :] = 0  # Set the values outside the desired frequency range to zero

t = 5  # Time frame index to analyze

magnitude = np.abs(D_filtered[:, t])  # Get the magnitude of the STFT
freqs = librosa.fft_frequencies(sr=sr, n_fft=window)  # Get the corresponding frequencies for the STFT bins

# Find peaks in the frequency data
peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.04)  # Adjust height as needed
peak_freqs = freqs[peaks]
peak_magnitudes = magnitude[peaks]

# Function to use a apply weight to a possible fundamental frequency
def weight_frequency(f0, peak_freqs):
    tolerance = 0.05  # Frequency tolerance as a percentage
    weight = 0

    for peak in peak_freqs:
        if peak < f0:
            continue

        nearest_harmonic = round(peak / f0)

        if nearest_harmonic < 1:
            continue

        expected_freq = nearest_harmonic * f0
        error = abs(peak - expected_freq) / expected_freq

        if error < tolerance:
            weight += 1 / nearest_harmonic # Higher harmonics contribute less to the weight

    return weight

# Estimate the fundamental frequency (f0) using a weighted approach
estimated_f0 = None
lowest_weight = -1

for f0 in peak_freqs:
    weight = weight_frequency(f0, peak_freqs)

    if weight > lowest_weight:

        estimated_f0 = f0
        lowest_weight = weight

        corrected_f0 = f0

        for divisor in range(2, 5):  # Check for subharmonics up to the 4th harmonic
            subharmonic = f0 / divisor

            if np.any(np.abs(peak_freqs - subharmonic) < 5):  # Check if there's a peak near the subharmonic
                corrected_f0 = subharmonic  # Update the estimated f0 to the subharmonic
                break

# Print the peaks and their corresponding frequencies
print("Peaks and their corresponding frequencies:")
for peak, freq in zip(peaks, peak_freqs):
    print(f"Peak at index {peak} corresponds to frequency {freq:.2f} Hz")

# Print the estimated fundamental frequency
try:
    print(f"Estimated fundamental frequency (f0) at time frame {t}: {corrected_f0:.2f} Hz")
except TypeError:
    print(f"Estimated fundamental frequency (f0) at time frame {t}: None") # Handle the case where no valid f0 is found

# Convert the estimated fundamental frequency to RPM
rpm = frequencies * 60  # Assuming the audio is at 60 Hz (standard for audio)

plt.figure(figsize=(10, 4))
plt.plot(rpm)
plt.title('Estimated RPM Over Time')
plt.xlabel('Time (frames)')
plt.ylabel('RPM')
plt.grid()
plt.tight_layout()
plt.show()