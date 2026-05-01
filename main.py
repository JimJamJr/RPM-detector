import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sounddevice as sd

# Allows you to pick the engine type, and applies the appropriate conversion.
conversion_factors = {
    'single-cylinder': 1,  # For a single-cylinder engine, the fundamental frequency corresponds directly to the RPM
    'twin-cylinder': 2,  # For a twin-cylinder engine, the fundamental frequency corresponds to half the RPM
    'flat-4': 2,  # For a flat-4 engine, the fundamental frequency corresponds to half the RPM
    'flat-6': 3,  # For a flat-6 engine, the fundamental frequency corresponds to one-third of the RPM
    'flat-8': 4,  # For a flat-8 engine, the fundamental frequency corresponds to one-fourth of the RPM
    'inline-4': 2,  # For an inline-4 engine, the fundamental frequency corresponds to half the RPM
    'inline-6': 3,  # For an inline-6 engine, the fundamental frequency corresponds to one-third of the RPM
    'v2': 1,       # For a V2 engine, the fundamental frequency corresponds directly to the RPM
    'v4': 2,       # For a V4 engine, the fundamental frequency corresponds to half the RPM
    'v6': 3,       # For a V6 engine, the fundamental frequency corresponds to one-third of the RPM
    'v8': 4        # For a V8 engine, the fundamental frequency corresponds to one-fourth of the RPM
}

# Load the audio file
audio_path = 'test_audio/sample_set/flat-4/start-idle.wav'
engine_type = 'flat-4'  # Change this to the appropriate engine type

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

# Second estimated fundamental frequency (f0) using a harmonic product spectrum approach
def harmonic_product_spectrum(magnitude, freqs, max_harmonic=5):
    hps = magnitude.copy()

    for harmonic in range(2, max_harmonic + 1):
        downsampled = librosa.resample(magnitude, orig_sr=len(magnitude), target_sr=len(magnitude) // harmonic)
        hps[:len(downsampled)] *= downsampled

    peak_index = np.argmax(hps)
    estimated_f0 = freqs[peak_index]

    return estimated_f0

# combine the two f0 estimations by applying the weighted approach to the HPS estimation as well
def refine_f0(f0, peak_freqs):
    lowest_weight = 0.5  # Define a threshold for the lowest acceptable weight, adjust as needed based on experimentation
    weight = weight_frequency(f0, peak_freqs)

    if weight > lowest_weight:
        return f0

    for divisor in range(2, 5):  # Check for subharmonics up to the 4th harmonic
        subharmonic = f0 / divisor

        if np.any(np.abs(peak_freqs - subharmonic) < 5):  # Check if there's a peak near the subharmonic # Adjust the tolerance as needed
            return subharmonic  # Update the estimated f0 to the subharmonic

    return f0

# We will now attempt to smooth the RPM estimation by assuming that the RPM changes gradually over time.
def rpm_smoothing(rpm, previous_rpm=None, alpha=0.85): # Alpha is the smoothing factor, adjust it as needed for more or less smoothing. A higher alpha gives more weight to the current RPM, while a lower alpha gives more weight to the previous RPM.
    if previous_rpm is None:
        print("No previous RPM available, using current RPM without smoothing")  # Print a message if there is no previous RPM available for smoothing
        smoothed_rpm = rpm
    else:
        smoothed_rpm = alpha * rpm + (1 - alpha) * previous_rpm

    return smoothed_rpm

# Apply fundamental freuency isolation to each time frame to get a more accurate RPM estimation over time
def refine_rpm_over_time(D, freqs):

    magnitude = np.abs(D[:, 1])  # Get the initial magnitude

    rpm_values = []
    previous_f0 = None
    previous_rpm = None

    for t in range(D.shape[1]):  # Loop through each time frame in the STFT

        max_change = 150 # Limit the maximum change in RPM between frames to 50 RPM, adjust as needed based on the expected acceleration of the engine

        magnitude = np.abs(D[:, t])  # Get the magnitude of the STFT for the current time frame

        f0 = harmonic_product_spectrum(magnitude, freqs)  # Estimate the fundamental frequency for the current time frame

        # Get the peak frequencies for the current time frame, adjust prominence as needed
        peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.04)  # Adjust height as needed
        try:
            peak_freqs = freqs[peaks]
        except IndexError:
            peak_freqs = None

        if peak_freqs is None or len(peak_freqs) == 0:
            refined_f0 = f0  # Refine the estimated f0 using the weighted approach
        else:
            refined_f0 = refine_f0(f0, peak_freqs)  # Refine the estimated f0 using the weighted approach

        if previous_f0 is None:
            previous_f0 = refined_f0
            previous_rpm = previous_f0 * 60 / conversion_factors[engine_type]  # Convert the initial f0 to RPM for the first frame

        ratios = [1,2,3,0.5,1/3]

        tolerance = max(5, previous_f0 * 0.02)  # Set a tolerance for validating the current f0, adjust as needed based on experimentation and the expected variability of the engine sound

        valid = any(abs(refined_f0-previous_f0 * ratio) < tolerance for ratio in ratios)  # Check if the current f0 is within a reasonable range of the previous f0, adjust tolerance as needed
        if not valid:
            refined_f0 = previous_f0 * 0.9 + refined_f0 * 0.1  # If the current f0 is not valid, use the previous f0 instead, but apply some smoothing to allow for gradual changes

        max_delta_f0 = 15  # Limit the maximum change in f0 between frames to 20 Hz, adjust as needed based on the expected acceleration of the engine
        if abs(refined_f0 - previous_f0) > max_delta_f0:
            refined_f0 = previous_f0 + np.sign(refined_f0 - previous_f0) * max_delta_f0  # Limit the change in f0 to prevent unrealistic jumps

        current_rpm = refined_f0 * 60 / conversion_factors[engine_type]  # Convert the refined f0 to RPM

        # Limit the change in RPM to prevent unrealistic jumps
        if abs(current_rpm - previous_rpm) > max_change:
            current_rpm = previous_rpm + np.sign(current_rpm - previous_rpm) * max_change
        
        smoothed_rpm = rpm_smoothing(current_rpm, previous_rpm, alpha=0.4)  # Smooth the RPM estimation, adjust alpha as needed for more or less smoothing
        rpm_values.append(smoothed_rpm)
        previous_rpm = smoothed_rpm  # Update the previous RPM for the next iteration
        previous_f0 = refined_f0  # Update the previous f0 for the next iteration
    return rpm_values

# ----------------------------------------------------------------------------------
# Runs the Real-Time estimation of RPM over time

def exists_near(value, set, tolerance):
    return any(abs(c - value) < tolerance for c in set)

# selects the most likely fundamental peak in the range of lags
def select_peaks(autocorr, search_range, min_lag, previous_lag):
    global rpm_estimates
    threshold = 0.3

    peaks, properties = find_peaks(search_range, prominence=threshold) # Extract peaks

    peaks = peaks + min_lag # Realign peak indices

    strong_peaks = np.sort(peaks) # The most prominent peaks in order of highest frequency to lowest
    candidates = set(strong_peaks) # For easy look-up of harmonics
    strengths = autocorr[strong_peaks] # the strengths of all the strong peaks
    supports = []
    penalties = []
    fundamentals = []
    continuities = []

    tolerance = 20
    scores = []
    for peak in strong_peaks:
        tolerance = (int)(0.1 * peak)
        # take the confidence of the autocorrelation into account
        strength = abs(autocorr[peak])

        # if there are harmonics above, award, if there are harmonics below, penalise
        harmonic_support = 1
        harmonic_penalty = 1
        for divisor in range(2,5):
            if exists_near(peak / divisor, candidates, tolerance):
                harmonic_penalty *= 2
            if exists_near(peak * divisor, candidates, tolerance):
                harmonic_support *= 2
        
        supports.append(harmonic_support)
        penalties.append(harmonic_penalty)

        # reward higher lags (lower frequencies)
        fundamental = peak
        fundamentals.append(fundamental)

        # rewards more temporally continuous lags
        if previous_lag is not None:
            continuity = np.exp(-abs(peak - previous_lag) / (300))
            if abs(peak - previous_lag) > 0.5 * previous_lag:
                continuity = 1  # ignore continuity
        else:
            continuity = 1

        continuities.append(continuity)

        # Combine attributes to form a score
        score = fundamental * continuity * harmonic_support / harmonic_penalty
        scores.append(score)

    if len(scores) != 0:
        lag = strong_peaks[np.argmax(scores)]
        confidence = np.max(scores)

        print(f"Strong peaks: {strong_peaks}")
        print(f"Strengths: {strengths}")
        print(f"Fundamentals: {fundamentals}")
        print(f"Continuities: {continuities}")
        print(f"Harmonic Supports: {supports}")
        print(f"Harmonic penalties: {penalties}")

        print(f"Scores: {scores}")
        print(f"Using peak {lag}")

        frequency = (sr/ lag)
        print(f"Raw frequency estimate: {frequency}")

        rpm_estimate = frequency * 60 / conversion_factors[engine_type]
        rpm_estimates.append(rpm_estimate)
        print(f"Raw rpm estimation: {rpm_estimate}")

        return lag, confidence
    else:
        print("No strong peaks")
        return None, 0




# Returns the waveform data from the audio stream, and applies the same processing steps as the offline analysis to estimate the RPM in real-time
def get_signal(indata):
    global buffer
    buffer = np.concatenate((buffer, indata[:, 0]))  # Append the new audio data to the buffer

    if len(buffer) >= window:  # Check if the buffer has enough data for processing
        signal = buffer[:window]  # Get the current window of audio data for processing
        buffer = buffer[blocksize:]  # Remove the processed data from the buffer, using the same block size as the hop length for consistency
        return signal
    else:
        return None  # Return None if there is not enough data in the buffer for processing

# Use auto-correlation to estimate the fundamental frequency from the audio signal, which can be used to calculate the RPM in real-time
def estimate_f0(signal):
    global previous_lag

    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    min_lag = int(sr / 200)  # Minimum lag corresponding to the maximum expected frequency (10 Hz)
    max_lag = int(sr / 10)   # Maximum lag corresponding to the minimum expected frequency (200 Hz)

    search_range = autocorr[min_lag:max_lag]  # Limit the search range for peaks to the expected frequency range
    if len(search_range) == 0:
        print("No valid lags in the search range for f0 estimation")  # Print a message if there are no valid lags in the search range
        return None, 0  # Return None and zero confidence if there are no valid lags in the search range

    peak_lag, confidence = select_peaks(autocorr, search_range, min_lag, previous_lag)  # Find the lag of the peak in the autocorrelation
    
    previous_lag = peak_lag

    if peak_lag is None:
        return None, 0

    f0 = sr / peak_lag  # Convert the lag to frequency

    return f0, confidence

# validate the estimated f0 by checking the confidence of the estimation, the harmonic structure of the frequency and the consistency of the estimation over time, similar to the approach used in the offline analysis
def validate_f0(f0, confidence, previous_f0):
    if confidence < 0.5:  # Check if the confidence of the estimation is above a certain threshold, adjust as needed based on experimentation
        print("Low confidence in f0 estimation")  # Print a message if the confidence is low
        return False

    if previous_f0 is not None:
        if abs(f0 - previous_f0) > 20:  # Check if the estimated f0 is within a reasonable range of the previous f0, adjust as needed based on experimentation and the expected variability of the engine sound
            print("f0 estimation is inconsistent with previous values")  # Print a message if the estimation is inconsistent
            return False

    return True

# Update the RPM estimation in real-time by applying the same conversion factors and smoothing techniques as the offline analysis, while also ensuring that the estimation is validated and consistent over time
def update_rpm_estimation(f0):
    global previous_f0, previous_rpm

    if f0 is None:
        return previous_rpm  # If the estimated f0 is None, return the previous RPM estimation

    current_rpm = f0 * 60 / conversion_factors[engine_type]  # Convert the estimated f0 to RPM
    print(f"Current RPM: {current_rpm}")

    # Limit the change in RPM to prevent unrealistic jumps
    max_change = 150  # Adjust as needed based on the expected acceleration of the engine
    if previous_rpm is not None and abs(current_rpm - previous_rpm) > max_change:
        current_rpm = previous_rpm + np.sign(current_rpm - previous_rpm) * max_change

    smoothed_rpm = rpm_smoothing(current_rpm, previous_rpm, alpha=0.8)  # Smooth the RPM estimation, adjust alpha as needed for more or less smoothing

    if previous_f0 is not None and abs(f0 - previous_f0) < 20:
        previous_f0 = f0  # Update the previous f0 for the next iteration
    previous_rpm = smoothed_rpm  # Update the previous RPM for the next iteration

    return smoothed_rpm

# Define a callback function to process the audio data in real-time
def callback(indata, frames, time, status):
    global buffer
    global previous_f0, previous_rpm

    signal = get_signal(indata)  # Get the current window of audio data for processing
    if signal is not None:
        f0, confidence = estimate_f0(signal)  # Estimate the fundamental frequency from the audio signal
        if validate_f0(f0, confidence, previous_f0):  # Validate the estimated f0, replace previous_f0 with actual previous f0 value for consistency
            rpm = update_rpm_estimation(f0)  # Update the RPM estimation, replace previous_f0 and previous_rpm with actual values for consistency
            print(f"Estimated RPM: {rpm:.2f}\n")  # Print the estimated RPM in real-time
            return rpm  # Return the estimated RPM from the callback function, you can modify this to send the RPM value to a display or another part of your application as needed

        else:
            print("Invalid f0 estimation, skipping RPM update\n")  # Print a message if the f0 estimation is not valid
    else:
        print("Not enough data in buffer for processing\n")  # Print a message if there is not enough data in the buffer for processing


# Below is the test code to run the real-time estimation on a test audio file, you can replace the callback function with the actual processing code to estimate the RPM in real-time from the audio stream.

sr = 44100  # Sample rate for real-time audio processing
window = 2048  # Window size for real-time audio processing, using the same window

# === LOAD FILE ===
audio, sr = librosa.load(audio_path, sr=sr, mono=True)

blocksize = window // 2  # same as your stream
pointer = blocksize  # Start processing from the first block of audio data

previous_f0 = None  # Initialize previous_f0 for validation
previous_rpm = None  # Initialize previous_rpm for smoothing
previous_lag = None # Initialize previous_lag for peak selection

# Get the autocorrelation of the first chunk of audio data and display the graph to visualize the peaks and the estimated fundamental frequency

def autocorrelation(signal):
    
    correlation = np.correlate(signal, signal, mode='full')
    correlation = correlation[len(correlation)//2:]

    max_lag = int(sr / 200)  # Maximum lag corresponding to the minimum expected frequency (10 Hz)
    min_lag = int(sr / 10)  # Minimum lag corresponding to the maximum expected frequency (200 Hz)
    correlation[min_lag:max_lag] = 0  # Set the values outside the desired frequency range to zero

    # convert to frequency
    freqs = sr / np.arange(1, len(correlation) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, correlation)
    plt.title(f'Frequency estimations of audio')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('autocorrelation')
    plt.xlim(0, 220)  # Limit the x-axis to the expected frequency range
    plt.grid()

    return correlation


# === LOAD FILE ===
audio, sr = librosa.load(audio_path, sr=None, mono=True)

window = 2048  # Window size for real-time audio processing, using the same window size as the offline analysis for consistency
sr = 44100  # Sample rate for real-time audio processing

buffer = np.zeros(window)  # Initialize a buffer to hold the audio data for processing
blocksize = window // 2  # same as your stream
pointer = blocksize  # Start processing from the first block of audio data

previous_f0 = None  # Initialize previous_f0 for validation
previous_rpm = None  # Initialize previous_rpm for smoothing

# Split the audio into blocks for processing
blocks = [audio[i:i + blocksize] for i in range(0, len(audio), blocksize)]

rpm = []
rpm_estimates = []

for block in blocks:
    if len(block) < blocksize:
        block = np.pad(block, (0, blocksize - len(block)))  # Pad the last block if it's shorter than the blocksize

    indata = block.reshape(-1, 1)  # Reshape to match InputStream format
    rpm.append(callback(indata, blocksize, None, None))  # Call the callback function to process the audio data

    pointer += blocksize  # Move the pointer to the next block of audio data

plt.figure(figsize=(10, 4))
plt.plot(rpm)
plt.title(f'RPM over time')
plt.xlabel('Time (blocks)') 
plt.ylabel('Estimated RPM')
plt.ylim(0,6000)
plt.grid()

plt.figure(figsize=(10, 4))
plt.plot(rpm_estimates)
plt.title(f'Estimated RPM over time')
plt.xlabel('Time (blocks)') 
plt.ylabel('Estimated RPM')
plt.ylim(0,6000)
plt.grid()

plt.show()