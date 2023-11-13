import numpy as np
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import os
import wave
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt
import math


# Previous Assignment Functions
def block_audio(x, block_size, hop_size, fs):
    x = x.astype(float)
    numberOfSamples = len(x)
    # Calculate the number of zeros to pad
    left_over_samples = numberOfSamples % hop_size
    numZeros = (block_size - left_over_samples) % block_size
    x = np.pad(x, (0, numZeros), 'constant')
    # Recalculate the number of samples after padding
    numberOfSamples = len(x)
    numberOfBlocks = (numberOfSamples - block_size) // hop_size + 1
    # Create the block indices
    block_indices = np.arange(numberOfBlocks) * hop_size
    xb = x[block_indices[:, None] + np.arange(block_size)]

    timeInSec = block_indices / fs
    return xb, timeInSec

# Hann Windowing & FFT from in-class exercise
def hann_fft(xb):
    window_size = len(xb[0])
    windows = np.hanning(window_size)
    xb = xb * windows
    fft_xb = np.abs(np.array(np.fft.rfft(xb)))
    return fft_xb


# A.1
def get_spectral_peaks(X):
    spectralPeaks = []
    # here each column is each audio block
    for block in X:
        peaks, _ = find_peaks(block)
        # sort the peaks found
        top20peaks = sorted(peaks, key=lambda p: block[p], reverse=True)[:20]
        spectralPeaks.append(top20peaks)
    return spectralPeaks


# A.2
def freq_to_midi(freq):
    return 69 + 12 * np.log2(freq / 440.0)
def calculate_cent_deviation(freq):
    midi_note = freq_to_midi(freq)
    nearest_midi_note = round(midi_note)
    cent_deviation = 100 * (midi_note - nearest_midi_note)
    return cent_deviation

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    # block audio, fft
    xb, _ = block_audio(x, blockSize, hopSize, fs)
    X = hann_fft(xb)
    # top 20 peak bins
    spectralPeaks = get_spectral_peaks(X)

    deviations = []
    # iterate to find deviations of each in cents
    for block_peaks in spectralPeaks:
        for peak in block_peaks:
            # bin to freq
            peak_freq = (peak * fs) / blockSize
            cent_deviation = calculate_cent_deviation(peak_freq)
            deviations.append(cent_deviation)

    histogram, bin_edges = np.histogram(deviations, 100)
    max_bin_index = np.argmax(histogram)
    most_common_deviation = bin_edges[max_bin_index]
    tfInHz = 440 * 2 ** (most_common_deviation / 1200)
    return tfInHz


# B.1
def generate_pc_filters(fs, spec_length, tfInHz):
    f_mid = 261.63 * (tfInHz/440)  # Starting at C4
    num_octaves = 4
    # Adjust the number of octaves to fit within the Nyquist frequency
    while f_mid * 2 ** num_octaves > fs / 2:
        num_octaves -= 1
    
    H = np.zeros((12, spec_length))
    
    for i in range(12):
        # Calculate semitone boundaries in Hz
        af_bounds = 2 ** np.array([-1/24, 1/24]) * f_mid * 2 * (spec_length - 1) / fs
        for j in range(num_octaves):
            i_bounds = [int(np.ceil(2 ** (j - 1) * af_bounds[0])), 
                        int(np.floor(2 ** (j - 1) * af_bounds[1]))]
            # Ensure that the end index is not less than the start index
            if i_bounds[1] >= i_bounds[0]:
                H[i, i_bounds[0]:i_bounds[1]+1] = 1 / (i_bounds[1] - i_bounds[0] + 1)
        
        # Increment to the next semitone
        f_mid *= 2 ** (1/12)
    
    return H
def extract_pitch_chroma(X, fs, tfInHz):
    spec_length = X.shape[0]
    H = generate_pc_filters(fs, spec_length, tfInHz)
    pitchChroma = np.dot(H, X ** 2)
    pitchChroma /= np.sum(pitchChroma, axis=0, keepdims=True)
    # avoid 0s
    pitchChroma[:, np.sum(X, axis=0) == 0] = 0

    return pitchChroma


# B.2
def detect_key(x, blockSize, hopSize, fs, bTune):
    # tuning freq
    if bTune:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
    else :
        tfInHz = 440
    # block and spectrogram
    xb, _ = block_audio(x, blockSize, hopSize, fs)
    X = hann_fft(xb).T # cloumns are blocks
    # extract pitch chroma
    pitchChroma = extract_pitch_chroma(X, fs, tfInHz)
    # Krumhansl key profiles
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    major_profiles = np.array([np.roll(t_pc[0], i) for i in range(12)])
    minor_profiles = np.array([np.roll(t_pc[1], i) for i in range(12)])

    major_profiles = major_profiles / np.linalg.norm(major_profiles, axis=1)[:, np.newaxis]
    minor_profiles = minor_profiles / np.linalg.norm(minor_profiles, axis=1)[:, np.newaxis]
    # aggregate pitch chroma across all blocks and normalize them
    agg_chroma = np.sum(pitchChroma, axis=1)
    agg_chroma /= np.linalg.norm(agg_chroma)

    min_dist_major = float('inf')
    min_dist_minor = float('inf')
    keyEstimate_major = None
    keyEstimate_minor = None
    key_names_major = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_names_minor = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

    # compare aggregated chroma w/ each major/minor profile
    for i in range(12):
        # Major
        dist_major = np.linalg.norm(agg_chroma - major_profiles[i])
        if dist_major < min_dist_major:
            min_dist_major = dist_major
            keyEstimate_major = key_names_major[i]
        # minor
        dist_minor = np.linalg.norm(agg_chroma - minor_profiles[i])
        if dist_minor < min_dist_minor:
            min_dist_minor = dist_minor
            keyEstimate_minor = key_names_minor[i]

    # find if the closest match is major or minor
    keyEstimate = keyEstimate_major if min_dist_major < min_dist_minor else keyEstimate_minor
    return keyEstimate

# C1

blockSize = 4096
hopSize = 2048

num_peaks = 20

# Set the cents scale for tuning frequency estimation
cents_scale = np.arange(0, 1200, 100)

# Set the tuning frequency correction flag
bTune = True  


def load_audio(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        fs = audio_file.getframerate()
        num_frames = audio_file.getnframes()
        x = np.frombuffer(audio_file.readframes(num_frames), dtype=np.int16)
    return x, fs

def eval_tfe(pathToAudio, pathToGT):
    # Get a list of audio file names
    audio_files = os.listdir(pathToAudio)

    total_deviation = 0.0
    num_files = 0

    for audio_file in audio_files:
        if audio_file.endswith(".wav"):
            x, fs = load_audio(os.path.join(pathToAudio, audio_file))

            # Compute the tuning frequency estimation
            tf_estimate = estimate_tuning_freq(x, blockSize=4096, hopSize=2048, fs=fs)

            # Load the ground truth tuning frequency
            gt_file = os.path.join(pathToGT, audio_file.replace(".wav", ".txt"))
            with open(gt_file, "r") as f:
                gt_tf = float(f.read().strip())

            # Calculate the absolute deviation in cents
            deviation = 1200 * np.log2(tf_estimate / gt_tf)

            total_deviation += np.abs(deviation)
            num_files += 1

    # Calculate the average absolute deviation in cents
    avgDeviation = total_deviation / num_files

    return avgDeviation


# C2

def eval_key_detection(pathToAudio, pathToGT):
    # Get a list of audio file names
    audio_files = os.listdir(pathToAudio)

    accuracy_with_tune = 0
    accuracy_without_tune = 0
    num_files = 0

    for audio_file in audio_files:
        if audio_file.endswith(".wav"):
            x, fs = load_audio(os.path.join(pathToAudio, audio_file))
            # Compute key detection with tuning frequency correction
            key_estimate_with_tune = detect_key(x, blockSize=4096, hopSize=2048, fs=fs, bTune=True)
            # Compute key detection without tuning frequency correction
            key_estimate_without_tune = detect_key(x, blockSize=4096, hopSize=2048, fs=fs, bTune=False)

            # Load the ground truth key label
            gt_file = os.path.join(pathToGT, audio_file.replace(".wav", ".txt"))
            with open(gt_file, "r") as f:
                gt_key = f.read().strip()
            gt_key = int(gt_key)
            label_to_key_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#',
                                 'a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
            gt_key = label_to_key_names[gt_key]
            print(audio_file, " estimate:", key_estimate_with_tune, "   gt:", gt_key)
            # Compare key estimates with ground truth
            if key_estimate_with_tune == gt_key:
                accuracy_with_tune += 1
            if key_estimate_without_tune == gt_key:
                accuracy_without_tune += 1

            num_files += 1

    # Calculate accuracy for key detection with and without tuning frequency correction
    accuracy = np.array([accuracy_with_tune / num_files, accuracy_without_tune / num_files])

    return accuracy


# C3

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):

    avg_deviationInCent = eval_tfe(pathToAudioTf, pathToGTTf)

    avg_accuracy = eval_key_detection(pathToAudioKey, pathToGTKey)

    return avg_accuracy, avg_deviationInCent


##----------------------------------------------- Evaluation Function Call -----------------------------------------------##

# Replace with the path to your data directory
mainDirectory = "/Users/sut2/Documents/GTcourses/MUSI6201/key_tf" 

# Set the directories for audio and ground truth data
audio_key_dir = os.path.join(mainDirectory, "key_eval/audio")
gt_key_dir = os.path.join(mainDirectory, "key_eval/GT")
audio_tf_dir = os.path.join(mainDirectory, "tuning_eval/audio")
gt_tf_dir = os.path.join(mainDirectory, "tuning_eval/GT")


avg_accuracy, avg_deviationInCent = evaluate(audio_key_dir, gt_key_dir, audio_tf_dir, gt_tf_dir)


print("Average Accuracy (with tune, without tune):", avg_accuracy)
print("Average Deviation in Cents:", avg_deviationInCent)