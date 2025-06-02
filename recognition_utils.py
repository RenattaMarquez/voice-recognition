import numpy as np
import librosa
from scipy.io import wavfile
from util import *

filter_coefficient = 0.95
window_size = 320
sliding_factor = 128
h_window = np.hamming(window_size)
n_mfcc = 16

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def extract_quantized_sequence(filepath, quantizer):
    rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))
    signal = preemphasis(data, filter_coefficient)
    index = 0
    sequence = []
    while index + window_size < len(signal):
        block = h_window * signal[index:index+window_size]
        mfccs = librosa.feature.mfcc(y=block, sr=rate, n_mfcc=n_mfcc, n_fft=window_size, n_mels=20)
        for mfcc_vector in mfccs.T:
            idx, _ = quantizer.classify(mfcc_vector)
            sequence.append(idx)
        index += sliding_factor
    return sequence

def forward(obs_seq, A, B, pi):
    A = np.log(A + 1e-12)
    B = np.log(B + 1e-12)
    pi = np.log(pi + 1e-12)
    T, N = len(obs_seq), len(pi)
    if T == 0:
        return -np.inf
    alpha = np.zeros((T, N))
    alpha[0] = pi + B[:, obs_seq[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.logaddexp.reduce(alpha[t-1] + A[:, j]) + B[j, obs_seq[t]]
    return np.logaddexp.reduce(alpha[-1])

def recognize(filepath, quantizer, hmm_models):
    sequence = extract_quantized_sequence(filepath, quantizer)
    scores = {}
    for word, (A, B, pi) in hmm_models.items():
        score = forward(sequence, A, B, pi)
        scores[word] = score
    predicted_word = max(scores, key=scores.get)
    print("\nPredicciones:")
    for word, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {word}: {score:.2f}")
    print(f"\nPalabra reconocida: **{predicted_word.upper()}**")
