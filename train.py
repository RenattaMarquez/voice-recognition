import glob
import os
import numpy as np
import librosa
from scipy.io import wavfile
import pickle
import util  # Asegúrate de tener util.py en el mismo directorio o ruta de importación
from util import *

filter_coefficient = 0.95
window_size = 320
sliding_factor = 128
h_window = np.hamming(window_size)
n_mfcc = 16
num_centroids = 256
n_states = 4

folders = ['continua', 'derecha', 'detente', 'empieza', 'izquierda', 'levanta', 'pausa']

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

def initialize_hmm(N, M):
    A = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.00, 0.96, 0.03, 0.01],
        [0.00, 0.00, 0.97, 0.03],
        [0.00, 0.00, 0.01, 0.99]
    ])
    B = np.ones((N, M)) / M
    pi = np.zeros(N)
    pi[0] = 1.0
    return A, B, pi

def baum_welch(sequences, N, M, n_iter=10):
    A, B, pi = initialize_hmm(N, M)
    for _ in range(n_iter):
        A_num, A_den = np.zeros_like(A), np.zeros(N)
        B_num, B_den = np.zeros_like(B), np.zeros(N)
        pi_new = np.zeros(N)
        for obs_seq in sequences:
            T = len(obs_seq)
            if T < 2:
                continue
            alpha = np.zeros((T, N))
            alpha[0] = pi * B[:, obs_seq[0]]
            for t in range(1, T):
                for j in range(N):
                    alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs_seq[t]]
            beta = np.zeros((T, N))
            beta[T-1] = 1
            for t in range(T-2, -1, -1):
                for i in range(N):
                    beta[t, i] = np.sum(A[i] * B[:, obs_seq[t+1]] * beta[t+1])
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = np.divide(gamma, gamma_sum, where=gamma_sum != 0)
            xi = np.zeros((T-1, N, N))
            for t in range(T-1):
                denom = np.sum(alpha[t][:, None] * A * B[:, obs_seq[t+1]] * beta[t+1])
                if denom == 0 or np.isnan(denom):
                    continue
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = (
                            alpha[t, i] * A[i, j] * B[j, obs_seq[t+1]] * beta[t+1, j] / denom
                        )
            A_num += xi.sum(axis=0)
            A_den += gamma[:-1].sum(axis=0)
            for t in range(T):
                B_num[:, obs_seq[t]] += gamma[t]
            B_den += gamma.sum(axis=0)
            pi_new += gamma[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.divide(A_num, A_den[:, None], where=A_den[:, None] != 0)
            B = np.divide(B_num, B_den[:, None], where=B_den[:, None] != 0)
            pi = pi_new / np.sum(pi_new) if np.sum(pi_new) > 0 else np.ones(N) / N
    return A, B, pi

def forward(obs_seq, A, B, pi):
    A = np.log(A + 1e-12)
    B = np.log(B + 1e-12)
    pi = np.log(pi + 1e-12)
    T, N = len(obs_seq), len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi + B[:, obs_seq[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.logaddexp.reduce(alpha[t-1] + A[:, j]) + B[j, obs_seq[t]]
    return np.logaddexp.reduce(alpha[-1])

def train_all():
    quantizer = VectorQuantizer()
    vectors = []
    for folder in folders:
        for filepath in glob.iglob(f'assets/{folder}/train/*.wav'):
            rate, data = wavfile.read(filepath)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data = data / np.max(np.abs(data))
            signal = util.preemphasis(data, filter_coefficient)
            index = 0
            while index + window_size < len(signal):
                block = h_window * signal[index:index+window_size]
                mfccs = librosa.feature.mfcc(y=block, sr=rate, n_mfcc=n_mfcc, n_fft=window_size, n_mels=20)
                vectors.extend(mfccs.T)
                index += sliding_factor
    quantizer.train(np.array(vectors), 0.1, num_centroids)

    train_sequences = {word: [] for word in folders}
    for folder in folders:
        for path in glob.iglob(f'assets/{folder}/train/*.wav'):
            seq = extract_quantized_sequence(path, quantizer)
            if seq:
                train_sequences[folder].append(seq)

    hmm_models = {}
    for word in folders:
        A, B, pi = baum_welch(train_sequences[word], N=n_states, M=num_centroids, n_iter=1000)
        hmm_models[word] = (A, B, pi)

    return quantizer, hmm_models

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    quantizer, hmm_models = train_all()
    with open("models/quantizer.pkl", "wb") as f:
        pickle.dump(quantizer, f)
    with open("models/hmm_models.pkl", "wb") as f:
        pickle.dump(hmm_models, f)
    print(" Modelos entrenados y guardados en 'models/'.")
