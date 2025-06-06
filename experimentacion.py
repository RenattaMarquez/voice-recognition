# -*- coding: utf-8 -*-
"""experimentacion.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bvNgUhQ9f5Osacs6hcxALusUBWrZKLcb
"""

import glob
import os
import librosa
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import zipfile
import os
import util
from mfccVQ import VectorQuantizer

from google.colab import files
uploaded = files.upload()

with zipfile.ZipFile("assets.zip", 'r') as zip_ref:
    zip_ref.extractall()

# Verifica el contenido
os.listdir("assets")

folders = ['abajo', 'continua', 'derecha', 'detente', 'empieza', 'izquierda', 'levanta', 'pausa']
filter_coefficient = 0.95
window_size = 320
sliding_factor = 128
h_window = np.hamming(window_size)
p_size = 12
num_centroids = 256
n_states = 4
n_mfcc = 16

class VectorQuantizer:
    def __init__(self):
        self.centroids = [] # MFCC centroids

    def train(self, mfcc_vectors:np.ndarray, epsilon:float, partitions:int=256):
        self.centroids =  [mfcc_vectors.mean(0)]     # Initial centroid
        print(self.centroids[0].shape)
        while len(self.centroids) < partitions:
            self.centroids = [y for centroid in self.centroids for y in (centroid * 0.999, centroid * 1.001)]   # Centroid splitting
            dist_prev = -1
            dist_diff = -1
            while dist_diff >= epsilon or dist_diff == -1: # Centroid reposition
                clusters = [[] for _ in self.centroids] # We reset the regions
                dist_glob = 0
                for vector in mfcc_vectors:
                    index, dist = self.classify(vector)
                    clusters[index].append(vector)
                    dist_glob += dist
                if dist_prev != -1:
                    dist_diff = abs(dist_glob - dist_prev)
                dist_prev = dist_glob
                new_centroids = [np.zeros_like(centroid) for centroid in self.centroids]
                for i in range(len(clusters)):
                    if clusters[i]:
                        new_centroids[i] = np.mean(clusters[i], 0)
                    else:
                        new_centroids[i] = self.centroids[i]
                self.centroids = new_centroids

    def classify(self, vector:np.ndarray):
        index = np.argmin(np.linalg.norm(self.centroids - vector, axis=1))
        min_dist = np.linalg.norm(self.centroids - vector, axis=1)[index]
        return index, min_dist

folders = ['abajo', 'continua', 'derecha', 'detente', 'empieza', 'izquierda', 'levanta', 'pausa']
quantizer = VectorQuantizer()
vectors = []

for folder in folders:
    print("vector cuantizador folder: ", folder)
    for filepath in glob.iglob('assets/' + folder + '/train/*.wav'):
        rate, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Convert to mono if stereo
        data = data / np.max(np.abs(data))  # Normalize
        signal = util.preemphasis(data, filter_coefficient)
        index = 0
        while index + window_size < len(signal):
            block = h_window * signal[index:index+window_size]
            mfccs = librosa.feature.mfcc(y=block, sr=rate, n_mfcc=16)
            for mfcc_vector in mfccs.T:  # One vector per time step
                vectors.append(mfcc_vector)
            index += sliding_factor

vectors = np.array(vectors)
print(vectors[0].shape, vectors[0])
if len(vectors) > 0:
    quantizer.train(vectors, 0.1, 256)

def extract_quantized_sequence(filepath, quantizer):
    rate, data = wavfile.read(filepath)

    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Convert to mono
    data = data / np.max(np.abs(data))  # Normalize

    signal = preemphasis(data, filter_coefficient)
    index = 0
    sequence = []
    while index + window_size < len(signal):
        block = h_window * signal[index:index+window_size]
        mfccs = librosa.feature.mfcc(y=block, sr=rate, n_mfcc=n_mfcc)
        for mfcc_vector in mfccs.T:
            idx, _ = quantizer.classify(mfcc_vector)
            sequence.append(idx)
        index += sliding_factor
    return sequence

def initialize_hmm(N, M):
    # Matriz de transición fija (puedes ajustar esto si es necesario)
    A = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.00, 0.96, 0.03, 0.01],
        [0.00, 0.00, 0.97, 0.03],
        [0.00, 0.00, 0.01, 0.99]
    ])

    # Matriz de observación B (N x M)
    B = np.zeros((N,M))
    for i in range(N):
      for j in range(M):
        B[i, j] = 1/M


    # Vector inicial pi
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
                continue  # ignorar secuencias muy cortas

            # Forward
            alpha = np.zeros((T, N))
            alpha[0] = pi * B[:, obs_seq[0]]
            for t in range(1, T):
                for j in range(N):
                    alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs_seq[t]]

            # Backward
            beta = np.zeros((T, N))
            beta[T-1] = 1
            for t in range(T-2, -1, -1):
                for i in range(N):
                    beta[t, i] = np.sum(A[i] * B[:, obs_seq[t+1]] * beta[t+1])

            # Gamma
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = np.divide(gamma, gamma_sum, where=gamma_sum != 0)

            # Xi
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

            # Update accumulators
            A_num += xi.sum(axis=0)
            A_den += gamma[:-1].sum(axis=0)
            for t in range(T):
                B_num[:, obs_seq[t]] += gamma[t]
            B_den += gamma.sum(axis=0)
            pi_new += gamma[0]

        # Re-estimate parameters with safeguards
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.divide(A_num, A_den[:, None], where=A_den[:, None] != 0)
            B = np.divide(B_num, B_den[:, None], where=B_den[:, None] != 0)
            pi_sum = np.sum(pi_new)
            pi = pi_new / pi_sum if pi_sum > 0 else np.ones(N) / N

    return A, B, pi

def forward(obs_seq, A, B, pi):
    A = np.log(A + 1e-12)   # evitar log(0)
    B = np.log(B + 1e-12)
    pi = np.log(pi + 1e-12)

    T, N = len(obs_seq), len(pi)
    alpha = np.zeros((T, N))  # alpha[t, i] = log prob. of partial observation ending in state i

    # Inicialización
    alpha[0] = pi + B[:, obs_seq[0]]

    # Recursión
    for t in range(1, T):
        for j in range(N):
            logsum = np.logaddexp.reduce(alpha[t-1] + A[:, j])
            alpha[t, j] = logsum + B[j, obs_seq[t]]

    # Terminación: log-sum de las probabilidades finales
    log_likelihood = np.logaddexp.reduce(alpha[-1])
    return log_likelihood

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

train_sequences = {word: [] for word in folders}
for folder in folders:
    print("parte 2 folder: ", folder)
    for path in glob.iglob(f'assets/{folder}/train/*.wav'):
        seq = extract_quantized_sequence(path, quantizer)
        if seq:
            train_sequences[folder].append(seq)

hmm_models = {}
for word in folders:
    print("word en el hmm: ", word)
    A, B, pi = baum_welch(train_sequences[word], N=n_states, M=num_centroids, n_iter=1000)
    hmm_models[word] = (A, B, pi)

correct = 0
total = 0

for folder in folders:
    print(folder)
    for path in glob.iglob(f'assets/{folder}/test/*.wav'):
        obs_seq = extract_quantized_sequence(path, quantizer)
        if not obs_seq:
            continue
        scores = {word: forward(obs_seq, *hmm_models[word]) for word in folders}
        predicted = max(scores, key=scores.get)
        print(f"Archivo: {os.path.basename(path)} | Real: {folder} | Predicho: {predicted}")
        if predicted == folder:
            correct += 1
        total += 1

accuracy = 100 * correct / total
print(f"\nPrecisión total: {accuracy:.2f}%")