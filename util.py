import numpy as np

class VectorQuantizer:
    def __init__(self):
        self.centroids = []

    def train(self, mfcc_vectors, epsilon, partitions=256):
        self.centroids = [mfcc_vectors.mean(0)]
        while len(self.centroids) < partitions:
            self.centroids = [y for centroid in self.centroids for y in (centroid * 0.999, centroid * 1.001)]
            dist_prev = -1
            dist_diff = -1
            while dist_diff >= epsilon or dist_diff == -1:
                clusters = [[] for _ in self.centroids]
                dist_glob = 0
                for vector in mfcc_vectors:
                    index, dist = self.classify(vector)
                    clusters[index].append(vector)
                    dist_glob += dist
                if dist_prev != -1:
                    dist_diff = abs(dist_glob - dist_prev)
                dist_prev = dist_glob
                new_centroids = [np.mean(c, 0) if c else self.centroids[i] for i, c in enumerate(clusters)]
                self.centroids = new_centroids

    def classify(self, vector):
        index = np.argmin(np.linalg.norm(self.centroids - vector, axis=1))
        min_dist = np.linalg.norm(self.centroids - vector, axis=1)[index]
        return index, min_dist

def preemphasis(signal, coeff=0.95):
   return np.append(signal[0], signal[1:] - coeff * signal[:-1])


#def preemphasis(signal: np.ndarray, coefficient: float):
    #return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

def autocorrelation(x: np.ndarray, n: int = 0) -> float:
    corr = 0
    j = n
    while j < x.shape[0]:
        corr += x[j-n] * x[j]
        j += 1
    return corr

def AutocorrelationVector(x: np.ndarray, p: int = 0):
    vector = []
    for i in range(p+1):
        vector.append(autocorrelation(x, i))
    return vector

def ShortAutocorrelation(x: np.ndarray, p: int = 0):
    vector = []
    for i in range(p+1):
        vector.append(autocorrelation(x, i))
    return vector

def ItakuraSaito(r_a: np.ndarray, R: np.ndarray) -> float:
    is_dist = R[0] * r_a[0]
    for i in range(1, len(r_a)):
        is_dist += 2*(R[i] * r_a[i])
    return is_dist