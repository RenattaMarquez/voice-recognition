import os
import time
import numpy as np
import sounddevice as sd
import keyboard
import scipy.io.wavfile as wav
from hmmlearn import hmm
from python_speech_features import mfcc, delta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Configuración
words = ['abajo', 'empieza', 'levanta', 'detente', 'izquierda', 'derecha', 'pausa', 'continua']
base_folder = 'assets_multi'
sample_rate = 16000
n_mfcc = 13
n_components = 8
n_iter = 500
covariance_type = 'diag'
threshold_silence = 0.02  # Umbral para detectar silencios

# Función para recortar silencios al inicio y final
def trim_silence(signal, threshold=threshold_silence):
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Convertir a mono si es estéreo
    
    # Normalizar señal a rango [-1, 1]
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    elif signal.dtype == np.float32:
        signal = signal.copy()
    else:
        signal = signal.astype(np.float32) / np.max(np.abs(signal))
    
    # Calcular amplitud absoluta
    abs_signal = np.abs(signal)
    
    # Encontrar el primer punto donde la amplitud supera el umbral
    start = 0
    for i in range(len(abs_signal)):
        if abs_signal[i] > threshold:
            start = max(0, i - 100)  
            break
    
    # Encontrar el último punto donde la amplitud supera el umbral
    end = len(abs_signal)
    for i in range(len(abs_signal)-1, 0, -1):
        if abs_signal[i] > threshold:
            end = min(len(abs_signal), i + 100) 
            break
    
    return signal[start:end]

# Función para grabar con barra espaciadora
def record_audio_on_space(output_filename='grabacion.wav', duration=2):
    print("Presiona la BARRA ESPACIADORA para grabar...")   
    while True:
        if keyboard.is_pressed('space'):
            print("Grabando...")
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            
            # Aplicar recorte de silencios
            trimmed_recording = trim_silence(recording)
            
            # Guardar la versión recortada
            wav.write(output_filename, sample_rate, (trimmed_recording * 32767).astype(np.int16))
            print(f"Grabación guardada en: {output_filename} (duración: {len(trimmed_recording)/sample_rate:.2f}s)")
            break
        time.sleep(0.1)

# Extracción de características MFCC + delta + delta-delta
def extract_features(audio_path):
    sr, signal = wav.read(audio_path)
    
    # Aplicar recorte de silencios también al cargar archivos
    signal = trim_silence(signal)
    
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    
    mfcc_features = mfcc(
        signal,
        samplerate=sr,
        numcep=n_mfcc,
        winlen=0.025,
        winstep=0.01,
        nfilt=26,
        preemph=0.95
    )
    
    delta_features = delta(mfcc_features, 2)
    delta_delta_features = delta(delta_features, 2)
    
    combined = np.hstack([mfcc_features, delta_features, delta_delta_features])
    return combined

# Entrenamiento de modelos HMM
def train_models():
    models = {}
    scalers = {}
    for word in words:
        print(f"Entrenando modelo para: {word}")
        train_folder = os.path.join(base_folder, word, 'train')
        features = []

        for file in os.listdir(train_folder):
            if file.endswith('.wav'):
                path = os.path.join(train_folder, file)
                features.append(extract_features(path))
        
        lengths = [len(x) for x in features]
        all_features = np.vstack(features)
        scaler = StandardScaler()
        all_scaled = scaler.fit_transform(all_features)

        split_scaled = []
        pos = 0
        for l in lengths:
            split_scaled.append(all_scaled[pos:pos+l])
            pos += l

        model = hmm.GMMHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            n_mix=3,
            verbose=False
        )
        try:
            model.fit(np.vstack(split_scaled), lengths)
            models[word] = model
            scalers[word] = scaler
        except Exception as e:
            print(f"Error entrenando {word}: {e}")
            models[word] = None
            scalers[word] = None
   
    return models, scalers

# Reconocimiento de palabra desde archivo
def recognize_from_file(filepath, models, scalers):
    features = extract_features(filepath)
    scores = {}

    for word, model in models.items():
        if model is None:
            scores[word] = -float('inf')
            continue
        try:
            features_scaled = scalers[word].transform(features)
            score = model.score(features_scaled)
            scores[word] = score
        except:
            scores[word] = -float('inf')

    best_word = max(scores, key=scores.get)
    print("\nResultados de reconocimiento:")
    for w, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {w}: {s:.2f}")
    print(f"\n Palabra reconocida: **{best_word.upper()}**")

# Programa principal      
def main():
    print("Entrenando modelos...")
    models, scalers = train_models()
    print("\nListo para grabar y reconocer.")
    
    while True:
        record_audio_on_space("grabacion.wav", duration=2)
        recognize_from_file("grabacion.wav", models, scalers)
        print("\nPresiona Q para salir o cualquier tecla para otra grabación.")
        if keyboard.read_key() == 'q':
            print("Saliendo...")
            break

if __name__ == "__main__":
    main()