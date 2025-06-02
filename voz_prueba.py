import os
import numpy as np
from hmmlearn import hmm
from python_speech_features import mfcc, delta
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Enhanced Configuration
words = ['abajo', 'empieza', 'levanta', 'detente', 'izquierda', 'derecha', 'pausa', 'continua']
base_folder = 'assets'
n_mfcc = 13  # Number of MFCC coefficients
n_components = 8  # Increased number of states
n_iter = 1000  # More iterations for better convergence
covariance_type = 'diag'  # Try 'full' if you have enough data

# Enhanced feature extraction with deltas and delta-deltas
def extract_features(audio_path):
    sample_rate, signal = wavfile.read(audio_path)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    
    # Extract MFCCs with larger window and overlap
    mfcc_features = mfcc(
        signal, 
        samplerate=sample_rate, 
        numcep=n_mfcc,
        winlen=0.025,
        winstep=0.01,
        nfilt=26,
        preemph=0.97
    )
    
    # Add delta and delta-delta features
    delta_features = delta(mfcc_features, 2)
    delta_delta_features = delta(delta_features, 2)
    
    # Combine all features
    combined_features = np.hstack([mfcc_features, delta_features, delta_delta_features])
    
    return combined_features

def train_models():
    models = {}
    scalers = {}  # To store feature scalers for each word
    
    for word in words:
        print(f"Training model for: {word}")
        train_folder = os.path.join(base_folder, word, 'train')
        features = []
        
        # Collect features from all training files
        for filename in os.listdir(train_folder):
            if filename.endswith('.wav'):
                audio_path = os.path.join(train_folder, filename)
                features.append(extract_features(audio_path))
        
        # Feature normalization per word
        lengths = [len(x) for x in features]
        combined = np.vstack(features)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined)
        
        # Split back into individual sequences
        scaled_features_split = []
        current_pos = 0
        for l in lengths:
            scaled_features_split.append(scaled_features[current_pos:current_pos+l])
            current_pos += l
        
        # Create and train HMM with more components
        model = hmm.GMMHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            n_mix=3,  # Using Gaussian Mixture Model HMM
            verbose=True
        )
        
        try:
            model.fit(np.vstack(scaled_features_split), lengths=lengths)
            models[word] = model
            scalers[word] = scaler
        except Exception as e:
            print(f"Error training {word}: {str(e)}")
            models[word] = None
            scalers[word] = None
    
    return models, scalers

def evaluate_models(models, scalers):
    results = {word: {'correct': 0, 'total': 0, 'confusions': {}} for word in words}
    
    for true_word in words:
        test_folder = os.path.join(base_folder, true_word, 'test')
        
        for filename in os.listdir(test_folder):
            if filename.endswith('.wav'):
                audio_path = os.path.join(test_folder, filename)
                features = extract_features(audio_path)
                
                # Normalize features using the true_word's scaler
                if scalers[true_word] is not None:
                    features = scalers[true_word].transform(features)
                
                scores = {}
                for word, model in models.items():
                    if model is None:
                        scores[word] = -float('inf')
                        continue
                    
                    try:
                        # Score with each model
                        scores[word] = model.score(features)
                    except:
                        scores[word] = -float('inf')
                
                predicted_word = max(scores, key=scores.get)
                results[true_word]['total'] += 1
                
                if predicted_word == true_word:
                    results[true_word]['correct'] += 1
                else:
                    if predicted_word not in results[true_word]['confusions']:
                        results[true_word]['confusions'][predicted_word] = 0
                    results[true_word]['confusions'][predicted_word] += 1
    
    return results

def main():
    print("Training enhanced HMM models...")
    models, scalers = train_models()
    
    print("\nEvaluating models on test data...")
    results = evaluate_models(models, scalers)
    
    print("\nEnhanced Results:")
    for word in words:
        correct = results[word]['correct']
        total = results[word]['total']
        print(f"{word}: {correct}/{total}")
        
        if correct < total:
            print(f"  Confused with:")
            for confused_word, count in results[word]['confusions'].items():
                print(f"    {confused_word}: {count} times")

if __name__ == "__main__":
    main()