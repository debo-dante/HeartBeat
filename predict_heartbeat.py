import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def load_model_and_classes(model_path, classes_path):
    """Load the trained model and label classes"""
    model = tf.keras.models.load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True)
    return model, classes

def extract_features(file_path):
    """Extract enhanced features from audio file"""
    try:
        # Load audio
        x, sr = librosa.load(file_path, sr=22050, duration=5.0)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=x, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral features
        spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        # Zero-crossing rate
        zero_crossings = librosa.feature.zero_crossing_rate(y=x)
        zero_crossings_mean = np.mean(zero_crossings)

        # Concatenate all features
        features = np.concatenate([
            mfccs_mean, mfccs_std, chroma_mean, spectral_contrast_mean, [zero_crossings_mean]
        ])

        # Ensure consistent feature length
        expected_length = 60
        if len(features) < expected_length:
            features = np.pad(features, (0, expected_length - len(features)), mode='constant')
        elif len(features) > expected_length:
            features = features[:expected_length]
            
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_heartbeat(audio_file_path, model_path='/Users/vioborah/HeartBeat/heartbeat_model.h5', 
                     classes_path='/Users/vioborah/HeartBeat/label_classes.npy'):
    """Predict heartbeat class for a given audio file"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    if not os.path.exists(classes_path):
        print(f"Classes file not found at {classes_path}")
        return None
    
    # Load model and classes
    model, classes = load_model_and_classes(model_path, classes_path)
    
    # Extract features
    features = extract_features(audio_file_path)
    if features is None:
        return None
    
    # Make prediction
    features = features.reshape(1, -1)
    prediction = model.predict(features, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(prediction)
    predicted_class = classes[predicted_class_idx]
    confidence = np.max(prediction)
    
    # Get all class probabilities
    class_probabilities = {classes[i]: prediction[0][i] for i in range(len(classes))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': class_probabilities
    }

import random

def select_random_files(directory, count=10):
    """Select random WAV files from a directory"""
    if not os.path.exists(directory):
        return []
    all_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    return random.sample(all_files, min(count, len(all_files)))

def main():
    """Example usage"""
    data_path = '/Users/vioborah/HeartBeat'
    set_a_dir = os.path.join(data_path, 'set_a')
    set_b_dir = os.path.join(data_path, 'set_b')
    
    # Get random files from each directory
    set_a_files = select_random_files(set_a_dir, 5)
    set_b_files = select_random_files(set_b_dir, 5)
    
    print("Predicting on 10 random heartbeat audio files:")
    print("=" * 60)
    
    # Process set_a files
    for file_name in set_a_files:
        file_path = os.path.join(set_a_dir, file_name)
        if os.path.exists(file_path):
            result = predict_heartbeat(file_path)
            if result:
                print(f"File: {file_name} (set_a)")
                print(f"Predicted class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("All class probabilities:")
                for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {prob:.4f}")
                print()
            else:
                print(f"Failed to process: {file_name}")
                print()
        else:
            print(f"File not found: {file_name}")
            print()
    
    # Process set_b files
    for file_name in set_b_files:
        file_path = os.path.join(set_b_dir, file_name)
        if os.path.exists(file_path):
            result = predict_heartbeat(file_path)
            if result:
                print(f"File: {file_name} (set_b)")
                print(f"Predicted class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("All class probabilities:")
                for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {prob:.4f}")
                print()
            else:
                print(f"Failed to process: {file_name}")
                print()
        else:
            print(f"File not found: {file_name}")
            print()

if __name__ == "__main__":
    main()
