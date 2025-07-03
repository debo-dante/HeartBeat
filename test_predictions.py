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

def get_true_label_from_filename(filename):
    """Extract the true label from filename"""
    if 'normal' in filename:
        return 'normal'
    elif 'murmur' in filename:
        return 'murmur'
    elif 'artifact' in filename:
        return 'artifact'
    elif 'extrahls' in filename:
        return 'extrahls'
    elif 'extrastole' in filename:
        return 'extrastole'
    else:
        return 'unknown'

def test_multiple_files():
    """Test predictions on multiple files"""
    # Get some test files from each category
    test_files = []
    
    # From set_a
    set_a_path = '/Users/vioborah/HeartBeat/set_a'
    if os.path.exists(set_a_path):
        # Get a few files from each category
        for filename in sorted(os.listdir(set_a_path)):
            if filename.endswith('.wav') and not filename.startswith('.'):
                test_files.append(os.path.join(set_a_path, filename))
                if len(test_files) >= 20:  # Limit to first 20 files
                    break
    
    # From set_b
    set_b_path = '/Users/vioborah/HeartBeat/set_b'
    if os.path.exists(set_b_path):
        # Get a few files from each category
        count = 0
        for filename in sorted(os.listdir(set_b_path)):
            if filename.endswith('.wav') and not filename.startswith('.'):
                test_files.append(os.path.join(set_b_path, filename))
                count += 1
                if count >= 10:  # Limit to first 10 files from set_b
                    break
    
    print(f"Testing on {len(test_files)} files:")
    print("=" * 80)
    
    correct_predictions = 0
    total_predictions = 0
    
    for file_path in test_files:
        filename = os.path.basename(file_path)
        true_label = get_true_label_from_filename(filename)
        
        result = predict_heartbeat(file_path)
        if result:
            predicted_label = result['predicted_class']
            confidence = result['confidence']
            
            is_correct = predicted_label == true_label
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} File: {filename[:50]}")
            print(f"  True: {true_label} | Predicted: {predicted_label} | Confidence: {confidence:.3f}")
            
            # Show top 2 predictions
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            print(f"  Top predictions: {sorted_probs[0][0]} ({sorted_probs[0][1]:.3f}), {sorted_probs[1][0]} ({sorted_probs[1][1]:.3f})")
            print()
        else:
            print(f"✗ Failed to process: {filename}")
            print()
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print("=" * 80)
        print(f"Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    else:
        print("No successful predictions made.")

def test_specific_files():
    """Test on specific files from different categories"""
    specific_files = [
        # Normal files
        '/Users/vioborah/HeartBeat/set_a/normal__201101070538.wav',
        '/Users/vioborah/HeartBeat/set_a/normal__201102081152.wav',
        '/Users/vioborah/HeartBeat/set_a/normal__201103140132.wav',
        
        # Murmur files
        '/Users/vioborah/HeartBeat/set_a/murmur__201101051104.wav',
        '/Users/vioborah/HeartBeat/set_a/murmur__201102051443.wav',
        
        # Artifact files
        '/Users/vioborah/HeartBeat/set_a/artifact__201012172012.wav',
        '/Users/vioborah/HeartBeat/set_a/artifact__201105040918.wav',
        
        # Extrahls files
        '/Users/vioborah/HeartBeat/set_a/extrahls__201101070953.wav',
        '/Users/vioborah/HeartBeat/set_a/extrahls__201101091153.wav',
    ]
    
    print("Testing on specific files from different categories:")
    print("=" * 80)
    
    for file_path in specific_files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            true_label = get_true_label_from_filename(filename)
            
            result = predict_heartbeat(file_path)
            if result:
                predicted_label = result['predicted_class']
                confidence = result['confidence']
                
                is_correct = predicted_label == true_label
                status = "✓" if is_correct else "✗"
                
                print(f"{status} File: {filename}")
                print(f"  True: {true_label} | Predicted: {predicted_label} | Confidence: {confidence:.3f}")
                
                # Show all class probabilities
                print("  All probabilities:")
                for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"    {class_name}: {prob:.4f}")
                print()
            else:
                print(f"✗ Failed to process: {filename}")
                print()
        else:
            print(f"✗ File not found: {os.path.basename(file_path)}")
            print()

if __name__ == "__main__":
    print("Heartbeat Classification Testing")
    print("=" * 50)
    
    # Test on specific files first
    test_specific_files()
    
    print("\n" + "=" * 50)
    print("Testing on multiple files:")
    # Test on multiple files
    test_multiple_files()
