import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# File Paths
data_path = '/Users/vioborah/HeartBeat/'
set_a_path = os.path.join(data_path, 'set_a')
set_b_path = os.path.join(data_path, 'set_b')

def load_and_clean_data():
    """Load and clean the dataset, handling file path issues"""
    print("Loading CSV files...")
    df_a = pd.read_csv(os.path.join(data_path, 'set_a.csv'))
    df_b = pd.read_csv(os.path.join(data_path, 'set_b.csv'))
    
    # Combine datasets
    dataset = pd.concat([df_a, df_b], ignore_index=True)
    
    # Remove rows with empty labels
    dataset = dataset[dataset['label'].notna() & (dataset['label'] != '')]
    
    print(f"Total samples after cleaning: {len(dataset)}")
    print(f"Label distribution:\n{dataset['label'].value_counts()}")
    
    return dataset

def find_actual_file_path(fname, data_path):
    """Find the actual file path, handling naming inconsistencies"""
    original_path = os.path.join(data_path, fname)
    
    # If the original path exists, return it
    if os.path.exists(original_path):
        return original_path
    
    # Extract the base filename and directory
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    
    # Handle unlabelled files that start with "__"
    if base_name.startswith('__'):
        # Try with Aunlabelledtest prefix
        new_name = 'Aunlabelledtest' + base_name
        new_path = os.path.join(dir_name, new_name)
        if os.path.exists(new_path):
            return new_path
            
        # Try with Bunlabelledtest prefix for set_b
        new_name = 'Bunlabelledtest' + base_name
        new_path = os.path.join(dir_name, new_name)
        if os.path.exists(new_path):
            return new_path
    
    # Handle set_b files that have Btraining_ prefix in CSV but not in actual files
    if 'set_b' in dir_name and base_name.startswith('Btraining_'):
        # Remove the Btraining_ prefix
        actual_name = base_name.replace('Btraining_', '')
        new_path = os.path.join(dir_name, actual_name)
        if os.path.exists(new_path):
            return new_path
    
    # If nothing works, return None
    return None

def load_audio_data(file_path):
    """Load audio data with error handling"""
    try:
        x, sr = librosa.load(file_path, sr=22050, duration=5.0)
        return x, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_features(file_path):
    """Extract MFCC features from audio file"""
    x, sr = load_audio_data(file_path)
    
    if x is None:
        # Return zero features if file can't be loaded
        return np.zeros(60)
    
    try:
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
        print(f"Error extracting features from {file_path}: {e}")
        return np.zeros(60)

def prepare_data():
    """Prepare data for training"""
    dataset = load_and_clean_data()
    
    print("Processing audio files and extracting features...")
    features = []
    labels = []
    processed_count = 0
    
    for idx, row in dataset.iterrows():
        # Find the actual file path
        actual_path = find_actual_file_path(row['fname'], data_path)
        
        if actual_path and os.path.exists(actual_path):
            feature = extract_features(actual_path)
            features.append(feature)
            labels.append(row['label'])
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} files...")
        else:
            print(f"File not found: {row['fname']}")
    
    print(f"Successfully processed {len(features)} files")
    
    if len(features) == 0:
        raise ValueError("No valid audio files found!")
    
    X = np.array(features)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    return X, y, label_encoder

def build_model(num_classes):
    """Build the neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(60,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main training function"""
    print("Starting Heartbeat Classification Training...")
    print("=" * 60)
    
    # Prepare data
    X, y, label_encoder = prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Build model
    model = build_model(len(label_encoder.classes_))
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and label encoder
    model.save(os.path.join(data_path, 'heartbeat_model.h5'))
    
    # Save label encoder classes for later use
    np.save(os.path.join(data_path, 'label_classes.npy'), label_encoder.classes_)
    
    print(f"\nModel saved successfully!")
    print(f"Classes: {label_encoder.classes_}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
