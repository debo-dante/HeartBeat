import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class HeartbeatClassifier:
    def __init__(self, data_path='/Users/vioborah/HeartBeat/'):
        self.data_path = data_path
        self.set_a_path = os.path.join(data_path, 'set_a')
        self.set_b_path = os.path.join(data_path, 'set_b')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_length = 40
        
    def load_data(self):
        """Load and combine datasets from set_a and set_b"""
        print("Loading datasets...")
        df_a = pd.read_csv(os.path.join(self.data_path, 'set_a.csv'))
        df_b = pd.read_csv(os.path.join(self.data_path, 'set_b.csv'))
        
        # Clean and filter data - remove empty labels
        df_a_clean = df_a[df_a['label'].notna() & (df_a['label'] != '')]
        df_b_clean = df_b[df_b['label'].notna() & (df_b['label'] != '')]
        
        self.dataset = pd.concat([df_a_clean, df_b_clean], ignore_index=True)
        print(f"Total samples: {len(self.dataset)}")
        print(f"Label distribution:\n{self.dataset['label'].value_counts()}")
        
    def extract_advanced_features(self, file_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=22050, duration=5.0)
            
            # If audio is too short, pad it
            if len(y) < sr * 2:  # Less than 2 seconds
                y = np.pad(y, (0, sr * 2 - len(y)), mode='constant', constant_values=0)
            
            # Extract multiple features
            features = []
            
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            features.extend(mfccs_mean)
            features.extend(mfccs_std)
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ])
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
            
            # Ensure consistent feature length
            if len(features) < self.feature_length:
                features.extend([0] * (self.feature_length - len(features)))
            elif len(features) > self.feature_length:
                features = features[:self.feature_length]
                
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return np.zeros(self.feature_length)
    
    def prepare_data(self):
        """Extract features and prepare data for training"""
        print("Extracting features...")
        features = []
        labels = []
        
        for idx, row in self.dataset.iterrows():
            file_path = os.path.join(self.data_path, row['fname'])
            if os.path.exists(file_path):
                feature = self.extract_advanced_features(file_path)
                features.append(feature)
                labels.append(row['label'])
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(self.dataset)} files")
        
        self.X = np.array(features)
        self.y = self.label_encoder.fit_transform(labels)
        
        # Normalize features
        self.X = self.scaler.fit_transform(self.X)
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
    def build_cnn_model(self):
        """Build a 1D CNN model for heartbeat classification"""
        model = models.Sequential([
            # Reshape for 1D CNN
            layers.Reshape((self.feature_length, 1), input_shape=(self.feature_length,)),
            
            # 1D Convolutional layers
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output layer
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_dense_model(self):
        """Build a dense neural network model"""
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.feature_length,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model_type='cnn'):
        """Train the model with cross-validation"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Build model
        if model_type == 'cnn':
            self.model = self.build_cnn_model()
        else:
            self.model = self.build_dense_model()
            
        print(f"\nModel Architecture ({model_type}):")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.data_path, f'best_heartbeat_model_{model_type}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        self.plot_training_history(history)
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_path, 'confusion_matrix.png'), dpi=300)
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_path, 'training_history.png'), dpi=300)
        plt.show()
    
    def predict_heartbeat(self, audio_file_path):
        """Predict heartbeat class for a new audio file"""
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        features = self.extract_advanced_features(audio_file_path)
        features = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(features)
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)
        
        return predicted_class, confidence

def main():
    # Initialize classifier
    classifier = HeartbeatClassifier()
    
    # Load and prepare data
    classifier.load_data()
    classifier.prepare_data()
    
    # Train CNN model
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    history_cnn = classifier.train_model(model_type='cnn')
    
    # Train Dense model for comparison
    print("\n" + "="*60)
    print("TRAINING DENSE MODEL")
    print("="*60)
    history_dense = classifier.train_model(model_type='dense')
    
    print("\nTraining completed! Models and results saved.")

if __name__ == "__main__":
    main()
