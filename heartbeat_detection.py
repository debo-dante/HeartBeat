import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# File Paths
data_path = '/Users/vioborah/HeartBeat/'
set_a_path = os.path.join(data_path, 'set_a')
set_b_path = os.path.join(data_path, 'set_b')

# Load CSV files
df_a = pd.read_csv(os.path.join(data_path, 'set_a.csv'))
df_b = pd.read_csv(os.path.join(data_path, 'set_b.csv'))
dataset = pd.concat([df_a, df_b])

def load_audio_data(file_path):
    try:
        x, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        new_path = os.path.join(os.path.dirname(file_path), 'Aunlabelledtest' + os.path.basename(file_path).split('__')[1])
        x, sr = librosa.load(new_path, sr=None)
    return x, sr

def extract_features(file_path):
    x, sr = load_audio_data(file_path)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

dataset['features'] = dataset['fname'].apply(lambda x: extract_features(os.path.join(data_path, x)))

# Preparing data
X = np.array(dataset['features'].tolist())
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['label'])

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(40,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Saving the model
model.save(os.path.join(data_path, 'heartbeat_model.h5'))
