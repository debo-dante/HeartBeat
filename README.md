# HeartBeat Detection Project

A machine learning project for detecting and classifying heartbeat sounds using audio signal processing and deep learning techniques.

## Overview

This project implements a heartbeat detection system using audio analysis and neural networks. It processes heart sound recordings to extract features and classify different types of heartbeats using machine learning models.

## Features

- **Audio Processing**: Utilizes librosa for audio signal processing and feature extraction
- **Feature Extraction**: Extracts MFCC (Mel-frequency cepstral coefficients) features from audio files
- **Deep Learning**: Implements neural network models using TensorFlow/Keras
- **Classification**: Classifies different types of heartbeat sounds
- **Visualization**: Generates training results and prediction visualizations

## Project Structure

```
HeartBeat/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── heartbeat_detection.py            # Basic heartbeat detection implementation
├── advanced_heartbeat_detection.py   # Advanced version with enhanced features
├── fixed_heartbeat_detection.py      # Fixed/optimized version
├── predict_heartbeat.py              # Prediction script for new audio files
├── test_predictions.py               # Testing and validation script
├── heartbeat_model.h5                # Trained model file
├── label_classes.npy                 # Label encoder classes
├── training_results.png              # Training visualization
├── set_a/                            # Training dataset A
├── set_b/                            # Training dataset B
├── set_a.csv                         # Dataset A metadata
├── set_b.csv                         # Dataset B metadata
└── set_a_timing.csv                  # Timing information for dataset A
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd HeartBeat
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- numpy==1.24.3
- pandas==2.0.3
- librosa==0.10.1
- scikit-learn==1.3.0
- tensorflow==2.13.0
- matplotlib==3.7.2
- seaborn==0.12.2
- soundfile==0.12.1
- scipy==1.11.1

## Usage

### Training the Model

1. **Basic training:**
   ```bash
   python heartbeat_detection.py
   ```

2. **Advanced training with enhanced features:**
   ```bash
   python advanced_heartbeat_detection.py
   ```

### Making Predictions

1. **Predict on new audio files:**
   ```bash
   python predict_heartbeat.py
   ```

2. **Run tests and validation:**
   ```bash
   python test_predictions.py
   ```

## Model Architecture

The neural network consists of:
- Input layer: 40 MFCC features
- Hidden layers: Dense layers with ReLU activation
- Dropout layers: For regularization (0.3 dropout rate)
- Output layer: Softmax activation for multi-class classification

## Data Processing

1. **Audio Loading**: Uses librosa to load audio files
2. **Feature Extraction**: Extracts 40 MFCC coefficients per audio file
3. **Preprocessing**: Normalizes and prepares data for training
4. **Label Encoding**: Converts categorical labels to numerical format

## Results

The model training results and visualizations are saved as:
- `training_results.png`: Training accuracy and loss curves
- `heartbeat_model.h5`: Trained model weights
- `label_classes.npy`: Label encoder classes for predictions

## Dataset

The project uses two datasets:
- **Set A**: Training data with corresponding CSV metadata
- **Set B**: Additional training data with metadata
- Audio files are stored in respective directories (`set_a/`, `set_b/`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Thanks to the contributors of librosa for audio processing capabilities
- TensorFlow/Keras for the deep learning framework
- The dataset providers for the heartbeat audio samples

## Troubleshooting

### Common Issues

1. **Audio file loading errors**: Check file paths and formats
2. **Memory issues**: Reduce batch size or use data generators
3. **Model convergence**: Adjust learning rate or network architecture

### Support

For issues and questions, please create an issue in the repository.

---

**Note**: Make sure to update file paths in the Python scripts to match your local environment before running.
