# Emotion Detector

An AI-powered emotion detection system that analyzes emotions from facial expressions, voice tone, and text input. The system uses computer vision, speech processing, and natural language processing (NLP) to classify emotions.

## Features

- **Facial Emotion Recognition**: Uses OpenCV and deep learning models to detect emotions from facial expressions in real-time or from uploaded images.
- **Voice-Based Emotion Detection**: Analyzes tone, pitch, and speech patterns using Librosa and ML models to classify emotions.
- **Text Sentiment Analysis**: Uses transformer-based models to determine the emotional tone of text input.
- **Multi-Modal Analysis**: Combines face, voice, and text detection for comprehensive emotion analysis.
- **User-Friendly GUI**: Interactive interface for easy input and analysis.

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

The GUI will open with three tabs:
1. **Face Detection**: Upload images or use webcam for real-time emotion detection
2. **Voice Analysis**: Record audio for emotion analysis
3. **Text Analysis**: Input text for sentiment analysis

## Requirements

- Python 3.7+
- See requirements.txt for complete list of dependencies

## Technical Details

- Face Detection: Uses OpenCV and Haar Cascades for face detection
- Voice Analysis: Uses Librosa for audio feature extraction
- Text Analysis: Uses DistilRoBERTa-based model for emotion classification

## Applications

- Mental health monitoring
- Customer sentiment analysis
- AI-driven virtual assistants
- Human-computer interaction research