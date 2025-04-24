from transformers import pipeline
import torch

class TextEmotionDetector:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Emotion labels
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
    def analyze_text(self, text):
        """Analyze text and predict emotion"""
        if not text or text.isspace():
            return {
                'emotion': 'neutral',
                'confidence': 1.0
            }
            
        try:
            # Get predictions
            predictions = self.sentiment_analyzer(text)[0]
            
            # Find emotion with highest confidence
            max_score = max(predictions, key=lambda x: x['score'])
            
            return {
                'emotion': max_score['label'],
                'confidence': float(max_score['score']),
                'all_emotions': [
                    {
                        'emotion': pred['label'],
                        'confidence': float(pred['score'])
                    }
                    for pred in predictions
                ]
            }
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return {
                'emotion': 'neutral',
                'confidence': 1.0
            }
    
    def get_emotion_distribution(self, text):
        """Get distribution of emotions for visualization"""
        result = self.analyze_text(text)
        if 'all_emotions' not in result:
            return [], []
            
        emotions = [pred['emotion'] for pred in result['all_emotions']]
        scores = [pred['confidence'] for pred in result['all_emotions']]
        return emotions, scores