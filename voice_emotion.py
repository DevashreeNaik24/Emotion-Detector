import numpy as np
import sounddevice as sd
import librosa
import threading
from queue import Queue

class VoiceEmotionDetector:
    def __init__(self):
        # Audio recording parameters
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_queue = Queue()
        self.recording_thread = None
        
        # Emotion labels
        self.emotions = ['Angry', 'Happy', 'Sad', 'Neutral']
        
    def start_recording(self):
        """Start audio recording in a separate thread"""
        if self.recording:
            return
            
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
    def _record_audio(self):
        """Record audio in chunks"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f'Audio callback status: {status}')
            self.audio_queue.put(indata.copy())
        
        with sd.InputStream(callback=audio_callback,
                           channels=self.channels,
                           samplerate=self.sample_rate):
            while self.recording:
                sd.sleep(100)  # Sleep to prevent busy-waiting
                
    def stop_recording(self):
        """Stop audio recording and process the recorded audio"""
        if not self.recording:
            return None
            
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
        # Collect all audio data from queue
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
            
        if not audio_chunks:
            return None
            
        # Combine audio chunks
        audio_data = np.concatenate(audio_chunks)
        return self.analyze_audio(audio_data)
        
    def extract_features(self, audio_data):
        """Extract audio features using librosa"""
        try:
            # Extract various audio features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            
            # Extract additional features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
            
            # Compute statistics
            chroma_scaled = np.mean(chroma.T, axis=0)
            mel_scaled = np.mean(mel.T, axis=0)
            
            # Combine features
            features = np.concatenate([mfcc_scaled, chroma_scaled, np.mean(mel_scaled)])
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
            
    def analyze_audio(self, audio_data):
        """Analyze audio data and predict emotion"""
        # Extract features
        features = self.extract_features(audio_data)
        if features is None:
            return None
            
        # TODO: Add emotion prediction when model is implemented
        # For now, return a placeholder result
        emotion = 'Neutral'  # Placeholder
        confidence = 1.0     # Placeholder
        
        return {
            'emotion': emotion,
            'confidence': confidence
        }