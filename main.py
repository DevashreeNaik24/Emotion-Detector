import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class EmotionDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("800x600")

        # Initialize emotion detectors
        from face_emotion import FaceEmotionDetector
        self.face_detector = FaceEmotionDetector()
        self.camera_active = False

        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create tabs for different detection modes
        self.tab_control = ttk.Notebook(self.main_container)
        self.face_tab = ttk.Frame(self.tab_control)
        self.voice_tab = ttk.Frame(self.tab_control)
        self.text_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.face_tab, text='Face Detection')
        self.tab_control.add(self.voice_tab, text='Voice Analysis')
        self.tab_control.add(self.text_tab, text='Text Analysis')
        self.tab_control.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Setup individual tabs
        self.setup_face_tab()
        self.setup_voice_tab()
        self.setup_text_tab()

    def setup_face_tab(self):
        # Create frame for image display
        self.image_frame = ttk.Frame(self.face_tab)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5)

        # Create canvas for image display
        self.image_canvas = tk.Canvas(self.image_frame, width=640, height=480)
        self.image_canvas.grid(row=0, column=0)

        # Buttons frame
        button_frame = ttk.Frame(self.face_tab)
        button_frame.grid(row=1, column=0, pady=10)

        # Upload and Capture buttons
        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Start Camera", command=self.start_camera).grid(row=0, column=1, padx=5)

    def setup_voice_tab(self):
        # Voice analysis controls
        ttk.Label(self.voice_tab, text="Voice Emotion Analysis").grid(row=0, column=0, pady=20)
        ttk.Button(self.voice_tab, text="Start Recording", command=self.start_recording).grid(row=1, column=0, pady=10)
        ttk.Button(self.voice_tab, text="Stop Recording", command=self.stop_recording).grid(row=2, column=0, pady=10)

    def setup_text_tab(self):
        # Text analysis controls
        ttk.Label(self.text_tab, text="Text Emotion Analysis").grid(row=0, column=0, pady=20)
        self.text_input = tk.Text(self.text_tab, height=10, width=50)
        self.text_input.grid(row=1, column=0, pady=10, padx=10)
        ttk.Button(self.text_tab, text="Analyze Text", command=self.analyze_text).grid(row=2, column=0, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # TODO: Implement image processing and emotion detection
            pass

    def start_camera(self):
        if not self.camera_active:
            self.face_detector.start_video_capture()
            self.camera_active = True
            self.update_camera()
        else:
            self.face_detector.stop_video_capture()
            self.camera_active = False

    def update_camera(self):
        if self.camera_active:
            frame, results = self.face_detector.get_video_frame()
            if frame is not None:
                # Convert frame to PhotoImage for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(image=frame)
                
                # Update canvas
                self.image_canvas.create_image(0, 0, image=frame, anchor=tk.NW)
                self.image_canvas.image = frame  # Keep a reference
            
            # Schedule next update
            self.root.after(10, self.update_camera)

    def start_recording(self):
        # TODO: Implement voice recording
        pass

    def stop_recording(self):
        # TODO: Implement voice recording stop and analysis
        pass

    def analyze_text(self):
        # TODO: Implement text emotion analysis
        text = self.text_input.get("1.0", tk.END)
        pass

def main():
    root = tk.Tk()
    app = EmotionDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()