import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import face_recognition
import pickle
import numpy as np

class CameraBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraBoxLayout, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        with open('knn_model.clf', 'rb') as f:
            knn_clf = pickle.load(f)
            self.known_face_encodings = knn_clf['encodings']
            self.known_face_names = knn_clf['names']
        
        Clock.schedule_interval(self.update, 1.0/30.0)
    
    def capture_image(self):
        ret, frame = self.capture.read()
        if ret:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                
                self.ids.result_label.text = f"Detected: {name}"
    
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video_feed.texture = texture
    
    def on_stop(self):
        self.capture.release()

class FaceRecognitionApp(App):
    def build(self):
        return CameraBoxLayout()

if __name__ == '__main__':
    FaceRecognitionApp().run()
