# src/inou/face/processor.py

import time
import cv2
import av
import threading

from inou.face.embedder import FaceRecognition

class FaceProcessor:
    def __init__(self):
        self.last_time = time.perf_counter()
        self.face_recognizer = FaceRecognition()
        
        self.current_label = 'Unknown'
        self._lock = threading.Lock()
        
        self.registration_mode = False
        self.registration_name = None
        self.registration_frames_count = 0
        self.registration_frames_total = 5
        
    def start_registration(self, name, num_frames=5):
        with self._lock:
            self.registration_mode = True
            self.registration_name = name
            self.registration_frames_count = 0
            self.registration_frames_total = num_frames
            
    
    def get_results(self):
        with self._lock:
            return self.current_label
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        #fps
        current_time = time.perf_counter()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        fps = 1 / delta_time if delta_time > 0 else 0
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        # mode handling
        with self._lock:
            is_registring = self.registration_mode
            reg_name = self.registration_name
            frames_collected = self.registration_frames_count
            frames_total = self.registration_frames_total
        
        if is_registring and frames_collected < frames_total:
            success, bbox = self.face_recognizer.sample_embedding(img)
            if success and bbox:
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                with self._lock:
                    self.registration_frames_count += 1
                    frames_collected = self.registration_frames_count
            
                cv2.putText(img, f'Registering: {reg_name} ({frames_collected}/{frames_total})', (x, y), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            else:
                cv2.putText(img, f'No face detected - look at camera!', (10, 100), 
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                
            # finish collecting frames
            if frames_collected >= frames_total:
                self.face_recognizer.finalize_registration(reg_name)
                with self._lock:
                    self.registration_frames_count = 0
                    self.registration_mode = False
                    self.current_label = f'Registered: {reg_name}'
        else: 
            # face recognition
            label, confidence, bbox = self.face_recognizer.recognize(img)
            if bbox:
                x, y, w, h = bbox
                color = (0, 255, 0) if label != 'Unknown' else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f'{label}', (x, y), 
                            cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            with self._lock:
                self.current_label = label
        
        return av.VideoFrame.from_ndarray(img, format='bgr24')
        