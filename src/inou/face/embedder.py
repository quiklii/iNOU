# src/inou/face/embedder.py

import numpy as np
import torch
import cv2

from inou.face.model import load_model

class FaceRecognition:
    def __init__(self):
        self.known_faces: dict[str, np.ndarray] = {}
        self.model, self.device = load_model()
        self.temp_embeddings: list[np.ndarray] = []
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    def detect_faces(self, img: np.ndarray):
        face_img = img.copy()
        face_rect = self.face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
        
        if len(face_rect) == 0:
            return None, None
        
        # Select the largest face
        x, y, w, h = max(face_rect, key=lambda rect: rect[2] * rect[3])
        
        # crop and resize
        face_crop = face_img[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))
        
        return face_crop, (x, y, w, h)
        
    
    def get_embedding(self, img: np.ndarray):
        img = img.astype(np.float32)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_tensor = img_tensor / 255.0 * 2 - 1  # Normalize to [-1, 1]
        with torch.no_grad():
            embedding = self.model(img_tensor).cpu().numpy()
        return embedding.flatten()
    
    def sample_embedding(self, img: np.ndarray):
        face_img, bbox = self.detect_faces(img)
        if face_img is None:
            return False, None
        
        embedding = self.get_embedding(face_img)
        if embedding is not None:
            self.temp_embeddings.append(embedding)
            return True, bbox
        return False, None
    
    def finalize_registration(self, label: str):
        if not self.temp_embeddings:
            return False
        avg_embedding = np.mean(self.temp_embeddings, axis=0)
        #avg_embedding /= np.linalg.norm(avg_embedding)  # Normalize
        
        self.known_faces[label] = avg_embedding
        self.temp_embeddings = []
        return True
    
    def recognize(self, img: np.ndarray, threshold: float = 0.8):
        face_img, bbox = self.detect_faces(img)
        
        if face_img is None:
            return 'No face detected', -1, None
        
        if not self.known_faces:
            return 'Unknown', -1, bbox
        
        embedding = self.get_embedding(face_img)
        
        best_score = -1
        best_label = 'Unknown'
        
        for name, emb in self.known_faces.items():
            cosine = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
            if cosine > best_score and cosine > threshold:
                best_score = cosine
                best_label = name
        return best_label, best_score, bbox
        