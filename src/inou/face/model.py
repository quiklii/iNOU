# src/inou/face/model.py

from facenet_pytorch import InceptionResnetV1
import streamlit as st
from pathlib import Path
import torch

MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'facenet.pt'

@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if MODEL_PATH.exists():
        print(f'Loading model from {MODEL_PATH}')
        model =  torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval().to(device)
    else:
        print('Downloading pretrained model (first time only)...')
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device) 
        
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, MODEL_PATH)
        print(f'Model saved to {MODEL_PATH}')
    
    return model, device
    