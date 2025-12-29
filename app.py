# app.py

import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
from inou.face.processor import FaceProcessor

# class FPSProcessor:
#     def __init__(self):
#         self.last_time = time.time()
    
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         #print(img.shape)
#         current_time = time.time()
#         delta_time = current_time - self.last_time
#         self.last_time = current_time
        
#         fps = 1 / delta_time if delta_time > 0 else 0
#         cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN,
#                     3, (0, 255, 0), 2)
        
#         return av.VideoFrame.from_ndarray(img, format="bgr24")      

st.set_page_config(
    page_title='iNOU',
    page_icon='👋',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# --- SESSION STATE INITIALIZATION ---
if 'faces' not in st.session_state:
    st.session_state['faces'] = []

# --- APP LAYOUT ---
st.title('Welcome to iNOU!')
st.markdown('A face recognition application')

with st.container():
    col1, col2 = st.columns([5, 2])
    with col1:
        cam = webrtc_streamer(key='cam',
                        video_html_attrs=VideoHTMLAttributes(
                            autoPlay=True,
                            controls=False,
                            muted=True,
                            width='100%',),
                        video_processor_factory=FaceProcessor
        )
    with col2:
        with st.container(border=True):
            st.subheader('Add new faces:')
            faces = st.multiselect('Add labels for new face', options=[], accept_new_options=True, label_visibility='collapsed')
            if len(set(faces)) != len(faces):
                st.warning('There are duplicate labels. Please ensure all labels are unique.')
                faces = st.session_state['faces']
            elif set(faces) != set(st.session_state['faces']):
                if set(faces) > set(st.session_state['faces']):
                    if cam.video_processor:
                        new_label = set(faces) - set(st.session_state['faces'])
                        new_label = new_label.pop()
                        cam.video_processor.start_registration(new_label)
                        if cam.video_processor.registration_mode:
                            st.success(f'Started registration for {new_label}')
                    else:
                        st.warning('Please start camera first.')
                        faces = st.session_state['faces'] 
                else:
                    remove_label = set(st.session_state['faces']) - set(faces)
                    remove_label = remove_label.pop()
                    if cam.video_processor:
                        with cam.video_processor._lock:
                            cam.video_processor.face_recognizer.known_faces.pop(remove_label, None)
                    st.info(f'Removed label: {remove_label} from known faces.')
            st.session_state['faces'] = faces
            