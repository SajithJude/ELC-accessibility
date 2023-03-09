import streamlit as st
import threading
import cv2
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

model = load_model('keras_model.h5', compile = False)
lock = threading.Lock() 
img_container = {'img':None}

# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     with lock:
#         img_container['img'] = img
#     return frame


# st.title('Is it Pikachu or Eevee!?')
# st.subheader('The image detection tool you definitely do not need in your life')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Skin Analysis")
    st.write('Dark spots')
    Darkspots = st.progress(0)
    st.write('Puffy Eyes')
    Puffyeyes = st.progress(0)
    st.write('Wrinkles')
    Wrinkles = st.progress(0)
with col2:
    img_file_buffer = st.camera_input(f"Take a picture ")
    while img_file_buffer is not None:

        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(cv2_img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1
        probabilities = model.predict(img)
        st.write(probabilities)
        darkspots = int(probabilities[0,0] * 100)
        puffyeyes = int(probabilities[0,1] * 100)
        wrinkles = int(probabilities[0,2] * 100)

        Darkspots.progress(darkspots)
        Puffyeyes.progress(puffyeyes)
        Wrinkles.progress(wrinkles)
