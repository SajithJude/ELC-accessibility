import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        return img


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)



# import streamlit as st
# import threading
# import cv2
# import numpy as np
# from keras.models import load_model
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
# import av
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

# model = load_model('keras_model.h5', compile = False)
# lock = threading.Lock() 
# img_container = {'img':None}

# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     with lock:
#         img_container['img'] = img
#     return frame


# st.title('Is it Pikachu or Eevee!?')
# st.subheader('The image detection tool you definitely do not need in your life')

# ctx = webrtc_streamer(key="example", 
#                 video_frame_callback=video_frame_callback,
#                 rtc_configuration = RTC_CONFIGURATION,
#                 mode=WebRtcMode.SENDRECV)
# st.write('Pikachu')
# pikachu = st.progress(0)
# st.write('Eevee')
# eevee = st.progress(0)
# st.write('Eevee')
# et = st.progress(0)


# while ctx.state.playing:
#     with lock: 
#         img = img_container['img']
#     if img is None:
#         continue
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
#     img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
#     img = (img / 127.5) - 1
#     probabilities = model.predict(img)
    
#     pikachu_p = int(probabilities[0,0] * 100)
#     eevee_p = int(probabilities[0,1] * 100)
#     et_p = int(probabilities[0,2] * 100)

#     pikachu.progress(pikachu_p)
#     eevee.progress(eevee_p)
#     et.progress(et_p)
