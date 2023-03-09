import streamlit as st
import threading
import cv2
import numpy as np
from keras.models import load_model
import openai 
import os

# import os

openai.api_key =  os.getenv("APIKEY")


explan = ""
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
# import av
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

model = load_model('keras_model.h5', compile = False)
# lock = threading.Lock() 
# img_container = {'img':None}

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
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # Detect faces in the image

    img_file_buffer = st.camera_input(f"Take a picture ")
    if img_file_buffer is not None:

        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        # st.image(cv2_img)
        try:
            faces = face_cascade.detectMultiScale(cv2_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                # Extract the face region from the image
                face = cv2_img[y:y+h, x:x+w]
            # st.image(face)
            img = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            # st.image(img)
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img / 127.5) - 1
            probabilities = model.predict(img)
            # st.write(probabilities)
        

            darkspots = int(probabilities[0,0] * 100)
            puffyeyes = int(probabilities[0,1] * 100)
            wrinkles = int(probabilities[0,2] * 100)

            if darkspots >= puffyeyes and darkspots >= wrinkles:
                largest = {'name': 'darkspots', 'value': darkspots}
            elif puffyeyes >= darkspots and puffyeyes >= wrinkles:
                largest = {'name': 'puffyeyes', 'value': puffyeyes}
            else:
                largest = {'name': 'wrinkles', 'value': wrinkles}


            Darkspots.progress(darkspots)
            Puffyeyes.progress(puffyeyes)
            Wrinkles.progress(wrinkles)

        
            if st.button("Generate Care solution"):

                inpt = "What are the cosmetics that I should use for a face covered with " +  str({largest['name']} )
    # st.write(inpt)

                reply = openai.Completion.create(
                                                    engine="text-davinci-003",
                                                    prompt=inpt,
                                                    max_tokens=3600,
                                                    n=1,
                                                    stop=None,
                                                    temperature=0.5,
                                                    )
                explan= reply.choices[0].text.strip()
                
                st.stop()

        
        except Exception as err:
            st.write(err )
            st.stop()

st.caption(explan)