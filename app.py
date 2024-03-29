import streamlit as st
import threading
import cv2
import numpy as np
from keras.models import load_model
import openai 
import os
import pandas as pd






# import os
largest = {}
openai.api_key =  os.getenv("APIKEY")

# sessionstate = 

model = load_model('keras_model.h5', compile = False)


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

    

            # st.table(largest)

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
                st.session_state["largest"] = largest
            elif puffyeyes >= darkspots and puffyeyes >= wrinkles:
                largest = {'name': 'puffyeyes', 'value': puffyeyes}
                st.session_state["largest"] = largest

            else:
                largest = {'name': 'wrinkles', 'value': wrinkles}
                st.session_state["largest"] = largest



            Darkspots.progress(darkspots)
            Puffyeyes.progress(puffyeyes)
            Wrinkles.progress(wrinkles)

        
        except Exception as err:
            st.write(err )
            st.stop()

with st.expander("Product links and Names"):
        if st.button("Show Links"):
            eyes = pd.read_csv("dataELC - eyesdata.csv")
            largest = st.session_state["largest"] 


            ulla = "Select the products that can be used to cover" +  str(largest) + " out of the following cosmetic products with the relevant urls in the adjacent column and insert them into a table :" + str(eyes) 
    # st.write(inpt)

            reply = openai.Completion.create(
                                                    engine="text-davinci-003",
                                                    prompt=ulla,
                                                    max_tokens=1200,
                                                    n=1,
                                                    stop=None,
                                                    temperature=0.5,
                                                    )
            veliya= reply.choices[0].text.strip()
            st.session_state["veliya"] = veliya
            st.write(veliya)
            # st.stop()




with st.expander("Grooming tips for detected face"):
        if st.button("Generate Tips"):
            inpt = "Generate Grooming tips on how to apply the following  " +  str(st.session_state["veliya"])
            reply = openai.Completion.create(
                                                    engine="text-davinci-003",
                                                    prompt=inpt,
                                                    max_tokens=3600,
                                                    n=1,
                                                    stop=None,
                                                    temperature=0.5,
                                                    )
            explan= reply.choices[0].text.strip()
            st.write(explan)
            st.stop()


# if st.button("Generate Care solution"):

              