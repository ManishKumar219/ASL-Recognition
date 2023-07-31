import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration


# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Initializing the drawing utils for drawing the landmarks on image
mpDraw = mp.solutions.drawing_utils
mpDrawingStyle = mp.solutions.drawing_styles

# Load the Trained model of Sign Language
model = tf.keras.models.load_model('ASLmodelF.h5')

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
ASLimg = cv2.imread('ASLimg.JPG')


label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# cap = cv2.VideoCapture(0)
class signDetection:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lmsList = []
        result = hands.process(imgRGB)
        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]
            for lm in handLms.landmark:
                h, w, c = img.shape
                lmsList.append(lm.x)
                lmsList.append(lm.y)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mpDrawingStyle.get_default_hand_landmarks_style(),
                                  mpDrawingStyle.get_default_hand_connections_style())
            lmsList = [lmsList]
            lmsList = np.array(lmsList)
            r = model.predict(lmsList)
            r = np.argmax(r)
            cv2.putText(img, f'Result = {label[r]}', (50, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format='bgr24')

def main():
    st.title('Real Time Sign Language Detection')

    st.header('Webcam Live Feed')
    st.write("Click on start to use webcam and detect finger spellings")
    webrtc_streamer(key='key', rtc_configuration=RTC_CONFIGURATION, video_processor_factory=signDetection)

    with st.sidebar:
        st.header('Finger Spellings')
        st.image(image=ASLimg)
        st.markdown('**Created by:** Manish Kumar')
        st.markdown('Github Link: [ASL-Recognition](https://github.com/ManishKumar219/ASL-Recognition)')

if __name__ == "__main__":
    main()