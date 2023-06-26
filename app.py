from PIL import Image
import numpy as np
import cv2
import requests
import face_recognition
import os
from datetime import datetime
import streamlit as st

# Set page title and description
st.set_page_config(
    page_title="Attendance System Using Face Recognition",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title("Attendance System Using Face Recognition 📷")
st.markdown("This app recognizes faces in an image and updates attendance records with current timestamp & Location.")

# Load images for face recognition
Images = []
classnames = []
directory = "photos"

myList = os.listdir(directory)

for cls in myList:
    if os.path.splitext(cls)[1] in [".jpg", ".jpeg"]:
        img_path = os.path.join(directory, cls)
        curImg = cv2.imread(img_path)
        Images.append(curImg)
        classnames.append(os.path.splitext(cls)[0])

def findEncodings(Images):
    encodeList = []
    for img in Images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknown = findEncodings(Images)

# Take picture using the camera
img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    test_image = Image.open(img_file_buffer)
    image = np.asarray(test_image)

    imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    name = "Unknown"  # Default name for unknown faces

    if len(encodesCurFrame) > 0:
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListknown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListknown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classnames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if name != "Unknown":
                url = "https://mrvishal7705.000webhostapp.com"
                url1 = "/update.php"
                data1 = {'name': name}
                response = requests.post(url + url1, data=data1)

                if response.status_code == 200:
                    st.success("Data updated on: " + url)
                else:
                    st.warning("Data not updated")

    # Apply styling with CSS
    st.markdown('<style>img { animation: pulse 2s infinite; }</style>', unsafe_allow_html=True)
    st.image(image, use_column_width=True, output_format="PNG")

    if name == "Unknown":
        st.info("Face not detected. Please try again.")
