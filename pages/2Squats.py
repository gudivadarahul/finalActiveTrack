# imports for openCV, MediaPipe, and numpy
import cv2
import mediapipe as medPipe
import streamlit as st
import numpy as np
from streamlit_lottie import st_lottie

import requests
import time

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Get rid of default menu bar and footer from streamlit
sidebar = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """

st.markdown(sidebar, unsafe_allow_html=True)

# two libraries from mediapipe to recognize poses
drawings = medPipe.solutions.drawing_utils
poseSolutions = medPipe.solutions.pose

# calculate angle of each position between joints
def calcAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rads = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rads*180.0/np.pi)

    if ang > 180.0:
        ang = 360-ang

    return ang

def start(sets, reps, restAmt):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    setsCounter = 0

    while setsCounter < sets:
        repsCounter = 0
        motion = None

        with poseSolutions.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as poseTrack:
            cap.isOpened()
            while repsCounter < reps:
                ret, frame = cap.read()

                # change image to rgb to allow mediapipe to process images
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # detect the image for joints
                detections = poseTrack.process(image)

                # change image back to brg to allow openCV to process image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    bodyPositions = detections.pose_landmarks.landmark

                    # GET SQUAT COORDINATES
                    # only need one side so we are choosing left landmarks
                    L_hip = [bodyPositions[poseSolutions.PoseLandmark.LEFT_HIP.value].x,
                            bodyPositions[poseSolutions.PoseLandmark.LEFT_HIP.value].y]
                    L_knee = [bodyPositions[poseSolutions.PoseLandmark.LEFT_KNEE.value].x,
                            bodyPositions[poseSolutions.PoseLandmark.LEFT_KNEE.value].y]
                    L_ankle = [bodyPositions[poseSolutions.PoseLandmark.LEFT_ANKLE.value].x,
                            bodyPositions[poseSolutions.PoseLandmark.LEFT_ANKLE.value].y]                   
                    
                    cv2.rectangle(image, (0,0), (240,75), (245,117,16), -1)

                    # VARS FOR TEXT FIELD
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    color = (255, 255, 255)
                    thickness = 2

                    # Calculate the knee-joint angle and the hip-joint angle
                    kneeAngle = calcAngle(L_hip, L_knee, L_ankle)
                    
                    # Visualize angle
                    cv2.putText(image, str(kneeAngle), 
                                tuple(np.multiply(L_knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    # Render detections
                    drawings.draw_landmarks(image, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS,
                                            drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                    
                    if kneeAngle > 160:
                        motion = "down"
                    elif kneeAngle < 160 and kneeAngle > 100:
                        cv2.rectangle(image, (250,230), (420,260), (0,0,255), -1)
                        cv2.putText(image, 'GO LOWER', (270,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    elif kneeAngle < 100 and motion == "down":
                        motion = "up"
                        repsCounter += 1
                    
                    # Reps Info
                    cv2.putText(image, "REPS", (15, 12), font,0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(repsCounter), (10, 60), font,2, color, thickness, cv2.LINE_AA)

                    # Motion Info
                    cv2.putText(image, "MOTION", (65, 12), font,0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, motion, (60, 60), font,2, color, thickness, cv2.LINE_AA)  
                    
                    cv2.imshow('Mediapipe Feed', image)  
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break 
                    
                except:
                    cv2.imshow('Mediapipe Feed', image)
                    print("An exception occurred")
                    pass

            setsCounter += 1
            if setsCounter != sets:
                try:
                    cv2.rectangle(image, (50,180), (600,400), (0,255,0), -1)
                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                    cv2.putText(image, '  STAND UP' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', image)
                    cv2.waitKey(1)
                    time.sleep(3)     
                except:
                    cv2.imshow('Mediapipe Feed', image)
                    pass 

    cv2.rectangle(image, (50,180), (600,400), (0,255,0), -1)
    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(image, '    REST' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.imshow('Mediapipe Feed', image)
    cv2.waitKey(1) 
    time.sleep(restAmt)
    cap.release()
    cv2.destroyAllWindows()     

col1, col2 = st.columns([2, 1])
with col1:
    st.write("# Squats")
    st.write("For squats place your legs and feet in view of the camera!")
    st.write("### Enter Desired Amount of Sets and Reps")
    sets_input = st.number_input("Please enter sets amount: ")
    reps_input = st.number_input("Please enter rep amount: ")
    restAmt = st.slider('select your rest time', 0, 60, 30)
    st.write("Your selected break is: ", restAmt, "seconds")
    st.write("##### to manually exit the program please press q")



    options = st.button("Click me to begin")
    if options:
        start(sets_input, reps_input, restAmt)

with col2:
    lottie_diagram_url = 'https://assets2.lottiefiles.com/packages/lf20_cfrvib6d.json'
    lottie_diagram = load_lottieurl(lottie_diagram_url)
    st_lottie(lottie_diagram, key='diagram')        

                