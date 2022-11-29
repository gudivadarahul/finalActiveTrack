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
        switch_sides = False
        motion = None

        with poseSolutions.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as poseTrack:
            cap.isOpened()
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            while repsCounter < reps:
                ret, frame = cap.read()

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break


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

                    if switch_sides == False:
                        # GET LEFT ARM COORDINATES
                        shoulder = [bodyPositions[poseSolutions.PoseLandmark.LEFT_SHOULDER.value].x,
                                    bodyPositions[poseSolutions.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [bodyPositions[poseSolutions.PoseLandmark.LEFT_ELBOW.value].x,
                                bodyPositions[poseSolutions.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [bodyPositions[poseSolutions.PoseLandmark.LEFT_WRIST.value].x,
                                bodyPositions[poseSolutions.PoseLandmark.LEFT_WRIST.value].y]
                        hip = [bodyPositions[poseSolutions.PoseLandmark.LEFT_HIP.value].x,
                            bodyPositions[poseSolutions.PoseLandmark.LEFT_HIP.value].y]
                    else:
                        shoulder = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    bodyPositions[poseSolutions.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_ELBOW.value].x,
                                bodyPositions[poseSolutions.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_WRIST.value].x,
                                bodyPositions[poseSolutions.PoseLandmark.RIGHT_WRIST.value].y]
                        hip = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_HIP.value].x,
                            bodyPositions[poseSolutions.PoseLandmark.LEFT_HIP.value].y]
                    
                    
                    cv2.rectangle(image, (0,0), (240,75), (245,117,16), -1)

                    # VARS FOR TEXT FIELD
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    color = (255, 255, 255)
                    thickness = 2

                    # Reps Info
                    cv2.putText(image, "REPS", (12, 16), font,0.6, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(repsCounter), (15, 60), font,1.75, color, thickness, cv2.LINE_AA)

                    # Motion Info
                    cv2.putText(image, "MOTION", (110, 16), font,0.6, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, motion, (100, 55), font,1.5, color, thickness, cv2.LINE_AA)  

                    # Render detections
                    drawings.draw_landmarks(image, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS,
                                            drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                    
                    # calc angle for joints for bicep curl motion
                    angle = calcAngle(shoulder, elbow, wrist)

                    # calc angle for incorrect form if elbows move too much
                    angleForm = calcAngle(shoulder, elbow, hip) 
                    
                    # Visualize angle
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                         
                        
                    # If elbow moved, then indicate improper form
                    if angleForm < 167:
                        cv2.rectangle(image, (250,230), (420,260), (0,0,255), -1)
                        cv2.putText(image, 'INCORRECT FORM', (270,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        
                    else:
                        if angle > 160:
                            motion = "up"
                        elif angle < 45 and motion == "up":
                            motion = "down"
                            repsCounter += 1


                    if switch_sides == False and repsCounter == reps:
                        switch_sides = True
                        repsCounter = 0
                        cv2.rectangle(image, (50,180), (600,400), (0,255,0), -1)
                        cv2.putText(image, 'SWITCH SIDES' , (155,300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                        cv2.imshow('Mediapipe Feed', image)
                        cv2.waitKey(1) 
                        time.sleep(3)
                    else:
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
                    cv2.putText(image, 'SWITCH SIDES' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', image)
                    cv2.waitKey(1) 
                    time.sleep(3)
                except:
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
    st.write("# Bicep Curls")
    st.write("For bicep curls please place your arms and the left side of your body in full view of the camera!")
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
    lottie_diagram_url = 'https://assets5.lottiefiles.com/private_files/lf30_i5o0xxk6.json'
    lottie_diagram = load_lottieurl(lottie_diagram_url)
    st_lottie(lottie_diagram, key='diagram')        

                