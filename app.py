from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mail import Mail,Message
import dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from flask import jsonify
import time
from flask import Response
import random
import json
import time
from pprint import pprint

from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import smtplib

from landmarks import *

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

app = Flask(__name__)
app.secret_key = "fitcompass_secret_key"




currentDirectory = os.path.dirname(os.path.abspath(__file__))

db_path = os.path.join(currentDirectory, "UserLogins.db")

def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Create tables
connection = get_db_connection()
cursor = connection.cursor()

# Drop old table if it exists (WARNING: deletes old user data!) Only do when adding columns to the table and want total reset
# cursor.execute("DROP TABLE IF EXISTS UserLogins")
# cursor.execute("DROP TABLE IF EXISTS UserOutfits")

# first table
cursor.execute("""
CREATE TABLE IF NOT EXISTS UserLogins(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    goal TEXT,
    goal_other TEXT,
    workouts_per_week INTEGER,
    body_part TEXT,
    coins INTEGER DEFAULT 1000,
    history TEXT DEFAULT '[]',
    current_workout INTEGER,
    equipped_outfit TEXT,
    workout_plan TEXT
)
""")

#second table
cursor.execute("""
CREATE TABLE IF NOT EXISTS UserOutfits(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    outfit_name TEXT,
    FOREIGN KEY(user_id) REFERENCES UserLogins(id)
)
""")

connection.commit()
connection.close()

# Webcam setup
camera = cv2.VideoCapture(0)
#global variable for the latest detected frame

latest_detection = None

loggedInUsers={}
class User:
    def __init__ (self,id):
        self.user_ID = id
        self.latest_detection = None
        self.currentExercise= None
        self.exerciseManager = None

@app.route('/webcam_feed')
def webcam_feed():
    user_id = session.get('user_id')
    return Response(generate_frames(user_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#format is anglebetweenlines(endpoint one, vertex, endpoint two)
def angleBetweenLines(a,b,c):
    a = np.array(a) # end
    b = np.array(b) # vertext 
    c = np.array(c) # End 
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def landmarks_to_pixels(pose_landmarks, image_shape):
    h, w, _ = image_shape

    pixel_landmarks = []

    for lm in pose_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pixel_landmarks.append((x, y))

    return pixel_landmarks

class SitUpState:
    IDLE="IDLE"
    DOWN = "DOWN"
    RISING="RISING"
    TOP="TOP"
class SitUpController:
    def __init__(self):
        self.state=SitUpState.IDLE
        self.count=0
        self.bodyBendAngle=0 #the idle state means the angle between the rays from hip to head and hip to ankle has to be about flat
        self.kneeAngle=0 #knees should be bent in a situp
        self.heel_anchor=None #heel shouldnt move very much
        self.last_rep_time= None
        self.intervals=[]        
        #dont care about the arms for now

    def update(self,detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
        landmarks = detection_result.pose_landmarks[0] #only get the first person
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)  
        right_hip=pixel_landmarks[RIGHT_HIP]
        right_shoulder=pixel_landmarks[RIGHT_SHOULDER] #no head landmark
        right_knee=pixel_landmarks[RIGHT_KNEE]

        right_heel=pixel_landmarks[RIGHT_HEEL]

        self.bodyBendAngle=angleBetweenLines(right_shoulder, right_hip,right_heel)
        self.kneeAngle=angleBetweenLines(right_hip,right_knee,right_heel)

        if self.state==SitUpState.IDLE:
            self.heel_anchor = np.array(right_heel)
            #wait until body bend angle is less than some number
            if self.bodyBendAngle>165:
                self.state = SitUpState.IDLE #lying flat, stay in idle
                return
            
            elif self.bodyBendAngle<165 and self.kneeAngle<110: #knees must be bent and body must be bent enough to count as up 
                #transition to up state
                self.state=SitUpState.RISING
                return
        
        elif self.state==SitUpState.RISING:
            current_heel = np.array(right_heel)
            heel_displacement = np.linalg.norm(current_heel - self.heel_anchor)

            if self.heel_anchor is None:
                self.state = SitUpState.IDLE

            if self.kneeAngle>110 or heel_displacement > 80: #knees not bent enough! go back to idle
                self.state=SitUpState.IDLE
                return

            elif self.bodyBendAngle<110: #more and more bent
                self.state=SitUpState.TOP
                return
            
        elif self.state==SitUpState.TOP:
            if self.kneeAngle>110: #knees not bent enough! go back to idle
                self.state=SitUpState.IDLE
                return
            
            elif self.bodyBendAngle>165:
                current_time = time.monotonic()
                interval=0
                if self.last_rep_time is not None:
                    interval=current_time-self.last_rep_time    
                    self.intervals.append(interval)
                    print(f"time between reps: {interval}")
                
                self.last_rep_time=current_time    
                self.count=self.count+1
                self.state=SitUpState.IDLE
                return
    
    def getIntervals(self):
        return self.intervals
    
    def draw(self,image, detection_result):
        annotated_image = image.copy()
        if not detection_result.pose_landmarks:
            return annotated_image
        h, w, _ = image.shape

        for pose_landmarks in detection_result.pose_landmarks:
            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)
            head= to_pixel(pose_landmarks[NOSE])
            right_hip = to_pixel(pose_landmarks[RIGHT_HIP])
            right_knee = to_pixel(pose_landmarks[RIGHT_KNEE])
            right_ankle = to_pixel(pose_landmarks[RIGHT_HEEL])
            
            #head to hip, hip to ankle ignoring knee
            cv2.line(annotated_image, right_hip, head, (0, 0, 255), 2)
            cv2.line(annotated_image, right_hip, right_ankle, (0, 0, 255), 2)

            #knee angle
            cv2.line(annotated_image, right_hip, right_knee, (0, 255, 0), 2)
            cv2.line(annotated_image, right_knee, right_ankle, (0, 255, 0), 2)
        return annotated_image

class SquatState:
    IDLE="IDLE"
    BEGIN = "BEGIN"
    DOWN = "DOWN"
    RISE="RISE"
class SquatController:
    def __init__(self):
        self.state = SquatState.IDLE
        self.count = 0
        self.knee_angle = 0
        self.heel_anchor = None
        self.down_start_time = None
        self.start_time = 0
        self.end_time=0
        self.last_rep_time= None
        self.intervals=[]

    def update(self, detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
        landmarks = detection_result.pose_landmarks[0] #only get the first person
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)  
        left_hip = pixel_landmarks[LEFT_HIP] #both share a point at knee
        left_knee = pixel_landmarks[LEFT_KNEE]
        left_heel = pixel_landmarks[LEFT_HEEL]

        self.left_knee_angle=angleBetweenLines(left_hip,left_knee,left_heel)

        right_hip = pixel_landmarks[RIGHT_HIP] #both share a point at knee
        right_knee = pixel_landmarks[RIGHT_KNEE]
        right_heel = pixel_landmarks[RIGHT_HEEL]

        self.right_knee_angle=angleBetweenLines(right_hip,right_knee,right_heel)
              
        if self.state == SquatState.IDLE:
            
            if self.left_knee_angle>140 and self.right_knee_angle >140:
                self.heel_anchor = np.array(left_heel)
            left_hip = np.array(left_hip)
            left_heel = np.array(left_heel)

            dx = left_heel[0] - left_hip[0]
            dy = left_heel[1] - left_hip[1]
            slope = dy / dx

            #patching the angel franco office chair cheat
            if(not (-2> slope or  slope>2)): #line from hip to ankle must be mostly vertical
                return

            
            if self.left_knee_angle<140 and self.right_knee_angle <140:
                self.state=SquatState.BEGIN
                return
            
        elif self.state==SquatState.BEGIN:
            if self.heel_anchor is None:

                self.state = SquatState.IDLE
                return
            current_heel = np.array(left_heel)
            heel_displacement = np.linalg.norm(current_heel - self.heel_anchor)
            if heel_displacement > 80:
                self.state=SquatState.IDLE
                return

            if self.left_knee_angle<80 and self.right_knee_angle <80 : #80 degree squat
                self.start_time = time.time()

                self.state = SquatState.DOWN
                self.down_start_time = time.time()
                return

        elif self.state == SquatState.DOWN:
            if self.left_knee_angle > 100 or self.right_knee_angle> 100: # User started rising too early
                if (time.time() - self.down_start_time) >= 1.0:
                    self.state =  SquatState.RISE
                else:
                    self.state = SquatState.RISE

        elif self.state ==  SquatState.RISE:
            if self.left_knee_angle <160  and self.right_knee_angle<160 :
                current_time = time.monotonic()
                interval=0
                if self.last_rep_time is not None:
                    interval=current_time-self.last_rep_time    
                    self.intervals.append(interval)
                    print(f"time between reps: {interval}")
                
                self.last_rep_time=current_time    
                self.count += 1
                self.state = SquatState.IDLE
                print(f"Count: {self.count}")
    
    def getIntervals(self):
        return self.intervals
    
    def draw(self,image, detection_result):
        annotated_image = image.copy()
        if not detection_result.pose_landmarks:
            return annotated_image
        h, w, _ = image.shape

        for pose_landmarks in detection_result.pose_landmarks:
            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)
            left_hip = to_pixel(pose_landmarks[LEFT_HIP])
            left_knee = to_pixel(pose_landmarks[LEFT_KNEE])
            left_ankle = to_pixel(pose_landmarks[LEFT_HEEL])
            right_hip = to_pixel(pose_landmarks[RIGHT_HIP])
            right_knee = to_pixel(pose_landmarks[RIGHT_KNEE])
            right_ankle = to_pixel(pose_landmarks[RIGHT_HEEL])
            
            cv2.line(annotated_image, left_hip, left_ankle, (255, 0, 0), 2)

            cv2.line(annotated_image, left_hip, left_knee, (0, 255, 0), 2)
            cv2.line(annotated_image, left_knee, left_ankle, (0, 255, 0), 2)
            cv2.line(annotated_image, right_hip, right_knee, (0, 255, 0), 2)
            cv2.line(annotated_image, right_knee, right_ankle, (0, 255, 0), 2)
        return annotated_image

class LungeState:
    IDLE="IDLE"
    DESCENDING="DESCENDING" #left leg forward
    ASCENDING="ASCENDING"
    DOWN="DOWN"
class LungeController:
    def __init__(self):
        self.state=LungeState.IDLE
        self.count=0
        self.leftKneeAngle=0
        self.rightKneeAngle=0
        self.heelToHeelDistance=0
        self.calfLength=0 #this is a constant
        self.idleHipHeight=0
        self.last_rep_time= None
        self.intervals=[]
    
    def update(self,detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
        landmarks = detection_result.pose_landmarks[0] 
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)  
        left_hip = pixel_landmarks[LEFT_HIP] #both share a point at knee
        left_knee = pixel_landmarks[LEFT_KNEE]
        left_heel = pixel_landmarks[LEFT_HEEL]

        self.left_knee_angle=angleBetweenLines(left_hip,left_knee,left_heel)

        right_hip = pixel_landmarks[RIGHT_HIP] #both share a point at knee
        right_knee = pixel_landmarks[RIGHT_KNEE]
        right_heel = pixel_landmarks[RIGHT_HEEL]

        right_knee = np.array(right_knee)
        right_heel = np.array(right_heel)
        self.calfLength=np.linalg.norm(right_knee-right_heel) #this is a constant!!!!

        self.right_knee_angle=angleBetweenLines(right_hip,right_knee,right_heel)


        if self.state==LungeState.IDLE:

            left_heel = np.array(left_heel)
            right_heel = np.array(right_heel)
                
            self.heelToHeelDistance=np.linalg.norm(left_heel-right_heel) #this is a constant!!!!
            
            self.idleHipHeight = right_hip[1]

            if(abs(self.heelToHeelDistance)> 1.3 * self.calfLength):

                self.state=LungeState.DESCENDING
                return
            
        elif self.state==LungeState.DESCENDING:
            pass

            left_heel = np.array(left_heel)
            right_heel = np.array(right_heel)
                
            self.heelToHeelDistance=np.linalg.norm(left_heel-right_heel) 

            if(abs(self.heelToHeelDistance)< 1.3 * self.calfLength):
                self.state=LungeState.IDLE
                return
            rightCalfSlope = (right_knee[1]-right_heel[1])  / (right_knee[0]-right_heel[0])
            leftCalfSlope = (left_knee[1]-right_heel[1])  / (left_knee[0]-left_heel[0])
    
            self.right_knee_angle=angleBetweenLines(right_hip,right_knee,right_heel)
            self.left_knee_angle=angleBetweenLines(left_hip,left_knee,left_heel)
            backLeg="dumb"
            frontLeg="dummy"
            if(abs(rightCalfSlope) <0.75 ): #if the slope of the right calf is near flat
                frontLeg="left"
                backLeg="right"
            elif(abs(leftCalfSlope)<0.75):
                frontLeg="right"
                backLeg="left"

            if( frontLeg=="right"  and self.left_knee_angle < 110):
                self.state=LungeState.DOWN
                return
            elif(frontLeg=="left" and self.right_knee_angle< 110):
                self.state=LungeState.DOWN
                return
 
        elif self.state==LungeState.DOWN:

            left_heel = np.array(left_heel)
            right_heel = np.array(right_heel)
                
            self.heelToHeelDistance=np.linalg.norm(left_heel-right_heel) #this is a constant!!!!
            if(abs(self.heelToHeelDistance) < self.calfLength *1.3):
                self.state=LungeState.ASCENDING
                self.count=self.count+1
        
        elif self.state==LungeState.ASCENDING:
            self.right_knee_angle=angleBetweenLines(right_hip,right_knee,right_heel)
            self.left_knee_angle=angleBetweenLines(left_hip,left_knee,left_heel)
            if (self.right_knee_angle > 140 and self.left_knee_angle > 140):
                current_time = time.monotonic()
                if self.last_rep_time is not None:
                    interval = current_time - self.last_rep_time
                    self.intervals.append(interval)
                    print(f"Interval recorded: {interval:.2f}s")
                
                # ALWAYS update the last_rep_time, even on the first rep
                self.last_rep_time = current_time 
                
                self.state = LungeState.IDLE
                return
    def getIntervals(self):
        return self.intervals
    def draw(self,image, detection_result):
        annotated_image = image.copy()
        if not detection_result.pose_landmarks:
            return annotated_image
        h, w, _ = image.shape

        for pose_landmarks in detection_result.pose_landmarks:
            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)
            left_hip = to_pixel(pose_landmarks[LEFT_HIP])
            left_knee = to_pixel(pose_landmarks[LEFT_KNEE])
            left_ankle = to_pixel(pose_landmarks[LEFT_HEEL])
            right_hip = to_pixel(pose_landmarks[RIGHT_HIP])
            right_knee = to_pixel(pose_landmarks[RIGHT_KNEE])
            right_ankle = to_pixel(pose_landmarks[RIGHT_HEEL])

            left_ankle = np.array(left_ankle)
            right_ankle = np.array(right_ankle)
            
            ankleToAnkleDistance=np.linalg.norm(left_ankle-right_ankle) #this is a constant!!!!

            right_knee = np.array(right_knee)
            rightCalfLength=np.linalg.norm(right_knee-right_ankle) #this is a constant!!!!
            
            if(abs(ankleToAnkleDistance) > 1.5 * rightCalfLength):
                cv2.line(annotated_image, left_ankle, right_ankle, (0, 0, 255), 2)

            else:
                cv2.line(annotated_image, left_ankle, right_ankle, (0, 255, 0), 2)
            
            cv2.line(annotated_image, right_knee, right_ankle, (255, 0, 0), 2)
            cv2.line(annotated_image, left_hip, left_knee, (0, 255, 0), 2)
            cv2.line(annotated_image, left_knee, left_ankle, (0, 255, 0), 2)
            cv2.line(annotated_image, right_hip, right_knee, (0, 255, 0), 2)
        return annotated_image

class RunningState:
    TIMER = "TIMER"
class RunningController:
    def __init__(self):
        self.state = RunningState.TIMER
        self.count = 0

    def update(self, detection_result,image_shape):
        return

    def draw(self, image,detection_result):
        if image is None:
            return None
        # No extra drawing.
        return image.copy()

class JumpingJackState:
    TIMER = "TIMER"
class JumpingJacksController:
    def __init__(self):
        self.state = JumpingJackState.TIMER
        self.count = 0

    def update(self, detection_result,image_shape):
        return

    def draw(self,  image,detection_result):
        if image is None:
            return None
        return image.copy()

class GluteBridgeState():
    IDLE = "IDLE"
    UP = "UP"
class GluteBridgeController():
    def __init__(self):
        self.state = GluteBridgeState.IDLE
        self.count = 0
        self.hipAngle = 0
        self.kneeAngle = 0
        self.is_lying_down = False
    def update(self, detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
        landmarks = detection_result.pose_landmarks[0]
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)

        shoulder = pixel_landmarks[RIGHT_SHOULDER]
        hip = pixel_landmarks[RIGHT_HIP]
        knee = pixel_landmarks[RIGHT_KNEE]
        ankle = pixel_landmarks[RIGHT_HEEL]

        dx = abs(hip[0] - shoulder[0])
        dy = abs(hip[1] - shoulder[1])

        if dy > dx:
            self.is_lying_down = False
            return
        else:
            self.is_lying_down = True

        self.knee_angle = angleBetweenLines(hip, knee, ankle)
        self.hip_angle = angleBetweenLines(shoulder, hip, knee)

        if self.knee_angle > 135:
            return

        if self.state == GluteBridgeState.IDLE:
            if self.hip_angle > 165:
                self.state = GluteBridgeState.UP
        elif self.state == GluteBridgeState.UP:
            if self.hip_angle < 140:
                self.count+=1
                self.state = GluteBridgeState.IDLE

    def draw(self,image, detection_result):
        annotated_image = image.copy()
        if not detection_result.pose_landmarks:
            return annotated_image
        h, w, _ = image.shape
        for pose_landmarks in detection_result.pose_landmarks:
            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)

            shoulder = to_pixel(pose_landmarks[RIGHT_SHOULDER])
            hip = to_pixel(pose_landmarks[RIGHT_HIP])
            knee = to_pixel(pose_landmarks[RIGHT_KNEE])
            ankle = to_pixel(pose_landmarks[RIGHT_HEEL])

            cv2.line(annotated_image, shoulder, hip, (0, 255, 0), 4)
            cv2.line(annotated_image, hip, knee, (0, 0, 255), 4)
            cv2.line(annotated_image, knee, ankle, (255, 0, 0), 4)
            
            cv2.circle(annotated_image, hip, 6, (255, 255, 255), -1)
            cv2.circle(annotated_image, knee, 6, (255, 255, 255), -1)
            cv2.circle(annotated_image, ankle, 6, (0, 255, 255), -1)
            
        return annotated_image

class SupermanState:
    IDLE = "IDLE"
    UP = "UP"
class SupermanController:
    def __init__(self):
        self.state = SupermanState.IDLE
        self.count = 0
        self.angle = 0
    def update(self,detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
            
        landmarks = detection_result.pose_landmarks[0]
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)
        
        shoulder = pixel_landmarks[RIGHT_SHOULDER]
        hip = pixel_landmarks[RIGHT_HIP]
        knee = pixel_landmarks[RIGHT_KNEE]
        
        self.back_angle = angleBetweenLines(shoulder, hip, knee)
        
        if self.state == SupermanState.IDLE:
            if self.back_angle < 165:
                self.state = SupermanState.UP
                
        elif self.state == SupermanState.UP:
            if self.back_angle > 175:
                self.count += 1
                self.state = SupermanState.IDLE
    def draw(self, image, detection_result):
        annotated_image = image.copy()
        if not detection_result.pose_landmarks:
            return annotated_image
        h, w, _ = image.shape

        for pose_landmarks in detection_result.pose_landmarks:
            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)
            
            shoulder = to_pixel(pose_landmarks[RIGHT_SHOULDER])
            hip = to_pixel(pose_landmarks[RIGHT_HIP])
            knee = to_pixel(pose_landmarks[RIGHT_KNEE])
            
            color = (0, 255, 0) if self.state == GluteBridgeState.UP else (0, 0, 255)
            
            cv2.line(annotated_image, shoulder, hip, color, 4)
            cv2.line(annotated_image, hip, knee, color, 4)
        return annotated_image

IDEAL_BODY_ANGLE = 180
BODY_ANGLE_TOLERANCE = 15     # +-15 allowed

IDEAL_BODY_SLOPE = 0
BODY_SLOPE_TOLERANCE = 35     # px

MIN_ELBOW_DOWN = 90
MAX_ELBOW_UP   = 165

class PushUpState:
    IDLE = "IDLE"
    DOWN = "DOWN"

class PushUpController:
    def __init__(self):
        self.state = PushUpState.IDLE
        self.count = 0
    def update(self, detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
        landmarks = detection_result.pose_landmarks[0] 
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)

        shoulder = pixel_landmarks[RIGHT_SHOULDER]
        elbow = pixel_landmarks[RIGHT_ELBOW]
        wrist = pixel_landmarks[RIGHT_WRIST]
        hip = pixel_landmarks[RIGHT_HIP]
        knee = pixel_landmarks[RIGHT_KNEE]

        self.elbow_angle = angleBetweenLines(shoulder, elbow, wrist)
        self.body_alignment = angleBetweenLines(shoulder, hip, knee)

        if self.state == PushUpState.IDLE:
            if self.elbow_angle < 85 and self.body_alignment > 150:
                self.state = PushUpState.DOWN
        elif self.state == PushUpState.DOWN:
            if self.elbow_angle > 160:
                self.count += 1
                self.state = PushUpState.IDLE
    def draw(self, image, detection_result):
        annotated_image = image.copy()
        if not detection_result.pose_landmarks:
            return annotated_image
        h, w, _ = image.shape

        for pose_landmarks in detection_result.pose_landmarks:
            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)
            
            shoulder = to_pixel(pose_landmarks[RIGHT_SHOULDER])
            elbow = to_pixel(pose_landmarks[RIGHT_ELBOW])
            wrist = to_pixel(pose_landmarks[RIGHT_WRIST])
            hip = to_pixel(pose_landmarks[RIGHT_HIP])

            color = (0, 255, 0) if self.state == PushUpState.DOWN else (0, 0, 255)
            

            cv2.line(annotated_image, shoulder, elbow, color, 4)
            cv2.line(annotated_image, elbow, wrist, color, 4)
            cv2.line(annotated_image, shoulder, hip, (255, 255, 0), 2)
        return annotated_image

sitUpController = SitUpController()
squatController = SquatController()
lungeController = LungeController()
runningController = RunningController()
jumpingjacksController = JumpingJacksController()
pushupController = PushUpController()
supermanController = SupermanController()
glutebridgeController = GluteBridgeController()

class ExerciseManager:

    def __init__(self, today_exercises):
        self.all_controllers = {
            "squats": SquatController(),
            "situps": SitUpController(),
            "lunges": LungeController(),
            "pushups" : PushUpController(),
            "running" : RunningController(), 
            "jumpingjacks" : JumpingJacksController(),
            "glutebridges" : GluteBridgeController(),
            "supermans" : SupermanController()

        }

        self.exercises = {}

        self.allExerciseIntervals={}
        self.workout_begin_time=None
        self.workout_end_time=None

        self.total_reps=None

        for ex in today_exercises:

            clean = normalize_name(ex)

            if clean in self.all_controllers:
                self.exercises[clean] = self.all_controllers[clean]

        self.currentExercise = (
            list(self.exercises.keys())[0]
            if self.exercises else None
        )


    def getCurrentExercise(self): #this getse the CONTROLLER
        if not self.currentExercise:
            return None
        return self.exercises[self.currentExercise]

    def setCurrentExercise(self, name):
        clean = normalize_name(name)
        if clean in self.exercises:
            self.currentExercise = clean

    def finalize_all_intervals(self):
        """Call this right before workoutcomplete or printing stats."""
        for name, controller in self.exercises.items():
            # Only save if there's actually data
            if hasattr(controller, 'intervals') and controller.intervals:
                self.allExerciseIntervals[name] = controller.intervals

@app.route('/switch_exercise', methods=["POST"])
def switch_exercise():

    user_id = session.get("user_id")

    if not user_id or user_id not in loggedInUsers:
        return jsonify(error="User not active")

    currentUser = loggedInUsers[user_id]
    old_exercise_name = currentUser.exerciseManager.currentExercise
    currentExerciseController=currentUser.exerciseManager.getCurrentExercise()
    intervals=currentExerciseController.getIntervals()
    currentUser.exerciseManager.allExerciseIntervals[old_exercise_name]=intervals


    if not currentUser.exerciseManager:
        return jsonify(error="No active workout")

    data = request.get_json()
    new_exercise = data.get('exercise')

    currentUser.exerciseManager.setCurrentExercise(new_exercise)


    return jsonify(status="success", now_doing=new_exercise)

@app.route('/get_exercise_data')
def get_exercise_data():

    user_id = session.get("user_id")

    if not user_id or user_id not in loggedInUsers:
        return jsonify(error="User not active")

    currentUser = loggedInUsers[user_id]

    if not currentUser.exerciseManager:
        return jsonify(error="No active workout")

    multiple_detected = False

    if currentUser.latest_detection and currentUser.latest_detection.pose_landmarks:
        if len(currentUser.latest_detection.pose_landmarks) > 1:
            multiple_detected = True

    currentExercise = currentUser.exerciseManager.getCurrentExercise()

    if not currentExercise:
        return jsonify(error="No exercise loaded")

    return jsonify(
        currentExercise=currentUser.exerciseManager.currentExercise,
        count=currentExercise.count,
        state=currentExercise.state,
        multiple_detected=multiple_detected
    )

def generate_frames(user_id):
    if not user_id in loggedInUsers:
        return
    currentUser= loggedInUsers[user_id]

    

    global camera
    while True:

        success, frame = camera.read()
        if frame is None:
            continue
        if not success:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        currentUser.latest_detection = detector.detect(mp_image)     
        currentUser.currentExercise = currentUser.exerciseManager.getCurrentExercise()
        currentUser.currentExercise.update(currentUser.latest_detection, frame.shape)

        annotated_image = currentUser.currentExercise.draw(frame, currentUser.latest_detection)
        ret, buffer = cv2.imencode('.jpg', annotated_image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

all_selected_exercises = {}

# ---------------- EXERCISES ----------------
upper_body = ["pushups"]
lower_body = ["squats", "lunges","glutebridges"]
core = ["situps", "supermans"]
cardio = ["jumpingjacks",  "running"]

new_people_exercises = [
    "Glute Bridges", "Running", "Jumping Jacks",
    "Lunges", "Push-ups", "Sit-ups", "Squats", "Supermans"
]

# ---------------- REPS ----------------

#these are categories of exercises and suggested rep counts
rep_ranges = {
    "Beginner": {
        "core": "5",
        "lower_body": "6", 
        "upper_body": "3", 
        "cardio":"30"
    }
}

# ---------------- HELPERS ----------------

def pick_random(ex_list, num):
    return random.sample(ex_list, k=min(num, len(ex_list)))

def normalize_name(name):
    return name.lower().replace("-", "").replace(" ", "")

def format_exercise(ex, category, day_key):
    rows = ""
    for _, reps in rep_ranges.items():
        all_selected_exercises[day_key].append([ex, category, reps[category]])
        rows += f"{ex:<20} | {category:<10} | {reps[category]}\n"
    return rows

# ---------------- GENERATE PLAN ----------------
def generate_workout_plan(goal, days_per_week, body_part):
    all_selected_exercises.clear()
    plan = f"Goal: {goal}\nWorkouts per Week: {days_per_week}\n\n"

    mode = {
        "get fit": "balanced",
        "lose weight": "cardio",
        "gain strength": "strength"

    }.get(goal, "balanced")

    body_map = {
        "legs": lower_body,
        "arms": upper_body,
        "abdomen": core
    }

    focus = body_map.get(body_part, upper_body)

    for day in range(1, days_per_week + 1):

        day_key = f"day_{day}"
        all_selected_exercises[day_key] = []
        plan += f"DAY {day}\n"

        if mode == "cardio":
            ex = pick_random(cardio, 1)[0]
            plan += format_exercise(ex, "cardio", day_key)

        elif mode == "balanced":
            ex = pick_random(cardio, 1)[0]
            plan += format_exercise(ex, "cardio", day_key)
            for ex in pick_random(lower_body,2):
                plan += format_exercise(ex, "lower_body", day_key)
            
            ex = pick_random(upper_body, 1)[0]
            plan += format_exercise(ex, "upper_body", day_key)

            for ex in pick_random(core,2):
                plan += format_exercise(ex, "core", day_key)
        
        elif mode =="strength":

            for ex in pick_random(lower_body,2): #pick 2 lower body
                plan += format_exercise(ex, "lower_body", day_key)

            for ex in pick_random(core,2): #pick 2 core
                plan += format_exercise(ex, "core", day_key)

            for ex in pick_random(upper_body,1): #pick 1 upper body
                plan += format_exercise(ex, "upper_body", day_key)

    return plan



def get_user_exercises(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT goal_other, workouts_per_week
        FROM UserLogins
        WHERE username = ?
    """, (username,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return [], 0

    exercises = json.loads(row["goal_other"])

    return exercises, row["workouts_per_week"]

def get_weekly_workouts(username):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT goal_other, workouts_per_week
        FROM UserLogins
        WHERE username=?
    """, (username,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    workouts = json.loads(row["goal_other"])
    days_per_week = row["workouts_per_week"]

    week = {}

    for i in range(1, 8):  # Mon–Sun
        key = f"day_{i}"
        week[key] = [{"name":ex[0], "reps":ex[2]} for ex in workouts.get(key, [])] #exercises are stored as "name","category" and then "reps"

    return week

def get_today_exercises(username, day_number=1):

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT goal_other FROM UserLogins WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return []

    workouts = json.loads(row["goal_other"])
    day_key = f"day_{day_number}"

    if day_key not in workouts:
        return []

    return [{"name": ex[0], "reps": ex[2]} for ex in workouts[day_key]]



# -------------------------
# Login
# -------------------------
@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    user_id = session.get('user_id')
    if user_id not in loggedInUsers:
        return jsonify(status="error"), 401
        
    currentUser = loggedInUsers[user_id]
    # Fetch the controller instance
    current_ex_obj = currentUser.exerciseManager.getCurrentExercise()
    
    current_ex_obj.count = 0
    current_ex_obj.state = "IDLE" 
    
    return jsonify({
        "status": "success", 
        "message": f"Counter reset for {currentUser.exerciseManager.currentExercise}"
    })

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        raw_password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM UserLogins WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], raw_password):
            session['user_id'] = user["id"]
            session['username'] = username

            loggedInUsers.update({user["id"]: User(user["id"])})

            return redirect(url_for('home'))

        flash("Invalid username or password")
        return redirect(url_for('login'))

    return render_template('login.html')

def send_fit_email(recipient_email, subject, html_content):
    sender_email = "fitcompass8@gmail.com"
    sender_password = "pbfp hulf zrvi qumq" # Use Google App Password
    
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        print(f"Email failed: {e}")

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        coins = 1000

        goal = request.form.get('goal')
        goal_other = request.form.get('goal_other') if goal == 'other' else None
        workouts_per_week = int(request.form.get('workouts_per_week', 0))
        body_part = request.form.get('body_part')

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            
            workout_plan = generate_workout_plan(goal, workouts_per_week, body_part)

          
            current_workout = 0

            if 'all_selected_exercises' in globals():
                goal_other = json.dumps(all_selected_exercises)

           
            history = json.dumps([workout_plan])

            cursor.execute("""
                INSERT INTO UserLogins
                (username, email, password, goal, goal_other, workouts_per_week,
                 body_part, workout_plan, history, current_workout,coins)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            """, (
                username,
                email,
                password,
                goal,
                goal_other,
                workouts_per_week,
                body_part,
                workout_plan,
                history,
                current_workout,
                coins
            ))

            conn.commit()
            conn.close()
            # Prepare Welcome Email
            welcome_html = f"""
                <h1>Welcome to Fit Compass, {username}!</h1>
                <p>We are excited to help you reach your goal: <strong>{goal}</strong>.</p>
                <p>Your journey starts now. Log in and get moving!</p>
            """
            send_fit_email(email, "Welcome to Fit Compass!", welcome_html)

            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            conn.close()
            flash("Username or email already exists")
            return redirect(url_for('register'))
    return render_template('register.html')
# -------------------------
# Home
# -------------------------
@app.route('/home')
def home():

    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT coins, current_workout, workouts_per_week ,equipped_outfit
        FROM UserLogins
        WHERE username = ?
    """, (username,))

    row = cursor.fetchone()
    conn.close()

    if row:
        coins = int(row["coins"] or 0)
        current_workout = int(row["current_workout"] or 0)
        workouts_per_week = int(row["workouts_per_week"] or 1)
    else:
        coins = 0
        current_workout = 0
        workouts_per_week = 1

   
    if workouts_per_week > 0:
        goal_percent = int((current_workout / workouts_per_week) * 100)
    else:
        goal_percent = 0


    equipped = row["equipped_outfit"] if row and row["equipped_outfit"] else "Business"
    weekly_workouts = get_weekly_workouts(username)

    return render_template(
        "home.html",
        username=username,
        weekly_workouts=weekly_workouts,
        goal_percent=goal_percent,
        points=coins,
        equipped=equipped
    )

@app.route('/set_day', methods=['POST'])
def set_day():
    session['selected_day'] = request.get_json()['day']
    return jsonify(status="success")


#this func is ai
@app.route('/start_workout', methods=['POST'])
def start_workout():
    if "username" not in session or session.get("user_id") not in loggedInUsers:
        return "", 403

    user_id = session["user_id"]
    loggedInUsers[user_id].exerciseManager.workout_begin_time = time.time()

    return "", 204

@app.route('/workoutSession')
#when workoutsession is loaded, grab initialize exercise manager with exercises
def workoutSession():
    if "username" not in session or session.get("user_id") not in loggedInUsers:
        return redirect(url_for("login"))
    user_id = session["user_id"]
    day_number = session.get('selected_day', 1)
    today_exercises = get_today_exercises(session["username"], day_number)
    if not today_exercises:
        today_exercises = [{"name": "squats", "reps": "5"}]

    names = [ex["name"] for ex in today_exercises]

    loggedInUsers[user_id].exerciseManager = ExerciseManager(names)

    # loggedInUsers[user_id].exerciseManager.workout_begin_time=time.time()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT equipped_outfit FROM UserLogins WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    return render_template("workoutSession.html", exercises=today_exercises,
            equipped=user["equipped_outfit"] if user["equipped_outfit"] else "")

@app.route('/library', methods=['GET','POST'])
def library():
    if "username" not in session or session.get("user_id") not in loggedInUsers:
        session.clear()
        return redirect(url_for("login"))
        
    user_id = session["user_id"]
    if request.method == 'POST':
        choice = request.form.get('routines')
        
        routines = {
            "I_Love_Pushups": [
                {"name": "pushups", "reps": "3"},
                {"name": "running", "reps": "45"},
                {"name": "pushups", "reps": "3"},
                {"name": "running", "reps": "45"},
                {"name": "pushups", "reps": "3"},
                {"name": "running", "reps": "45"},
                {"name": "pushups", "reps": "3"}
            ],
            "Legs_Are_Great": [
                {"name": "squats", "reps": "3"},
                {"name": "lunges", "reps": "3"},
                {"name": "glutebridges", "reps": "3"},
                {"name": "running", "reps": "45"},
                {"name": "squats", "reps": "3"},
                {"name": "lunges", "reps": "3"},
                {"name": "glutebridges", "reps": "3"}
            ],
            "cardio_cardio_cardio": [
                {"name": "running", "reps": "45"},
                {"name": "jumpingjacks", "reps": "45"},
                {"name": "lunges", "reps": "3"},
                {"name": "running", "reps": "45"},
                {"name": "jumpingjacks", "reps": "45"},
                {"name": "lunges", "reps": "3"}
            ],
            "MichalisMode1": [
                {"name": "squats", "reps": "1"},
                {"name": "lunges", "reps": "1"},
            ],
            "MichalisMode2": [
                {"name": "squats", "reps": "2"},
                {"name": "lunges", "reps": "2"},
            ],
            "PresentationDemo": [
                {"name": "squats", "reps": "4"},
                {"name": "situps", "reps": "6"},
                {"name": "lunges", "reps": "4"},
            ],
        }
        
        if choice in routines:
            chosen_routine = routines[choice]
        
            names_only = [ex["name"] for ex in chosen_routine]
            loggedInUsers[user_id].exerciseManager = ExerciseManager(names_only) 
            
            return render_template("workoutSession.html", exercises=chosen_routine) 
            
    return render_template("library.html")

import statistics

#flask route
@app.route('/get_interval_std')
def get_consistency_stats():
    if "username" not in session or session.get("user_id") not in loggedInUsers:
        return redirect(url_for("login"))
    user_id = session["user_id"]
    loggedInUsers[session["user_id"]].exerciseManager.finalize_all_intervals()
    intervalsMap=loggedInUsers[user_id].exerciseManager.allExerciseIntervals
    consistencyScoreMap = {}
    for exercise_name, intervals in intervalsMap.items():
        if len(intervals)>1:
            std_dev = statistics.stdev(intervals)
            cappedStd = min(std_dev, 20)
            score = (1 - cappedStd / 20) * 100
            consistencyScoreMap[exercise_name]=round(score, 2)
        else:
            consistencyScoreMap[exercise_name]=100
    return jsonify(consistencyScoreMap)

#non flask route for sending to email.
def get_formatted_consistency_scores(user_id):
    if user_id not in loggedInUsers:
        return None     
    manager = loggedInUsers[user_id].exerciseManager
    manager.finalize_all_intervals()
    intervals_map = manager.allExerciseIntervals
    
    scores = {}
    for exercise_name, intervals in intervals_map.items():
        if len(intervals) > 1:
            std_dev = statistics.stdev(intervals)
            capped_std = min(std_dev, 20)
            score = (1 - capped_std / 20) * 100
            scores[exercise_name] = round(score, 2)
        else:
            scores[exercise_name] = 100
    html = '<ul class="consistency-list">'
    for exercise, score in scores.items():
        color = "green" if score > 60 else "red"
        html += f'<li>{exercise.capitalize()}: <span style="color:{color}">{score}%</span></li>'
    html += '</ul>'
    return html
    #now consistency scores is moved away from the flask route
   
@app.route('/send_total_reps', methods=['POST'])
def send_total_reps():
    data = request.get_json()
    total_reps = data.get("total_reps")
    user_id = session["user_id"]
    loggedInUsers[user_id].exerciseManager.total_reps = total_reps

    return {"status": "ok"}

@app.route('/workoutcomplete')
def workoutcomplete():

    # if "username" not in session:
    #     return redirect(url_for("login"))
    if "username" not in session or session.get("user_id") not in loggedInUsers:
        return redirect(url_for("login"))
    user_id = session["user_id"]
    loggedInUsers[session["user_id"]].exerciseManager.finalize_all_intervals()
    pprint(loggedInUsers[user_id].exerciseManager.allExerciseIntervals)
    
    loggedInUsers[user_id].exerciseManager.workout_end_time=time.time()
    workout_duration=loggedInUsers[user_id].exerciseManager.workout_end_time-loggedInUsers[user_id].exerciseManager.workout_begin_time
    intervalsMap=loggedInUsers[user_id].exerciseManager.allExerciseIntervals

    total_reps = loggedInUsers[user_id].exerciseManager.total_reps

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT email, coins, current_workout, equipped_outfit
        FROM UserLogins
        WHERE username = ?
    """, (session["username"],))

    user = cursor.fetchone()

    if user:
        user_email = user["email"]
        current_coins = int(user["coins"] or 0)
        current_workout = int(user["current_workout"] or 0)

        new_coins = current_coins + 150
        new_current_workout = current_workout + 1

        cursor.execute("""
            UPDATE UserLogins
            SET coins = ?, current_workout = ?
            WHERE username = ?
        """, (
            new_coins,
            new_current_workout,
            session["username"]
        ))
        conn.commit()
        consistency_snippet = get_formatted_consistency_scores(user_id)
        report_html = f"""
            <div style="font-family: sans-serif;">
                <h2>Workout Summary!</h2>
                <p><strong>Total Reps:</strong> {total_reps}</p>
                <p><strong>Duration:</strong> {workout_duration:.1f} seconds</p>
                <hr>
                {consistency_snippet}
            </div>
        """
        send_fit_email(user_email, "Your Fit Compass Workout Report", report_html)
        equipped = user["equipped_outfit"] if user["equipped_outfit"] else "business"
    conn.close()

    
    '''consistency_snippet = get_formatted_consistency_scores(user_id)
    msg = MIMEMultipart()
    msg['Subject'] = "Workout Summary!"
    msg['From'] =  "fitcompass8@gmail.com" #dummy fitcompass email
    msg['To'] = "amberlin618@gmail.com" #change this to the email the user provides
    report_html = f"""
        <div class="workout-report">
            <h2>Workout Summary!</h2>
            <p><strong>Total Reps:</strong> {total_reps} reps</p>
            <p><strong>Duration:</strong> {workout_duration:.1f} seconds</p>
            <h3>Consistency Scores:</h3>
            {consistency_snippet}
        </div>
        """
    # part = MIMEText(report_html, 'html')
    # msg.attach(part)
    # smtp = smtplib.SMTP('smtp.gmail.com', 587)
    # smtp.starttls()
    #first argument is email, second is google app password
    #smtp.login('sender email@gmail.com', 'sender app password')
    smtp.sendmail(msg['From'], [msg['To']], msg.as_string())   
    smtp.quit()''' 
    

    return render_template("workoutcomplete.html",this_workout_duration= workout_duration,this_total_reps=total_reps,equipped=equipped )

# -------------------------
# Placeholder
# -------------------------
@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, email, goal,
               workouts_per_week,
               coins,
               current_workout,
               equipped_outfit
        FROM UserLogins
        WHERE username = ?
    """, (session["username"],))

    user = cursor.fetchone()
    conn.close()

    if not user:
        return redirect(url_for("login"))

    workouts_per_week = int(user["workouts_per_week"] or 1)
    current_workout = int(user["current_workout"] or 0)

    if workouts_per_week > 0:
        goal_percent = int((current_workout / workouts_per_week) * 100)
    else:
        goal_percent = 0

    if goal_percent >= 100:
        goal_percent = 100

    equipped = user["equipped_outfit"] if user and user["equipped_outfit"] else "Business"

    return render_template(
        "profile.html",
        username=user["username"],
        email=user["email"],
        goal=user["goal"],
        workouts_per_week=workouts_per_week,
        points=int(user["coins"] or 0),
        goal_percent=goal_percent,
        equipped=equipped
    )

@app.route("/profileEdit", methods=["GET", "POST"])
def profileEdit():

    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == "POST":

        new_email = request.form["email"]
        goal = request.form["goal"]
        workouts_per_week = int(request.form["workouts_per_week"])
        body_part = request.form["body_part"]
        current_workout = 0

        workout_plan = generate_workout_plan(goal, workouts_per_week, body_part)

        if 'all_selected_exercises' in globals():
            goal_other = json.dumps(all_selected_exercises)
        else:
            goal_other = None

        cursor.execute(
            "SELECT email, history FROM UserLogins WHERE username = ?",
            (session["username"],)
        )
        row = cursor.fetchone()

        old_email = row["email"]

        cursor.execute(
            "SELECT username FROM UserLogins WHERE email = ? AND username != ?",
            (new_email, session["username"])
        )
        existing_user = cursor.fetchone()

        if existing_user:
            email_to_save = old_email
        else:
            email_to_save = new_email

        if row and row["history"]:
            history = json.loads(row["history"])
        else:
            history = []

        history.append(workout_plan)
        updated_history = json.dumps(history)

        cursor.execute("""
            UPDATE UserLogins
            SET email = ?, 
                goal = ?, 
                workouts_per_week = ?, 
                body_part = ?, 
                workout_plan = ?,
                history = ?,
                goal_other = ?,
                current_workout =?
            WHERE username = ?
        """, (
            email_to_save,
            goal,
            workouts_per_week,
            body_part,
            workout_plan,
            updated_history,
            goal_other,
            current_workout,
            session["username"]
        ))

        conn.commit()
        conn.close()

        if existing_user:
            flash("Email already exists. Kept your old email. Profile updated!")
        else:
            flash("Profile updated and workout plan added to history!")

        return redirect(url_for("profile"))

    cursor.execute("""
        SELECT email, goal, goal_other,
               workouts_per_week, 
               body_part, workout_plan
        FROM UserLogins
        WHERE username = ?
    """, (session["username"],))

    user = cursor.fetchone()
    conn.close()

    return render_template(
        "profileEdit.html",
        email=user["email"],
        goal=user["goal"],
        goal_other=user["goal_other"],
        workouts_per_week=user["workouts_per_week"],
        body_part=user["body_part"]
    )




@app.route("/workoutLog")
def workoutLog():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT coins, history
        FROM UserLogins
        WHERE username = ?
    """, (session["username"],))

    user = cursor.fetchone()
    conn.close()

    if not user:
        return redirect(url_for("login"))

    
    if user["history"]:
        history = json.loads(user["history"])
    else:
        history = []

    return render_template(
        "workoutLog.html",
        points=user["coins"],
        history=history
    )

@app.route('/history')
def history():
    return redirect(url_for('workoutLog'))



@app.route('/shop')
def shop():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()

    #get user data (Coins & Equipped)
    cursor.execute("SELECT coins, equipped_outfit FROM UserLogins WHERE id=?", (uid,))
    user = cursor.fetchone()

    #make sure login
    if user is None:
        conn.close()
        session.clear()
        return redirect(url_for("login"))

    #get outfits
    cursor.execute("SELECT outfit_name FROM UserOutfits WHERE user_id=?", (uid,))
    owned = [row["outfit_name"] for row in cursor.fetchall()]

    conn.close()

    return render_template(
        "shop.html",
        coins=user["coins"],
        owned=owned,
        equipped=user["equipped_outfit"] if user["equipped_outfit"] else ""
    )

@app.route('/buy_costume', methods=["POST"])
def buy_costume():
    if "user_id" not in session:
        return jsonify(status="error")

    data = request.get_json()
    outfit = data["costume"]
    cost = data["cost"]
    uid = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor()

    #get coins
    cursor.execute("SELECT coins FROM UserLogins WHERE id=?", (uid,))
    user = cursor.fetchone()

    if not user or user["coins"] < cost:
        conn.close()
        return jsonify(status="not_enough")

    #check owned outfts
    cursor.execute("SELECT * FROM UserOutfits WHERE user_id=? AND outfit_name=?", (uid, outfit))
    if cursor.fetchone():
        conn.close()
        return jsonify(status="already_owned")

    #buy and save new outfits
    try:
        new_coins = user["coins"] - cost
        cursor.execute("UPDATE UserLogins SET coins=? WHERE id=?", (new_coins, uid))
        cursor.execute("INSERT INTO UserOutfits (user_id, outfit_name) VALUES (?, ?)", (uid, outfit))
        conn.commit()
        status = "success"
    except sqlite3.Error:
        conn.rollback()
        status = "error"
    finally:
        conn.close()

    return jsonify(status=status, new_coins=new_coins)

@app.route('/wear_costume', methods=["POST"])
def wear_costume():
    if "user_id" not in session:
        return jsonify(status="error")

    data = request.get_json()
    outfit = data["costume"]
    uid = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor()

    #check they own it
    cursor.execute("SELECT * FROM UserOutfits WHERE user_id=? AND outfit_name=?", (uid, outfit))
    if not cursor.fetchone():
        conn.close()
        return jsonify(status="not_owned")

    #change UserLogin main outfit
    cursor.execute("UPDATE UserLogins SET equipped_outfit=? WHERE id=?", (outfit, uid))
    conn.commit()
    conn.close()

    return jsonify(status="success")

@app.route('/settings')
def settings():
    return "Settings page coming soon"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)