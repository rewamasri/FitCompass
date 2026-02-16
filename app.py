from flask import Flask, render_template, request, redirect, url_for, session, flash
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

cursor.execute("""
CREATE TABLE IF NOT EXISTS UserLogins(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    goal TEXT,
    goal_other TEXT,
    workouts_per_week INTEGER,
    body_part TEXT
)
""")

connection.commit()



connection.close()

# Webcam setup
camera = cv2.VideoCapture(0)
#global variable for the latest detected frame

# latest_detection = None

loggedInUsers={}
class User:
    def __init__ (self,id):
        self.user_ID = id
        self.latest_detection = None
        self.currentExercise= None
        self.exerciseManager = ExerciseManager()
        


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
                self.count=self.count+1
                self.state=SitUpState.IDLE
                return
        

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

            if self.left_knee_angle<120 and self.right_knee_angle <120:
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
                self.count += 1
                self.state = SquatState.IDLE
                print(f"Count: {self.count}")
    
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
 
        if self.state==LungeState.DOWN:

            left_heel = np.array(left_heel)
            right_heel = np.array(right_heel)
                
            self.heelToHeelDistance=np.linalg.norm(left_heel-right_heel) #this is a constant!!!!
            if(abs(self.heelToHeelDistance) < self.calfLength *1.3):
                self.state=LungeState.ASCENDING
                self.count=self.count+1
                return
        
        if self.state==LungeState.ASCENDING:
            self.right_knee_angle=angleBetweenLines(right_hip,right_knee,right_heel)
            self.left_knee_angle=angleBetweenLines(left_hip,left_knee,left_heel)
            if(self.right_knee_angle > 140 and self.left_knee_angle >140):
                self.state=LungeState.IDLE
                return
 
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

class ExerciseManager():
    def __init__(self):
        self.exercises={"squats": SquatController(), "situps" : SitUpController(), "lunges" : LungeController(), "running" : RunningController(), "jumpingjacks" : JumpingJacksController(), "pushups": PushUpController()}
    
        self.currentExercise="pushups"
    def getCurrentExercise(self):
        return self.exercises[self.currentExercise]
    def setCurrentExercise(self,exerciseName):
        self.currentExercise=exerciseName




@app.route('/switch_exercise',methods=["POST"])
def switch_exercise():

    user_id = session.get('user_id')
    if not user_id in loggedInUsers:
        return
    currentUser= loggedInUsers[user_id]
    data = request.get_json()
    new_exercise = data.get('exercise')
    currentUser.exerciseManager.setCurrentExercise(new_exercise)
    return jsonify(status="success", now_doing=new_exercise)

@app.route('/get_exercise_data')
def get_exercise_data():
    user_id = session.get('user_id')
    if not user_id in loggedInUsers:
        return
    currentUser= loggedInUsers[user_id]

    currentExerciseObject=currentUser.exerciseManager.getCurrentExercise()
    return jsonify(
        currentExercise=currentUser.exerciseManager.currentExercise,
        count=currentExerciseObject.count,
        state=currentExerciseObject.state,
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

# -------------------------
# Login
# -------------------------

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

# -------------------------
# Register + Intake Quiz
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        goal = request.form.get('goal')
        goal_other = request.form.get('goal_other') if goal == 'other' else None
        workouts_per_week = request.form.get('workouts_per_week')
        body_part = request.form.get('body_part')

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Insert user
            cursor.execute(
                """
                INSERT INTO UserLogins (username, email, password, goal, goal_other, workouts_per_week, body_part)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, email, password, goal, goal_other, workouts_per_week, body_part)
            )
            user_id = cursor.lastrowid

            conn.commit()
            conn.close()

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
    if 'username' not in session:
        return redirect(url_for('login'))

    return render_template(
        'home.html',
        username=session['username'],
        points=120,
        goal_percent=62
    )


@app.route('/workoutSession')
def workoutSession():
    squat_count=0
    knee_angle=0

    return render_template("workoutSession.html",squat_count=squat_count,knee_angle=knee_angle)

@app.route('/workoutcomplete')
def workoutcomplete():


    return render_template("workoutcomplete.html")


# -------------------------
# Placeholder
# -------------------------
@app.route('/profile')
def profile():
    return "Profile page coming soon"

@app.route('/history')
def history():
    return render_template("workoutLog.html")

@app.route('/library')
def library():
    return "Library page coming soon"

@app.route('/shop')
def shop():
    return render_template("shop.html")

@app.route('/settings')
def settings():
    return "Settings page coming soon"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)

