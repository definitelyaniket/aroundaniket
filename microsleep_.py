import cv2
import mediapipe as mp
import pyttsx3
import datetime
import os
from math import hypot


speaker = pyttsx3.init()
def alert_user(message):
    print(f"[ALERT] {message}")
    speaker.say(message)
    speaker.runAndWait()


event_folder = "microsleep_events"
os.makedirs(event_folder, exist_ok=True)


face_mesh_model = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)


EYE_CLOSED_THRESHOLD = 0.25
FRAMES_FOR_MICROSLEEP = 15
MOUTH_OPEN_THRESHOLD = 0.7


left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]
mouth_indices = [13, 14, 17, 0]


def distance(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_landmarks):
    A = distance(eye_landmarks[1], eye_landmarks[5])
    B = distance(eye_landmarks[2], eye_landmarks[4])
    C = distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_landmarks):
    vertical = distance(mouth_landmarks[0], mouth_landmarks[1])
    horizontal = distance(mouth_landmarks[2], mouth_landmarks[3])
    return vertical / horizontal

def save_frame(frame, event_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(event_folder, f"{event_name}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)


camera = cv2.VideoCapture(0)


eye_frame_counter = 0
yawn_frame_counter = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  
    frame_height, frame_width = frame.shape[:2] #(height, width, channels)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh_model.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            
            left_eye = [(int(face_landmarks.landmark[i].x * frame_width),
                         int(face_landmarks.landmark[i].y * frame_height)) for i in left_eye_indices]
            right_eye = [(int(face_landmarks.landmark[i].x * frame_width),
                          int(face_landmarks.landmark[i].y * frame_height)) for i in right_eye_indices]

            
            for point in left_eye + right_eye:
                
                cv2.circle(frame, point, 1, (220, 255, 220), -1)
                


            
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            average_ear = (ear_left + ear_right) / 2.0
            cv2.putText(frame, f"EAR: {average_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            
            if average_ear < EYE_CLOSED_THRESHOLD:
                eye_frame_counter += 1
                if eye_frame_counter >= FRAMES_FOR_MICROSLEEP:
                    save_frame(frame, "microsleep")
                    alert_user("Microsleep detected")
                    cv2.putText(frame, "Microsleep Detected!", (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    eye_frame_counter = 0
            else:
                eye_frame_counter = 0

            
            mouth = [(int(face_landmarks.landmark[i].x * frame_width),
                      int(face_landmarks.landmark[i].y * frame_height)) for i in mouth_indices]

            for point in mouth:
                cv2.circle(frame, point, 1, (220, 255, 220), -1)

            mar = calculate_mar(mouth)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            
            if mar > MOUTH_OPEN_THRESHOLD:
                yawn_frame_counter += 1
                if yawn_frame_counter > 10:
                    save_frame(frame, "yawning")
                    alert_user("Yawning detected")
                    cv2.putText(frame, "Yawning Detected!", (200, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    yawn_frame_counter = 0
            else:
                yawn_frame_counter = 0

    cv2.imshow("Microsleep & Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
