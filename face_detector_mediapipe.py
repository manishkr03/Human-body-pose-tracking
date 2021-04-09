import cv2
import mediapipe as mp
import imutils

mp_face_detction = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

video_path_1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"

# For webcam input:
cap = cv2.VideoCapture(video_path_1)
with mp_face_detction.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=800)
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        #image.flags.writeable = False
        results = face_detection.process(image)
    
        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
          for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    else:
        break
      
cap.release()
cv2.destroyAllWindows()







