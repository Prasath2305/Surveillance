import cv2
import winsound
from datetime import datetime
import argparse
import os
from twilio.rest import Client

# Create folders for saving face and motion-captured images
face_folder = 'face_images'
motion_folder = 'motion_images'

if not os.path.exists(face_folder):
    os.makedirs(face_folder)

if not os.path.exists(motion_folder):
    os.makedirs(motion_folder)

# Face detection setup
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Motion detection setup
cam = cv2.VideoCapture(0) # Change the video source here (can provide video and camera IP)

# Output video setup
output_file = 'output_video.mp4'
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (frame_width, frame_height))

# Twilio SMS setup
twilio_account_sid = 'Your_account_id'
twilio_auth_token = 'Your_auth_token'
twilio_phone_number = '+12058946237'
owner_phone_number = '+939876543210'

client = Client(twilio_account_sid, twilio_auth_token)

# Function to send SMS alert using Twilio
def send_sms_alert(body, media_url=None):
    message = client.messages.create(
        body=body,
        from_=twilio_phone_number,
        to=owner_phone_number
    )

    print("SMS alert sent successfully. SID:", message.sid)

while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # Face detection
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
        exact_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S-%f')
        face_image_path = os.path.join(face_folder, "face_detected_" + str(exact_time) + ".jpg")
        cv2.imwrite(face_image_path, img)

    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the motion-captured image in the specified folder
        motion_capture_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S-%f')
        motion_image_path = os.path.join(motion_folder, "motion_captured_" + str(motion_capture_time) + ".jpg")
        cv2.imwrite(motion_image_path, frame1)

        # Send SMS alert with the motion-captured image
        send_sms_alert("Motion detected at your property!", media_url=motion_image_path)

        winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

    # Write frame to the output video
    out.write(frame1)

    cv2.imshow('Integrated Cam', frame1)

    if cv2.waitKey(10) == ord('q'):
        break

# Cleanup
cam.release()
out.release()
cv2.destroyAllWindows()
