import csv
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
import picamera

# Set up GPIO
GPIO.setmode(GPIO.BCM)

# Define the PIR sensor pin
pir_pin = 6  # You can use any available GPIO pin

# Set up GPIO pins
GPIO.setup(pir_pin, GPIO.IN)  # PIR sensor pin as input

# Initialize the camera
camera = picamera.PiCamera()

# Face recognizer setup
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None', 'Bala', 'Rajarajan', 'Tamilarasan']
minW = 0.1 * 640
minH = 0.1 * 480

# Create the directory if it doesn't exist
output_folder = "recognized_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load existing data from CSV file, if it exists
csv_file = 'user_recognition.csv'
user_data = {}
if os.path.exists(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_data[row['User ID']] = {'Timestamp': row['Timestamp'], 'Confidence': int(row['Confidence'])}

def detect_motion(pir_pin):
    print("Motion detected!")

    # Capture an image when motion is detected
    timestamp = time.strftime("%Y%m%d%H%M%S")
    image_filename = f"motion_{timestamp}.jpg"
    camera.capture(image_filename)
    print(f"Image captured: {image_filename}")

    # Read the captured image
    img = cv2.imread(image_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    # Process only if faces are detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                id = names[id]
                confidence = round(100 - confidence)
            else:
                id = "unknown"
                confidence = round(100 - confidence)

            cv2.putText(img, f"ID: {id}", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Confidence: {confidence}%", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # Update dictionary with latest data for each user ID
            if id in user_data:
                if confidence > user_data[id]['Confidence']:
                    user_data[id] = {'Timestamp': timestamp, 'Confidence': confidence}
            else:
                user_data[id] = {'Timestamp': timestamp, 'Confidence': confidence}

        # Save the image with recognition details in the recognized_faces folder
        output_filename = f"{output_folder}/recognized_{timestamp}.jpg"
        cv2.imwrite(output_filename, img)
        print(f"Image saved with recognition details: {output_filename}")

    # Write data to CSV file, removing duplicates
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'User ID', 'Confidence'])
        for id, data in user_data.items():
            writer.writerow([data['Timestamp'], id, data['Confidence']])

try:
    print("PIR Motion Sensor Test")
    GPIO.add_event_detect(pir_pin, GPIO.RISING, callback=detect_motion)
    while True:
        time.sleep(1)  # Keep the program running
except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.cleanup()
    camera.close()
    
#remdul 1CSV
