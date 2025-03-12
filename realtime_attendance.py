import cv2
import numpy as np
import os
import csv
import pickle
from datetime import datetime

# Initialize video capture
video = cv2.VideoCapture(0)

# Load Haar Cascade XML file
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if the cascade classifier was loaded successfully
if facedetect.empty():
    print("Error: Could not load Haar Cascade XML file.")
    exit()

# Create directories if they don't exist
if not os.path.exists('data/images'):
    os.makedirs('data/images')

if not os.path.exists('data/trainer'):
    os.makedirs('data/trainer')

# CSV file for attendance
attendance_file = 'data/attendance.csv'

# Check if the CSV file exists, if not create it with headers
if not os.path.exists(attendance_file):
    with open(attendance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp"])

# Function to add a new user
def add_new_user():
    name = input("Enter student name: ")
    user_folder = f"data/images/{name}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    face_data = []
    i = 0

    print(f"Capturing images for {name}. Please look at the camera.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame from the camera.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (100, 100))

            if len(face_data) < 10 and i % 5 == 0:  # Collect 10 faces (every 5th frame)
                face_data.append(resized_img)
                cv2.putText(frame, f"Images Captured: {len(face_data)}/10", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Save the captured image
                cv2.imwrite(f"{user_folder}/{len(face_data)}.jpg", resized_img)

        # Display the current number of images captured
        cv2.putText(frame, f"Images Captured: {len(face_data)}/10", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

        i += 1  # Increment counter
        if len(face_data) == 10:  # Stop after collecting 10 faces
            cv2.destroyAllWindows()
            break

    print(f"Images captured for {name} and saved in {user_folder}.")

# Function to train the face recognition model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    # Load images and labels
    for root, dirs, files in os.walk("data/images"):
        for dir_name in dirs:
            label_ids[dir_name] = current_id
            for file in os.listdir(f"{root}/{dir_name}"):
                if file.endswith("jpg"):
                    img_path = f"{root}/{dir_name}/{file}"
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(img)
                    labels.append(current_id)
            current_id += 1

    # Save the label_ids dictionary to a file
    with open("data/trainer/label_ids.pkl", "wb") as f:
        pickle.dump(label_ids, f)

    # Train the model
    recognizer.train(faces, np.array(labels))
    recognizer.save("data/trainer/trainer.yml")
    print("Model trained and saved.")

# Function to mark attendance
def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("data/trainer/trainer.yml")

    # Load the label_ids dictionary
    with open("data/trainer/label_ids.pkl", "rb") as f:
        label_ids = pickle.load(f)

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame from the camera.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Recognize the face
            label, confidence = recognizer.predict(roi_gray)
            if confidence < 100:  # Confidence threshold
                name = list(label_ids.keys())[list(label_ids.values()).index(label)]
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Mark attendance
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, timestamp])
                print(f"Attendance marked for {name} at {timestamp}.")
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        cv2.imshow("frame", frame)
        if cv2.waitKey(0):  # Press 'Esc' to exit
            cv2.destroyAllWindows()
            break
        
# Main menu
while True:
    print("\nWelcome to Mark Attendance")
    print("------------------------------")
    print("1. Add New User")
    print("2. Train Model")
    print("3. Mark Attendance")
    print("4. Exit")
    choice = input("Enter your choice: ")

    if choice == "1":
        add_new_user()
        
    elif choice == "2":
        train_model()
        
    elif choice == "3":
        mark_attendance()
        
    elif choice == "4":
        break
    else:
        print("Invalid choice. Please try again.")

video.release()
cv2.destroyAllWindows()