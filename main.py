import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

path = 'Training_images'

images = []
classNames = []

# Get list of filenames in the training image directory
myList = os.listdir(path)

# Iterate through each filename
for cl in myList:
    # Load the current image
    curImg = cv2.imread(f'{path}/{cl}')

    # Check if image loading was successful
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])  # Extract class name from filename
    else:
        print(f"Error reading image: {cl}")  # Handle failed image reads

print(classNames)  # Print the extracted class names


def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert the image to RGB color space (expected by face_recognition)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Dictionary to track attendance for each person
attendance_records = {}

def markAttendance(name):
    global attendance_records
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    if name not in attendance_records:
        with open('Attendance.csv', 'a') as f:
            f.writelines(f'\n{name},{dtString}')
        attendance_records[name] = True 

# Encode the training images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Capture video from webcam (change index for other cameras)
cap = cv2.VideoCapture(0)

# Reduce framerate for smoother processing (adjust value as needed)
frame_rate = 5
cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_rate)

while True:
    success, img = cap.read()
    if not success:
        print("Error capturing frame from webcam")
        break

    # Resize the captured frame significantly for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find faces in the resized frame
    facesCurFrame = face_recognition.face_locations(imgS)

    # Encode the faces in the resized frame (potentially skip frames for speed)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Iterate over each detected face
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare the face encoding with known encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Find the index of the best matching face
        matchIndex = np.argmin(faceDis)

        # If there's a match
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            # Scale the face location coordinates back to the original frame size
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2

            # Draw a rectangle around the matched face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create a filled rectangle for the name text
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

            # Put the detected name on the frame
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance only once per person
            markAttendance(name) 

    # Display the resulting image with detections
    cv2.imshow('Webcam', img)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Mark exit attendance for all recorded entries
        for name in attendance_records:
            markAttendance(name + "_EXIT", "EXIT") 

        # Clear attendance.csv file
        try:
            os.remove('Attendance.csv')
        except FileNotFoundError:
            pass  # Ignore if the file doesn't exist

        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()