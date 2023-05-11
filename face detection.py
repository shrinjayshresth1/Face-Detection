

import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("C:/Users/hp/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Open a video capture object to capture frames from the webcam
video_capture = cv2.VideoCapture(0)

# Start an infinite loop to continuously read frames from the video capture object and perform face detection on each frame
while True:
    # Read a frame from the video capture object
    ret, frame = video_capture.read()

    # Convert the frame from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the detected faces in the original BGR image
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the BGR image with rectangles drawn around the detected faces
    cv2.imshow("Video", frame)

    # Wait for a key press and break out of the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()





