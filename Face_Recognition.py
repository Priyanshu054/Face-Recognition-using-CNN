import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Trained_Model/CNN_Model.keras')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('Cascade_Classifier/haarcascade_frontalface_default.xml')

person = ["Amit", "Bill_Gates", "Elon Musk", "Modi Ji", "Priyanshu", "Rahul", "Steve Rogers", "Thor", "Tony Stark"]

# Function to preprocess image for model input
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image to match the model input size
    resized = cv2.resize(gray, (128, 128))
    # Normalize pixel values
    normalized = resized / 255.0
    # Add batch dimension
    processed = np.expand_dims(normalized, axis=0)
    return processed

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale for face detection
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection using Haar Cascade
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the detected face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the detected face image
        processed_image = preprocess_image(face_roi)
        
        # Perform face recognition
        predictions = model.predict(processed_image)
        
        # Get the predicted label and confidence
        predicted_label = np.argmax(predictions)
        confidence = predictions[0][predicted_label]

        # Display the rectangle around the face and the predicted label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Print name of identified person and confidence value
        if confidence > 0.4:
            cv2.putText(frame, person[predicted_label], (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "unknown", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:    # press 'ESC' to quit
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
