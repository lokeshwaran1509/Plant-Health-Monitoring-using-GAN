import tensorflow.lite as tflite
import numpy as np
import cv2
import urllib.request

# ESP32-CAM URL (Modify this if needed)
url = 'http://192.168.50.131/cam-hi.jpg'

# Load Weed Detection Model
weed_interpreter = tflite.Interpreter(model_path="weed_detection_model.tflite")
weed_interpreter.allocate_tensors()
weed_input_details = weed_interpreter.get_input_details()
weed_output_details = weed_interpreter.get_output_details()
weed_img_height, weed_img_width = weed_input_details[0]['shape'][1:3]

# Load Plant Disease Detection Model
disease_interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
disease_interpreter.allocate_tensors()
disease_input_details = disease_interpreter.get_input_details()
disease_output_details = disease_interpreter.get_output_details()
disease_img_height, disease_img_width = disease_input_details[0]['shape'][1:3]

# Define class labels for plant disease classification
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Jasmine__healthy', 'Jasmine__leaf__spot'
]  # Modify if needed

# Function to preprocess image for both models
def preprocess_image(image, img_width, img_height):
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# Start video capture from ESP32-CAM
while True:
    try:
        # Fetch image from ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        if frame is None:
            print("Failed to retrieve frame.")
            continue

        h, w, _ = frame.shape  # Get frame dimensions

        # --- Weed Detection ---
        weed_input_data = preprocess_image(frame, weed_img_width, weed_img_height)
        weed_interpreter.set_tensor(weed_input_details[0]['index'], weed_input_data)
        weed_interpreter.invoke()
        weed_output_data = weed_interpreter.get_tensor(weed_output_details[0]['index'])
        weed_prediction = weed_output_data[0][0]

        # Draw bounding box only if weed is detected
        if weed_prediction > 0.5:
            startX, startY, endX, endY = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
            cv2.putText(frame, "Weed Detected", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Not a Weed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Plant Disease Detection ---
        disease_input_data = preprocess_image(frame, disease_img_width, disease_img_height)
        disease_interpreter.set_tensor(disease_input_details[0]['index'], disease_input_data)
        disease_interpreter.invoke()
        disease_output_data = disease_interpreter.get_tensor(disease_output_details[0]['index'])
        disease_prediction = np.argmax(disease_output_data)

        # Show plant disease result only if the model detects a disease confidently
        confidence_score = np.max(disease_output_data)
        if confidence_score > 0.7:  # Adjust confidence threshold as needed
            disease_label = class_labels[disease_prediction]
            cv2.putText(frame, f"Disease: {disease_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame with detections
        cv2.imshow("Live Weed & Plant Disease Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error fetching frame: {e}")
        break

cv2.destroyAllWindows()
