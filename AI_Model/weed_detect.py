import tensorflow.lite as tflite
import numpy as np
import cv2

# Load TFLite model
interpreter = tflite.Interpreter(model_path="weed_detection_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

cap = cv2.VideoCapture(0)  # Change index if using an external camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    h, w, _ = frame.shape
    startX, startY, endX, endY = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)

    if prediction > 0.5:
        # Weed detected: Show red bounding box and label
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
        cv2.putText(frame, "Weed Detected", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # No weed detected: Show "Not a Weed" in green text
        cv2.putText(frame, "Not a Weed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Live Weed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
