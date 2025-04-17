from flask import Flask, request, jsonify
from flask_cors import CORS 
from copy import deepcopy

from ultralytics import YOLO
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import io
from PIL import Image
import pyttsx3
import threading

app = Flask(__name__)
CORS(app) 

yolo_model = YOLO("yolov5s.pt")
efficientnet_model = EfficientNetB2(weights="imagenet")

def detect_objects(image, results_dict):
    results = yolo_model(image)
    detected_objects = {}
    for result in results:
        for obj in result.boxes:
            class_id = int(obj.cls)
            label = yolo_model.names[class_id]
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1
    print(f"Detected Objects: {detected_objects}")
    results_dict["yolo"] = detected_objects

def classify_dominant_object(image, results_dict):
    image = image.convert("RGB").resize((260, 260))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = efficientnet_model.predict(img_array)
    label = decode_predictions(predictions, top=1)[0][0][1].replace("_", " ")
    print(f"Dominant Object: {label}")
    results_dict["efficientnet"] = label

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 0.8)
    engine.say(text)
    engine.runAndWait()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    file.stream.seek(0)
    image = Image.open(io.BytesIO(file.read()))
    results_dict = {}
    yolo_thread = threading.Thread(target=detect_objects, args=(image.copy(), results_dict))
    efficientnet_thread = threading.Thread(target=classify_dominant_object, args=(image.copy(), results_dict))
    yolo_thread.start()
    efficientnet_thread.start()
    yolo_thread.join()
    efficientnet_thread.join()
    detected_objects = results_dict.get("yolo", {})
    dominant_object = results_dict.get("efficientnet", "Unknown")
    if detected_objects:
        detected_text = ", ".join([f"{obj} ({count})" for obj, count in detected_objects.items()])
        speech_text = f"Detected objects are: {detected_text}. The most dominant object is likely a {dominant_object}."
    else:
        speech_text = f"No objects detected. However, the most dominant object is predicted as a {dominant_object}."
    tts_thread = threading.Thread(target=text_to_speech, args=(speech_text,))
    tts_thread.start()
    return jsonify({
        "detected_objects": detected_objects,
        "dominant_object": dominant_object,
        "message": speech_text
    })

if __name__ == "__main__":
    app.run(debug=True)
