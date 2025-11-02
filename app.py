from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sqlite3
from werkzeug.utils import secure_filename

# ✅ Load trained model
model = load_model('face_emotionModel.h5')

# Emotion labels (adjust if needed)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT,
                     matric TEXT,
                     emotion TEXT,
                     image_path TEXT)''')
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    matric = request.form['matric']
    file = request.files['image']

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess the image for the model
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))  # resize to model input size
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict emotion
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        # Friendly message
        message = f"You seem {emotion.lower()}!" if emotion else "Emotion not detected."

        # Save data to database
        conn = sqlite3.connect('database.db')
        conn.execute("INSERT INTO users (name, matric, emotion, image_path) VALUES (?, ?, ?, ?)",
                     (name, matric, emotion, filepath))
        conn.commit()
        conn.close()

        return render_template('index.html', emotion=emotion, message=message, image_path=filepath)

    return "No image uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
