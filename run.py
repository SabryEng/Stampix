from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Create a file named app.py
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('printability_model.h5')

# Define image dimensions
img_height = 28
img_width = 28

#full path for the uploads folder
uploads_dir = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Update the preprocess_image function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((img_height, img_width))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img_path = os.path.join(uploads_dir, file.filename)
            file.save(img_path)
            
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)
            printability_score = prediction[0][0]

            return render_template('result.html', score=printability_score, image_path=img_path)
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

