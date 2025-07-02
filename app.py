import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from mtcnn import MTCNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # maksimal 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load Keras model
model_path = 'model/age_classifier_best_model.keras'
model = tf.keras.models.load_model(model_path)

# Define age class names to match your model
AGE_CLASSES = [
    'Child (0-12)', 
    'Teen (13-18)', 
    'Young Adult (19-30)', 
    'Adult (31-45)', 
    'Middle Age (46-60)', 
    'Senior (60+)'
]

# Initialize face detector
detector = MTCNN()

def crop_faces(image_path):
    """
    Crop all detected faces from the image using MTCNN
    """
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    results = detector.detect_faces(image_array)

    faces = []
    if results:
        for i, face_info in enumerate(results):
            x, y, width, height = face_info['box']
            margin = 20
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            cropped_face = image.crop((x, y, x + width + margin, y + height + margin))
            # Save cropped face
            crop_filename = f"face_{i+1}_{os.path.basename(image_path)}"
            crop_path = os.path.join(app.config['UPLOAD_FOLDER'], 'crops', crop_filename)
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            cropped_face.save(crop_path)
            faces.append((crop_filename, cropped_face))
    return faces


def preprocess_image(face_image):
    """
    Preprocess image for the Keras model
    """
    # Resize to match model's expected input
    face_resized = face_image.resize((128, 128))
    
    # Convert to numpy array
    img_array = img_to_array(face_resized)
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image (assuming your model expects values between 0 and 1)
    img_array = img_array / 255.0
    
    return img_array

def predict_age_for_faces(image_path):
    """
    Detect all faces and predict age for each
    """
    faces = crop_faces(image_path)
    if not faces:
        return None, "No faces detected"
    
    predictions = []
    for crop_filename, face_img in faces:
        processed_image = preprocess_image(face_img)
        pred = model.predict(processed_image)[0]
        predicted_class_index = np.argmax(pred)
        predicted_class = AGE_CLASSES[predicted_class_index]
        confidence = pred[predicted_class_index] * 100
        predictions.append({
            'crop_filename': crop_filename,
            'age': predicted_class,
            'confidence': confidence
        })
    return predictions, None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    filename = None
    predictions = None
    error_message = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = 'Tidak ada file pada form.'
            return render_template('upload.html', error_message=error_message)

        file = request.files['file']
        if file.filename == '':
            error_message = 'Tidak ada gambar yang dipilih.'
            return render_template('upload.html', error_message=error_message)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                predictions, error_message = predict_age_for_faces(filepath)
                if not predictions:
                    filename = None  # Optional: Jangan tampilkan gambar
            except Exception as e:
                error_message = f"Terjadi kesalahan: {str(e)}"
                predictions = None

    return render_template('upload.html',
                           filename=filename,
                           predictions=predictions,
                           error_message=error_message)
    
@app.route('/timeseries')
def timeseries():
    return render_template('timeseries.html') 
    
if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)