# Age Classification by Photo using CNN

A web-based application that predicts a person’s **age group** from a facial image using a Convolutional Neural Network (CNN). The project uses **TensorFlow/Keras** for deep learning, **MTCNN** for face detection, and **Flask** for the web interface.

## 📌 Features
- Predicts age group from a photo:
  - Child (0–12)
  - Teen (13–18)
  - Young Adult (19–30)
  - Adult (31–45)
  - Middle Age (46–60)
  - Senior (60+)
- Upload an image or use an existing one from the app.
- Real-time age prediction with face detection.
- Lightweight web interface using Flask.

## 🧠 Tech Stack
- Python 3.x
- TensorFlow / Keras
- Flask
- MTCNN (Face Detection)
- OpenCV & Pillow (Image Processing)
- HTML / CSS (Jinja2 Templates)

## 📂 Project Structure
```
Project/
│
├── app.py // Main Flask application
├── model/ // Trained CNN models
│ ├── age_classifier_best_model.keras
│ └── age_classifier_final_model.keras
│
├── static/
│ ├── images/
│ │ └── forecasting.jpg // Placeholder or additional asset
│ └── uploads/ // Uploaded and cropped images
│ └── crops/ // Face-cropped versions
│
├── templates/
│ ├── upload.html // Upload page
│ └── timeseries.html // Additional/unused template
│
├── README.md
├── requirements.txt (optional)
├── pyproject.toml / uv.lock // Dependency management
└── .venv / .python-version // Virtual environment (optional)
```

## 🚀 How to Run the App
### 1. Clone the Repository
git clone https://github.com/your-username/age-classification-cnn.git
cd age-classification-cnn
### 2. Create a Virtual Environment (Recommended)
python -m venv .venv
source .venv/bin/activate   # On Linux/macOS
.venv\Scripts\activate      # On Windows
### 3. Install Dependencies
pip install -r requirements.txt
### 4. Run the App
python app.py
Visit http://127.0.0.1:5000 in your browser.

## 🔍 How It Works
1. User uploads an image from the browser.
2. MTCNN detects and crops the face.
3. Image is resized and normalized for model input.
4. CNN model classifies the face into one of six age categories.
5. The result is displayed along with the original and cropped images.

## 🧠 Model Information
- CNN model built using TensorFlow/Keras.
- Trained on a dataset of facial images with labeled age categories.
- Exported as .keras for optimal deployment.
- Loaded at runtime via Flask (age_classifier_best_model.keras).

## 📷 Example
Uploaded Image → Cropped Face → Predicted Age Group: Young Adult (19–30)

## 🤝 Credits
Developed by **Davin Williem**  
Politeknik Caltex Riau – 2025

## 📜 License
This project is for educational purposes only. Free to use and modify under MIT License.
