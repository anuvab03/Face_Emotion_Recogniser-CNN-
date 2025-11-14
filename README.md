# Facial Emotion Recognition using CNN 🎭

Facial Emotion Recognition (FER) is a deep learning task where a model predicts human emotions based on facial expressions.  
This project uses a **Convolutional Neural Network (CNN)** trained on the **FER2013** dataset and provides a modern, fully interactive **Streamlit web application**.

🌐 Live App:  
👉 https://face-er.streamlit.app

---

## 🔍 What This Project Does

- Detects one or more faces in an image  
- Classifies each detected face into 7 emotions:  
  Angry • Disgust • Fear • Happy • Sad • Surprise • Neutral  
- Draws bounding boxes and overlays:
  - Emotion labels  
  - Confidence scores  
  - Emoji indicators  
- Provides a smooth dark-themed Streamlit UI:
  - Capture from webcam  
  - Upload an image  
  - Analyze emotions  
  - Gradient UI components

---

## 🧠 Tech Stack

- TensorFlow / Keras  
- Streamlit  
- OpenCV  
- Pillow  
- NumPy / Pandas  

---

## 📦 Repository Structure

.
├── ferapp.py                           # Main Streamlit app  
├── Facial_Emotion_Recognition_using_CNN.ipynb   # Training notebook  
├── emotion_cnn.keras                   # Trained CNN model  
├── class_indices.npy                   # Optional: class index mapping  
├── models/                             # Optional model directory  
│   └── emotion_cnn.keras  
├── requirements.txt                    # Dependencies  
└── README.md  

The app automatically looks for a model in:  
1. `emotion_cnn.keras`  
2. `models/emotion_cnn.keras`  
3. Any `.keras` in these folders  

---

## 🧪 Dataset

Trained using the FER2013 dataset from Kaggle:  
👉 https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge

Includes:  
- 48×48 grayscale images  
- 7 emotion classes  
- 35k+ samples  

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone the repository

git clone https://github.com/<your-username>/face_emotion_recogniser-cnn-.git  
cd face_emotion_recogniser-cnn-  

### 2️⃣ Create virtual environment

python -m venv venv  
Windows: venv\Scripts\activate  
Linux/macOS: source venv/bin/activate  

### 3️⃣ Install dependencies

pip install --upgrade pip  
pip install -r requirements.txt  

(opencv-python-headless is used to avoid libGL errors)

---

## 🚀 Running the Streamlit App

Place your `.keras` model in root or inside `models/`

streamlit run ferapp.py  

Then open:

http://localhost:8501  

---

## 🖥 App Features

Inside the UI:  
- Capture from webcam  
- Upload images  
- Analyze emotions to:
  - detect faces  
  - classify emotions  
  - view confidence bars  
  - get an annotated image output  

---

## 📚 Training (Notebook)

Training occurs in:

Facial_Emotion_Recognition_using_CNN.ipynb

Includes:  
- Data preprocessing  
- CNN architecture  
- Callbacks (Checkpoint, EarlyStopping, ReduceLROnPlateau)  
- Save model as `.keras`  

---

## 🚧 Future Enhancements

- Switch Haarcascade → MTCNN  
- Live real-time video  
- TFLite mobile model  
- Emotion timeline graph  

---

## 💬 Support & Contributions

You can:  
- Open an issue  
- Submit a PR  
- Suggest improvements  

---

Thanks for checking out the project — enjoy experimenting with emotion detection! 😄
