Here is your **updated, cleaned, polished README** with:

✔ Your **live Streamlit app link**
✔ Your **new dataset link**
✔ Removed license section
✔ Improved formatting and clarity

You can directly paste this into `README.md`.

---

```markdown
# Facial Emotion Recognition using CNN 🎭

Facial Emotion Recognition (FER) is a deep learning task where a model predicts human emotions based on facial expressions.  
This project uses a **Convolutional Neural Network (CNN)** trained on the **FER2013** dataset and provides a modern, fully interactive **Streamlit web application**.

🌐 **Live App:**  
👉 https://face-er.streamlit.app

---

## 🔍 What This Project Does

- Detects one or more faces in an image
- Classifies each detected face into 7 emotions:

  `Angry • Disgust • Fear • Happy • Sad • Surprise • Neutral`

- Draws bounding boxes and overlays:
  - Emotion labels  
  - Confidence scores  
  - Emoji indicators  
- Offers a smooth **dark-themed Streamlit UI**:
  - 📷 Capture image from webcam  
  - 🖼 Upload image  
  - 🔎 Generate predictions with confidence bars  
  - 🎨 Beautiful gradient-based UI components

---

## 🧠 Tech Stack

- **TensorFlow / Keras** – CNN model  
- **Streamlit** – Web UI  
- **OpenCV** – Face detection  
- **Pillow (PIL)** – Image handling  
- **NumPy / Pandas** – Preprocessing  

---

## 📦 Repository Structure

```

.
├── ferapp.py                          # Main Streamlit app
├── Facial_Emotion_Recognition_using_CNN.ipynb   # Notebook for training/modeling
├── emotion_cnn.keras                  # Trained CNN model (saved model format)
├── models/                            # Optional model folder
│   └── emotion_cnn.keras
├── requirements.txt                   # App dependencies
└── README.md

````

The Streamlit app automatically looks for a model in:

1. `emotion_cnn.keras` (root directory), or  
2. `models/emotion_cnn.keras`, or  
3. Any `*.keras` file in either of those directories.

---

## 🧪 Dataset

This project uses the **FER2013** dataset available on Kaggle:  
👉 https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge

Dataset includes:

- 48×48 grayscale facial images  
- 7 emotion classes  
- Over 35,000 images  

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/face_emotion_recogniser-cnn-.git
cd face_emotion_recogniser-cnn-
````

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

(Uses `opencv-python-headless` to avoid `libGL.so.1` errors on servers.)

---

## 🚀 Running the Streamlit App

Place your `emotion_cnn.keras` model in:

* The project root
  **or**
* A folder named `models/`

Then run:

```bash
streamlit run ferapp.py
```

Open in your browser at:

```
http://localhost:8501
```

---

## 🖥 Web App Features

Inside the app, you can:

* **📷 Capture live image** via webcam
* **🖼 Upload an image file**
* **🔍 Click “Analyze Emotion”** to:

  * Detect all faces in the image
  * Predict their emotions using the CNN
  * Show per-face prediction cards with confidence bars
  * Display a fully annotated version of the image

The UI uses gradient cards, colorful badges, and modern styling for a clean aesthetic.

---

## 📚 Training (Notebook)

Model training is done in:

```
Facial_Emotion_Recognition_using_CNN.ipynb
```

It includes:

* FER2013 preprocessing
* CNN architecture
* Callbacks:

  * ModelCheckpoint
  * ReduceLROnPlateau
  * EarlyStopping
* Model export to `.keras` format
* Visualization & interpretation

You can retrain or modify the model easily.

---

## 🚧 Future Enhancements

* Replace Haarcascade with MTCNN for improved face detection
* Add live video feed support
* Multi-model selection (lightweight / advanced CNN)
* Convert model to TFLite for mobile
* Add emotion timeline graphs for video

---

## 💬 Support & Contributions

If you'd like to improve the interface, optimize the CNN, or contribute new features:

* Open an **issue**
* Create a **pull request**
* Share ideas to enhance UI/UX or accuracy

---

Thanks for checking out the project — enjoy experimenting with emotion detection! 😄

```
