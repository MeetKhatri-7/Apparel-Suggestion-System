# 👕 Apparel Suggestion System

A deep learning-based **apparel recommendation system** using **TensorFlow, Keras, and Streamlit**.  
Upload an apparel image and get visually similar product suggestions instantly.

---

## ✨ Features
- 📤 Upload images (`JPG`, `PNG`, `WEBP`)
- 🔍 Feature extraction using **ResNet50**
- 🎯 Top 5 similar apparel recommendations
- 🌐 Simple and interactive **Streamlit** interface

---

## ⚙️ Installation
```bash
git clone https://github.com/MeetKhatri-7/Apparel-Suggestion-System.git
cd Apparel-Suggestion-System
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate   # For Mac/Linux
pip install -r requirements.txt
```

## 🚀 Usage

1. 📥 Download the dataset: Kaggle Dataset

2. 🛠 Generate embeddings (if missing):
```bash
python main.py
```
3. ▶️ Run the app:
```bash
streamlit run app.py
```
Then open your browser at: http://localhost:8501

## 🛠 Tech Stack

- 🧠 TensorFlow + Keras – Feature extraction
- 🔢 NumPy, scikit-learn – Data processing
- 🎨 Streamlit, Pillow – Web interface & image handling

## ⚠️ Notes

- Do not upload the dataset or .pkl files to GitHub.
- Add the following to .gitignore:
```bash
.venv/
tensorflow_env/
images/
uploads/
*.pkl
```

### ✌️ Thanks for viewing this Project
