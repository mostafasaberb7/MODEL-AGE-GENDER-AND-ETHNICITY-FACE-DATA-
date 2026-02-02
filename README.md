# 🧠 Age, Gender & Ethnicity Prediction App

Streamlit web application for predicting age, gender, and ethnicity from face images using a Multi-Head CNN.

## 🚀 Quick Start

### Local Deployment

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Make sure your model file is in the same directory:**
```
📁 your-project/
  ├── app.py
  ├── requirements.txt
  ├── best_multihead_model.keras  ← Your trained model
  └── README.md
```

3. **Run the app:**
```bash
streamlit run app.py
```

4. **Open in browser:**
The app will automatically open at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (FREE!)

### Step 1: Prepare Your Files

Create a GitHub repository with:
```
📁 your-repo/
  ├── app.py
  ├── requirements.txt
  ├── best_multihead_model.keras
  └── README.md
```

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"!

🎉 Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## 📦 Alternative: Deploy to Hugging Face Spaces

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (select Streamlit)
3. Upload files:
   - `app.py`
   - `requirements.txt`
   - `best_multihead_model.keras`
4. Your app is live!

---

## 🎯 Features

- ✅ **Age Prediction** (1-116 years)
- ✅ **Gender Classification** (Male/Female)
- ✅ **Ethnicity Classification** (5 categories)
- ✅ **Confidence Scores**
- ✅ **Beautiful UI**
- ✅ **Download Results**
- ✅ **Mobile Responsive**

---

## 🖼️ Usage

1. Upload a clear face image (JPG, JPEG, or PNG)
2. Wait for AI to process
3. View predictions with confidence scores
4. Download results if needed

---

## 📊 Model Details

- **Architecture:** Multi-Head CNN
- **Input Size:** 64x64 RGB
- **Outputs:** 3 heads (age, gender, ethnicity)
- **Dataset:** UTK Face (27,305 images)
- **Framework:** TensorFlow/Keras

---

## 🛠️ Troubleshooting

### Model not found error
Make sure `best_multihead_model.keras` is in the same directory as `app.py`

### Memory issues on Streamlit Cloud
The free tier has 1GB RAM limit. If your model is too large, consider:
- Using model quantization
- Deploying on Hugging Face (2GB RAM)

### Slow predictions
This is normal on free tiers. Consider:
- Using GPU-enabled hosting (paid)
- Optimizing model size

---

## 📝 License

MIT License - feel free to use and modify!

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📧 Contact

Questions? Open an issue on GitHub!

---

Made with ❤️ using TensorFlow & Streamlit