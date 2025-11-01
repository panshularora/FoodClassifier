# 🍛 Indian Food Classifier

A simple **Deep Learning + FastAPI web app** that classifies Indian food images into different categories.  
Built using **PyTorch**, **FastAPI**, and **HTML/CSS frontend**.

---

## 🚀 Features
- 🧠 Deep learning model trained on Indian food dataset  
- ⚡ Backend powered by **FastAPI**  
- 🌐 Simple, clean frontend for image upload and prediction  
- 🔍 Real-time image classification  
- 🐳 Ready to deploy on Docker, Render, or any cloud platform

---

## 🧩 Tech Stack
**Backend:** FastAPI, Python  
**Model:** PyTorch / TensorFlow (depending on your version)  
**Frontend:** HTML, CSS, JS  
**Deployment:** GitHub + (optional) Render / Docker

---

## 🛠️ Project Structure
**food-classifier**
**app.py # FastAPI backend**
**model/ # Trained model files**
**static/ # Frontend (HTML, CSS, JS)**
**requirements.txt # Dependencies**
**README.md**

---

## ⚙️ Installation

Clone the repository:
git clone https://github.com/panshularora/indian-food-classifier.git
cd indian-food-classifier
Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate  # For Windows
Install dependencies:

pip install -r requirements.txt
Run the app:

uvicorn app:app --reload
Open in browser:
👉 http://127.0.0.1:8000/

🖼️ Example
Upload an image of any Indian dish like biryani, idli, or samosa and the model predicts the food name with accuracy.

🌐 Deployment
You can easily deploy this app using:
Render
Railway
Docker
Hugging Face Spaces

💡 Future Improvements
Add more food categories 🍲
Improve model accuracy
Create a React frontend
Add nutrition or calorie prediction

🧑‍💻 Author
Panshul Arora
📬 Contributions & suggestions are welcome!
⭐ Don’t forget to star the repo if you found it useful :)

