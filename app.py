from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms, models
import io
from fastapi.staticfiles import StaticFiles



# ============================================================
# ✅ Initialize app
# ============================================================
app = FastAPI(title="🍛 Indian Food Classifier API")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
# ============================================================
# ✅ Load model and labels
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify this list according to your dataset folders
class_names = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature',
               'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa',
               'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']

# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, len(class_names))
)
model.load_state_dict(torch.load("indian_food_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# ============================================================
# ✅ Define endpoint
# ============================================================
@app.get("/")
def root():
    return {"message": "Welcome to the Indian Food Classifier API 🍛"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform
        img_t = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_t)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        return JSONResponse({"predicted_class": label})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
