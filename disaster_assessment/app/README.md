# 🛰️ AI Disaster Damage Assessment — Backend & Integration

**College EPICS Project | Task 5: Model Serving & Systems Integration**

This module provides the **FastAPI backend** that connects the building detection model,
damage assessment model, and Streamlit frontend into a working system.

---

## 📁 Project Structure

```
disaster_assessment/
│
├── app/                          # FastAPI application package
│   ├── __init__.py
│   ├── main.py                   # App entry point, startup events
│   │
│   ├── routes/                   # API endpoint definitions
│   │   ├── __init__.py
│   │   ├── health.py             # GET /health
│   │   └── predict.py            # POST /predict
│   │
│   ├── models/                   # Model loading & registry
│   │   ├── __init__.py
│   │   └── model_loader.py       # ⭐ INTEGRATION POINT for teammates
│   │
│   └── core/                     # Business logic
│       ├── __init__.py
│       ├── config.py             # Settings (env-var driven)
│       ├── preprocessing.py      # OpenCV image preprocessing
│       └── inference.py          # Detection & assessment pipeline
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py               # Python test suite (requests)
│   └── curl_examples.sh          # curl API examples
│
├── sample_images/                # Auto-generated demo images
├── streamlit_app.py              # Streamlit web frontend
├── generate_test_image.py        # Script to create sample images
├── requirements.txt
├── .env                          # Environment configuration
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or later
- pip (comes with Python)

### Step 1 — Clone or navigate to the project
```powershell
cd d:\EPICS\disaster_assessment
```

### Step 2 — Create a virtual environment (recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```powershell
pip install -r requirements.txt
```

> **Note:** PyTorch installation may take several minutes. For GPU support, visit
> [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command.

---

## 🚀 Running the System

### Start the FastAPI Backend
```powershell
cd d:\EPICS\disaster_assessment
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server starts at **http://localhost:8000**

> When no model weights are found, the system automatically runs in **MOCK MODE**
> (simulated predictions). The console will log a warning to confirm this.

### Start the Streamlit Frontend
Open a **second terminal window**:
```powershell
cd d:\EPICS\disaster_assessment
venv\Scripts\activate
streamlit run streamlit_app.py
```

Streamlit opens at **http://localhost:8501**

### Generate Sample Test Images
```powershell
python generate_test_image.py
```
Creates `sample_images/before_sample.png` and `sample_images/after_sample.png`
that you can upload to the Streamlit UI immediately.

---

## 🌐 API Endpoints

### `GET /health`
Check whether the backend is running and whether real models are loaded.

**Response:**
```json
{
  "status": "ok",
  "app_name": "AI Disaster Damage Assessment",
  "version": "1.0.0",
  "mock_mode": true,
  "mock_mode_message": "⚠️ Running in MOCK MODE – predictions are simulated.",
  "building_detector_loaded": false,
  "damage_classifier_loaded": false
}
```

---

### `POST /predict`
Upload before and after satellite images; receive building detections and damage assessments.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `before_image` | file | Satellite image before the disaster |
| `after_image` | file | Satellite image after the disaster |

**Supported formats:** JPEG, PNG, TIFF  
**Max file size:** 10 MB per image

**Response:**
```json
{
  "status": "success",
  "mock_mode": true,
  "mock_warning": "⚠️ MOCK MODE ACTIVE: predictions are simulated.",
  "total_buildings": 4,
  "processing_time_ms": 42.7,
  "buildings": [
    {
      "building_id": 1,
      "bounding_box": { "x1": 45, "y1": 120, "x2": 145, "y2": 220, "width": 100, "height": 100 },
      "detection_confidence": 0.87,
      "damage_class": "major_damage",
      "damage_severity": "Major Damage",
      "damage_confidence": 0.79,
      "is_mock": true
    }
  ],
  "summary": {
    "total_buildings": 4,
    "no_damage_count": 1,
    "minor_damage_count": 1,
    "major_damage_count": 1,
    "destroyed_count": 1
  }
}
```

---

### Interactive Docs
- **Swagger UI:** http://localhost:8000/docs  
- **ReDoc:**      http://localhost:8000/redoc

---

## 🧪 Running Tests

```powershell
# Make sure the FastAPI server is running first
cd d:\EPICS\disaster_assessment
python tests/test_api.py
```

**Test cases:**
1. ✅ Health check — `GET /health` returns 200 with status=ok
2. ✅ Valid prediction — `POST /predict` returns structured JSON
3. ✅ Corrupted image rejected — HTTP 422
4. ✅ Unsupported format rejected — HTTP 422
5. ✅ Swagger UI accessible — `GET /docs` returns 200

---

## 🔗 Connecting the Streamlit Frontend

The Streamlit app (`streamlit_app.py`) calls the FastAPI backend over HTTP.

**How it works:**
1. User uploads before/after images in the Streamlit UI.
2. Streamlit POSTs the images to `http://localhost:8000/predict`.
3. FastAPI preprocesses them, runs the pipeline, and returns JSON.
4. Streamlit displays detected buildings, damage severity, and confidence scores.

**To change the backend URL** (e.g., for a remote server):
Edit line 18 in `streamlit_app.py`:
```python
BACKEND_URL = "http://your-server-ip:8000"
```

---

## 🤝 Integration Guide for Teammates

### How to Connect Your Trained Models

#### Building Detection Model (Teammate responsible for detection)

**File:** `app/models/model_loader.py` → function `load_building_detector()`

1. Export your model:
   ```python
   torch.save(model.state_dict(), "models/building_detector.pth")
   ```

2. Edit `load_building_detector()` and replace the placeholder:
   ```python
   # Replace this:
   model = torch.load(model_path, map_location="cpu")

   # With your actual architecture:
   from your_package.building_model import YourDetector
   model = YourDetector(num_classes=1)
   model.load_state_dict(torch.load(model_path, map_location="cpu"))
   ```

3. The model must accept input tensor of shape `(1, 3, 512, 512)`  
   and return bounding-box detections.

#### Damage Assessment Model (Teammate responsible for damage classification)

**File:** `app/models/model_loader.py` → function `load_damage_classifier()`

1. Export your model:
   ```python
   torch.save(model.state_dict(), "models/damage_classifier.pth")
   ```

2. Edit `load_damage_classifier()` and replace the placeholder with your class.

3. The model must accept **two** tensors (before, after) of shape `(1, 3, 512, 512)`  
   and return class scores for `[no_damage, minor_damage, major_damage, destroyed]`.

#### Output Parsing (Inference layer)

**File:** `app/core/inference.py`

- Replace the TODO blocks in `run_building_detection()` and `run_damage_assessment()`.
- Match your model's raw output format to the expected dict structure.
- Once done, set `MOCK_MODE=False` in `.env`.

---

## ⚙️ Environment Variables

Edit `.env` to override defaults:

| Variable | Default | Description |
|---|---|---|
| `MOCK_MODE` | `True` | Set `False` when real models are available |
| `BUILDING_MODEL_PATH` | `models/building_detector.pth` | Path to building detector weights |
| `DAMAGE_MODEL_PATH` | `models/damage_classifier.pth` | Path to damage classifier weights |
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum upload size |
| `IMAGE_HEIGHT` | `512` | Resize height |
| `IMAGE_WIDTH` | `512` | Resize width |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection threshold |
| `DEBUG` | `True` | Verbose logging |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Frontend | Streamlit |
| Validation | Pydantic v2 |
| Dataset | xBD (xView2) |

---

## 📝 Notes

- **MOCK MODE** is the default. All predictions are clearly labelled as simulated.
- The system is designed for **Windows** (PowerShell commands shown).
- Models are loaded once at **startup** to avoid repeated disk I/O on each request.
- CORS is enabled so the Streamlit frontend can call the API from any origin.
- All preprocessing (resize → normalise → tensor) happens server-side.

---

*AI Disaster Damage Assessment | EPICS Project — Task 5: Model Serving & Systems Integration*
