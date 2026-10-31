# NeuralCord — Spinal Cord MRI Diagnostic Pipeline

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=docker&logoColor=white)

NeuralCord is an end-to-end AI-powered diagnostic and segmentation platform for Spinal Cord MRI scans. It predicts 4 clinical disease categories: **Tumor, MS Lesions, Injury, and Normal**, providing full **Grad-CAM & SHAP** explainability, localized lesion discovery, and automatic spinal cord segmentation.

---

## 🛡️ Non-Spinal Cord Image Verification & Rejection (New)

NeuralCord includes an automated **Spinal Cord Verifier** trained to detect and reject non-spinal cord MR images (such as **Brain MRI**, **Knee / Musculoskeletal MRI**, **Chest / Abdominal CT**, and non-medical images).

### How It Works:
- **`SpinalCordVerifierNet` (Deep CNN)**: Multi-class convolutional neural network with Squeeze-and-Excitation channel attention.
- **`AnatomicalSpineDetector` (Medical CV Priors)**:
  - **Columnar Continuity**: Verifies continuous longitudinal spinal canal across the sagittal axis.
  - **Cranial Vault & Ventricle Detection**: Identifies circular skull envelopes, bilateral cerebral symmetry, and lateral ventricles characteristic of Brain MRI.
  - **Articular Joint Detector**: Identifies condylar bone gaps characteristic of Knee MRI.
  - **U-Net Cord Plausibility Check**: Verifies that the U-Net produces a valid anatomical cord contour.
- If a non-spinal scan is uploaded, inference halts with `HTTP 422 Unprocessable Entity` and displays a detailed rejection alert with clinical rationale.

---

## 🚀 1-Click Deployment

### Option A: Cloud Deployment (Render / Hugging Face Spaces / Railway)
- **Hugging Face Spaces (Free 16GB RAM + 2 vCPUs)**:
  1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
  2. Select **Docker** SDK (Blank template).
  3. Push this repository to your Space. The included multi-stage `Dockerfile` automatically builds both the React dashboard and the FastAPI PyTorch backend.
- **Render.com**:
  1. Create a new **Web Service** from your GitHub repo.
  2. Select **Docker** (it automatically uses `Dockerfile` and `render.yaml`).
  3. Deploy!

### Option B: Local Docker Compose
```bash
docker compose up --build
```
Navigate to `http://localhost:8000` to access the full application.

For detailed deployment options (Vercel, Railway, native Windows/Linux), see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 🛠️ Local Development Setup

### 1. Backend Setup
```bash
cd app/app/backend
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install PyTorch CPU and backend dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run the FastAPI server
uvicorn server:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd app/app/frontend
npm install --legacy-peer-deps
npm start
```
The React frontend will be live at `http://localhost:3000`.

---

## 🧠 Diagnostic Pipeline

1. **Upload & Verification:** User uploads an MRI slice. The dual-layer verifier validates that the scan is a genuine Spinal Cord MRI and rejects Brain, Knee, and non-spinal scans.
2. **Preprocessing:** Contrast Limited Adaptive Histogram Equalization (CLAHE) and grayscale tensor normalization.
3. **U-Net Cord Segmentation:** Deep neural segmentation predicts the exact spinal cord column mask.
4. **Lesion Localization:** Segmented cord is scanned for hyperintense plaques (MS), focal expansive masses (Tumor), or cord deformation (Injury).
5. **Explainability & Classification:** Dual Grad-CAM and SHAP attribution overlays confirm radiological rationale.

---

## 📄 License
This project is licensed under the MIT License.
