# NeuralCord Deployment & Non-Spinal MRI Rejection Guide

This guide explains how to deploy the **NeuralCord — Spinal Cord MRI Diagnostic Pipeline** and details the newly integrated **Spinal Cord Verifier** trained to detect and reject non-spinal MR images (e.g., Brain MRIs, Knee MRIs, Chest/Abdominal scans, and non-medical images).

---

## 🛡️ New Feature: Non-Spinal Cord Image Rejection

### Problem Solved
Previously, standard 4-class classification models (`Tumor`, `MS`, `Injury`, `Normal`) forced softmax probabilities over any input image. An uploaded Brain MRI or Knee scan would be mistakenly diagnosed with a spinal cord condition.

### Solution Architecture
The pipeline now integrates a **dual-layer Spinal Cord Verification System**:

1. **`SpinalCordVerifierNet` (Deep CNN Checkpoint)**:
   - Residual depthwise convolutional encoder with Squeeze-and-Excitation (SE) channel attention.
   - Dual-head output:
     - Binary verification: `P(is_spinal_cord)`
     - Modality classification: `Spinal Cord (Sagittal)`, `Spinal Cord (Axial)`, `Brain MRI`, `Knee / Joint MRI`, `Other Non-Spinal Scan`.
2. **`AnatomicalSpineDetector` (Medical Computer Vision Priors)**:
   - **Columnar Continuity Analysis**: Tests for the continuous longitudinal spinal canal characteristic of sagittal spinal MRI.
   - **Cranial Vault & Ventricle Detection**: Identifies circular skull envelopes, bilateral cerebral hemisphere symmetry, and lateral ventricle CSF spaces characteristic of Brain MRI.
   - **Articular Joint Space Test**: Identifies opposing femoral/tibial bone blocks and transverse joint spaces characteristic of Knee MRI.
   - **U-Net Mask Plausibility Gate**: Ensures the segmented mask represents a structurally coherent spinal cord.

If an image fails these checks, the backend halts inference and returns `HTTP 422 Unprocessable Entity`:
```json
{
  "detail": "Non-Spinal Scan Rejected: Detected round cranial envelope, bilateral cerebral hemisphere symmetry, and brain parenchyma. (Detected: Brain MRI, Confidence: 92.0%). NeuralCord is strictly calibrated for Spinal Cord MRI scans. Please upload a sagittal or axial spinal cord scan."
}
```

The frontend immediately renders a dedicated warning alert with full diagnostic reasoning.

---

## 🚀 Deployment Options

### Option 1: 1-Click Unified Docker Deployment (Recommended)

A production-ready multi-stage `Dockerfile` is included in the repository. It builds the modern React dashboard and serves both the frontend and the FastAPI backend from a single URL without CORS issues.

#### A. Free Deployment on Hugging Face Spaces (16GB RAM Free)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your Space (e.g. `spinalcord-mri-diagnostics`).
3. Select **Docker** as the Space SDK and choose the **Blank** template.
4. Clone your Space repository locally or push this repo directly to Hugging Face:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```
5. Hugging Face will automatically build the `Dockerfile` and provide a live public HTTPS URL with **16GB RAM + 2 vCPUs completely free**.

---

#### B. Deployment on Render.com (via Blueprint or Docker)
1. Push this project to your GitHub repository.
2. Log into [Render.com](https://render.com).
3. Click **New +** -> **Blueprint**, and select your GitHub repository (it will automatically detect `render.yaml`).
4. Alternatively, click **New +** -> **Web Service**:
   - Environment: **Docker**
   - Docker Context: `.`
   - Dockerfile Path: `Dockerfile`
   - Plan: **Starter** or higher (>= 1GB RAM recommended for PyTorch inference)
5. Click **Create Web Service**. Render builds both the React UI and FastAPI server and deploys them to a `https://your-service.onrender.com` URL.

---

#### C. Deployment on Railway.app
1. Go to [Railway.app](https://railway.app) and create a **New Project**.
2. Select **Deploy from GitHub repo**.
3. Select this repository. Railway automatically detects the `Dockerfile`, injects the `$PORT` variable, and starts the container.
4. Go to **Settings** -> **Generate Domain** to get your public HTTPS link.

---

### Option 2: Local Deployment with Docker Compose

If you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on Windows, Mac, or Linux:

1. Open your terminal in the project root:
   ```bash
   docker compose up --build
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
3. Both the React dashboard and the FastAPI API with the PyTorch model and optional MongoDB will be running.

---

### Option 3: Separated Cloud Deployment (Vercel Frontend + Render Backend)

If you prefer hosting the React frontend on Vercel's global CDN:

#### 1. Deploy Backend to Render or Railway
- Deploy `app/app/backend` (Docker or Python).
- Note your backend URL (e.g., `https://spinalcord-api.onrender.com`).

#### 2. Deploy Frontend to Vercel
- Import repository into [Vercel](https://vercel.com).
- Set **Root Directory** to `app/app/frontend`.
- Add Environment Variable:
  - `REACT_APP_BACKEND_URL` = `https://spinalcord-api.onrender.com`
- Click **Deploy**.

---

### Option 4: Local Native Development Setup (Windows)

#### 1. Backend Setup
```bash
cd app/app/backend
python -m venv venv
venv\Scripts\activate

# Install PyTorch CPU and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run backend
uvicorn server:app --reload --port 8000
```

#### 2. Frontend Setup
In a second terminal:
```bash
cd app/app/frontend
npm install --legacy-peer-deps
npm start
```
The React dashboard will open on `http://localhost:3000`.

---

## 🧪 Testing the Non-Spinal Rejection Feature

To verify that non-spinal MR images are rejected:
1. Open the dashboard.
2. In the upload area, upload a **Brain MRI** or **Knee MRI** image.
3. Observe the rejection banner:
   - **Status**: `SCAN REJECTED · NON-SPINAL MRI DETECTED`
   - **Reasoning**: Outlines detected cranial envelope, cerebral symmetry, or joint condyles.
4. Upload any genuine spinal cord MRI (or click one of the preset sample buttons: `Tumor`, `MS`, `Injury`, `Normal`):
   - The scan is verified and proceeds to U-Net segmentation, lesion localization, and clinical diagnosis.
