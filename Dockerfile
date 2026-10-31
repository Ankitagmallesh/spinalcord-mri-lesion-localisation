# ==============================================================================
# Multi-Stage Dockerfile for NeuralCord (Spinal Cord MRI Diagnostic Pipeline)
# Combines React Frontend & FastAPI + PyTorch Backend into a Single Service
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Modern React Frontend
# ------------------------------------------------------------------------------
FROM node:20-bullseye-slim AS frontend-builder
WORKDIR /build

# Disable strict CI failure on ESLint warnings and optimize memory
ENV CI=false
ENV DISABLE_ESLINT_PLUGIN=true
ENV GENERATE_SOURCEMAP=false
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV REACT_APP_BACKEND_URL=""

# Install dependencies
COPY app/app/frontend/package*.json ./
RUN npm install --legacy-peer-deps --no-audit --no-fund

# Copy frontend source code and configs
COPY app/app/frontend/ ./

# Build production bundle
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python Backend with PyTorch & OpenCV
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Install system runtime libraries required by OpenCV & image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU PyTorch & TorchVision (optimized for lightweight cloud services)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies (FastAPI, OpenCV headless, Pydantic, etc.)
COPY app/app/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ML engine, models, and training dataset
COPY app/app/ml /app/app/ml

# Copy backend server and API logic
COPY app/app/backend /app/app/backend

# Copy built frontend from Stage 1 into the location expected by FastAPI server.py
COPY --from=frontend-builder /build/build /app/app/frontend/build

WORKDIR /app/app/backend

# Port support: 8000 (standard), 7860 (Hugging Face Spaces), 10000 (Render default)
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV CORS_ORIGINS="*"

EXPOSE 8000 7860 10000

# Run FastAPI backend via Uvicorn
CMD ["sh", "-c", "python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
