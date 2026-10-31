# ==============================================================================
# Multi-Stage Dockerfile for NeuralCord (Spinal Cord MRI Diagnostic Pipeline)
# Combines React Frontend & FastAPI + PyTorch Backend into a Single Service
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Modern React Frontend
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /build

# Install dependencies with caching
COPY app/app/frontend/package*.json app/app/frontend/yarn*.lock ./
RUN npm install --legacy-peer-deps --no-audit --no-fund

# Copy frontend source code and configs
COPY app/app/frontend/ ./

# In single-container deployment, the frontend accesses the backend on the same host
ENV REACT_APP_BACKEND_URL=""
ENV GENERATE_SOURCEMAP=false

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

# Install CPU PyTorch & TorchVision (optimized for cloud web services)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY app/app/backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt

# Copy ML engine, models, and training dataset
COPY app/app/ml /app/app/ml

# Copy backend server and API logic
COPY app/app/backend /app/app/backend

# Copy built frontend from Stage 1 into the location expected by FastAPI server.py
COPY --from=frontend-builder /build/build /app/app/frontend/build

WORKDIR /app/app/backend

# Render / Railway / Cloud Run injects $PORT dynamically
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV CORS_ORIGINS="*"

EXPOSE 8000

# Run FastAPI backend via Uvicorn
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
