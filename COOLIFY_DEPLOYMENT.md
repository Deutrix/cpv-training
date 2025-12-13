# Coolify Deployment Guide for CPV Decoder

## Overview

This guide explains how to deploy the CPV Decoder application to Coolify. The application is a FastAPI service that serves an AI model for predicting CPV (Common Procurement Vocabulary) codes from text descriptions.

## Architecture

### Model Serving
**Yes, the model will be served from this folder.** The Docker container includes:
- The trained model files (`models/cpv-decoder/`)
- CPV codes database (`cpv_codes.csv`)
- Pre-computed embeddings (`cpv_embeddings.pt`)
- Static web UI (`static/index.html`)

The entire application is self-contained in the Docker image, making it portable and easy to deploy.

## Prerequisites

1. **Coolify instance** - Running and accessible
2. **Git repository** - Your code pushed to a Git repository (GitHub, GitLab, etc.)
3. **Model files** - Ensure `models/cpv-decoder/` directory contains your trained model

## Deployment Steps

### 1. Prepare Your Repository

Ensure these files are committed to your repository:
- `Dockerfile` ✓
- `.dockerignore` ✓
- `requirements.txt` ✓
- `main.py` ✓
- `predictor.py` ✓
- `cpv_codes.csv` ✓
- `cpv_embeddings.pt` ✓
- `models/cpv-decoder/` directory ✓
- `static/` directory ✓

### 2. Create Application in Coolify

1. Log into your Coolify dashboard
2. Click **+ New Resource**
3. Select **Application**
4. Choose your Git repository
5. Select the branch to deploy (e.g., `main`)

### 3. Configure Build Settings

In the application settings:

**Build Pack:** `Dockerfile`

**Dockerfile Location:** `./Dockerfile` (default)

**Port:** `8000`

### 4. Build Arguments (Required)

**IMPORTANT:** Set these as **Build Arguments** in Coolify (not environment variables):

1. Go to your application settings in Coolify
2. Find **Build Arguments** section
3. Add these arguments:

```bash
HF_MODEL_ID=Deutrix/cpv-decoder
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**To get your HF_TOKEN:**
1. Go to https://huggingface.co/settings/tokens
2. Copy your token (the one you used for login)
3. Paste it as the `HF_TOKEN` build argument

> **Note:** The model will be downloaded from Hugging Face during the Docker build process.

### 5. Environment Variables (Optional)

You can add these runtime environment variables if needed:

```bash
# Optional: Set log level
LOG_LEVEL=info
```

### 6. Resource Allocation

Recommended settings:

- **Memory:** 2GB minimum (4GB recommended for better performance)
- **CPU:** 1-2 cores
- **Storage:** 1GB minimum (depends on model size)

> **Note:** The model and embeddings are loaded into memory at startup. Ensure sufficient RAM is allocated.

### 7. Health Check

The application includes a built-in health check endpoint:

```
GET /health
```

Coolify will automatically use the Docker `HEALTHCHECK` directive. You can also configure custom health checks in Coolify:

- **Path:** `/health`
- **Port:** `8000`
- **Interval:** `30s`

### 8. Deploy

Click **Deploy** in Coolify. The deployment process will:

1. Clone your repository
2. Build the Docker image
3. Run the container
4. Perform health checks
5. Route traffic to your application

## Endpoints

After deployment, your application will expose:

- **Web UI:** `https://your-domain.com/`
- **API Prediction:** `POST https://your-domain.com/predict`
- **Health Check:** `GET https://your-domain.com/health`
- **Debug Info:** `GET https://your-domain.com/debug_load`

## API Usage

### Predict CPV Code

```bash
curl -X POST https://your-domain.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "laptop computers"}'
```

**Response:**
```json
{
  "keyword_matches": [...],
  "rag_matches": [...],
  "model_predictions": [...],
  "final_decision": [
    {
      "cpv_code": "30213000",
      "name": "Personal computers",
      "confidence": 0.85,
      "sources": ["keyword", "rag", "model"]
    }
  ]
}
```

## Troubleshooting

### Container Fails to Start

1. Check logs in Coolify dashboard
2. Verify model files are included in the repository
3. Check memory allocation (increase if needed)

### Model Not Loading

1. Access debug endpoint: `GET /debug_load`
2. Verify `model_dir_exists` is `true`
3. Check that `models/cpv-decoder/` contains:
   - `config.json`
   - `pytorch_model.bin` or `model.safetensors`
   - `tokenizer_config.json`
   - Other tokenizer files

### Health Check Failing

1. Increase `start-period` in Dockerfile if model takes long to load
2. Check if port 8000 is accessible
3. Verify the `/health` endpoint returns 200 OK

## Performance Optimization

### 1. Use GPU (Optional)

If your Coolify server has GPU support:

1. Update Dockerfile to use CUDA base image:
   ```dockerfile
   FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
   ```

2. Install PyTorch with CUDA support in `requirements.txt`:
   ```
   torch>=2.6.0+cu118
   ```

3. Configure Coolify to use GPU resources

### 2. Enable Multiple Workers

For higher throughput, increase workers in Dockerfile:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

> **Warning:** Each worker loads the model into memory. Ensure sufficient RAM.

### 3. Add Caching

Consider adding Redis for caching predictions:

1. Add Redis service in Coolify
2. Update application to cache frequent queries

## Monitoring

### Logs

Access logs via Coolify dashboard or CLI:

```bash
coolify logs <app-name>
```

### Metrics

Monitor these metrics:
- Response time for `/predict` endpoint
- Memory usage (should be stable after startup)
- CPU usage during predictions
- Health check success rate

## Scaling

### Horizontal Scaling

Coolify supports horizontal scaling:

1. Go to application settings
2. Increase replica count
3. Coolify will load balance across instances

> **Note:** Each replica loads the full model into memory.

### Vertical Scaling

If predictions are slow:
1. Increase CPU allocation
2. Increase memory allocation
3. Consider GPU acceleration

## Security

The Dockerfile includes security best practices:

- ✓ Non-root user (`appuser`)
- ✓ Minimal base image (`python:3.9-slim`)
- ✓ No unnecessary packages
- ✓ Health checks enabled

### Additional Security (Optional)

1. **Add API authentication:**
   - Implement API keys in `main.py`
   - Use Coolify's environment variables for secrets

2. **Rate limiting:**
   - Add rate limiting middleware to FastAPI
   - Protect against abuse

3. **HTTPS:**
   - Coolify automatically provides SSL certificates
   - Ensure "Force HTTPS" is enabled

## Backup

Important files to backup:
- `models/cpv-decoder/` - Your trained model
- `cpv_codes.csv` - CPV codes database
- `cpv_embeddings.pt` - Pre-computed embeddings

Store these in a separate location (S3, Git LFS, etc.) for disaster recovery.

## Updates

To update the application:

1. Push changes to your Git repository
2. Coolify will auto-deploy (if enabled)
3. Or manually trigger deployment in Coolify dashboard

## Cost Estimation

Typical resource usage:
- **Memory:** 2-4GB
- **CPU:** 0.5-1 core (idle), 1-2 cores (under load)
- **Storage:** 500MB - 2GB (depending on model size)
- **Bandwidth:** Depends on traffic

## Support

For issues:
1. Check application logs in Coolify
2. Review this deployment guide
3. Test locally with Docker: `docker build -t cpv-decoder . && docker run -p 8000:8000 cpv-decoder`

## Summary

✅ Model is served from the Docker container (self-contained)  
✅ Dockerfile optimized for production  
✅ Health checks configured  
✅ Security best practices implemented  
✅ Ready for Coolify deployment  

Your CPV Decoder is now ready to deploy to Coolify!
