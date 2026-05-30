# DerMind API - Docker Setup

This guide explains how to build and run the DerMind API application using Docker with Python 3.11.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- API_KEY environment variable configured

## Quick Start

### 1. Setup Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and set your `API_KEY`:
```env
API_KEY=your-secure-api-key-here
```

### 2. Build and Run with Docker Compose

**Development Mode:**
```bash
docker-compose up -d
```

**Production Mode:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Access the Application

- **API Documentation:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

## Available Commands

Using Makefile (if available):
```bash
make build          # Build Docker image
make up             # Start containers in development
make down           # Stop containers
make logs           # View container logs
make shell          # Access container shell
make restart        # Restart containers
make build-prod     # Build production image
make up-prod        # Start production containers
```

Using Docker Compose directly:
```bash
# Development
docker-compose build
docker-compose up -d
docker-compose down
docker-compose logs -f
docker-compose exec dermind-api sh

# Production
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml down
```

## Container Details

- **Base Image:** `python:3.11-slim`
- **Port:** 8000 (configurable)
- **Container Name:** `dermind-api`
- **Health Check:** Enabled, checks `/docs` endpoint every 30 seconds

## System Dependencies

The Dockerfile includes:
- `build-essential` - For building Python extensions
- `libopenblas-dev`, `liblapack-dev`, `gfortran` - Required for NumPy, SciPy, scikit-learn (ML libraries)

## Environment Variables

Key variables to configure:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | Yes | - | Authentication key for API |
| `PYTHONUNBUFFERED` | No | 1 | Unbuffered Python output |

## Volume Mounting

**Development:** The source code is mounted as a volume, allowing live code changes.

**Production:** Code is copied into the image during build (immutable).

## Building the Image

### Development Build:
```bash
docker-compose build
```

### Production Build:
```bash
docker build -t dermind-api:latest .
```

### Custom Tag:
```bash
docker build -t dermind-api:v1.0.0 .
```

## Running Tests

```bash
# Access container shell
docker-compose exec dermind-api sh

# Inside container, run your tests
python -m pytest
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs dermind-api
```

### Port already in use
Edit `docker-compose.yml` and change port mapping:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Permission issues
```bash
docker-compose exec dermind-api chown -R nobody:nogroup /app
```

### Rebuild without cache
```bash
docker-compose build --no-cache
```

## Performance Tips

1. **Layer Caching:** Requirements are copied before code to maximize cache reuse
2. **Slim Base Image:** Uses `python:3.11-slim` for smaller image size (~150MB vs ~1GB)
3. **Multi-stage Build:** Can be added if image size becomes critical
4. **Production:** Use restart policy `always` for automatic recovery

## Deployment

### Using Docker Hub
```bash
docker tag dermind-api:latest yourusername/dermind-api:latest
docker push yourusername/dermind-api:latest
```

### Using Private Registry
```bash
docker tag dermind-api:latest registry.example.com/dermind-api:latest
docker push registry.example.com/dermind-api:latest
```

### Kubernetes Deployment
See `k8s/` directory for Kubernetes manifests (if available).

## File Structure

```
.
├── Dockerfile                 # Development/Production Dockerfile
├── docker-compose.yml         # Development compose config
├── docker-compose.prod.yml    # Production compose config
├── .dockerignore              # Files to exclude from Docker build
├── .env.example               # Environment variables template
├── Makefile                   # Helper commands
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── routers/                   # API route handlers
├── models/                    # Data models
└── wrappers/                  # Utility wrappers
```

## Additional Notes

- The application uses FastAPI with Uvicorn as the ASGI server
- API requires authentication via `X-Dermind-Key` header
- Supports models for skin analysis, mental health, and chatbot
- Google Gen AI integration for enhanced features

For more information, see the main README or API documentation at `/docs`.
