# ORIGYN API

> **High-Accuracy AI Content Detection Backend API**  
> Powered by an ensemble of 23 independent detection engines — neural classifiers, statistical tests, and linguistic forensics. Run locally or deploy anywhere via Docker.

---

## Features

- **23 Independent Detection Engines:** Neural classifiers (DeBERTa, RoBERTa, E5 LoRA), statistical distributions (GLTR, Perplexity, Binoculars), and structural heuristics.
- **Pure REST & Streaming API:** FastAPI-powered with sub-second quick-scoring and real-time Server-Sent Events (SSE) streaming.
- **Zero Third-Party Data Sharing:** Runs fully containerized on your infrastructure.
- **Hardware Adaptive:** Auto-detects available CPU/RAM/GPU and selects optimal profiles (`lite`, `standard`, `performance`).
- **Production Ready:** Pre-configured with Docker, health checks, configurable CORS, and asynchronous queue management.

---

## Quick Start

### 1. Local Development (Python 3.10+)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at: **`http://localhost:8000/docs`**

---

### 2. Docker & Docker Compose

```bash
# Build and run with persistent storage for models & database
docker compose up -d
```

Check server health:
```bash
curl http://localhost:8000/health
```

---

## Hardware Profiles

Configure the resource footprint via the `ORIGYN_PROFILE` environment variable in your `.env` or container settings:

| Profile | Target Hardware | Description |
| :--- | :--- | :--- |
| **`lite`** | `< 8 GB RAM`, 2 vCPUs | Runs the 4 best neural classifiers. Low latency, lightweight. |
| **`standard`** *(Default)* | `8–16 GB RAM`, 4 vCPUs | Runs all 23 detection engines across parallel CPU workers. |
| **`performance`** | `16+ GB RAM` or NVIDIA GPU | Maximum throughput with CUDA acceleration and multi-replica pools. |

---

## API Endpoints

All endpoints accept JSON payloads. Text inputs must be at least **50 characters**.

| Method | Endpoint | Description | Typical Latency |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API status and endpoint directory | `< 10ms` |
| `GET` | `/docs` | OpenAPI / Swagger interactive UI | `< 50ms` |
| `GET` | `/health` | Health check & queue status | `< 20ms` |
| `POST` | `/api/quick-score` | Fast scan using 4 ML classifiers | `~0.5s – 2s` |
| `POST` | `/api/analyze` | Full 23-engine ensemble scan | `~2s – 8s` |
| `POST` | `/api/analyze/start` | Async scan (returns `report_id` for streaming) | `< 50ms` |
| `GET` | `/api/stream/{report_id}` | Real-time SSE stream of engine results | Streaming |
| `GET` | `/api/report/{report_id}` | Fetch previous analysis report by ID | `< 20ms` |
| `POST` | `/api/paragraph-score` | Per-paragraph AI score breakdown | `~1s – 4s` |
| `POST` | `/api/scan/snippets` | Ultra-fast batch scan for search snippets | Batch |
| `GET` | `/api/engines` | Metadata & ensemble weights of all engines | `< 20ms` |

---

## Integration Examples

### Fast Scan (`POST /api/quick-score`)

#### JavaScript / TypeScript
```javascript
const res = await fetch("http://localhost:8000/api/quick-score", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "Artificial intelligence has rapidly evolved over recent years, transforming numerous industries and reshaping everyday technology."
    // Or URL: url: "https://example.com/article"
  })
});

const data = await res.json();
console.log(`Score: ${data.score}/100 | Verdict: ${data.verdict}`);
```

#### Python
```python
import requests

res = requests.post(
    "http://localhost:8000/api/quick-score",
    json={"text": "Artificial intelligence has rapidly evolved over recent years..."}
)
print(res.json())
```

**Response:**
```json
{
  "score": 88.5,
  "verdict": "Likely AI-generated",
  "confidence": 0.94,
  "raw": {
    "classifier_desklib": 0.91,
    "classifier_e5": 0.89,
    "classifier_superannotate": 0.87,
    "classifier_remodetect": 0.88
  }
}
```

---

### Real-Time Streaming Scan (`SSE`)

For a live user experience where engines report results as they finish:

1. **Start analysis:**
   ```bash
   curl -X POST "http://localhost:8000/api/analyze/start" \
     -H "Content-Type: application/json" \
     -d '{"text": "Sample text exceeding 50 characters..."}'
   # Returns: {"status": "started", "report_id": "ab12cd34"}
   ```

2. **Connect to SSE stream:**
   ```javascript
   const eventSource = new EventSource("http://localhost:8000/api/stream/ab12cd34");

   eventSource.onmessage = (event) => {
     const data = JSON.parse(event.data);
     if (data.done) {
       eventSource.close();
       console.log("Analysis complete!");
     } else {
       console.log(`Engine ${data.engine_name}: ${data.score} (${data.verdict})`);
       console.log(`Running overall score: ${data.overall_score}`);
     }
   };
   ```

---

## Deploy to Microsoft Azure

### Azure Container Apps (Recommended)

```bash
# 1. Login to Azure
az login

# 2. Create resource group & Azure Container Registry (ACR)
az group create --name origyn-rg --location eastus
az acr create --resource-group origyn-rg --name origynacr$RANDOM --sku Basic --admin-enabled true

# 3. Build container directly in Azure cloud (no local Docker required)
az acr build --registry <YOUR_ACR_NAME> --image origyn-api:latest .

# 4. Create Container App environment
az containerapp env create \
  --name origyn-env \
  --resource-group origyn-rg \
  --location eastus

# 5. Deploy the API
az containerapp create \
  --name origyn-api \
  --resource-group origyn-rg \
  --environment origyn-env \
  --image <YOUR_ACR_NAME>.azurecr.io/origyn-api:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 2.0 \
  --memory 4.0Gi \
  --env-vars ORIGYN_PROFILE=lite ORIGYN_CORS_ORIGINS="*"
```

---

## Configuration Reference

Set these in your `.env` or container environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ORIGYN_PORT` | `8000` | HTTP port the server listens on |
| `ORIGYN_HOST` | `0.0.0.0` | Bind host address |
| `ORIGYN_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated or `*`) |
| `ORIGYN_PROFILE` | *(auto)* | `lite`, `standard`, or `performance` |
| `ORIGYN_DATA_DIR` | `./data` | Directory for SQLite scan logs & database |
| `HF_HOME` | `./models` | Directory for cached Hugging Face model weights |

---

## License

MIT License.
