# R1-Genie RAG System Setup Guide

Complete setup guide for the RAG-based vendor mapping system for O-RAN R1 API generation.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Qdrant Vector Database Setup](#qdrant-setup)
7. [Usage Examples](#usage-examples)
8. [MCP Server Setup](#mcp-server-setup)
9. [REST API Deployment](#rest-api-deployment)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt  # Or: uv pip install -r requirements.txt

# 2. Start Qdrant vector database
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Upload vendor documentation
python embedding_pipeline.py vendor_api_spec.yaml viavi

# 5. Run tests
python test_rag_system.py

# 6. Start using RAG!
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Query                    │
│          "Get RSRP for Cell-1 using Viavi equipment"        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Vendor Detector (3-tier detection)              │
│  Tier 1: Explicit → Tier 2: Context → Tier 3: Default      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Vendor Mapper                         │
│  1. Query Building + Embedding                               │
│  2. Hybrid Search (Semantic + Metadata Filter)               │
│  3. Confidence Scoring                                       │
│  4. Validation (Prevent Hallucinations)                      │
│  5. Fallback to YAML if confidence < 0.75                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Vendor-Specific REST API Call                   │
│  GET https://viavi.smo.com/api/v1/ue/measurements/rsrp     │
└─────────────────────────────────────────────────────────────┘
```

### Components

- **document_parsers.py**: Parse OpenAPI/PDF/Markdown vendor docs
- **embedding_pipeline.py**: Generate embeddings and store in Qdrant
- **vendor_detector.py**: Auto-detect vendor from queries
- **rag_vendor_mapper.py**: Core RAG retrieval + YAML fallback
- **r1_dataset_builder_v3.py**: Training (canonical) & Inference (RAG) modes
- **mcp_rag_server.py**: Claude Desktop integration
- **vendor_doc_api.py**: FastAPI REST endpoints

---

## Prerequisites

### System Requirements
- **Python**: 3.13+ (or 3.11+)
- **Docker**: For Qdrant vector database
- **RAM**: 4GB+ (8GB+ recommended)
- **Disk**: 2GB+ for vector storage

### API Keys Required
- **OpenAI**: For embeddings (`text-embedding-3-small`)
- **Anthropic**: For LLM validation (optional but recommended)
- **LlamaParse**: For PDF parsing (optional)

### Qdrant Vector Database
- Self-hosted (Docker) or Qdrant Cloud
- Port 6333 (default)

---

## Installation

### Option 1: Using pip
```bash
cd /path/to/r1-genie/data_generator

# Install all dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (faster)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

### Option 3: Using Poetry
```bash
cd /path/to/r1-genie

# Install from pyproject.toml
poetry install
```

### Verify Installation
```bash
python -c "import qdrant_client, openai, anthropic; print('✅ All RAG dependencies installed')"
```

---

## Configuration

### Environment Variables (.env)

Copy the `.env` file from `data_generator/` and configure:

```bash
# ==========================================
# RAG System Configuration
# ==========================================

# Enable RAG (set to 'true' for inference mode)
ENABLE_RAG=true

# Vector Database (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=vendor_apis

# API Keys
OPENAI_API_KEY=sk-proj-...           # Required for embeddings
ANTHROPIC_API_KEY=sk-ant-...         # Optional for validation
LLAMA_CLOUD_API_KEY=llx_...          # Optional for PDF parsing

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# RAG Configuration
RAG_CONFIDENCE_THRESHOLD=0.75   # Minimum confidence for RAG
RAG_TOP_K=5                     # Number of retrieval results
DEFAULT_VENDOR=generic           # Fallback vendor

# Caching (Redis) - Optional
ENABLE_CACHE=false
REDIS_HOST=localhost
REDIS_PORT=6379

# Vendor Detection
VENDOR_DETECTION_CONFIDENCE_MIN=0.60
```

### Get API Keys

**OpenAI** (Required):
1. Visit https://platform.openai.com/api-keys
2. Create API key
3. Add to `.env` as `OPENAI_API_KEY`

**Anthropic** (Optional but recommended):
1. Visit https://console.anthropic.com/settings/keys
2. Create API key
3. Add to `.env` as `ANTHROPIC_API_KEY`

**LlamaParse** (Optional for PDF parsing):
1. Visit https://cloud.llamaindex.ai
2. Get API key
3. Add to `.env` as `LLAMA_CLOUD_API_KEY`

---

## Qdrant Setup

### Option 1: Docker (Recommended)
```bash
# Start Qdrant with persistent storage
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify Qdrant is running
curl http://localhost:6333/
```

### Option 2: Docker Compose
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
```

```bash
docker-compose up -d
```

### Option 3: Qdrant Cloud
1. Visit https://cloud.qdrant.io
2. Create cluster
3. Update `.env`:
   ```bash
   QDRANT_HOST=your-cluster.qdrant.io
   QDRANT_PORT=6333
   ```

### Create Collection
```bash
# The embedding pipeline will create the collection automatically
# Or create manually:
curl -X PUT 'http://localhost:6333/collections/vendor_apis' \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    }
  }'
```

---

## Usage Examples

### 1. Upload Vendor Documentation

#### Upload OpenAPI Spec
```bash
python embedding_pipeline.py /path/to/viavi_api_v3.0.yaml viavi
```

#### Upload PDF Manual
```bash
python embedding_pipeline.py /path/to/ericsson_api_manual.pdf ericsson
```

#### Upload Markdown Docs
```bash
python embedding_pipeline.py /path/to/nokia_api_docs.md nokia
```

### 2. Test RAG Mapping

```python
from rag_vendor_mapper import RAGVendorMapper
from r1_dataset_builder_v3 import CANONICAL_KPIS

# Initialize mapper
mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)

# Map canonical KPI to vendor
result = mapper.map_to_vendor(
    canonical_kpi="rsrp_dbm",
    vendor="viavi"
)

print(f"Endpoint: {result['vendor_api']['endpoint']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Source: {result['source']}")  # 'rag' or 'yaml_fallback'
```

### 3. Vendor Auto-Detection

```python
from vendor_detector import VendorDetector

detector = VendorDetector()

# Detect from query
result = detector.detect("Get RSRP values using Viavi equipment")

print(f"Vendor: {result['vendor']}")       # 'viavi'
print(f"Confidence: {result['confidence']}")  # 0.95
print(f"Method: {result['method']}")       # 'explicit'
```

### 4. Generate Inference Dataset

```python
from r1_dataset_builder_v3 import build_rest_sample

# Training mode (canonical only)
sample_training = build_rest_sample(mode="training")

# Inference mode (with RAG vendor mapping)
sample_inference = build_rest_sample(mode="inference")

# Check vendor-specific API
if "vendor_api" in sample_inference:
    print(sample_inference["vendor_api"])
```

### 5. End-to-End Workflow

```python
from vendor_detector import VendorDetector
from rag_vendor_mapper import RAGVendorMapper
from r1_dataset_builder_v3 import CANONICAL_KPIS

# User query
query = "Get RSRP and throughput for Cell-1 using Viavi"

# Step 1: Detect vendor
detector = VendorDetector()
vendor_info = detector.detect(query)

# Step 2: Map KPIs
mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)
mapping = mapper.map_to_vendor("rsrp_dbm", vendor_info["vendor"])

# Step 3: Build API call
print(f"Vendor: {vendor_info['vendor']}")
print(f"Endpoint: {mapping['vendor_api']['endpoint']}")
print(f"Confidence: {mapping['confidence']:.2f}")
```

---

## MCP Server Setup

### For Claude Desktop Integration

1. **Start MCP Server**:
   ```bash
   python mcp_rag_server.py
   ```

2. **Configure Claude Desktop**:

   Edit `~/.config/claude-desktop/config.json` (macOS/Linux) or
   `%APPDATA%\Claude\config.json` (Windows):

   ```json
   {
     "mcpServers": {
       "r1-vendor-mapping": {
         "command": "python",
         "args": ["/absolute/path/to/r1-genie/data_generator/mcp_rag_server.py"],
         "env": {
           "QDRANT_HOST": "localhost",
           "QDRANT_PORT": "6333",
           "OPENAI_API_KEY": "sk-proj-...",
           "ANTHROPIC_API_KEY": "sk-ant-..."
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**

4. **Test MCP Tools** in Claude Desktop:
   ```
   You: "Upload Viavi API docs from /path/to/viavi_api.yaml"

   Claude: [Uses upload_vendor_docs tool]

   You: "Map rsrp_dbm to Viavi API"

   Claude: [Uses map_canonical_to_vendor tool]
   ```

---

## REST API Deployment

### Development Server
```bash
python vendor_doc_api.py
```

Access at: http://localhost:8000
API Docs: http://localhost:8000/docs

### Production Deployment (with Gunicorn)
```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn vendor_doc_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker Deployment
Create `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Copy requirements
COPY pyproject.toml ./
RUN pip install poetry && poetry install --no-dev

# Copy application
COPY *.py ./
COPY *.yaml ./

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "vendor_doc_api.py"]
```

```bash
docker build -t r1-vendor-api .
docker run -p 8000:8000 \
  -e QDRANT_HOST=host.docker.internal \
  -e OPENAI_API_KEY=sk-... \
  r1-vendor-api
```

---

## Testing

### Run Full Test Suite
```bash
python test_rag_system.py
```

### Run Individual Tests
```bash
# Test canonical KPI implementation
python test_canonical_kpis.py

# Test dataset generation
python r1_dataset_builder_v3.py
```

### Test MCP Server
```bash
# In one terminal
python mcp_rag_server.py

# In another terminal
python -c "
from mcp_rag_server import call_tool
result = call_tool('get_canonical_kpis', {})
print(result)
"
```

### Test REST API
```bash
# Start API
python vendor_doc_api.py &

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/kpis
curl -X POST http://localhost:8000/api/v1/vendors/detect \
  -H 'Content-Type: application/json' \
  -d '{"query": "Get RSRP using Viavi"}'
```

---

## Troubleshooting

### Issue: Qdrant Connection Failed
```
Error: Cannot connect to Qdrant at localhost:6333
```

**Solution**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if not running
docker run -p 6333:6333 qdrant/qdrant

# Verify connection
curl http://localhost:6333/
```

### Issue: OpenAI API Key Error
```
Error: Incorrect API key provided
```

**Solution**:
- Check `.env` file has correct `OPENAI_API_KEY`
- Verify key at https://platform.openai.com/api-keys
- Ensure key starts with `sk-proj-` or `sk-`

### Issue: Low RAG Confidence (Always Falling Back to YAML)
```
Warning: Confidence 0.65 below threshold 0.75
```

**Solution**:
1. Check if vendor docs are uploaded:
   ```bash
   curl http://localhost:6333/collections/vendor_apis
   ```

2. Upload more vendor documentation

3. Lower confidence threshold in `.env`:
   ```bash
   RAG_CONFIDENCE_THRESHOLD=0.60
   ```

### Issue: PDF Parsing Fails
```
Error: LLAMA_CLOUD_API_KEY not set
```

**Solution**:
- Get LlamaParse key: https://cloud.llamaindex.ai
- Add to `.env`: `LLAMA_CLOUD_API_KEY=llx_...`
- Or convert PDFs to Markdown manually

### Issue: MCP Server Not Showing in Claude Desktop
**Solution**:
1. Check config path: `~/.config/claude-desktop/config.json`
2. Use absolute paths in config
3. Check MCP server logs for errors
4. Restart Claude Desktop

---

## Performance Optimization

### 1. Enable Caching (Redis)
```bash
# Install Redis
docker run -d -p 6379:6379 redis

# Enable in .env
ENABLE_CACHE=true
REDIS_HOST=localhost
```

### 2. Batch Uploads
```bash
# Upload multiple files at once
for file in vendor_docs/*.yaml; do
    python embedding_pipeline.py "$file" viavi &
done
wait
```

### 3. Optimize Embedding Batch Size
In `embedding_pipeline.py`:
```python
EmbeddingGenerator(batch_size=200)  # Increase for faster processing
```

---

## Next Steps

1. **Upload all vendor documentation** to Qdrant
2. **Configure vendor-to-cell mappings** in `vendor_detector.py`
3. **Train LLM** using generated datasets
4. **Deploy REST API** to production
5. **Monitor RAG confidence scores** and refine thresholds

---

## Support & Documentation

- **GitHub Issues**: https://github.com/yourusername/r1-genie/issues
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **O-RAN Specs**: https://www.o-ran.org/specifications
- **MCP Protocol**: https://spec.modelcontextprotocol.io/

---

**Last Updated**: 2025-01-26
**Version**: 1.0.0
