#!/usr/bin/env python3
"""
Vendor Documentation REST API
FastAPI wrapper for R1-Genie RAG system
"""

import os
import sys
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

# Import FastAPI dependencies
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Query
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("❌ FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Import r1-genie components
try:
    from document_parsers import DocumentParserFactory
    from embedding_pipeline import EmbeddingPipeline
    from rag_vendor_mapper import RAGVendorMapper
    from vendor_detector import VendorDetector
    from r1_dataset_builder_v3 import CANONICAL_KPIS
except ImportError as e:
    print(f"❌ Failed to import r1-genie components: {e}")
    sys.exit(1)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class VendorMappingRequest(BaseModel):
    """Request model for canonical to vendor mapping"""
    canonical_kpi: str = Field(..., description="Canonical KPI name (e.g., 'rsrp_dbm')")
    vendor: str = Field(..., description="Vendor name (e.g., 'viavi')")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Optional context")

class VendorDetectionRequest(BaseModel):
    """Request model for vendor detection"""
    query: str = Field(..., description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Optional network context")

class SearchRequest(BaseModel):
    """Request model for vendor doc search"""
    query: str = Field(..., description="Search query")
    vendor: Optional[str] = Field(None, description="Filter by vendor")
    limit: int = Field(5, description="Number of results", ge=1, le=50)

class UploadResponse(BaseModel):
    """Response model for document upload"""
    status: str
    vendor: str
    file: str
    doc_type: str
    chunks_processed: int
    embeddings_generated: int
    points_uploaded: int

class MappingResponse(BaseModel):
    """Response model for KPI mapping"""
    status: str
    canonical_kpi: str
    vendor: str
    confidence: float
    vendor_api: Dict[str, Any]
    source: str
    validation: Dict[str, Any]

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="R1 Vendor Documentation API",
    description="RAG-based vendor-specific API mapping for O-RAN R1 interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG components
embedding_pipeline: Optional[EmbeddingPipeline] = None
rag_mapper: Optional[RAGVendorMapper] = None
vendor_detector: Optional[VendorDetector] = None

# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global embedding_pipeline, rag_mapper, vendor_detector

    print("🚀 Initializing RAG components...")

    try:
        embedding_pipeline = EmbeddingPipeline()
        rag_mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)
        vendor_detector = VendorDetector()
        print("✅ RAG system initialized")
    except Exception as e:
        print(f"⚠️  RAG initialization failed: {e}")
        print("   API will return errors until Qdrant is running")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down API server")

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    rag_status = "healthy" if embedding_pipeline is not None else "degraded"

    return {
        "status": "ok",
        "service": "r1-vendor-doc-api",
        "version": "1.0.0",
        "rag_system": rag_status
    }

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "service": "R1 Vendor Documentation API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

# =============================================================================
# VENDOR DOCUMENTATION ENDPOINTS
# =============================================================================

@app.post("/api/v1/vendors/{vendor}/docs/upload", tags=["Documentation"])
async def upload_vendor_documentation(
    vendor: str,
    file: UploadFile = File(...),
    doc_type: str = Query("auto", enum=["auto", "openapi", "pdf", "markdown"])
):
    """
    Upload vendor API documentation for RAG indexing

    Args:
        vendor: Vendor name (viavi, ericsson, nokia, etc.)
        file: Documentation file (OpenAPI YAML/JSON, PDF, Markdown)
        doc_type: Document type (auto-detect if not specified)

    Returns:
        Upload statistics and collection info
    """
    if embedding_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    # Save uploaded file temporarily
    temp_path = f"/tmp/{vendor}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Parse document
        parser_factory = DocumentParserFactory(CANONICAL_KPIS)

        if doc_type == "auto":
            chunks = parser_factory.parse(temp_path, vendor)
        else:
            parser = parser_factory.get_parser(temp_path)
            chunks = parser.parse(temp_path, vendor)

        # Process through embedding pipeline
        result = embedding_pipeline.process_document(
            chunks,
            create_collection=True,
            force_recreate=False
        )

        return JSONResponse({
            "status": "success",
            "vendor": vendor,
            "file": file.filename,
            "doc_type": doc_type,
            "chunks_processed": result["chunks_processed"],
            "embeddings_generated": result["embeddings_generated"],
            "points_uploaded": result["points_uploaded"],
            "collection_stats": result["collection_stats"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# =============================================================================
# MAPPING ENDPOINTS
# =============================================================================

@app.post("/api/v1/mappings/canonical-to-vendor", tags=["Mapping"])
async def map_canonical_to_vendor(request: VendorMappingRequest):
    """
    Map canonical KPI to vendor-specific API

    Args:
        request: Mapping request with canonical_kpi, vendor, and optional context

    Returns:
        Vendor-specific API details with confidence score
    """
    if rag_mapper is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = rag_mapper.map_to_vendor(
            canonical_kpi=request.canonical_kpi,
            vendor=request.vendor,
            context=request.context
        )

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/vendors/{vendor}/mappings/{canonical_kpi}", tags=["Mapping"])
async def get_vendor_mapping(
    vendor: str,
    canonical_kpi: str,
    cell_id: Optional[str] = None,
    slice_id: Optional[str] = None
):
    """
    Get vendor-specific mapping for canonical KPI (GET version)

    Args:
        vendor: Vendor name
        canonical_kpi: Canonical KPI name
        cell_id: Optional cell ID for context
        slice_id: Optional network slice ID

    Returns:
        Vendor-specific API details
    """
    if rag_mapper is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    context = {}
    if cell_id:
        context["cellId"] = cell_id
    if slice_id:
        context["slice_id"] = slice_id

    try:
        result = rag_mapper.map_to_vendor(
            canonical_kpi=canonical_kpi,
            vendor=vendor,
            context=context
        )

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# VENDOR DETECTION
# =============================================================================

@app.post("/api/v1/vendors/detect", tags=["Vendor Detection"])
async def detect_vendor(request: VendorDetectionRequest):
    """
    Auto-detect vendor from natural language query

    Args:
        request: Detection request with query and optional context

    Returns:
        Detected vendor with confidence score and method
    """
    if vendor_detector is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = vendor_detector.detect(request.query, request.context)

        return JSONResponse({
            "status": "success",
            "query": request.query,
            "vendor": result["vendor"],
            "confidence": result["confidence"],
            "detection_method": result["method"],
            "alternatives": result.get("alternatives", [])
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================

@app.post("/api/v1/search", tags=["Search"])
async def search_vendor_docs(request: SearchRequest):
    """
    Semantic search across vendor documentation

    Args:
        request: Search request with query, optional vendor filter, and limit

    Returns:
        Search results with relevance scores
    """
    if embedding_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Generate query embedding
        query_embedding = embedding_pipeline.embedding_generator.generate_embedding(request.query)

        # Build filter conditions
        filter_conditions = None
        if request.vendor:
            filter_conditions = {
                "must": [{"key": "vendor", "match": {"value": request.vendor}}]
            }

        # Search
        results = embedding_pipeline.vector_store.search(
            query_vector=query_embedding,
            limit=request.limit,
            filter_conditions=filter_conditions
        )

        return JSONResponse({
            "status": "success",
            "query": request.query,
            "results_count": len(results),
            "results": [
                {
                    "score": r["score"],
                    "vendor": r["payload"].get("vendor"),
                    "doc_type": r["payload"].get("doc_type"),
                    "text": r["payload"].get("text", "")[:500],
                    "canonical_kpis": r["payload"].get("canonical_kpis", []),
                    "endpoint": r["payload"].get("path", "N/A"),
                    "method": r["payload"].get("method", "N/A")
                }
                for r in results
            ]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# CANONICAL KPI ENDPOINTS
# =============================================================================

@app.get("/api/v1/kpis", tags=["KPIs"])
async def list_canonical_kpis(
    category: Optional[str] = None,
    vendor: Optional[str] = None
):
    """
    List canonical O-RAN KPIs with 3GPP references

    Args:
        category: Filter by KPI category (optional)
        vendor: Show mappings for specific vendor (optional)

    Returns:
        List of canonical KPIs with metadata
    """
    kpis = CANONICAL_KPIS.get("canonical_kpis", {})

    # Filter by category
    if category:
        filtered_kpis = {
            name: kpi_def
            for name, kpi_def in kpis.items()
            if kpi_def.get("category") == category
        }
    else:
        filtered_kpis = kpis

    # Format response
    kpi_list = []
    for name, kpi_def in filtered_kpis.items():
        kpi_entry = {
            "canonical_name": name,
            "full_name": kpi_def.get("full_name"),
            "unit": kpi_def.get("unit"),
            "category": kpi_def.get("category"),
            "range": kpi_def.get("range"),
            "threegpp": kpi_def.get("threegpp", {}),
            "use_cases": kpi_def.get("use_cases", [])
        }

        # Add vendor-specific mapping if requested
        if vendor:
            vendor_mappings = kpi_def.get("vendor_mappings", {})
            kpi_entry["vendor_mapping"] = vendor_mappings.get(vendor) or vendor_mappings.get("generic")

        kpi_list.append(kpi_entry)

    # Get unique categories
    categories = list(set(kpi_def.get("category", "other") for kpi_def in kpis.values()))

    return {
        "total_kpis": len(kpi_list),
        "categories": categories,
        "kpis": kpi_list
    }

@app.get("/api/v1/kpis/{canonical_kpi}", tags=["KPIs"])
async def get_canonical_kpi_details(canonical_kpi: str):
    """
    Get detailed information about a specific canonical KPI

    Args:
        canonical_kpi: Canonical KPI name

    Returns:
        Full KPI definition with 3GPP references and vendor mappings
    """
    kpis = CANONICAL_KPIS.get("canonical_kpis", {})
    kpi_def = kpis.get(canonical_kpi)

    if not kpi_def:
        raise HTTPException(status_code=404, detail=f"KPI '{canonical_kpi}' not found")

    return {
        "canonical_name": canonical_kpi,
        "full_name": kpi_def.get("full_name"),
        "description": kpi_def.get("description"),
        "unit": kpi_def.get("unit"),
        "category": kpi_def.get("category"),
        "range": kpi_def.get("range"),
        "threegpp_reference": kpi_def.get("threegpp", {}),
        "vendor_mappings": kpi_def.get("vendor_mappings", {}),
        "use_cases": kpi_def.get("use_cases", [])
    }

# =============================================================================
# STATISTICS
# =============================================================================

@app.get("/api/v1/stats", tags=["Statistics"])
async def get_statistics():
    """Get RAG system statistics"""
    if embedding_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        stats = embedding_pipeline.vector_store.get_collection_stats()

        return {
            "collection": stats["name"],
            "total_vectors": stats["vectors_count"],
            "total_points": stats["points_count"],
            "status": stats["status"],
            "canonical_kpis_count": len(CANONICAL_KPIS.get("canonical_kpis", {}))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")

    print(f"🚀 Starting R1 Vendor Documentation API on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
