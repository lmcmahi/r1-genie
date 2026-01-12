#!/usr/bin/env python3
"""
MCP RAG Server for R1-Genie
Enables Claude Desktop to interact with vendor documentation RAG system
"""

import os
import sys
import json
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

# Import MCP SDK components
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("❌ MCP SDK not installed. Install with: pip install mcp")
    print("   Or: uv pip install mcp")
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

# Initialize MCP server
server = Server("r1-vendor-mapping")

# Global RAG components
embedding_pipeline = None
rag_mapper = None
vendor_detector = None

def init_rag_components():
    """Initialize RAG components"""
    global embedding_pipeline, rag_mapper, vendor_detector

    try:
        embedding_pipeline = EmbeddingPipeline()
        rag_mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)
        vendor_detector = VendorDetector()
        return True
    except Exception as e:
        print(f"⚠️  RAG initialization failed: {e}", file=sys.stderr)
        return False

# =============================================================================
# MCP TOOLS
# =============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="upload_vendor_docs",
            description="Upload vendor API documentation for RAG indexing. Supports OpenAPI/Swagger (YAML/JSON), PDF manuals, and Markdown docs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor": {
                        "type": "string",
                        "description": "Vendor name (e.g., 'viavi', 'ericsson', 'nokia')"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to documentation file"
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["auto", "openapi", "pdf", "markdown"],
                        "description": "Document type (auto-detect if not specified)"
                    },
                    "create_collection": {
                        "type": "boolean",
                        "description": "Create Qdrant collection if it doesn't exist"
                    }
                },
                "required": ["vendor", "file_path"]
            }
        ),
        Tool(
            name="map_canonical_to_vendor",
            description="Map a canonical O-RAN KPI name to vendor-specific API endpoint using RAG retrieval with YAML fallback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical_kpi": {
                        "type": "string",
                        "description": "Canonical KPI name (e.g., 'rsrp_dbm', 'throughput_dl_mbps')"
                    },
                    "vendor": {
                        "type": "string",
                        "description": "Vendor name (e.g., 'viavi', 'ericsson')"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context (cellId, network info, etc.)"
                    }
                },
                "required": ["canonical_kpi", "vendor"]
            }
        ),
        Tool(
            name="detect_vendor",
            description="Auto-detect vendor from natural language query using 3-tier detection (explicit mention, network context, default).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query from user"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional network context (cellId, rapp_id, etc.)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="query_vendor_docs",
            description="Semantic search across vendor documentation in Qdrant vector database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "vendor": {
                        "type": "string",
                        "description": "Filter by specific vendor (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_vendors",
            description="List all vendors with indexed documentation in the RAG system.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_canonical_kpis",
            description="Get list of canonical O-RAN KPI names with 3GPP references and vendor mappings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by KPI category (optional)"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls"""

    # Initialize RAG components if not already done
    if embedding_pipeline is None:
        if not init_rag_components():
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "RAG system initialization failed. Check Qdrant connection."
                }, indent=2)
            )]

    try:
        if name == "upload_vendor_docs":
            result = await upload_vendor_docs(arguments)
        elif name == "map_canonical_to_vendor":
            result = await map_canonical_to_vendor(arguments)
        elif name == "detect_vendor":
            result = await detect_vendor(arguments)
        elif name == "query_vendor_docs":
            result = await query_vendor_docs(arguments)
        elif name == "list_vendors":
            result = await list_vendors(arguments)
        elif name == "get_canonical_kpis":
            result = await get_canonical_kpis(arguments)
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": str(e),
                "tool": name
            }, indent=2)
        )]

# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

async def upload_vendor_docs(args: Dict) -> Dict:
    """Upload vendor documentation to RAG system"""
    vendor = args["vendor"]
    file_path = args["file_path"]
    doc_type = args.get("doc_type", "auto")
    create_collection = args.get("create_collection", False)

    # Check file exists
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "message": f"File not found: {file_path}"
        }

    # Parse document
    parser_factory = DocumentParserFactory(CANONICAL_KPIS)

    if doc_type == "auto":
        chunks = parser_factory.parse(file_path, vendor)
    else:
        parser = parser_factory.get_parser(file_path)
        chunks = parser.parse(file_path, vendor)

    # Process through embedding pipeline
    result = embedding_pipeline.process_document(
        chunks,
        create_collection=create_collection,
        force_recreate=False
    )

    return {
        "status": "success",
        "vendor": vendor,
        "file": os.path.basename(file_path),
        "doc_type": doc_type,
        "chunks_processed": result["chunks_processed"],
        "embeddings_generated": result["embeddings_generated"],
        "points_uploaded": result["points_uploaded"],
        "collection_stats": result["collection_stats"]
    }

async def map_canonical_to_vendor(args: Dict) -> Dict:
    """Map canonical KPI to vendor-specific API"""
    canonical_kpi = args["canonical_kpi"]
    vendor = args["vendor"]
    context = args.get("context", {})

    result = rag_mapper.map_to_vendor(
        canonical_kpi=canonical_kpi,
        vendor=vendor,
        context=context
    )

    return {
        "status": result["status"],
        "canonical_kpi": canonical_kpi,
        "vendor": vendor,
        "confidence": result["confidence"],
        "vendor_api": result["vendor_api"],
        "source": result["source"],
        "validation": result.get("validation", {}),
        "alternatives": result.get("alternatives", [])[:2]  # Top 2 alternatives
    }

async def detect_vendor(args: Dict) -> Dict:
    """Detect vendor from query"""
    query = args["query"]
    context = args.get("context", {})

    result = vendor_detector.detect(query, context)

    return {
        "status": "success",
        "query": query,
        "vendor": result["vendor"],
        "confidence": result["confidence"],
        "detection_method": result["method"],
        "alternatives": result.get("alternatives", [])
    }

async def query_vendor_docs(args: Dict) -> Dict:
    """Search vendor documentation"""
    query = args["query"]
    vendor_filter = args.get("vendor")
    limit = args.get("limit", 5)

    # Generate query embedding
    query_embedding = embedding_pipeline.embedding_generator.generate_embedding(query)

    # Build filter conditions
    filter_conditions = None
    if vendor_filter:
        filter_conditions = {
            "must": [{"key": "vendor", "match": {"value": vendor_filter}}]
        }

    # Search
    results = embedding_pipeline.vector_store.search(
        query_vector=query_embedding,
        limit=limit,
        filter_conditions=filter_conditions
    )

    return {
        "status": "success",
        "query": query,
        "results_count": len(results),
        "results": [
            {
                "score": r["score"],
                "vendor": r["payload"].get("vendor"),
                "doc_type": r["payload"].get("doc_type"),
                "text": r["payload"].get("text", "")[:300],  # Truncate for display
                "canonical_kpis": r["payload"].get("canonical_kpis", []),
                "endpoint": r["payload"].get("path", "N/A")
            }
            for r in results
        ]
    }

async def list_vendors(args: Dict) -> Dict:
    """List all indexed vendors"""
    try:
        # Query Qdrant for unique vendors
        # Note: This is a simplified implementation
        # In production, you'd scroll through all points

        stats = embedding_pipeline.vector_store.get_collection_stats()

        return {
            "status": "success",
            "message": "Vendor listing requires scrolling through collection",
            "collection_stats": stats,
            "known_vendors": ["viavi", "ericsson", "nokia", "huawei", "generic"],
            "note": "Use query_vendor_docs with vendor filter to check if vendor is indexed"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def get_canonical_kpis(args: Dict) -> Dict:
    """Get canonical KPI information"""
    category_filter = args.get("category")

    kpis = CANONICAL_KPIS.get("canonical_kpis", {})

    if category_filter:
        filtered_kpis = {
            name: kpi_def
            for name, kpi_def in kpis.items()
            if kpi_def.get("category") == category_filter
        }
    else:
        filtered_kpis = kpis

    # Format for display
    kpi_list = []
    for name, kpi_def in filtered_kpis.items():
        kpi_list.append({
            "canonical_name": name,
            "full_name": kpi_def.get("full_name"),
            "unit": kpi_def.get("unit"),
            "category": kpi_def.get("category"),
            "threegpp_spec": kpi_def.get("threegpp", {}).get("spec"),
            "vendor_mappings": list(kpi_def.get("vendor_mappings", {}).keys())
        })

    # Get unique categories
    categories = list(set(kpi_def.get("category", "other") for kpi_def in kpis.values()))

    return {
        "status": "success",
        "total_kpis": len(kpi_list),
        "categories": categories,
        "kpis": kpi_list
    }

# =============================================================================
# MAIN SERVER
# =============================================================================

async def main():
    """Run MCP server"""
    print("🚀 R1 Vendor Mapping MCP Server starting...", file=sys.stderr)

    # Initialize RAG components
    if init_rag_components():
        print("✅ RAG system initialized", file=sys.stderr)
    else:
        print("⚠️  RAG system initialization failed - tools may not work", file=sys.stderr)

    print("📡 MCP server ready for connections", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
