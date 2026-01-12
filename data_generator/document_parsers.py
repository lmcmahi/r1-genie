#!/usr/bin/env python3
"""
Document Parsers for RAG System
Handles OpenAPI/Swagger, PDF, and Markdown vendor documentation
"""

import os
import re
import yaml
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import validation and parsing libraries
try:
    from openapi_spec_validator import validate_spec
    from openapi_spec_validator.readers import read_from_filename
except ImportError:
    print("⚠️  openapi-spec-validator not installed. OpenAPI parsing disabled.")
    validate_spec = None

try:
    from llama_parse import LlamaParse
except ImportError:
    print("⚠️  llama-parse not installed. PDF parsing disabled.")
    LlamaParse = None

try:
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownNodeParser
except ImportError:
    print("⚠️  llama-index not installed. Markdown parsing disabled.")
    Document = None
    MarkdownNodeParser = None


class DocumentChunk:
    """Represents a parsed document chunk with metadata"""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata
        }


class OpenAPIParser:
    """Parse OpenAPI/Swagger specifications"""

    def __init__(self, canonical_kpis: Optional[Dict] = None):
        self.canonical_kpis = canonical_kpis or {}

    def parse(self, file_path: str, vendor: str) -> List[DocumentChunk]:
        """
        Parse OpenAPI spec and extract API operations

        Args:
            file_path: Path to OpenAPI YAML/JSON file
            vendor: Vendor name (e.g., 'viavi', 'ericsson')

        Returns:
            List of DocumentChunk objects
        """
        if validate_spec is None:
            raise ImportError("openapi-spec-validator is required for OpenAPI parsing")

        chunks = []

        # Read and validate OpenAPI spec
        try:
            spec_dict, spec_url = read_from_filename(file_path)
            validate_spec(spec_dict)
        except Exception as e:
            print(f"❌ OpenAPI validation failed: {e}")
            # Try to parse anyway
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    spec_dict = json.load(f)
                else:
                    spec_dict = yaml.safe_load(f)

        # Extract API version
        api_version = spec_dict.get('info', {}).get('version', 'unknown')

        # Parse each path/operation
        for path, methods in spec_dict.get('paths', {}).items():
            for method, operation in methods.items():
                if method.upper() not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    continue  # Skip non-HTTP methods like 'parameters'

                # Extract operation details
                summary = operation.get('summary', '')
                description = operation.get('description', '')
                operation_id = operation.get('operationId', '')

                # Extract parameters
                parameters = []
                for param in operation.get('parameters', []):
                    parameters.append({
                        "name": param.get('name'),
                        "in": param.get('in'),
                        "required": param.get('required', False),
                        "description": param.get('description', '')
                    })

                # Extract request body schema
                request_body = None
                if 'requestBody' in operation:
                    content = operation['requestBody'].get('content', {})
                    if 'application/json' in content:
                        request_body = content['application/json'].get('schema')

                # Extract response schema
                responses = {}
                for status_code, response in operation.get('responses', {}).items():
                    content = response.get('content', {})
                    if 'application/json' in content:
                        responses[status_code] = content['application/json'].get('schema')

                # Detect canonical KPIs in operation
                canonical_kpis = self._extract_kpis_from_operation(
                    summary, description, parameters, request_body
                )

                # Build comprehensive text representation
                text_parts = [
                    f"{method.upper()} {path}",
                    f"Summary: {summary}" if summary else "",
                    f"Description: {description}" if description else "",
                    f"Operation ID: {operation_id}" if operation_id else "",
                    f"Parameters: {json.dumps(parameters)}" if parameters else "",
                    f"Canonical KPIs: {', '.join(canonical_kpis)}" if canonical_kpis else ""
                ]
                text = "\n".join([p for p in text_parts if p])

                # Create chunk with rich metadata
                chunk = DocumentChunk(
                    text=text,
                    metadata={
                        "vendor": vendor,
                        "doc_type": "openapi",
                        "api_version": api_version,
                        "path": path,
                        "method": method.upper(),
                        "operation_id": operation_id,
                        "summary": summary,
                        "canonical_kpis": canonical_kpis,
                        "parameters": parameters,
                        "request_body": request_body,
                        "responses": responses,
                        "source_file": Path(file_path).name
                    }
                )
                chunks.append(chunk)

        print(f"✅ Parsed {len(chunks)} API operations from {Path(file_path).name}")
        return chunks

    def _extract_kpis_from_operation(
        self,
        summary: str,
        description: str,
        parameters: List[Dict],
        request_body: Optional[Dict]
    ) -> List[str]:
        """Extract canonical KPI names from operation details"""
        kpis = []

        # Search in text content
        search_text = f"{summary} {description}".lower()

        # Load canonical KPI names if available
        if self.canonical_kpis:
            canonical_names = self.canonical_kpis.get("canonical_kpis", {}).keys()
            for kpi_name in canonical_names:
                # Check for exact matches or similar patterns
                if kpi_name in search_text or kpi_name.replace('_', ' ') in search_text:
                    kpis.append(kpi_name)

        # Common KPI patterns
        kpi_patterns = [
            r'\b(rsrp|rsrq|sinr|throughput|prb|latency|handover)\b',
            r'\bmeasurement[s]?\b',
            r'\bkpi[s]?\b',
            r'\bmetric[s]?\b'
        ]

        for pattern in kpi_patterns:
            if re.search(pattern, search_text, re.IGNORECASE):
                # Try to map to canonical names
                if 'rsrp' in search_text and 'rsrp_dbm' not in kpis:
                    kpis.append('rsrp_dbm')
                if 'throughput' in search_text and 'throughput_dl_mbps' not in kpis:
                    kpis.append('throughput_dl_mbps')
                if 'prb' in search_text and 'prb_usage_dl' not in kpis:
                    kpis.append('prb_usage_dl')

        return list(set(kpis))  # Remove duplicates


class PDFParser:
    """Parse PDF vendor documentation using LlamaParse"""

    def __init__(self, canonical_kpis: Optional[Dict] = None):
        self.canonical_kpis = canonical_kpis or {}
        self.llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

        if not self.llama_api_key:
            print("⚠️  LLAMA_CLOUD_API_KEY not set. PDF parsing will fail.")

    def parse(self, file_path: str, vendor: str) -> List[DocumentChunk]:
        """
        Parse PDF using LlamaParse

        Args:
            file_path: Path to PDF file
            vendor: Vendor name

        Returns:
            List of DocumentChunk objects
        """
        if LlamaParse is None:
            raise ImportError("llama-parse is required for PDF parsing")

        if not self.llama_api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable not set")

        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=self.llama_api_key,
            result_type="markdown",  # Convert tables to markdown
            verbose=False
        )

        # Parse PDF
        print(f"📄 Parsing PDF: {Path(file_path).name} (this may take a minute)...")
        documents = parser.load_data(file_path)

        # Chunk by semantic sections using MarkdownNodeParser
        if MarkdownNodeParser:
            node_parser = MarkdownNodeParser()
            nodes = node_parser.get_nodes_from_documents(documents)
        else:
            # Fallback: simple chunking
            nodes = self._simple_chunk(documents)

        # Convert to DocumentChunks
        chunks = []
        for idx, node in enumerate(nodes):
            # Extract KPIs from text
            canonical_kpis = self._extract_kpis_from_text(node.text)

            chunk = DocumentChunk(
                text=node.text,
                metadata={
                    "vendor": vendor,
                    "doc_type": "pdf",
                    "page": node.metadata.get("page_label", idx),
                    "canonical_kpis": canonical_kpis,
                    "source_file": Path(file_path).name,
                    "chunk_index": idx
                }
            )
            chunks.append(chunk)

        print(f"✅ Parsed {len(chunks)} chunks from {Path(file_path).name}")
        return chunks

    def _extract_kpis_from_text(self, text: str) -> List[str]:
        """Extract canonical KPI names from text"""
        kpis = []
        text_lower = text.lower()

        # Load canonical KPI names
        if self.canonical_kpis:
            canonical_names = self.canonical_kpis.get("canonical_kpis", {}).keys()
            for kpi_name in canonical_names:
                if kpi_name in text_lower or kpi_name.replace('_', ' ') in text_lower:
                    kpis.append(kpi_name)

        # KPI keyword patterns
        kpi_keywords = {
            'rsrp': 'rsrp_dbm',
            'reference signal received power': 'rsrp_dbm',
            'rsrq': 'rsrq_db',
            'sinr': 'sinr_db',
            'signal to interference': 'sinr_db',
            'throughput downlink': 'throughput_dl_mbps',
            'throughput uplink': 'throughput_ul_mbps',
            'prb utilization': 'prb_utilization_dl_pct',
            'physical resource block': 'prb_usage_dl',
            'handover success rate': 'handover_success_rate_pct',
            'latency': 'latency_dl_ms'
        }

        for keyword, canonical_name in kpi_keywords.items():
            if keyword in text_lower and canonical_name not in kpis:
                kpis.append(canonical_name)

        return list(set(kpis))

    def _simple_chunk(self, documents: List, chunk_size: int = 512) -> List:
        """Simple text chunking fallback"""
        chunks = []
        for doc in documents:
            text = doc.text
            words = text.split()

            for i in range(0, len(words), chunk_size):
                chunk_text = ' '.join(words[i:i + chunk_size])
                chunks.append(type('Node', (), {
                    'text': chunk_text,
                    'metadata': doc.metadata
                })())

        return chunks


class MarkdownParser:
    """Parse Markdown vendor documentation"""

    def __init__(self, canonical_kpis: Optional[Dict] = None):
        self.canonical_kpis = canonical_kpis or {}

    def parse(self, file_path: str, vendor: str) -> List[DocumentChunk]:
        """
        Parse Markdown documentation

        Args:
            file_path: Path to Markdown file
            vendor: Vendor name

        Returns:
            List of DocumentChunk objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if Document and MarkdownNodeParser:
            # Use LlamaIndex for hierarchical parsing
            doc = Document(text=content, metadata={"source": file_path})
            parser = MarkdownNodeParser()
            nodes = parser.get_nodes_from_documents([doc])
        else:
            # Fallback: split by headers
            nodes = self._simple_markdown_split(content)

        # Convert to DocumentChunks
        chunks = []
        for idx, node in enumerate(nodes):
            canonical_kpis = self._extract_kpis_from_text(node.text)

            chunk = DocumentChunk(
                text=node.text,
                metadata={
                    "vendor": vendor,
                    "doc_type": "markdown",
                    "canonical_kpis": canonical_kpis,
                    "source_file": Path(file_path).name,
                    "chunk_index": idx,
                    "heading": self._extract_heading(node.text)
                }
            )
            chunks.append(chunk)

        print(f"✅ Parsed {len(chunks)} sections from {Path(file_path).name}")
        return chunks

    def _extract_kpis_from_text(self, text: str) -> List[str]:
        """Extract canonical KPI names from text (same logic as PDF parser)"""
        kpis = []
        text_lower = text.lower()

        if self.canonical_kpis:
            canonical_names = self.canonical_kpis.get("canonical_kpis", {}).keys()
            for kpi_name in canonical_names:
                if kpi_name in text_lower or kpi_name.replace('_', ' ') in text_lower:
                    kpis.append(kpi_name)

        return list(set(kpis))

    def _extract_heading(self, text: str) -> str:
        """Extract the first heading from markdown text"""
        match = re.match(r'^(#+)\s+(.+)$', text, re.MULTILINE)
        return match.group(2) if match else ""

    def _simple_markdown_split(self, content: str) -> List:
        """Simple markdown splitting by headers"""
        sections = re.split(r'\n(?=#+\s)', content)
        return [type('Node', (), {'text': section, 'metadata': {}})() for section in sections if section.strip()]


class DocumentParserFactory:
    """Factory for creating appropriate document parser"""

    def __init__(self, canonical_kpis: Optional[Dict] = None):
        self.canonical_kpis = canonical_kpis

    def get_parser(self, file_path: str):
        """Get appropriate parser based on file extension"""
        ext = Path(file_path).suffix.lower()

        if ext in ['.yaml', '.yml', '.json']:
            return OpenAPIParser(self.canonical_kpis)
        elif ext == '.pdf':
            return PDFParser(self.canonical_kpis)
        elif ext in ['.md', '.markdown']:
            return MarkdownParser(self.canonical_kpis)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def parse(self, file_path: str, vendor: str) -> List[DocumentChunk]:
        """Parse document using appropriate parser"""
        parser = self.get_parser(file_path)
        return parser.parse(file_path, vendor)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python document_parsers.py <file_path> <vendor>")
        print("Example: python document_parsers.py viavi_api.yaml viavi")
        sys.exit(1)

    file_path = sys.argv[1]
    vendor = sys.argv[2]

    # Load canonical KPIs
    try:
        from r1_dataset_builder_v3 import CANONICAL_KPIS
    except:
        CANONICAL_KPIS = {}

    factory = DocumentParserFactory(CANONICAL_KPIS)
    chunks = factory.parse(file_path, vendor)

    print(f"\n📊 Parsing Summary:")
    print(f"   File: {file_path}")
    print(f"   Vendor: {vendor}")
    print(f"   Chunks: {len(chunks)}")

    # Show first chunk
    if chunks:
        print(f"\n📝 First Chunk:")
        print(f"   Text: {chunks[0].text[:200]}...")
        print(f"   Metadata: {json.dumps(chunks[0].metadata, indent=2)}")
