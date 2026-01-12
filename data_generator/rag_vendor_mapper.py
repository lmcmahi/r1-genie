#!/usr/bin/env python3
"""
RAG Vendor Mapper
Core RAG retrieval logic for mapping canonical KPIs to vendor-specific APIs
"""

import os
import json
from typing import Dict, List, Optional, Any
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Import dependencies
try:
    from embedding_pipeline import EmbeddingGenerator, QdrantVectorStore
    from vendor_detector import VendorDetector
except ImportError as e:
    print(f"⚠️  Import error: {e}")

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class RAGVendorMapper:
    """
    Map canonical KPIs to vendor-specific APIs using RAG retrieval

    Workflow:
    1. Build query from canonical KPI + 3GPP spec
    2. Generate query embedding
    3. Hybrid search in Qdrant (semantic + metadata filter)
    4. Calculate confidence score
    5. Validate result
    6. Fallback to YAML if confidence < threshold
    """

    def __init__(
        self,
        canonical_kpis: Optional[Dict] = None,
        confidence_threshold: float = None
    ):
        """
        Initialize RAG vendor mapper

        Args:
            canonical_kpis: Canonical KPI definitions (from r1_canonical_kpis.yaml)
            confidence_threshold: Minimum confidence for RAG results (default: from env)
        """
        self.canonical_kpis = canonical_kpis or {}
        self.confidence_threshold = confidence_threshold or float(
            os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.75")
        )
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))

        # Initialize components
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = QdrantVectorStore()

        # Optional LLM for validation
        self.anthropic_client = None
        if Anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = Anthropic(api_key=api_key)

        print(f"✅ RAG vendor mapper initialized (threshold: {self.confidence_threshold})")

    def map_to_vendor(
        self,
        canonical_kpi: str,
        vendor: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Map canonical KPI to vendor-specific API

        Args:
            canonical_kpi: Canonical KPI name (e.g., 'rsrp_dbm')
            vendor: Vendor name (e.g., 'viavi')
            context: Optional additional context

        Returns:
            {
                "status": "success"|"fallback"|"error",
                "vendor": str,
                "canonical_kpi": str,
                "confidence": float,
                "vendor_api": {
                    "endpoint": str,
                    "method": str,
                    "parameters": dict,
                    "description": str
                },
                "source": "rag"|"yaml_fallback"|"3gpp_generic",
                "validation": dict
            }
        """
        context = context or {}

        # Step 1: Build search query
        query = self._build_search_query(canonical_kpi, vendor, context)

        # Step 2: Get query embedding
        query_embedding = self.embedding_gen.generate_embedding(query)

        # Step 3: Search in Qdrant with metadata filters
        filter_conditions = self._build_filter_conditions(canonical_kpi, vendor)

        try:
            search_results = self.vector_store.search(
                query_vector=query_embedding,
                limit=self.top_k,
                filter_conditions=filter_conditions
            )
        except Exception as e:
            print(f"⚠️  Qdrant search failed: {e}")
            # Fallback to YAML
            return self._yaml_fallback(canonical_kpi, vendor)

        # Step 4: Process results
        if not search_results:
            print(f"📭 No RAG results found for {canonical_kpi} + {vendor}")
            return self._yaml_fallback(canonical_kpi, vendor)

        # Get best result
        best_result = search_results[0]
        vector_score = best_result["score"]

        # Step 5: Calculate comprehensive confidence score
        confidence = self._calculate_confidence(
            best_result,
            canonical_kpi,
            vendor
        )

        # Step 6: Validate result (prevent hallucinations)
        validation = self._validate_rag_result(best_result, canonical_kpi, vendor)

        if not validation["is_valid"]:
            print(f"⚠️  RAG validation failed: {validation['reason']}")
            return self._yaml_fallback(canonical_kpi, vendor)

        # Step 7: Check confidence threshold
        if confidence < self.confidence_threshold:
            print(f"📊 Confidence {confidence:.2f} below threshold {self.confidence_threshold}")
            return self._yaml_fallback(canonical_kpi, vendor)

        # Step 8: Extract vendor API details
        vendor_api = self._extract_vendor_api(best_result["payload"])

        return {
            "status": "success",
            "vendor": vendor,
            "canonical_kpi": canonical_kpi,
            "confidence": confidence,
            "vendor_api": vendor_api,
            "source": "rag",
            "validation": validation,
            "alternatives": search_results[1:3]  # Top 2 alternatives
        }

    def _build_search_query(
        self,
        canonical_kpi: str,
        vendor: str,
        context: Dict
    ) -> str:
        """Build semantic search query"""
        # Get 3GPP spec for canonical KPI
        threegpp_spec = self._get_3gpp_spec(canonical_kpi)

        # Build comprehensive query
        query_parts = [
            f"{canonical_kpi} API endpoint for {vendor}",
            f"3GPP specification: {threegpp_spec}" if threegpp_spec else "",
            f"Full name: {self._get_kpi_full_name(canonical_kpi)}",
        ]

        # Add context-specific details
        if "operation" in context:
            query_parts.append(f"Operation: {context['operation']}")

        query = "\n".join([p for p in query_parts if p])
        return query

    def _build_filter_conditions(
        self,
        canonical_kpi: str,
        vendor: str
    ) -> Optional[Dict]:
        """Build Qdrant metadata filters"""
        # Build filter for vendor and KPI
        return {
            "must": [
                {"key": "vendor", "match": {"value": vendor}},
                # Also filter by KPI if available in metadata
                # Note: canonical_kpis is a list in metadata
            ]
        }

    def _calculate_confidence(
        self,
        search_result: Dict,
        canonical_kpi: str,
        vendor: str
    ) -> float:
        """
        Calculate comprehensive confidence score

        Components:
        - Vector similarity score (60%)
        - Metadata match quality (30%)
        - Document recency (10%)
        """
        # 1. Vector similarity (0.0-1.0)
        vector_score = search_result["score"]

        # 2. Metadata match quality
        payload = search_result["payload"]

        metadata_score = 0.0
        checks = 0

        # Check vendor match
        if payload.get("vendor") == vendor:
            metadata_score += 1.0
            checks += 1

        # Check KPI match
        payload_kpis = payload.get("canonical_kpis", [])
        if canonical_kpi in payload_kpis:
            metadata_score += 1.0
            checks += 1

        # Check doc type (prefer openapi over pdf/markdown)
        doc_type = payload.get("doc_type", "")
        if doc_type == "openapi":
            metadata_score += 0.8
        elif doc_type == "markdown":
            metadata_score += 0.5
        elif doc_type == "pdf":
            metadata_score += 0.3
        checks += 1

        metadata_score = metadata_score / checks if checks > 0 else 0.0

        # 3. Document recency (simplified - could check upload date)
        recency_score = 0.7  # Placeholder

        # Weighted combination
        confidence = (
            vector_score * 0.6 +
            metadata_score * 0.3 +
            recency_score * 0.1
        )

        return confidence

    def _validate_rag_result(
        self,
        search_result: Dict,
        canonical_kpi: str,
        vendor: str
    ) -> Dict[str, Any]:
        """
        Validate RAG result to prevent hallucinations

        Checks:
        1. Endpoint path sanity
        2. HTTP method sanity
        3. Vendor match
        4. Optional: LLM-based semantic validation
        """
        payload = search_result["payload"]

        # Check 1: Vendor match
        if payload.get("vendor") != vendor:
            return {
                "is_valid": False,
                "reason": f"Vendor mismatch: expected {vendor}, got {payload.get('vendor')}"
            }

        # Check 2: Endpoint path sanity (for OpenAPI results)
        if payload.get("doc_type") == "openapi":
            path = payload.get("path", "")
            if not path.startswith("/"):
                return {
                    "is_valid": False,
                    "reason": f"Invalid endpoint path: {path}"
                }

            # Check method sanity
            method = payload.get("method", "").upper()
            if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                return {
                    "is_valid": False,
                    "reason": f"Invalid HTTP method: {method}"
                }

        # Check 3: KPI relevance (optional LLM validation)
        if self.anthropic_client and search_result["score"] < 0.85:
            llm_validation = self._llm_validate(payload, canonical_kpi, vendor)
            if not llm_validation["is_valid"]:
                return llm_validation

        return {
            "is_valid": True,
            "reason": "All validation checks passed"
        }

    def _llm_validate(
        self,
        payload: Dict,
        canonical_kpi: str,
        vendor: str
    ) -> Dict[str, Any]:
        """Use LLM to validate mapping correctness"""
        if not self.anthropic_client:
            return {"is_valid": True, "reason": "LLM validation skipped"}

        prompt = f"""Validate if this vendor API correctly maps to the canonical KPI.

Canonical KPI: {canonical_kpi}
3GPP Spec: {self._get_3gpp_spec(canonical_kpi)}
Full Name: {self._get_kpi_full_name(canonical_kpi)}

Vendor: {vendor}
Vendor API Endpoint: {payload.get('path', 'N/A')}
API Method: {payload.get('method', 'N/A')}
Description: {payload.get('text', '')[:300]}

Is this mapping correct? Respond with only "YES" or "NO" followed by a brief reason."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()

            if result_text.startswith("NO"):
                return {
                    "is_valid": False,
                    "reason": f"LLM validation failed: {result_text}"
                }

            return {
                "is_valid": True,
                "reason": "LLM validation passed"
            }

        except Exception as e:
            print(f"⚠️  LLM validation error: {e}")
            return {"is_valid": True, "reason": "LLM validation error (defaulting to valid)"}

    def _extract_vendor_api(self, payload: Dict) -> Dict[str, Any]:
        """Extract vendor API details from payload"""
        if payload.get("doc_type") == "openapi":
            return {
                "endpoint": payload.get("path", ""),
                "method": payload.get("method", "GET").upper(),
                "parameters": payload.get("parameters", []),
                "request_body": payload.get("request_body"),
                "responses": payload.get("responses"),
                "description": payload.get("summary", ""),
                "operation_id": payload.get("operation_id", "")
            }
        else:
            # PDF/Markdown - extract from text
            return {
                "endpoint": self._extract_endpoint_from_text(payload.get("text", "")),
                "method": "GET",  # Default assumption
                "parameters": [],
                "description": payload.get("text", "")[:200],
                "source_doc": payload.get("source_file", "")
            }

    def _extract_endpoint_from_text(self, text: str) -> str:
        """Extract API endpoint from text using regex"""
        import re

        # Common API endpoint patterns
        patterns = [
            r'/api/[^\s]+',
            r'/v\d+/[^\s]+',
            r'https?://[^\s]+/api/[^\s]+'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return ""

    def _get_3gpp_spec(self, canonical_kpi: str) -> str:
        """Get 3GPP specification for canonical KPI"""
        if not self.canonical_kpis:
            return ""

        kpi_def = self.canonical_kpis.get("canonical_kpis", {}).get(canonical_kpi, {})
        threegpp = kpi_def.get("threegpp", {})

        spec = threegpp.get("spec", "")
        section = threegpp.get("section", "")
        measurement = threegpp.get("measurement_name", "")

        if spec and section and measurement:
            return f"{spec} Section {section} - {measurement}"
        elif spec:
            return spec
        else:
            return ""

    def _get_kpi_full_name(self, canonical_kpi: str) -> str:
        """Get full name of canonical KPI"""
        if not self.canonical_kpis:
            return canonical_kpi

        kpi_def = self.canonical_kpis.get("canonical_kpis", {}).get(canonical_kpi, {})
        return kpi_def.get("full_name", canonical_kpi)

    def _yaml_fallback(self, canonical_kpi: str, vendor: str) -> Dict[str, Any]:
        """
        Fallback to static YAML mappings

        Args:
            canonical_kpi: Canonical KPI name
            vendor: Vendor name

        Returns:
            Mapping result with source='yaml_fallback'
        """
        if not self.canonical_kpis:
            return self._generic_3gpp_fallback(canonical_kpi)

        kpi_def = self.canonical_kpis.get("canonical_kpis", {}).get(canonical_kpi, {})

        if not kpi_def:
            return self._generic_3gpp_fallback(canonical_kpi)

        vendor_mappings = kpi_def.get("vendor_mappings", {})
        vendor_name = vendor_mappings.get(vendor) or vendor_mappings.get("generic")

        if not vendor_name:
            return self._generic_3gpp_fallback(canonical_kpi)

        return {
            "status": "fallback",
            "vendor": vendor,
            "canonical_kpi": canonical_kpi,
            "confidence": 0.60,  # Medium confidence for YAML
            "vendor_api": {
                "endpoint": f"/api/v1/measurements/{vendor_name.lower().replace('.', '/')}",
                "method": "GET",
                "parameters": [],
                "description": f"{vendor} mapping from YAML",
                "vendor_param_name": vendor_name
            },
            "source": "yaml_fallback",
            "validation": {"is_valid": True, "reason": "Static YAML mapping"}
        }

    def _generic_3gpp_fallback(self, canonical_kpi: str) -> Dict[str, Any]:
        """Final fallback to generic 3GPP name"""
        if not self.canonical_kpis:
            generic_name = canonical_kpi.upper().replace('_', '-')
        else:
            kpi_def = self.canonical_kpis.get("canonical_kpis", {}).get(canonical_kpi, {})
            threegpp = kpi_def.get("threegpp", {})
            generic_name = threegpp.get("measurement_name", canonical_kpi)

        return {
            "status": "fallback",
            "vendor": "generic",
            "canonical_kpi": canonical_kpi,
            "confidence": 0.40,
            "vendor_api": {
                "endpoint": f"/api/v1/measurements/{generic_name.lower().replace('-', '/')}",
                "method": "GET",
                "parameters": [],
                "description": f"Generic 3GPP mapping for {generic_name}",
                "threegpp_name": generic_name
            },
            "source": "3gpp_generic",
            "validation": {
                "is_valid": True,
                "reason": "Generic 3GPP fallback (vendor-specific mapping not found)"
            }
        }

    @lru_cache(maxsize=1000)
    def map_to_vendor_cached(
        self,
        canonical_kpi: str,
        vendor: str,
        context_str: str = ""
    ) -> Dict[str, Any]:
        """Cached version of map_to_vendor for performance"""
        context = json.loads(context_str) if context_str else {}
        return self.map_to_vendor(canonical_kpi, vendor, context)


# Example usage
if __name__ == "__main__":
    import sys

    # Load canonical KPIs
    try:
        from r1_dataset_builder_v3 import CANONICAL_KPIS
    except:
        print("⚠️  Could not load CANONICAL_KPIS. Using empty dict.")
        CANONICAL_KPIS = {}

    # Initialize RAG mapper
    mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)

    # Test mappings
    test_cases = [
        ("rsrp_dbm", "viavi"),
        ("throughput_dl_mbps", "ericsson"),
        ("prb_usage_dl", "nokia"),
        ("latency_dl_ms", "unknown_vendor")  # Should fallback
    ]

    print("\n" + "=" * 80)
    print("RAG VENDOR MAPPING TESTS")
    print("=" * 80)

    for canonical_kpi, vendor in test_cases:
        print(f"\n📊 Mapping: {canonical_kpi} → {vendor}")

        result = mapper.map_to_vendor(canonical_kpi, vendor)

        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Source: {result['source']}")
        print(f"   Endpoint: {result['vendor_api'].get('endpoint', 'N/A')}")
        print(f"   Method: {result['vendor_api'].get('method', 'N/A')}")

        if result['source'] == 'yaml_fallback':
            print(f"   Vendor Param: {result['vendor_api'].get('vendor_param_name', 'N/A')}")
