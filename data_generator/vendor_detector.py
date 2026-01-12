#!/usr/bin/env python3
"""
Vendor Auto-Detection System
Implements 3-tier detection strategy for vendor identification
"""

import os
import re
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

# Import optional dependencies
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class VendorDetector:
    """
    Auto-detect vendor from user queries and context

    Detection Strategy:
    1. Tier 1: Explicit mention in query (pattern matching)
    2. Tier 2: Network context (cell ID, rApp registration, etc.)
    3. Tier 3: Default vendor configuration
    """

    def __init__(self, anthropic_api_key: str = None):
        """
        Initialize vendor detector

        Args:
            anthropic_api_key: Anthropic API key for LLM-based extraction
        """
        self.default_vendor = os.getenv("DEFAULT_VENDOR", "generic")
        self.min_confidence = float(os.getenv("VENDOR_DETECTION_CONFIDENCE_MIN", "0.60"))

        # Initialize Anthropic client for ambiguous cases
        self.anthropic_client = None
        if Anthropic:
            api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = Anthropic(api_key=api_key)

        # Vendor name patterns
        self.vendor_patterns = {
            # Viavi / Spirent
            r'\b(viavi|spirit[ao]|spirit)\b': 'viavi',

            # Ericsson
            r'\b(ericsson|erics+on)\b': 'ericsson',

            # Nokia
            r'\b(nokia|nsn|nokia\s+siemens)\b': 'nokia',

            # Huawei
            r'\b(huawei|hw)\b': 'huawei',

            # Samsung
            r'\b(samsung)\b': 'samsung',

            # ZTE
            r'\b(zte)\b': 'zte',

            # Mavenir
            r'\b(mavenir)\b': 'mavenir',

            # Parallel Wireless
            r'\b(parallel\s+wireless|pwireless)\b': 'parallel_wireless',

            # Generic/Standard
            r'\b(o-?ran|generic|standard)\b': 'generic'
        }

        # Network context mappings (can be loaded from external config)
        self.cell_vendor_map = {}  # cell_id -> vendor
        self.slice_vendor_map = {}  # slice_id -> List[vendor]
        self.rapp_vendor_map = {}  # rapp_id -> vendor

        print(f"✅ Vendor detector initialized (default: {self.default_vendor})")

    def detect(
        self,
        user_query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Detect vendor from query and context

        Args:
            user_query: Natural language query from user
            context: Optional context dict with network information

        Returns:
            {
                "vendor": str,
                "confidence": float,
                "method": str,  # explicit|context|llm|default
                "alternatives": List[Dict]  # Other possible vendors
            }
        """
        context = context or {}

        # Tier 1: Explicit mention
        explicit_result = self._detect_explicit_vendor(user_query)
        if explicit_result["confidence"] >= 0.90:
            return explicit_result

        # Tier 2: Network context
        context_result = self._detect_from_context(context)
        if context_result["confidence"] >= 0.75:
            return context_result

        # Tier 2.5: LLM-based extraction (for ambiguous cases)
        if self.anthropic_client and not explicit_result["vendor"]:
            llm_result = self._llm_vendor_extraction(user_query)
            if llm_result["confidence"] >= 0.70:
                return llm_result

        # Tier 3: Use explicit result if found (even with lower confidence)
        if explicit_result["vendor"]:
            return explicit_result

        # Tier 3: Default vendor
        return {
            "vendor": self.default_vendor,
            "confidence": 0.50,
            "method": "default",
            "alternatives": []
        }

    def _detect_explicit_vendor(self, query: str) -> Dict:
        """
        Detect vendor from explicit mentions in query

        Args:
            query: User query text

        Returns:
            Detection result with vendor and confidence
        """
        query_lower = query.lower()

        # Check all patterns
        matches = []
        for pattern, vendor in self.vendor_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                matches.append({
                    "vendor": vendor,
                    "pattern": pattern,
                    "confidence": 0.95
                })

        if not matches:
            return {
                "vendor": None,
                "confidence": 0.0,
                "method": "explicit",
                "alternatives": []
            }

        # Return best match (first match if multiple)
        best_match = matches[0]

        # If multiple vendors mentioned, reduce confidence
        if len(matches) > 1:
            alternatives = matches[1:]
            confidence = 0.70  # Lower confidence when ambiguous
        else:
            alternatives = []
            confidence = 0.95

        return {
            "vendor": best_match["vendor"],
            "confidence": confidence,
            "method": "explicit",
            "alternatives": alternatives
        }

    def _detect_from_context(self, context: Dict) -> Dict:
        """
        Detect vendor from network context

        Args:
            context: Context dict with network information

        Returns:
            Detection result
        """
        # Strategy 1: Cell ID to vendor mapping
        if "cellId" in context or "cell_id" in context:
            cell_id = context.get("cellId") or context.get("cell_id")
            vendor = self.cell_vendor_map.get(cell_id)
            if vendor:
                return {
                    "vendor": vendor,
                    "confidence": 0.85,
                    "method": "context_cell",
                    "alternatives": []
                }

        # Strategy 2: Network slice to vendor mapping
        if "network_slice" in context or "slice_id" in context:
            slice_id = context.get("network_slice") or context.get("slice_id")
            vendors = self.slice_vendor_map.get(slice_id, [])
            if vendors:
                if len(vendors) == 1:
                    return {
                        "vendor": vendors[0],
                        "confidence": 0.80,
                        "method": "context_slice",
                        "alternatives": []
                    }
                else:
                    # Multiple vendors in slice
                    return {
                        "vendor": vendors[0],
                        "confidence": 0.60,
                        "method": "context_slice",
                        "alternatives": [{"vendor": v} for v in vendors[1:]]
                    }

        # Strategy 3: rApp registration vendor
        if "rapp_id" in context:
            rapp_id = context["rapp_id"]
            vendor = self.rapp_vendor_map.get(rapp_id)
            if vendor:
                return {
                    "vendor": vendor,
                    "confidence": 0.80,
                    "method": "context_rapp",
                    "alternatives": []
                }

        # Strategy 4: Explicit vendor field in context
        if "vendor" in context or "network_vendor" in context:
            vendor = context.get("vendor") or context.get("network_vendor")
            if vendor:
                return {
                    "vendor": vendor,
                    "confidence": 0.90,
                    "method": "context_explicit",
                    "alternatives": []
                }

        # No context-based detection
        return {
            "vendor": None,
            "confidence": 0.0,
            "method": "context",
            "alternatives": []
        }

    def _llm_vendor_extraction(self, query: str) -> Dict:
        """
        Use LLM to extract vendor from ambiguous query

        Args:
            query: User query

        Returns:
            Detection result
        """
        if not self.anthropic_client:
            return {
                "vendor": None,
                "confidence": 0.0,
                "method": "llm",
                "alternatives": []
            }

        prompt = f"""Extract the vendor/equipment manufacturer name from this O-RAN R1 query.

Known vendors: Viavi, Ericsson, Nokia, Huawei, Samsung, ZTE, Mavenir, Parallel Wireless, Generic/O-RAN

Query: {query}

Respond with ONLY the vendor name from the list above, or "unknown" if not found. Do not add any explanation."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Fast model for extraction
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )

            vendor_name = response.content[0].text.strip().lower()

            # Normalize vendor name
            vendor_map = {
                "viavi": "viavi",
                "spirent": "viavi",
                "ericsson": "ericsson",
                "nokia": "nokia",
                "huawei": "huawei",
                "samsung": "samsung",
                "zte": "zte",
                "mavenir": "mavenir",
                "parallel wireless": "parallel_wireless",
                "generic": "generic",
                "o-ran": "generic"
            }

            vendor = vendor_map.get(vendor_name)

            if vendor:
                return {
                    "vendor": vendor,
                    "confidence": 0.75,
                    "method": "llm",
                    "alternatives": []
                }

        except Exception as e:
            print(f"⚠️  LLM vendor extraction failed: {e}")

        return {
            "vendor": None,
            "confidence": 0.0,
            "method": "llm",
            "alternatives": []
        }

    def register_cell_vendor(self, cell_id: str, vendor: str):
        """Register cell ID to vendor mapping"""
        self.cell_vendor_map[cell_id] = vendor
        print(f"📍 Registered cell '{cell_id}' → vendor '{vendor}'")

    def register_slice_vendors(self, slice_id: str, vendors: List[str]):
        """Register network slice to vendor(s) mapping"""
        self.slice_vendor_map[slice_id] = vendors
        print(f"🔀 Registered slice '{slice_id}' → vendors {vendors}")

    def register_rapp_vendor(self, rapp_id: str, vendor: str):
        """Register rApp to vendor mapping"""
        self.rapp_vendor_map[rapp_id] = vendor
        print(f"📱 Registered rApp '{rapp_id}' → vendor '{vendor}'")

    def load_mappings_from_file(self, file_path: str):
        """
        Load vendor mappings from YAML file

        File format:
        cell_vendors:
          Cell-1: viavi
          Cell-2: ericsson
        slice_vendors:
          slice-1: [viavi, nokia]
        rapp_vendors:
          rapp-ml-001: viavi
        """
        import yaml

        with open(file_path, 'r') as f:
            mappings = yaml.safe_load(f)

        if "cell_vendors" in mappings:
            self.cell_vendor_map.update(mappings["cell_vendors"])
            print(f"✅ Loaded {len(mappings['cell_vendors'])} cell-vendor mappings")

        if "slice_vendors" in mappings:
            self.slice_vendor_map.update(mappings["slice_vendors"])
            print(f"✅ Loaded {len(mappings['slice_vendors'])} slice-vendor mappings")

        if "rapp_vendors" in mappings:
            self.rapp_vendor_map.update(mappings["rapp_vendors"])
            print(f"✅ Loaded {len(mappings['rapp_vendors'])} rApp-vendor mappings")


class MultiVendorResolver:
    """Resolve queries that might target multiple vendors"""

    def __init__(self, vendor_detector: VendorDetector):
        self.detector = vendor_detector

    def resolve(
        self,
        canonical_kpi: str,
        context: Dict
    ) -> List[Dict]:
        """
        Resolve potential multi-vendor scenarios

        Args:
            canonical_kpi: Canonical KPI name
            context: Network context

        Returns:
            List of vendor candidates with confidence scores
        """
        # Detect primary vendor
        detection = self.detector.detect("", context)

        # Check if multiple vendors in context
        if "vendors" in context:
            # User explicitly specified multiple vendors
            vendors = context["vendors"]
            return [
                {
                    "vendor": v,
                    "confidence": 0.70,
                    "reason": "User specified multiple vendors"
                }
                for v in vendors
            ]

        # Check if network slice has multiple vendors
        if detection["alternatives"]:
            candidates = [detection] + detection["alternatives"]
            return candidates

        # Single vendor detected
        return [detection]


# Example usage
if __name__ == "__main__":
    import sys

    detector = VendorDetector()

    # Test queries
    test_queries = [
        "Get RSRP values for Cell-1 using Viavi equipment",
        "Show me Ericsson throughput metrics",
        "What's the signal strength in Nokia cells?",
        "Query performance data",  # Ambiguous
        "Get KPIs from O-RAN compliant SMO"
    ]

    print("\n" + "=" * 80)
    print("VENDOR DETECTION TESTS")
    print("=" * 80)

    for query in test_queries:
        result = detector.detect(query)

        print(f"\nQuery: {query}")
        print(f"  → Vendor: {result['vendor']}")
        print(f"  → Confidence: {result['confidence']:.2f}")
        print(f"  → Method: {result['method']}")
        if result["alternatives"]:
            print(f"  → Alternatives: {[a['vendor'] for a in result['alternatives']]}")

    # Test context-based detection
    print("\n" + "=" * 80)
    print("CONTEXT-BASED DETECTION")
    print("=" * 80)

    detector.register_cell_vendor("Cell-1", "viavi")
    detector.register_cell_vendor("Cell-2", "ericsson")

    context_query = "Get RSRP for Cell-1"
    result = detector.detect(context_query, context={"cellId": "Cell-1"})

    print(f"\nQuery: {context_query}")
    print(f"Context: cellId=Cell-1")
    print(f"  → Vendor: {result['vendor']}")
    print(f"  → Confidence: {result['confidence']:.2f}")
    print(f"  → Method: {result['method']}")
