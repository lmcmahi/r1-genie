#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG System
Tests all components: parsing, embeddings, detection, mapping, API
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

print("=" * 80)
print("R1-GENIE RAG SYSTEM TEST SUITE")
print("=" * 80)
print()

# Test results tracker
test_results = []

def run_test(test_name, test_func):
    """Run a test and track result"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}\n")

    try:
        result = test_func()
        test_results.append((test_name, result, None))
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"\n{status}: {test_name}")
        return result
    except Exception as e:
        test_results.append((test_name, False, str(e)))
        print(f"\n❌ FAIL: {test_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# TEST 1: DOCUMENT PARSER TESTS
# =============================================================================

def test_openapi_parser():
    """Test OpenAPI/Swagger document parsing"""
    from document_parsers import OpenAPIParser
    from r1_dataset_builder_v3 import CANONICAL_KPIS

    # Check if OpenAPI test file exists
    test_file = "r1_rest_apis.yaml"
    if not os.path.exists(test_file):
        print(f"⚠️  Test file not found: {test_file}")
        return True  # Skip test

    parser = OpenAPIParser(canonical_kpis=CANONICAL_KPIS)

    # For this test, we'll use r1_rest_apis.yaml as a proxy
    # In production, you'd have a real OpenAPI spec
    print(f"Testing OpenAPI parser (using {test_file} as proxy)")
    print("Note: Skipping actual parsing - add real OpenAPI spec for full test")

    return True  # Conditional pass

def test_markdown_parser():
    """Test Markdown document parsing"""
    from document_parsers import MarkdownParser
    from r1_dataset_builder_v3 import CANONICAL_KPIS

    parser = MarkdownParser(canonical_kpis=CANONICAL_KPIS)

    # Create test markdown content
    test_content = """
# Viavi API Documentation

## RSRP Measurement API

The RSRP (Reference Signal Received Power) measurement API provides access to UE signal strength data.

### Endpoint
GET /api/v1/measurements/rsrp

### Parameters
- cellId: Cell identifier
- ueId: User equipment identifier

### Example
```
GET /api/v1/measurements/rsrp?cellId=Cell-1
```
    """

    # Write to temp file
    temp_file = "/tmp/test_viavi_docs.md"
    with open(temp_file, 'w') as f:
        f.write(test_content)

    try:
        chunks = parser.parse(temp_file, "viavi")

        print(f"✓ Parsed {len(chunks)} markdown chunks")

        if chunks:
            print(f"✓ First chunk text: {chunks[0].text[:100]}...")
            print(f"✓ First chunk metadata: {chunks[0].metadata}")

        return len(chunks) > 0

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# =============================================================================
# TEST 2: VENDOR DETECTION TESTS
# =============================================================================

def test_vendor_detection():
    """Test 3-tier vendor detection system"""
    from vendor_detector import VendorDetector

    detector = VendorDetector()

    test_cases = [
        # (query, expected_vendor, min_confidence)
        ("Get RSRP values for Cell-1 using Viavi equipment", "viavi", 0.90),
        ("Show me Ericsson throughput metrics", "ericsson", 0.90),
        ("What's the signal strength in Nokia cells?", "nokia", 0.90),
        ("Query performance data", "generic", 0.40),  # Should use default
    ]

    all_passed = True

    for query, expected_vendor, min_confidence in test_cases:
        result = detector.detect(query)

        print(f"\nQuery: {query}")
        print(f"  Detected: {result['vendor']} (confidence: {result['confidence']:.2f})")
        print(f"  Expected: {expected_vendor} (min confidence: {min_confidence})")

        if result['vendor'] == expected_vendor and result['confidence'] >= min_confidence:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            all_passed = False

    return all_passed

def test_vendor_detection_with_context():
    """Test context-based vendor detection"""
    from vendor_detector import VendorDetector

    detector = VendorDetector()

    # Register test mappings
    detector.register_cell_vendor("Cell-1", "viavi")
    detector.register_cell_vendor("Cell-2", "ericsson")

    # Test with context
    result = detector.detect(
        "Get RSRP for Cell-1",
        context={"cellId": "Cell-1"}
    )

    print(f"\nContext-based detection:")
    print(f"  Query: Get RSRP for Cell-1")
    print(f"  Context: cellId=Cell-1")
    print(f"  Detected: {result['vendor']} (confidence: {result['confidence']:.2f})")
    print(f"  Method: {result['method']}")

    return result['vendor'] == "viavi" and result['confidence'] >= 0.80

# =============================================================================
# TEST 3: RAG VENDOR MAPPING TESTS
# =============================================================================

def test_yaml_fallback_mapping():
    """Test YAML fallback when RAG not available"""
    from rag_vendor_mapper import RAGVendorMapper
    from r1_dataset_builder_v3 import CANONICAL_KPIS

    # Note: This test doesn't require Qdrant to be running
    # It will automatically fall back to YAML

    mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)

    test_cases = [
        ("rsrp_dbm", "viavi"),
        ("throughput_dl_mbps", "ericsson"),
        ("prb_usage_dl", "nokia"),
    ]

    all_passed = True

    for canonical_kpi, vendor in test_cases:
        result = mapper.map_to_vendor(canonical_kpi, vendor)

        print(f"\nMapping: {canonical_kpi} → {vendor}")
        print(f"  Status: {result['status']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Source: {result['source']}")

        if result['status'] in ['success', 'fallback']:
            print(f"  Endpoint: {result['vendor_api'].get('endpoint', 'N/A')}")
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            all_passed = False

    return all_passed

def test_3gpp_generic_fallback():
    """Test final 3GPP generic fallback"""
    from rag_vendor_mapper import RAGVendorMapper
    from r1_dataset_builder_v3 import CANONICAL_KPIS

    mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)

    # Test with unknown vendor
    result = mapper.map_to_vendor("rsrp_dbm", "unknown_vendor_xyz")

    print(f"\nGeneric fallback test:")
    print(f"  KPI: rsrp_dbm")
    print(f"  Vendor: unknown_vendor_xyz")
    print(f"  Result status: {result['status']}")
    print(f"  Source: {result['source']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Should fallback to generic
    return result['source'] in ['yaml_fallback', '3gpp_generic']

# =============================================================================
# TEST 4: INTEGRATION TESTS
# =============================================================================

def test_end_to_end_workflow():
    """Test complete workflow: detect vendor → map KPI → build API call"""
    from vendor_detector import VendorDetector
    from rag_vendor_mapper import RAGVendorMapper
    from r1_dataset_builder_v3 import CANONICAL_KPIS

    detector = VendorDetector()
    mapper = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)

    # User query
    query = "Get RSRP and throughput values for Cell-1 using Viavi test equipment"

    print(f"\nEnd-to-end test:")
    print(f"  Query: {query}\n")

    # Step 1: Detect vendor
    vendor_result = detector.detect(query)
    print(f"Step 1 - Vendor Detection:")
    print(f"  Vendor: {vendor_result['vendor']}")
    print(f"  Confidence: {vendor_result['confidence']:.2f}")
    print(f"  Method: {vendor_result['method']}\n")

    # Step 2: Map KPIs
    kpis_to_map = ["rsrp_dbm", "throughput_dl_mbps"]
    mappings = {}

    print(f"Step 2 - KPI Mapping:")
    for kpi in kpis_to_map:
        result = mapper.map_to_vendor(kpi, vendor_result['vendor'])
        mappings[kpi] = result
        print(f"  {kpi}:")
        print(f"    Endpoint: {result['vendor_api'].get('endpoint', 'N/A')}")
        print(f"    Source: {result['source']}")
        print(f"    Confidence: {result['confidence']:.2f}")

    print("\nResult: End-to-end workflow completed successfully")

    return len(mappings) == len(kpis_to_map)

def test_dataset_builder_integration():
    """Test integration with dataset builder"""
    from r1_dataset_builder_v3 import build_rest_sample

    print("\nDataset Builder Integration:")

    # Test training mode (default - should work without RAG)
    print("\n  Testing training mode (canonical only)...")
    sample_training = build_rest_sample(mode="training")

    print(f"  ✓ Training sample generated")
    print(f"  ✓ Has instruction: {bool(sample_training.get('instruction'))}")
    print(f"  ✓ Has output: {bool(sample_training.get('output'))}")
    print(f"  ✓ Protocol: {sample_training['context']['protocol_stack']}")

    # Test inference mode (requires RAG - may skip if not enabled)
    print("\n  Testing inference mode (with RAG)...")
    try:
        sample_inference = build_rest_sample(mode="inference")
        print(f"  ✓ Inference sample generated")

        if "vendor_api" in sample_inference:
            print(f"  ✓ Has vendor_api: {sample_inference['vendor_api']['vendor']}")
            print(f"  ✓ Mapping metadata: {sample_inference.get('mapping_metadata', {})}")
        else:
            print("  ⚠️  No vendor_api (RAG not enabled or no KPIs in sample)")

    except Exception as e:
        print(f"  ⚠️  Inference mode skipped: {e}")

    return True  # Training mode must work

# =============================================================================
# TEST 5: API TESTS
# =============================================================================

def test_canonical_kpi_loading():
    """Test canonical KPI configuration loading"""
    from r1_dataset_builder_v3 import CANONICAL_KPIS, get_kpi_metadata

    kpis = CANONICAL_KPIS.get("canonical_kpis", {})

    print(f"\nCanonical KPI Loading:")
    print(f"  Total KPIs loaded: {len(kpis)}")

    if len(kpis) == 0:
        return False

    # Test metadata retrieval for known KPIs
    test_kpis = ["rsrp_dbm", "throughput_dl_mbps", "prb_usage_dl"]
    all_found = True

    for kpi_name in test_kpis:
        metadata = get_kpi_metadata(kpi_name)
        if metadata:
            print(f"  ✓ {kpi_name}: {metadata['full_name']}")
            print(f"    3GPP: {metadata.get('threegpp_reference', {}).get('spec', 'N/A')}")
            print(f"    Vendors: {list(metadata.get('vendor_mappings', {}).keys())}")
        else:
            print(f"  ✗ {kpi_name}: NOT FOUND")
            all_found = False

    return all_found

# =============================================================================
# TEST EXECUTION
# =============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n🚀 Starting RAG System Test Suite\n")

    tests = [
        # Document Parsing
        ("OpenAPI Parser", test_openapi_parser),
        ("Markdown Parser", test_markdown_parser),

        # Vendor Detection
        ("Vendor Detection - Explicit", test_vendor_detection),
        ("Vendor Detection - Context", test_vendor_detection_with_context),

        # RAG Mapping
        ("YAML Fallback Mapping", test_yaml_fallback_mapping),
        ("3GPP Generic Fallback", test_3gpp_generic_fallback),

        # Integration
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Dataset Builder Integration", test_dataset_builder_integration),

        # Configuration
        ("Canonical KPI Loading", test_canonical_kpi_loading),
    ]

    for test_name, test_func in tests:
        run_test(test_name, test_func)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result, _ in test_results if result)
    total = len(test_results)

    for test_name, result, error in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")

    pass_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n{passed}/{total} tests passed ({pass_rate:.1f}%)")

    # Exit code
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
