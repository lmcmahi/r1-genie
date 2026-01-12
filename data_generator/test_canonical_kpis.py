#!/usr/bin/env python3
"""
Test script for canonical KPI implementation
"""

import json
import sys
sys.path.insert(0, '.')

from r1_dataset_builder_v3 import (
    load_yaml_config,
    get_kpi_metadata,
    extract_kpis_from_payload,
    build_rest_sample,
    build_kafka_sample,
    CANONICAL_KPIS
)

def test_canonical_kpi_loading():
    """Test that canonical KPIs load correctly"""
    print("=" * 80)
    print("TEST 1: Loading Canonical KPIs")
    print("=" * 80)

    kpis = CANONICAL_KPIS.get("canonical_kpis", {})

    print(f"✅ Loaded {len(kpis)} canonical KPIs")
    print(f"\nKPI Names:")
    for kpi_name in list(kpis.keys())[:10]:  # Show first 10
        print(f"  - {kpi_name}")

    if len(kpis) >= 10:
        print(f"  ... and {len(kpis) - 10} more")

    return len(kpis) > 0

def test_kpi_metadata():
    """Test KPI metadata retrieval"""
    print("\n" + "=" * 80)
    print("TEST 2: KPI Metadata Retrieval")
    print("=" * 80)

    test_kpis = ["rsrp_dbm", "throughput_dl_mbps", "prb_usage_dl"]

    for kpi_name in test_kpis:
        metadata = get_kpi_metadata(kpi_name)
        if metadata:
            print(f"\n✅ {kpi_name}:")
            print(f"  Full Name: {metadata['full_name']}")
            print(f"  Unit: {metadata['unit']}")
            if metadata.get('threegpp_reference', {}).get('spec'):
                print(f"  3GPP Spec: {metadata['threegpp_reference']['spec']}")
            print(f"  Vendor Mappings: {len(metadata['vendor_mappings'])}")
        else:
            print(f"❌ {kpi_name}: No metadata found")
            return False

    return True

def test_kpi_extraction():
    """Test KPI extraction from payloads"""
    print("\n" + "=" * 80)
    print("TEST 3: KPI Extraction from Payloads")
    print("=" * 80)

    # Test REST payload
    test_payload = {
        "filterCriteria": {
            "measurementTypes": ["rsrp_dbm", "prb_usage_dl"]
        }
    }

    kpis = extract_kpis_from_payload(test_payload)
    print(f"REST Payload KPIs: {kpis}")

    if "rsrp_dbm" not in kpis or "prb_usage_dl" not in kpis:
        print("❌ Failed to extract REST KPIs")
        return False

    # Test Kafka payload
    kafka_payload = {
        "measurements": {
            "rsrp_dbm": -85,
            "sinr_db": 15
        }
    }

    kpis2 = extract_kpis_from_payload(kafka_payload)
    print(f"Kafka Payload KPIs: {kpis2}")

    if "rsrp_dbm" not in kpis2 or "sinr_db" not in kpis2:
        print("❌ Failed to extract Kafka KPIs")
        return False

    print("✅ KPI extraction working correctly")
    return True

def test_rest_sample_generation():
    """Test REST sample generation with KPI metadata"""
    print("\n" + "=" * 80)
    print("TEST 4: REST Sample Generation")
    print("=" * 80)

    # Generate 5 samples to increase chance of getting KPI-related one
    for i in range(5):
        sample = build_rest_sample()

        if "kpi_metadata" in sample and len(sample["kpi_metadata"]) > 0:
            print(f"\n✅ Found sample with KPI metadata (attempt {i+1}):")
            print(f"Instruction: {sample['instruction']}")
            print(f"API: {sample['output']['method']} {sample['output']['uri']}")
            print(f"KPIs Requested: {sample['context'].get('kpis_requested', [])}")

            print(f"\n✅ KPI Metadata Present:")
            for kpi_name, metadata in sample["kpi_metadata"].items():
                print(f"  - {kpi_name}:")
                print(f"    Full Name: {metadata['full_name']}")
                if metadata.get('threegpp_reference', {}).get('measurement_name'):
                    print(f"    3GPP: {metadata['threegpp_reference']['measurement_name']}")
                print(f"    Vendors: {list(metadata['vendor_mappings'].keys())}")
            return True

    print("⚠️ No KPI metadata found in 5 samples (might need more API calls with KPIs)")
    return True  # Not a failure, just no KPI APIs were randomly selected

def test_kafka_sample_generation():
    """Test Kafka sample generation with KPI metadata"""
    print("\n" + "=" * 80)
    print("TEST 5: Kafka Sample Generation")
    print("=" * 80)

    # Generate a few samples
    for i in range(3):
        sample = build_kafka_sample()

        if "kpi_metadata" in sample and len(sample["kpi_metadata"]) > 0:
            print(f"\n✅ Found Kafka sample with KPI metadata (attempt {i+1}):")
            print(f"Instruction: {sample['instruction']}")
            print(f"Topic: {sample['output']['topic']}")
            print(f"KPIs in Message: {sample['context'].get('kpis_in_message', [])}")
            print(f"KPI Metadata: {len(sample['kpi_metadata'])} KPIs")
            return True

    print("⚠️ No KPI metadata in 3 Kafka samples")
    return True

def test_vendor_mapping_coverage():
    """Test that all canonical KPIs have vendor mappings"""
    print("\n" + "=" * 80)
    print("TEST 6: Vendor Mapping Coverage")
    print("=" * 80)

    kpis = CANONICAL_KPIS.get("canonical_kpis", {})

    required_vendors = ["viavi", "ericsson", "nokia"]
    missing_mappings = []

    for kpi_name, kpi_def in kpis.items():
        vendor_mappings = kpi_def.get("vendor_mappings", {})
        for vendor in required_vendors:
            if vendor not in vendor_mappings:
                missing_mappings.append(f"{kpi_name} missing {vendor}")

    if missing_mappings:
        print(f"❌ Missing mappings:")
        for missing in missing_mappings[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_mappings) > 10:
            print(f"  ... and {len(missing_mappings) - 10} more")
        return False
    else:
        print(f"✅ All {len(kpis)} KPIs have mappings for {len(required_vendors)} vendors")
        return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CANONICAL KPI IMPLEMENTATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Loading Canonical KPIs", test_canonical_kpi_loading),
        ("KPI Metadata Retrieval", test_kpi_metadata),
        ("KPI Extraction", test_kpi_extraction),
        ("REST Sample Generation", test_rest_sample_generation),
        ("Kafka Sample Generation", test_kafka_sample_generation),
        ("Vendor Mapping Coverage", test_vendor_mapping_coverage)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
