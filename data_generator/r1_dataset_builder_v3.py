"""
O-RAN R1 Dataset Builder - PRODUCTION VERSION (Modular)
================================================
Modular orchestration system using YAML configs.
Loads O-RAN R1 REST, Kafka, Schema, and Intent definitions dynamically.
Generates datasets for SFT + GRPO training aligned with:
- O-RAN.WG2.TS.R1GAP-R004-v12.00
- O-RAN.WG2.TS.R1AP-R004-v9.00
- O-RAN.WG2.TS.R1TP-R004-v04.03

Complete Coverage:
✅ All 7 R1GAP service categories (Sections 5.1-5.7)
✅ 70+ REST API operations
✅ 8 Kafka streaming topics
✅ JSON Schema validation
✅ Natural language intent variations
✅ Audit logging (CSV + JSON)
================================================
"""

import os
import json
import csv
import uuid
import random
import yaml
from datetime import datetime
from tqdm import tqdm
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".")
API_ROOT = os.getenv("API_ROOT", "https://smo.example.com")

DATASET_SIZE_SFT = int(os.getenv("DATASET_SIZE_SFT", 5000))
DATASET_SIZE_GRPO = int(os.getenv("DATASET_SIZE_GRPO", 1500))

REST_RATIO = float(os.getenv("REST_RATIO", 0.65))
KAFKA_RATIO = float(os.getenv("KAFKA_RATIO", 0.30))
LIFECYCLE_RATIO = float(os.getenv("LIFECYCLE_RATIO", 0.05))

# Output directory and files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_VALIDATION_LOG = os.path.join(OUTPUT_DIR, "r1_validation_audit_modular.csv")
JSON_SUMMARY_LOG = os.path.join(OUTPUT_DIR, "r1_dataset_summary_modular.json")

# =============================================================================
# LOAD YAML CONFIGURATIONS
# =============================================================================


def load_yaml_config(name):
    """Load configuration YAML"""
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


print("📦 Loading YAML configurations...")
REST_APIS = load_yaml_config("r1_rest_apis.yaml")
KAFKA_TOPICS = load_yaml_config("r1_kafka_topics.yaml")
SCHEMAS = load_yaml_config("r1_schemas.yaml")
INTENTS = load_yaml_config("r1_intents.yaml")
CANONICAL_KPIS = load_yaml_config("r1_canonical_kpis.yaml")
print(f"✅ Loaded {len(REST_APIS)} REST API categories")
print(f"✅ Loaded {len(KAFKA_TOPICS)} Kafka topic categories")
print(f"✅ Loaded {len(SCHEMAS)} schema definitions")
print(f"✅ Loaded {len(INTENTS)} intent categories")
print(f"✅ Loaded {len(CANONICAL_KPIS.get('canonical_kpis', {}))} canonical KPIs")

# =============================================================================
# RAG SYSTEM INITIALIZATION (Optional - for inference mode only)
# =============================================================================
ENABLE_RAG = os.getenv("ENABLE_RAG", "false").lower() == "true"
RAG_VENDOR_MAPPER = None
VENDOR_DETECTOR = None

if ENABLE_RAG:
    try:
        from rag_vendor_mapper import RAGVendorMapper
        from vendor_detector import VendorDetector

        RAG_VENDOR_MAPPER = RAGVendorMapper(canonical_kpis=CANONICAL_KPIS)
        VENDOR_DETECTOR = VendorDetector()
        print("✅ RAG system initialized for inference mode")
    except ImportError as e:
        print(f"⚠️  RAG system not available: {e}")
        print("   Install RAG dependencies: pip install qdrant-client llama-index openai")
        ENABLE_RAG = False
except Exception as e:
    print(f"⚠️  RAG initialization failed: {e}")
    ENABLE_RAG = False

# =============================================================================
# VALIDATION AUDIT TRACKER
# =============================================================================
validation_audit = []

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_kpi_metadata(canonical_kpi_name):
    """Get metadata for canonical KPI including 3GPP references and vendor mappings"""
    kpi_def = CANONICAL_KPIS.get("canonical_kpis", {}).get(canonical_kpi_name)
    if not kpi_def:
        return None

    return {
        "canonical_name": canonical_kpi_name,
        "full_name": kpi_def.get("full_name", ""),
        "unit": kpi_def.get("unit", ""),
        "range": kpi_def.get("range"),
        "threegpp_reference": {
            "spec": kpi_def.get("threegpp", {}).get("spec"),
            "section": kpi_def.get("threegpp", {}).get("section"),
            "measurement_name": kpi_def.get("threegpp", {}).get("measurement_name")
        } if "threegpp" in kpi_def else {},
        "vendor_mappings": kpi_def.get("vendor_mappings", {}),
        "use_cases": kpi_def.get("use_cases", [])
    }


def extract_kpis_from_payload(payload):
    """Extract KPI names from API payload"""
    kpis = []

    if not isinstance(payload, dict):
        return kpis

    # Check measurementTypes in filterCriteria
    filter_criteria = payload.get("filterCriteria", {})
    measurement_types = filter_criteria.get("measurementTypes", [])
    if measurement_types:
        kpis.extend(measurement_types)

    # Check nested measurements
    if "measurements" in payload:
        measurements = payload["measurements"]
        if isinstance(measurements, dict):
            kpis.extend(measurements.keys())

    # Check nested kpis
    if "kpis" in payload:
        kpi_obj = payload["kpis"]
        if isinstance(kpi_obj, dict):
            kpis.extend(kpi_obj.keys())

    # Check metrics
    if "metrics" in payload:
        metrics = payload["metrics"]
        if isinstance(metrics, dict):
            kpis.extend(metrics.keys())

    return list(set(kpis))  # Remove duplicates


def substitute_path_params(uri):
    """Replace path parameters with realistic values"""
    uri = uri.replace("{apiRoot}", API_ROOT)

    replacements = {
        "{apfId}": str(uuid.uuid4()),
        "{serviceApiId}": str(uuid.uuid4()),
        "{rAppId}": str(uuid.uuid4()),
        "{registrationId}": str(uuid.uuid4()),
        "{dataJobId}": str(uuid.uuid4()),
        "{requestId}": str(uuid.uuid4()),
        "{subscriptionId}": str(uuid.uuid4()),
        "{policyId}": str(uuid.uuid4()),
        "{policyTypeId}": random.choice(
            ["QoS-policy", "Traffic-steering", "Congestion-control"]
        ),
        "{modelId}": str(uuid.uuid4()),
        "{trainingJobId}": str(uuid.uuid4()),
        "{dmeTypeId}": f"{random.choice(['oran-dmf', 'oran-udr'])}:kpi-v{random.randint(1,3)}",
        "{eiTypeId}": str(uuid.uuid4()),
        "{offerId}": str(uuid.uuid4()),
        "{cellId}": f"NRCell-{random.randint(1,999)}",
        "{alarmId}": str(uuid.uuid4()),
        "{jobId}": str(uuid.uuid4()),
    }

    for param, value in replacements.items():
        uri = uri.replace(param, value)

    return uri


def soft_validate(payload, schema_name):
    """Validate JSON payload against schema"""
    if schema_name not in SCHEMAS:
        return True, "No schema defined"

    try:
        validate(instance=payload, schema=SCHEMAS[schema_name])
        return True, "Schema valid"
    except ValidationError as e:
        return False, f"Schema error: {e.message[:50]}"


def generate_mock_jwt():
    """Generate mock JWT token"""
    return f"{uuid.uuid4().hex}{uuid.uuid4().hex}"[:64]


def generate_r1_compliant_headers(method, api_config, resource_id=None):
    """
    Generate R1/CAPIF compliant HTTP headers per 3GPP TS 29.122 and TS 29.222

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        api_config: API configuration from YAML
        resource_id: Optional resource ID for Location header

    Returns:
        Dict of R1-compliant headers
    """
    headers = {
        # Standard HTTP headers per RFC 7231
        "Content-Type": "application/json",
        "Accept": "application/json",

        # OAuth 2.0 Bearer token per RFC 6750
        "Authorization": f"Bearer {generate_mock_jwt()}",

        # Request correlation per 3GPP TS 29.122
        "X-Request-ID": str(uuid.uuid4()),

        # 3GPP SBI headers per TS 29.500
        "3gpp-Sbi-Message-Priority": str(random.randint(0, 31)),
    }

    # Add callback header for async operations
    if method == "POST" and api_config.get("response_code") == 201:
        headers["3gpp-Sbi-Callback"] = "http://rapp.example.com/callbacks"

    # Add target API root for forwarded requests
    if random.random() < 0.3:  # 30% of requests have forwarding context
        headers["3gpp-Sbi-Target-apiRoot"] = API_ROOT

    return headers


def generate_r1_response(method, api_config, request_body=None, resource_id=None):
    """
    Generate R1-compliant HTTP response structure

    Args:
        method: HTTP method
        api_config: API configuration from YAML
        request_body: Request body (for echoing in response)
        resource_id: Resource ID for created resources

    Returns:
        Dict containing response structure
    """
    response_code = api_config.get("response_code", 200)

    response = {
        "status_code": response_code,
        "headers": {
            "Content-Type": "application/json",
            "X-Request-ID": str(uuid.uuid4()),
        }
    }

    # Add Location header for 201 Created responses per RFC 7231
    if response_code == 201 and api_config.get("response_headers", {}).get("Location"):
        location = api_config["response_headers"]["Location"]
        location = location.replace("{apiRoot}", API_ROOT)
        if resource_id:
            # Replace any remaining path parameters with the resource ID
            for param in ["{serviceApiId}", "{registrationId}", "{subscriptionId}",
                         "{policyId}", "{modelId}", "{rAppId}", "{infoJobId}",
                         "{trainingJobId}", "{inferenceJobId}", "{artifactId}",
                         "{capabilityId}", "{offerId}", "{pmJobId}", "{eiJobId}",
                         "{eiTypeId}", "{nfDeploymentId}"]:
                location = location.replace(param, resource_id)
        response["headers"]["Location"] = location

    # Generate response body based on method and status code
    if response_code == 204:
        response["body"] = None
    elif response_code in [200, 201] and request_body:
        # Echo back request with added server-side fields
        response_body = dict(request_body) if request_body else {}
        if response_code == 201:
            response_body["id"] = resource_id or str(uuid.uuid4())
            response_body["createdAt"] = datetime.utcnow().isoformat() + "Z"
        response_body["lastModifiedAt"] = datetime.utcnow().isoformat() + "Z"
        response["body"] = response_body
    elif api_config.get("example_response"):
        response["body"] = api_config["example_response"]

    return response


# =============================================================================
# RAG INFERENCE FUNCTIONS (for inference mode only)
# =============================================================================


def enhance_sample_with_rag(sample, mode="training"):
    """
    Enhance sample with RAG-based vendor mapping (inference mode only)

    Args:
        sample: Base sample dict
        mode: 'training' or 'inference'

    Returns:
        Enhanced sample with vendor-specific API mapping
    """
    if mode != "inference" or not ENABLE_RAG or not RAG_VENDOR_MAPPER:
        return sample

    # Extract canonical KPIs from sample
    canonical_kpis = sample["context"].get("kpis_requested", []) or \
                     sample["context"].get("kpis_in_message", [])

    if not canonical_kpis:
        return sample  # No KPIs to map

    # Detect vendor from instruction
    vendor_info = VENDOR_DETECTOR.detect(
        user_query=sample["instruction"],
        context=sample.get("context", {})
    )

    # Map each KPI to vendor-specific API
    vendor_mappings = {}
    rag_used_count = 0
    fallback_used_count = 0

    for kpi in canonical_kpis:
        result = RAG_VENDOR_MAPPER.map_to_vendor(
            canonical_kpi=kpi,
            vendor=vendor_info["vendor"],
            context=sample.get("context", {})
        )

        vendor_mappings[kpi] = result

        if result["source"] == "rag":
            rag_used_count += 1
        elif result["source"] in ["yaml_fallback", "3gpp_generic"]:
            fallback_used_count += 1

    # Build vendor-specific API call
    if vendor_mappings:
        sample["vendor_api"] = build_vendor_api_call(
            vendor_mappings,
            vendor_info["vendor"],
            sample["output"]
        )

        sample["mapping_metadata"] = {
            "vendor": vendor_info["vendor"],
            "vendor_detection_method": vendor_info["method"],
            "vendor_confidence": vendor_info["confidence"],
            "rag_mappings_used": rag_used_count,
            "fallback_mappings_used": fallback_used_count,
            "total_kpis_mapped": len(vendor_mappings)
        }

    return sample


def build_vendor_api_call(vendor_mappings, vendor, base_output):
    """
    Build vendor-specific API call from RAG mappings

    Args:
        vendor_mappings: Dict of {kpi: mapping_result}
        vendor: Vendor name
        base_output: Base output from training mode

    Returns:
        Vendor-specific API call dict
    """
    # Get the highest confidence mapping to use as primary endpoint
    primary_mapping = max(
        vendor_mappings.values(),
        key=lambda x: x["confidence"]
    )

    vendor_api = primary_mapping["vendor_api"]

    # Build vendor-specific call
    return {
        "vendor": vendor,
        "protocol": base_output.get("protocol", "HTTPS"),
        "method": vendor_api.get("method", "GET"),
        "endpoint": vendor_api.get("endpoint", ""),
        "parameters": vendor_api.get("parameters", []),
        "headers": base_output.get("headers", {}),
        "transport": base_output.get("transport", {}),
        "kpi_mappings": {
            kpi: {
                "vendor_param": mapping["vendor_api"].get("vendor_param_name",
                                                         mapping["canonical_kpi"]),
                "confidence": mapping["confidence"],
                "source": mapping["source"]
            }
            for kpi, mapping in vendor_mappings.items()
        },
        "description": vendor_api.get("description", ""),
        "source": primary_mapping["source"]
    }


# =============================================================================
# DATASET BUILDERS
# =============================================================================


def build_rest_sample(mode="training"):
    """
    Build REST-based SFT sample - R1/CAPIF compliant

    Args:
        mode: 'training' (canonical only) or 'inference' (with RAG vendor mapping)
    """
    # Select random service category and API
    category = random.choice(list(REST_APIS.keys()))
    api = random.choice(REST_APIS[category])

    # Get natural language intent variation
    intent_variants = INTENTS.get(category, {}).get(api["method"], [api["name"]])
    instruction = random.choice(intent_variants)

    # Get payload
    body = api.get("example_payload")

    # Validate payload
    is_valid, validation_msg = True, "No body required"
    if body:
        is_valid, validation_msg = soft_validate(body, category)

    # Generate resource ID for created resources
    resource_id = str(uuid.uuid4())

    # Log validation
    validation_audit.append(
        {
            "category": category,
            "method": api["method"],
            "schema_valid": is_valid,
            "validation_msg": validation_msg,
            "timestamp": datetime.now(tz=None).isoformat(),
        }
    )

    # Select protocol per CAPIF spec - HTTP/1.1 mandatory, HTTP/2 recommended
    # Use enum values per 3GPP TS 29.222: HTTP_1_1, HTTP_2
    protocol_choice = random.choices(
        ["HTTP_1_1", "HTTP_2"],
        weights=[0.3, 0.7],  # Prefer HTTP/2 as recommended
        k=1
    )[0]

    # Build R1-compliant request output
    output = {
        "protocol": "HTTPS",
        "method": api["method"],
        "uri": substitute_path_params(api["uri"]),
        "headers": generate_r1_compliant_headers(api["method"], api, resource_id),
        "transport": {
            # Protocol per CAPIF enum (3GPP TS 29.222)
            "http_version": protocol_choice,
            # TLS per 3GPP TS 33.122
            "tls_version": "TLS_1_3",
            "port": 443,
            "tcp_enabled": True,
            # Security method per CAPIF
            "security_method": random.choice(["OAUTH", "PKI", "PSK"]),
        },
    }

    if body:
        output["body"] = body

    # Generate R1-compliant response
    response = generate_r1_response(api["method"], api, body, resource_id)

    # Extract KPIs from payload
    kpis_in_request = extract_kpis_from_payload(body) if body else []

    # Get KPI metadata for all KPIs in request
    kpi_metadata = {}
    for kpi_name in kpis_in_request:
        metadata = get_kpi_metadata(kpi_name)
        if metadata:
            kpi_metadata[kpi_name] = metadata

    # Build sample with request and response
    sample = {
        "instruction": instruction,
        "context": {
            "protocol_stack": "REST",
            "service_category": category,
            "schema_valid": is_valid,
            "api_name": api["name"],
            "response_code": api.get("response_code", 200),
            "kpis_requested": kpis_in_request,
        },
        "request": output,
        "response": response,
        # Keep output for backward compatibility
        "output": output,
    }

    # Add KPI metadata if available
    if kpi_metadata:
        sample["kpi_metadata"] = kpi_metadata

    # Enhance with RAG vendor mapping if in inference mode
    sample = enhance_sample_with_rag(sample, mode)

    return sample


def build_kafka_sample(mode="training"):
    """
    Build Kafka-based SFT sample

    Args:
        mode: 'training' (canonical only) or 'inference' (with RAG vendor mapping)
    """
    # Select random topic category and message
    topic_category = random.choice(list(KAFKA_TOPICS.keys()))
    msg = random.choice(KAFKA_TOPICS[topic_category])

    output = {
        "protocol": "Kafka",
        "operation": msg["operation"],
        "topic": msg["topic"],
        "transport": {
            "kafka_version": "3.0+",
            "security_protocol": "SASL_SSL",
            "port": 9093,
            "tls_enabled": True,
        },
    }

    # Add payload for PRODUCE operations
    if msg["operation"] == "PRODUCE":
        output["key"] = f"partition-key-{random.randint(1,100)}"
        output["value"] = msg["example"]
        output["headers"] = {
            "source": f"rApp-{uuid.uuid4().hex[:8]}",
            "correlation-id": str(uuid.uuid4()),
        }
    else:  # CONSUME
        output["consumer_group"] = msg["example"].get(
            "consumer_group", f"rapp-consumer-{random.randint(1,10)}"
        )
        output["auto_offset_reset"] = msg["example"].get("auto_offset_reset", "latest")

    # Extract KPIs from message example
    kpis_in_message = extract_kpis_from_payload(msg.get("example", {}))

    # Get KPI metadata
    kpi_metadata = {}
    for kpi_name in kpis_in_message:
        metadata = get_kpi_metadata(kpi_name)
        if metadata:
            kpi_metadata[kpi_name] = metadata

    # Build sample
    sample = {
        "instruction": msg["description"],
        "context": {
            "protocol_stack": "Kafka",
            "topic_category": topic_category,
            "operation": msg["operation"],
            "kpis_in_message": kpis_in_message,
        },
        "output": output,
    }

    # Add KPI metadata if available
    if kpi_metadata:
        sample["kpi_metadata"] = kpi_metadata

    # Enhance with RAG vendor mapping if in inference mode
    sample = enhance_sample_with_rag(sample, mode)

    return sample


def build_sft_dataset(n, mode="training"):
    """Build SFT dataset with proper distribution"""
    sft = []

    n_rest = int(n * REST_RATIO)
    n_kafka = int(n * KAFKA_RATIO)
    n_lifecycle = n - n_rest - n_kafka

    for _ in tqdm(range(n_rest), desc="Building REST SFT"):
        sft.append(build_rest_sample(mode))

    for _ in tqdm(range(n_kafka), desc="Building Kafka SFT"):
        sft.append(build_kafka_sample(mode))

    for _ in tqdm(range(n_lifecycle), desc="Building Lifecycle SFT"):
        # Lifecycle samples are also REST-based
        sft.append(build_rest_sample(mode))

    random.shuffle(sft)
    return sft


def build_grpo_dataset(n, mode="training"):
    """Build GRPO dataset with rewards and feedback"""
    grpo = []

    for _ in tqdm(range(n), desc="Building GRPO"):
        # Use REST samples for GRPO
        sample = build_rest_sample(mode)

        # Calculate reward based on validation and completeness
        base_reward = 0.9
        is_valid = sample["context"].get("schema_valid", True)

        if not is_valid:
            base_reward = 0.7
            feedback = "Schema validation failed - check payload structure"
        elif "Authorization" not in sample["output"]["headers"]:
            base_reward = 0.4
            feedback = "Missing OAuth 2.0 authentication header"
        elif sample["output"]["transport"]["tls_version"] != "1.3":
            base_reward = 0.5
            feedback = "TLS version must be 1.3 per O-RAN security requirements"
        elif (
            sample["output"]["method"] in ["POST", "PUT", "PATCH"]
            and "body" not in sample["output"]
        ):
            base_reward = 0.6
            feedback = "Missing request body for mutating operation"
        else:
            feedback = "Valid O-RAN R1 API request - fully compliant"

        # Add randomness
        reward = round(min(1.0, max(0.0, base_reward + random.uniform(-0.05, 0.05))), 2)

        grpo.append(
            {
                "prompt": sample["instruction"],
                "response": json.dumps(sample["output"], indent=2),
                "reward": reward,
                "feedback": feedback,
                "metadata": sample["context"],
            }
        )

    return grpo


# =============================================================================
# SAVE & AUDIT FUNCTIONS
# =============================================================================


def save_jsonl(data, filename):
    """Save dataset as JSONL"""
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved {len(data)} samples to {filename}")


def save_validation_audit():
    """Save validation audit to CSV"""
    if not validation_audit:
        print("⚠️  No validation audit data to save")
        return

    with open(CSV_VALIDATION_LOG, "w", newline="") as csvfile:
        fieldnames = validation_audit[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_audit)

    print(f"✅ Validation audit saved to {CSV_VALIDATION_LOG}")


def save_summary(sft, grpo):
    """Save comprehensive dataset summary"""
    # Category distribution
    category_dist = {}
    for s in sft:
        cat = s["context"].get("service_category", "other")
        category_dist[cat] = category_dist.get(cat, 0) + 1

    # Protocol distribution
    rest_count = len([s for s in sft if s["context"]["protocol_stack"] == "REST"])
    kafka_count = len([s for s in sft if s["context"]["protocol_stack"] == "Kafka"])

    # Validation stats
    total_validated = len([v for v in validation_audit if "schema_valid" in v])
    valid_count = len([v for v in validation_audit if v.get("schema_valid")])

    # GRPO reward distribution
    reward_distribution = {
        "high": len([g for g in grpo if g["reward"] >= 0.8]),
        "medium": len([g for g in grpo if 0.6 <= g["reward"] < 0.8]),
        "low": len([g for g in grpo if g["reward"] < 0.6]),
    }

    summary = {
        "generation_timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "rest_ratio": REST_RATIO,
            "kafka_ratio": KAFKA_RATIO,
            "lifecycle_ratio": LIFECYCLE_RATIO,
        },
        "dataset_sizes": {
            "sft_total": len(sft),
            "grpo_total": len(grpo),
        },
        "protocol_distribution": {
            "rest": rest_count,
            "kafka": kafka_count,
            "rest_percentage": round(rest_count / len(sft) * 100, 1),
            "kafka_percentage": round(kafka_count / len(sft) * 100, 1),
        },
        "service_category_distribution": category_dist,
        "validation_stats": {
            "total_validated": total_validated,
            "valid_count": valid_count,
            "invalid_count": total_validated - valid_count,
            "validation_rate": round(valid_count / max(total_validated, 1) * 100, 1),
        },
        "grpo_reward_distribution": reward_distribution,
        "specifications_aligned": [
            "O-RAN.WG2.TS.R1GAP-R004-v12.00 (General Aspects & Principles)",
            "O-RAN.WG2.TS.R1AP-R004-v9.00 (Application Protocols)",
            "O-RAN.WG2.TS.R1TP-R004-v04.03 (Transport Protocols)",
        ],
        "service_coverage": {
            "total_categories": len(REST_APIS),
            "categories": list(REST_APIS.keys()),
        },
    }

    with open(JSON_SUMMARY_LOG, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to {JSON_SUMMARY_LOG}")
    print(f"\n📊 Statistics:")
    print(f"   - Service Categories: {len(category_dist)}")
    print(f"   - Validation Rate: {summary['validation_stats']['validation_rate']}%")
    print(
        f"   - GRPO High Rewards: {reward_distribution['high']} ({reward_distribution['high']/len(grpo)*100:.1f}%)"
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("O-RAN R1 Dataset Builder - PRODUCTION (Modular)")
    print("=" * 80)
    print(f"Target: {DATASET_SIZE_SFT} SFT + {DATASET_SIZE_GRPO} GRPO samples")
    print(
        f"Distribution: REST {REST_RATIO*100:.0f}%, Kafka {KAFKA_RATIO*100:.0f}%, Lifecycle {LIFECYCLE_RATIO*100:.0f}%"
    )
    print("=" * 80 + "\n")

    # Generate datasets
    sft = build_sft_dataset(DATASET_SIZE_SFT)
    grpo = build_grpo_dataset(DATASET_SIZE_GRPO)

    # Save datasets
    print("\n📝 Saving datasets...")
    sft_file = os.path.join(OUTPUT_DIR, "r1_sft_dataset_modular.jsonl")
    grpo_file = os.path.join(OUTPUT_DIR, "r1_grpo_dataset_modular.jsonl")
    save_jsonl(sft, sft_file)
    save_jsonl(grpo, grpo_file)

    # Save audit logs
    print("\n📊 Generating audit logs...")
    save_validation_audit()
    save_summary(sft, grpo)

    # Print sample
    print("\n" + "=" * 80)
    print("Sample SFT Entry (REST):")
    print("=" * 80)
    rest_samples = [s for s in sft if s["context"]["protocol_stack"] == "REST"]
    if rest_samples:
        print(json.dumps(rest_samples[0], indent=2))

    print("\n" + "=" * 80)
    print("Sample SFT Entry (Kafka):")
    print("=" * 80)
    kafka_samples = [s for s in sft if s["context"]["protocol_stack"] == "Kafka"]
    if kafka_samples:
        print(json.dumps(kafka_samples[0], indent=2))

    print("\n" + "=" * 80)
    print("✅ COMPLETE - All datasets and audit logs generated!")
    print("=" * 80)
    print(f"\nGenerated Files:")
    print(f"  📄 {sft_file} ({len(sft)} samples)")
    print(f"  📄 {grpo_file} ({len(grpo)} samples)")
    print(f"  📄 {CSV_VALIDATION_LOG} (validation audit)")
    print(f"  📄 {JSON_SUMMARY_LOG} (comprehensive summary)")
    print(f"\nConfiguration Files Used:")
    print(f"  ⚙️  configs/r1_rest_apis.yaml")
    print(f"  ⚙️  configs/r1_kafka_topics.yaml")
    print(f"  ⚙️  configs/r1_schemas.yaml")
    print(f"  ⚙️  configs/r1_intents.yaml")
    print("\n🎉 Dataset generation complete!")
