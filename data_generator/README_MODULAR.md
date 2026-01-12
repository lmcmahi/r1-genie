# O-RAN R1 Dataset Builder - Production Modular System

## 🎯 Overview

This is a **production-grade modular system** for generating O-RAN R1 interface training datasets. All API definitions, schemas, and intents are externalized into YAML configuration files for easy maintenance and updates.

## 📋 Specification Alignment

✅ **O-RAN.WG2.TS.R1GAP-R004-v12.00** - General Aspects & Principles (ALL 7 sections)  
✅ **O-RAN.WG2.TS.R1AP-R004-v9.00** - Application Protocols  
✅ **O-RAN.WG2.TS.R1TP-R004-v04.03** - Transport Protocols  

## 🏗️ Architecture

```
r1_dataset_builder_modular.py    # Main orchestration script
│
├── configs/
│   ├── r1_rest_apis.yaml        # 70+ REST API definitions
│   ├── r1_kafka_topics.yaml     # 8 Kafka streaming topics
│   ├── r1_schemas.yaml          # JSON Schema definitions
│   └── r1_intents.yaml          # Natural language variations
│
└── outputs/
    ├── r1_sft_dataset_modular.jsonl         # SFT training data
    ├── r1_grpo_dataset_modular.jsonl        # GRPO reward data
    ├── r1_validation_audit_modular.csv      # Validation audit log
    └── r1_dataset_summary_modular.json      # Statistics summary
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pyyaml jsonschema tqdm
```

### 2. Run Dataset Generation

```bash
python r1_dataset_builder_modular.py
```

### 3. Output

The script generates 4 files:
- **r1_sft_dataset_modular.jsonl** (5000 samples)
- **r1_grpo_dataset_modular.jsonl** (1500 samples)
- **r1_validation_audit_modular.csv** (validation log)
- **r1_dataset_summary_modular.json** (stats)

## 📊 Dataset Composition

| Component | Percentage | Sample Count |
|-----------|-----------|--------------|
| REST APIs | 65% | 3,250 |
| Kafka Streaming | 30% | 1,500 |
| Lifecycle (Bootstrap/rApp) | 5% | 250 |

## 📁 Configuration Files

### 1. `r1_rest_apis.yaml`

Defines all REST API endpoints organized by R1GAP sections:

```yaml
# Section 5.1: Service Management
service_registration:
  - name: "Register API service"
    uri: "{apiRoot}/published-apis/v1/{apfId}/service-apis"
    method: "POST"
    example_payload:
      apiName: "rApp-TrafficOptimization"
      apiVersion: "v1"
      protocol: "HTTP_2"
```

**Coverage:**
- ✅ Section 5.1: Service Management (Bootstrap, Registration, Discovery)
- ✅ Section 5.2: Data Management (Registration, Discovery, Request, Subscription, Offer)
- ✅ Section 5.3: A1 Services (Policy Management, Enrichment Info)
- ✅ Section 5.4: RAN OAM (Network Info, FM, PM, Config Management)
- ✅ Section 5.5: O2 Services (Infrastructure, Deployment)
- ✅ Section 5.6: AI/ML Workflow (8 sub-categories)
- ✅ Section 5.7: rApp Management

**Total: 70+ API operations**

### 2. `r1_kafka_topics.yaml`

Defines Kafka streaming topics per R1TP:

```yaml
ue_telemetry:
  - description: "Stream UE measurements"
    topic: "oran.ue-measurements.v1"
    operation: "PRODUCE"
    example:
      timestamp: "2024-01-15T10:30:00Z"
      ueId: "imsi-310150123456789"
      measurements:
        rsrp: -85
        throughput: 50000
```

**Topics:**
- ue_telemetry (UE measurements)
- cell_kpis (Cell performance metrics)
- alarms (Fault notifications)
- policy_notifications (A1 policy updates)
- model_performance (AI/ML metrics)
- enrichment_data (EI data)
- data_delivery (Subscribed data streams)

### 3. `r1_schemas.yaml`

JSON Schema definitions for payload validation:

```yaml
a1_policy_management:
  type: "object"
  properties:
    policyTypeId:
      type: "string"
    policyId:
      type: "string"
    policyData:
      type: "object"
  required: ["policyTypeId", "policyId", "policyData"]
```

**Schemas for:**
- Service registration
- Data management (registration, offer, subscription)
- A1 policy management
- AI/ML model registration & training
- rApp registration
- Configuration management

### 4. `r1_intents.yaml`

Natural language variations for each operation:

```yaml
a1_policy_management:
  POST:
    - "Create QoS policy for enterprise slice"
    - "Define traffic steering rules"
    - "Set up congestion control policy"
```

## 🔧 Customization

### Modify Dataset Size

Edit `r1_dataset_builder_modular.py`:

```python
DATASET_SIZE_SFT = 5000   # Change SFT sample count
DATASET_SIZE_GRPO = 1500  # Change GRPO sample count
```

### Adjust Protocol Distribution

```python
REST_RATIO = 0.65    # 65% REST APIs
KAFKA_RATIO = 0.30   # 30% Kafka streaming
LIFECYCLE_RATIO = 0.05  # 5% Lifecycle operations
```

### Add New API Operations

Edit `configs/r1_rest_apis.yaml`:

```yaml
my_new_service:
  - name: "My new operation"
    uri: "{apiRoot}/my-service/v1/resource"
    method: "POST"
    example_payload:
      field1: "value1"
```

### Add New Intent Variations

Edit `configs/r1_intents.yaml`:

```yaml
my_new_service:
  POST:
    - "Natural language instruction 1"
    - "Natural language instruction 2"
```

## 📈 Sample Output

### SFT Sample (REST)

```json
{
  "instruction": "Create QoS policy for enterprise slice",
  "context": {
    "protocol_stack": "REST",
    "service_category": "a1_policy_management",
    "schema_valid": true
  },
  "output": {
    "protocol": "HTTPS",
    "method": "POST",
    "uri": "https://smo.example.com/a1-policy-management/v1/policies",
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "Bearer abc123..."
    },
    "body": {
      "policyTypeId": "QoS-policy",
      "policyId": "policy-001",
      "policyData": {
        "scope": {
          "cellIdList": ["cell-1", "cell-2"]
        }
      }
    }
  }
}
```

### SFT Sample (Kafka)

```json
{
  "instruction": "Stream UE measurements to telemetry topic",
  "context": {
    "protocol_stack": "Kafka",
    "topic_category": "ue_telemetry",
    "operation": "PRODUCE"
  },
  "output": {
    "protocol": "Kafka",
    "operation": "PRODUCE",
    "topic": "oran.ue-measurements.v1",
    "value": {
      "timestamp": "2024-01-15T10:30:00Z",
      "ueId": "imsi-310150123456789",
      "cellId": "NRCell-100",
      "measurements": {
        "rsrp": -85,
        "throughput": 50000
      }
    }
  }
}
```

### GRPO Sample

```json
{
  "prompt": "Register my rApp instance",
  "response": "{\n  \"protocol\": \"HTTPS\",\n  \"method\": \"POST\",\n  ...\n}",
  "reward": 0.92,
  "feedback": "Valid O-RAN R1 API request - fully compliant",
  "metadata": {
    "protocol_stack": "REST",
    "service_category": "rapp_registration",
    "schema_valid": true
  }
}
```

## 📊 Audit & Logging

### Validation Audit (CSV)

Tracks every schema validation attempt:

```csv
category,method,schema_valid,validation_msg,timestamp
a1_policy_management,POST,True,Schema valid,2024-01-15T10:30:00
data_registration,POST,False,Schema error: 'namespace' is required,2024-01-15T10:30:01
```

### Summary Report (JSON)

Comprehensive statistics:

```json
{
  "generation_timestamp": "2024-01-15T10:30:00Z",
  "dataset_sizes": {
    "sft_total": 5000,
    "grpo_total": 1500
  },
  "validation_stats": {
    "validation_rate": 95.2
  },
  "service_coverage": {
    "total_categories": 24,
    "categories": [...]
  }
}
```

## 🎯 Key Features

### 1. **Complete R1GAP Coverage**
- All 7 service categories from R1GAP Sections 5.1-5.7
- 70+ REST API operations
- 8 Kafka streaming topics

### 2. **Modular Design**
- Easy to add new services
- Update intents without code changes
- Externalized schemas for validation

### 3. **Production Quality**
- JSON Schema validation
- Comprehensive audit logging
- Realistic payloads with proper O-RAN terminology

### 4. **Spec Compliance**
- OAuth 2.0 authentication
- TLS 1.3 transport security
- Proper HTTP methods per R1AP
- Kafka SASL_SSL per R1TP

## 🔍 Validation

The system validates all payloads against OpenAPI-derived schemas:

- ✅ **Required fields** checked
- ✅ **Data types** validated
- ✅ **Enum values** constrained
- ✅ **Nested objects** validated

Invalid samples receive lower GRPO rewards with specific feedback.

## 📝 Use Cases

### 1. Training LLMs for O-RAN R1 Interface
```python
# Use SFT dataset for supervised fine-tuning
dataset = load_dataset("r1_sft_dataset_modular.jsonl")
```

### 2. Reinforcement Learning with Rewards
```python
# Use GRPO dataset for reward-based training
dataset = load_dataset("r1_grpo_dataset_modular.jsonl")
```

### 3. Testing O-RAN R1 Implementations
```python
# Extract test cases from SFT samples
test_cases = [sample["output"] for sample in sft_data]
```

### 4. Documentation Generation
```python
# Generate API documentation from YAML configs
docs = generate_docs(REST_APIS, INTENTS)
```

## 🛠️ Maintenance

### Updating to New O-RAN Specifications

1. **Update `r1_rest_apis.yaml`**: Add new endpoints
2. **Update `r1_schemas.yaml`**: Add new payload schemas
3. **Update `r1_intents.yaml`**: Add natural language variations
4. **Update `r1_kafka_topics.yaml`**: Add new streaming topics

No Python code changes needed!

### Version Control

Track specification versions in YAML comments:

```yaml
# Updated per O-RAN.WG2.TS.R1GAP-R004-v13.00 (2025-03-01)
```

## 📚 References

- [O-RAN R1GAP Specification](https://orandownloadsweb.azurewebsites.net/specifications)
- [O-RAN R1AP Specification](https://orandownloadsweb.azurewebsites.net/specifications)
- [O-RAN R1TP Specification](https://orandownloadsweb.azurewebsites.net/specifications)

## 🤝 Contributing

To add new services:

1. Update appropriate YAML config
2. Run validation: `python validate_configs.py`
3. Regenerate dataset: `python r1_dataset_builder_modular.py`
4. Verify output in summary JSON

## 📄 License

Aligned with O-RAN ALLIANCE specifications. Check O-RAN license terms.

## ✅ Verification Checklist

- [x] All 7 R1GAP sections covered
- [x] JSON Schema validation enabled
- [x] Natural language intents included
- [x] Kafka streaming support
- [x] Audit logging (CSV + JSON)
- [x] OAuth 2.0 + TLS 1.3 security
- [x] Realistic O-RAN terminology
- [x] Modular YAML configuration
- [x] Production-ready code quality

---

**Version:** 1.0  
**Last Updated:** 2024-01-15  
**Specification Versions:** R1GAP v12.00, R1AP v9.00, R1TP v04.03
