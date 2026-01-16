# R1-Genie: O-RAN R1 Training Dataset Generator

A API Dependency Graph-guided dataset generator for training domain-specialized LLMs on O-RAN R1 interface operations.

## The Problem

Current general-purpose LLMs perform poorly on telecom-specific tasks:

| Benchmark | GPT-4/5 Performance | Task |
|-----------|---------------------|------|
| **TeleYAML** | ~27% | Intent → API configuration |
| **3GPP Standards** | <40% | Standards comprehension |
| **TeleQnA** | <75% | Domain knowledge Q&A |

Telecom networks require **precise API orchestration** - a wrong API call sequence can cause network outages. General LLMs lack the domain knowledge to reliably:
- Chain multiple APIs in valid dependency order
- Understand which data flows between API calls
- Diagnose network issues from KPI observations

## The Solution

R1-Genie generates **high-quality training data** to fine-tune a compact model (7B-13B) that can:

1. **Understand O-RAN R1 intents** - Parse natural language requests into API operations
2. **Orchestrate multi-step workflows** - Chain APIs with correct dependencies and data flow
3. **Reason about diagnostics** - Analyze KPIs to diagnose network issues
4. **Work offline at the edge** - No external API calls needed at inference time

### Key Innovation: API Dependency Graph at Training Time

```
TRAINING (with API Dependency Graph):
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ NL Intents  │ ──► │ API Dependency   │ ──► │ Valid Training  │
│             │     │ Graph Validates  │     │ Samples         │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │ Fine-tune Model │
                                             └─────────────────┘

INFERENCE (without API Dependency Graph):
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ User Intent │ ──► │ Fine-tuned Model │ ──► │ Valid API       │
│             │     │ (Graph knowledge │     │ Orchestration   │
│             │     │  in weights)     │     │                 │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Generate dataset (basic mode - single API samples)
cd data_generator
python r1_dataset_builder_v3.py

# Generate enhanced dataset (multi-step workflows with dependency validation)
python r1_dataset_builder_v3.py --enhanced --target-size 1000
```

## Interactive Tutorial

R1-Genie includes a **Marimo notebook** for hands-on learning of O-RAN R1 concepts:

```bash
# Install marimo
pip install marimo requests pandas plotly

# Run the tutorial
marimo edit r1_rapp_tutorial.py
```

The tutorial covers:
- O-RAN Architecture (SMO, Near-RT RIC, rApps)
- R1 Interface and Bootstrap Pattern
- Data Management & Exposure (DME) via ICS
- Building R1-compliant rApps
- A1 Policy Management
- AI/ML Model Integration

## O-RAN R1 Coverage

All 7 service categories from O-RAN R1GAP v11.00:

| Section | Category | Example Operations |
|---------|----------|-------------------|
| 5.1 | Service Management | Bootstrap, Registration, Discovery |
| 5.2 | Data Management | Subscribe to KPIs, Data offers |
| 5.3 | A1 Services | Policy management, Enrichment info |
| 5.4 | RAN OAM | Fault/Performance/Config management |
| 5.5 | O2 Services | Infrastructure, Deployment |
| 5.6 | AI/ML Workflow | Model training, Inference |
| 5.7 | rApp Management | Lifecycle management |

## GSMA Benchmark Alignment

Training data is structured to improve performance on [GSMA Open-Telco LLM Benchmarks](https://huggingface.co/blog/otellm/gsma-benchmarks-02):

| Sample Type | Target % | GSMA Benchmark | Description |
|-------------|----------|----------------|-------------|
| Intent-to-Orchestration | 30% | TeleYAML | Multi-API workflow from NL intent |
| Troubleshooting | 25% | TeleLogs | KPI-based diagnostic reasoning |
| Domain Q&A | 20% | TeleQnA | O-RAN concept explanations |
| Multi-turn Conversations | 15% | - | Interactive workflow execution |
| KPI Calculations | 10% | TeleMath | Threshold analysis |

## Project Structure

```
r1-genie/
├── data_generator/
│   ├── Core Generation
│   │   ├── r1_dataset_builder_v3.py      # Entry point
│   │   ├── api_dependency_graph.py       # API dependency validation
│   │   ├── workflow_sample_builder.py    # Multi-step workflows
│   │   ├── troubleshooting_generator.py  # Diagnostic samples
│   │   ├── domain_explanation_generator.py
│   │   └── multiturn_conversation_generator.py
│   │
│   ├── Configuration (YAML)
│   │   ├── r1_rest_apis.yaml             # 70+ API definitions (R1AP v08.00)
│   │   ├── r1_api_dependencies.yaml      # API relationships (R1AP + R1TD)
│   │   ├── r1_workflows.yaml             # Workflow templates (R1TD v04.01)
│   │   ├── r1_schemas.yaml               # Data type schemas (R1TD v04.01)
│   │   └── r1_troubleshooting_trees.yaml # Diagnostic trees
│   │
│   └── outputs/                          # Generated datasets
│
├── r1_rapp_tutorial.py                   # Interactive Marimo tutorial
├── tests/
├── ARCHITECTURE.md                       # Detailed design docs
└── pyproject.toml
```

## Sample Output

### Intent-to-Orchestration (TeleYAML-style)

```json
{
  "intent": "Set up monitoring for cell NRCell-100 to detect coverage issues",
  "reasoning": "Coverage monitoring requires RSRP, RSRQ, SINR KPIs. Need to: (1) discover available data types via ICS, (2) subscribe to coverage KPIs, (3) set up threshold alerts.",
  "orchestration": {
    "steps": [
      {"step": 1, "api": "GET /data-consumer/v1/info-types", "purpose": "Discover available DME types via ICS"},
      {"step": 2, "api": "PUT /data-consumer/v1/info-jobs/{id}", "purpose": "Subscribe to coverage KPIs"},
      {"step": 3, "api": "POST /ProvMnS/v1/AlarmList/subscriptions", "purpose": "Set up threshold alerts"}
    ],
    "dependency_validated": true
  }
}
```

### Troubleshooting (TeleLogs-style)

```json
{
  "symptom": "Users report poor video streaming quality in cell NRCell-100",
  "kpi_observations": {
    "rsrp_dbm": -82,
    "prb_utilization_dl_pct": 94,
    "throughput_dl_mbps": 28
  },
  "diagnosis": "CAPACITY_CONGESTION - PRB utilization 94% exceeds 85% threshold",
  "remediation": {
    "action": "Deploy load balancing A1 policy",
    "api_sequence": ["GET /a1-policy-management/v1/rics", "GET /a1-policy-management/v1/policy-types", "POST /a1-policy-management/v1/policies"]
  }
}
```

## Configuration

For LLM-assisted reasoning traces:

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
ENABLE_LLM=true
```

## Specification Compliance

YAML configurations are aligned with:

- **O-RAN.WG2.TS.R1GAP-R004-v11.00** - General Aspects & Principles
- **O-RAN.WG2.TS.R1AP-R004-v08.00** - Application Protocols (API endpoints, URIs)
- **O-RAN.WG2.TS.R1TP-R004-v04.03** - Transport Protocols
- **O-RAN.WG2.TS.R1TD-R004-v04.01** - Type Definitions (DME schemas, data structures)

Key R1TD v04.01 compliance features:
- `dmeTypeId` string format: `"namespace:name:version"` (e.g., `"oran:coverage-kpis:1.0.0"`)
- Data production schema: `dataSelector`, `targetSelector`, `timing`
- Standard DME types: `ORAN:RanOamPmData:1.0.0`, `ORAN:RanOamTraceMetrics:1.0.0`

## License

Aligned with O-RAN ALLIANCE specifications.
