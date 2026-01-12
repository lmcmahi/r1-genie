import sys
sys.path.insert(0, '.')
from r1_dataset_builder_v3 import load_yaml_config, build_rest_sample
import json

# Load configs
REST_APIS = load_yaml_config("r1_rest_apis.yaml")
KAFKA_TOPICS = load_yaml_config("r1_kafka_topics.yaml")
SCHEMAS = load_yaml_config("r1_schemas.yaml")
INTENTS = load_yaml_config("r1_intents.yaml")

# Test KPI-related API generation
print("=" * 80)
print("Testing KPI extraction for data_subscription (RSRP example)")
print("=" * 80)

# Find data_subscription category
if "data_subscription" in REST_APIS:
    api = REST_APIS["data_subscription"][0]  # First API in data_subscription
    sample = build_rest_sample("data_subscription", api, SCHEMAS, INTENTS)
    print(json.dumps(sample, indent=2))

print("\n" + "=" * 80)
print("Testing Kafka KPI streaming (UE telemetry with RSRP)")
print("=" * 80)

# Check Kafka topics
if "ue_telemetry" in KAFKA_TOPICS:
    topic_def = KAFKA_TOPICS["ue_telemetry"][0]
    print(json.dumps(topic_def, indent=2))

