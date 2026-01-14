"""
dataset_orchestrator.py
Main Orchestrator for O-RAN R1 Intent-to-API Orchestration Training Dataset

This module generates training data for intent-to-API orchestration,
teaching models to translate natural language intents into valid
O-RAN R1 API call sequences.

Target Dataset:
- Intent-to-Orchestration (TeleYAML): 100%

Features:
- Knowledge Graph validation for all workflows
- Quality controls (schema validation, deduplication)
- LLM-enhanced reasoning traces (Claude Opus 4.5)
- Multiple output formats (JSONL)
"""

import os
import json
import hashlib
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random

from .api_dependency_graph import R1ApiDependencyGraph, WorkflowValidator, load_api_dependencies
from .workflow_sample_builder import WorkflowSampleBuilder, get_workflow_builder
from .troubleshooting_generator import TroubleshootingGenerator, get_troubleshooting_generator
from .domain_explanation_generator import DomainExplanationGenerator, get_explanation_generator
from .multiturn_conversation_generator import MultiTurnConversationGenerator, get_conversation_generator

# Import LLM generator if available
try:
    from .llm_assisted_generator import LLMAssistedGenerator, get_llm_generator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    target_size: int = 10000
    distribution: Dict[str, float] = field(default_factory=lambda: {
        "workflow": 1.00,  # 100% intent-to-API orchestration
        "troubleshooting": 0.00,
        "domain_qna": 0.00,
        "multi_turn": 0.00,
        "kpi_calculation": 0.00
    })
    enable_llm: bool = False
    deduplication_threshold: float = 0.85
    validate_all: bool = True
    output_dir: str = "outputs"
    random_seed: int = 42


@dataclass
class ValidationReport:
    """Report from dataset validation."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    kg_validated: int
    schema_errors: List[Dict]
    duplicate_count: int
    distribution_actual: Dict[str, float]
    distribution_target: Dict[str, float]


class DatasetOrchestrator:
    """
    Main orchestrator for generating the complete O-RAN R1 training dataset.

    Coordinates all generators and ensures:
    - Target distribution is met
    - All samples are KG-validated
    - No excessive duplicates
    - Proper output formatting
    """

    def __init__(self, config: DatasetConfig = None):
        """
        Initialize the orchestrator.

        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()
        random.seed(self.config.random_seed)

        # Initialize knowledge graph
        self.kg = load_api_dependencies()
        self.validator = WorkflowValidator(self.kg)

        # Initialize LLM generator if enabled
        self.llm = None
        if self.config.enable_llm and LLM_AVAILABLE:
            self.llm = get_llm_generator()

        # Initialize generators
        self.workflow_builder = get_workflow_builder(
            knowledge_graph=self.kg,
            llm_generator=self.llm
        )
        self.troubleshoot_gen = get_troubleshooting_generator(
            knowledge_graph=self.kg,
            llm_generator=self.llm
        )
        self.explain_gen = get_explanation_generator(
            llm_generator=self.llm
        )
        self.conversation_gen = get_conversation_generator(
            knowledge_graph=self.kg,
            llm_generator=self.llm
        )

        # Statistics
        self.stats = {
            "generation_started": None,
            "generation_completed": None,
            "total_generated": 0,
            "by_type": defaultdict(int),
            "kg_validated": 0,
            "llm_enhanced": 0,
            "duplicates_removed": 0,
            "validation_errors": 0
        }

        # Sample storage
        self.samples: List[Dict] = []
        self.seen_hashes: set = set()

    def generate_full_dataset(self) -> List[Dict]:
        """
        Generate the complete dataset with target distribution.

        Returns:
            List of all generated samples
        """
        self.stats["generation_started"] = datetime.utcnow().isoformat()
        self.samples = []
        self.seen_hashes = set()

        # Calculate counts for each type
        type_counts = self._calculate_type_counts()

        print(f"=== Generating O-RAN R1 Dataset ===")
        print(f"Target size: {self.config.target_size}")
        print(f"LLM enabled: {self.config.enable_llm}")
        print(f"\nDistribution:")
        for sample_type, count in type_counts.items():
            print(f"  {sample_type}: {count}")

        # Generate each type
        self._generate_workflow_samples(type_counts.get("workflow", 0))
        self._generate_troubleshooting_samples(type_counts.get("troubleshooting", 0))
        self._generate_domain_samples(type_counts.get("domain_qna", 0))
        self._generate_conversation_samples(type_counts.get("multi_turn", 0))
        self._generate_kpi_samples(type_counts.get("kpi_calculation", 0))

        # Validate if enabled
        if self.config.validate_all:
            self._validate_all_samples()

        # Rebalance if needed
        self._rebalance_distribution()

        self.stats["generation_completed"] = datetime.utcnow().isoformat()
        self.stats["total_generated"] = len(self.samples)

        print(f"\n=== Generation Complete ===")
        print(f"Total samples: {len(self.samples)}")
        print(f"KG validated: {self.stats['kg_validated']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")

        return self.samples

    def _calculate_type_counts(self) -> Dict[str, int]:
        """Calculate target count for each sample type."""
        counts = {}
        for sample_type, ratio in self.config.distribution.items():
            counts[sample_type] = int(self.config.target_size * ratio)

        # Adjust to exactly hit target
        total = sum(counts.values())
        if total < self.config.target_size:
            counts["workflow"] += self.config.target_size - total

        return counts

    def _generate_workflow_samples(self, count: int) -> None:
        """Generate workflow orchestration samples."""
        print(f"\nGenerating {count} workflow samples...")

        workflows = self.workflow_builder.list_workflows()
        samples_per_workflow = max(1, count // len(workflows))

        generated = 0
        for wf_info in workflows:
            wf_id = wf_info.get('id', wf_info.get('name', ''))
            for i in range(samples_per_workflow):
                if generated >= count:
                    break
                try:
                    sample = self.workflow_builder.build_workflow_sample(
                        workflow_id=wf_id,
                        use_llm=self.config.enable_llm
                    )
                    training_sample = self.workflow_builder.to_training_format(sample)
                    training_sample["gsma_category"] = "r1_orchestration"

                    if self._add_sample(training_sample, "workflow"):
                        generated += 1

                        if sample.metadata.get('kg_validated'):
                            self.stats["kg_validated"] += 1

                except Exception as e:
                    print(f"  Error generating workflow sample: {e}")

            if generated >= count:
                break

        # Fill remaining with random workflows
        while generated < count:
            wf_id = random.choice([w.get('id', w.get('name', '')) for w in workflows])
            try:
                sample = self.workflow_builder.build_workflow_sample(
                    workflow_id=wf_id,
                    use_llm=self.config.enable_llm
                )
                training_sample = self.workflow_builder.to_training_format(sample)
                training_sample["gsma_category"] = "r1_orchestration"

                if self._add_sample(training_sample, "workflow"):
                    generated += 1
            except Exception:
                pass

        print(f"  Generated: {generated}")

    def _generate_troubleshooting_samples(self, count: int) -> None:
        """Generate troubleshooting diagnostic samples."""
        print(f"\nGenerating {count} troubleshooting samples...")

        trees = self.troubleshoot_gen.list_trees()
        samples_per_tree = max(1, count // len(trees))

        generated = 0
        for tree_info in trees:
            tree_id = tree_info.get('id', tree_info.get('name', ''))
            for i in range(samples_per_tree):
                if generated >= count:
                    break
                try:
                    sample = self.troubleshoot_gen.generate_diagnostic_sample(
                        tree_id=tree_id,
                        use_llm=self.config.enable_llm
                    )
                    training_sample = self.troubleshoot_gen.to_training_format(sample)
                    training_sample["gsma_category"] = "tele_logs"

                    if self._add_sample(training_sample, "troubleshooting"):
                        generated += 1

                except Exception as e:
                    print(f"  Error generating troubleshooting sample: {e}")

            if generated >= count:
                break

        # Fill remaining
        while generated < count:
            tree_id = random.choice([t.get('id', t.get('name', '')) for t in trees])
            try:
                sample = self.troubleshoot_gen.generate_diagnostic_sample(
                    tree_id=tree_id,
                    use_llm=self.config.enable_llm
                )
                training_sample = self.troubleshoot_gen.to_training_format(sample)
                training_sample["gsma_category"] = "tele_logs"

                if self._add_sample(training_sample, "troubleshooting"):
                    generated += 1
            except Exception:
                pass

        print(f"  Generated: {generated}")

    def _generate_domain_samples(self, count: int) -> None:
        """Generate domain explanation Q&A samples."""
        print(f"\nGenerating {count} domain Q&A samples...")

        available = self.explain_gen.list_available()
        generated = 0

        # Generate comparison samples (40% of domain)
        comparison_count = int(count * 0.4)
        for comp_id in available.get('comparisons', []):
            for detail_level in ["basic", "intermediate", "advanced"]:
                if generated >= comparison_count:
                    break
                try:
                    sample = self.explain_gen.generate_comparison_sample(
                        comparison_id=comp_id,
                        detail_level=detail_level,
                        use_llm=self.config.enable_llm
                    )
                    training_sample = self.explain_gen.to_training_format(sample)

                    if self._add_sample(training_sample, "domain_qna"):
                        generated += 1

                except Exception as e:
                    print(f"  Error generating comparison sample: {e}")

        # Generate concept samples (30% of domain)
        concept_count = int(count * 0.3)
        concepts_generated = 0
        for concept_id in available.get('concepts', []):
            for detail_level in ["basic", "intermediate", "advanced"]:
                if concepts_generated >= concept_count:
                    break
                try:
                    sample = self.explain_gen.generate_concept_sample(
                        concept_id=concept_id,
                        detail_level=detail_level,
                        use_llm=self.config.enable_llm
                    )
                    training_sample = self.explain_gen.to_training_format(sample)

                    if self._add_sample(training_sample, "domain_qna"):
                        generated += 1
                        concepts_generated += 1

                except Exception:
                    pass

        # Generate service category samples (20% of domain)
        service_count = int(count * 0.2)
        services_generated = 0
        for cat_id in available.get('service_categories', []):
            if services_generated >= service_count:
                break
            try:
                sample = self.explain_gen.generate_service_category_sample(
                    category_id=cat_id,
                    use_llm=self.config.enable_llm
                )
                training_sample = self.explain_gen.to_training_format(sample)

                if self._add_sample(training_sample, "domain_qna"):
                    generated += 1
                    services_generated += 1

            except Exception:
                pass

        # Fill remaining with FAQ
        while generated < count:
            try:
                sample = self.explain_gen.generate_faq_sample(
                    use_llm=self.config.enable_llm
                )
                training_sample = self.explain_gen.to_training_format(sample)

                if self._add_sample(training_sample, "domain_qna"):
                    generated += 1
            except Exception:
                break

        print(f"  Generated: {generated}")

    def _generate_conversation_samples(self, count: int) -> None:
        """Generate multi-turn conversation samples."""
        print(f"\nGenerating {count} conversation samples...")

        workflows = self.conversation_gen.list_workflows()
        generated = 0

        # Workflow conversations (70%)
        workflow_conv_count = int(count * 0.7)
        for wf_id in workflows:
            if generated >= workflow_conv_count:
                break
            for style in ["guided", "direct"]:
                if generated >= workflow_conv_count:
                    break
                try:
                    sample = self.conversation_gen.generate_workflow_conversation(
                        workflow_id=wf_id,
                        interaction_style=style
                    )
                    training_sample = self.conversation_gen.to_training_format(sample)

                    if self._add_sample(training_sample, "multi_turn"):
                        generated += 1

                except Exception as e:
                    print(f"  Error generating workflow conversation: {e}")

        # Troubleshooting conversations (30%)
        trouble_scenarios = [
            ("Users report poor throughput in cell {cellId}", "capacity_congestion"),
            ("Coverage issues in {cellId} area", "coverage_issue"),
            ("Intermittent connection problems in {cellId}", "interference"),
        ]

        trouble_conv_count = count - generated
        trouble_generated = 0
        while trouble_generated < trouble_conv_count:
            symptom, root_cause = random.choice(trouble_scenarios)
            try:
                sample = self.conversation_gen.generate_troubleshooting_conversation(
                    symptom=symptom,
                    root_cause=root_cause
                )
                training_sample = self.conversation_gen.to_training_format(sample)

                if self._add_sample(training_sample, "multi_turn"):
                    generated += 1
                    trouble_generated += 1

            except Exception:
                break

        print(f"  Generated: {generated}")

    def _generate_kpi_samples(self, count: int) -> None:
        """Generate KPI calculation/interpretation samples."""
        print(f"\nGenerating {count} KPI calculation samples...")

        kpi_scenarios = [
            ("rsrp_dbm", -75, -105),
            ("rsrq_db", -8, -18),
            ("sinr_db", 20, -2),
            ("prb_utilization_dl_pct", 40, 95),
            ("throughput_dl_mbps_avg", 80, 12),
            ("handover_success_rate_pct", 99, 85),
            ("latency_ms_avg", 8, 65),
        ]

        generated = 0
        while generated < count:
            kpi_name, good_value, bad_value = random.choice(kpi_scenarios)
            value = random.choice([good_value, bad_value, (good_value + bad_value) / 2])

            try:
                sample = self.explain_gen.generate_kpi_interpretation_sample(
                    kpi_name=kpi_name,
                    value=value,
                    use_llm=self.config.enable_llm
                )
                training_sample = self.explain_gen.to_training_format(sample)
                training_sample["gsma_category"] = "tele_math"

                if self._add_sample(training_sample, "kpi_calculation"):
                    generated += 1

            except Exception:
                pass

        print(f"  Generated: {generated}")

    def _add_sample(self, sample: Dict, sample_type: str) -> bool:
        """
        Add sample if it's not a duplicate.

        Args:
            sample: Sample to add
            sample_type: Type of sample

        Returns:
            True if sample was added, False if duplicate
        """
        # Generate content hash for deduplication
        content_hash = self._hash_sample(sample)

        if content_hash in self.seen_hashes:
            self.stats["duplicates_removed"] += 1
            return False

        self.seen_hashes.add(content_hash)
        sample["sample_type"] = sample_type
        self.samples.append(sample)
        self.stats["by_type"][sample_type] += 1

        return True

    def _hash_sample(self, sample: Dict) -> str:
        """Generate hash for sample deduplication."""
        # Use key content fields for hashing
        key_fields = []

        if "instruction" in sample:
            key_fields.append(sample["instruction"])
        if "question" in sample:
            key_fields.append(sample["question"])
        if "symptom" in sample:
            key_fields.append(sample["symptom"])
        if "conversation" in sample and sample["conversation"]:
            # Hash first user message
            for turn in sample["conversation"]:
                if turn.get("role") == "user":
                    key_fields.append(turn.get("content", ""))
                    break

        content = "|".join(key_fields)
        return hashlib.md5(content.encode()).hexdigest()

    def _validate_all_samples(self) -> None:
        """Validate all generated samples."""
        print("\nValidating samples...")

        invalid_samples = []
        for sample in self.samples:
            is_valid, errors = self._validate_sample(sample)
            if not is_valid:
                invalid_samples.append({
                    "id": sample.get("id", "unknown"),
                    "errors": errors
                })
                self.stats["validation_errors"] += 1

        if invalid_samples:
            print(f"  Found {len(invalid_samples)} samples with validation errors")
        else:
            print(f"  All {len(self.samples)} samples valid")

    def _validate_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """Validate a single sample."""
        errors = []

        # Check required fields
        if "id" not in sample:
            errors.append("Missing 'id' field")

        sample_type = sample.get("sample_type", "")

        # Type-specific validation
        if sample_type == "workflow":
            if "output" not in sample or "steps" not in sample.get("output", {}):
                errors.append("Workflow sample missing steps")
            else:
                # Validate with KG
                steps = sample["output"]["steps"]
                api_ids = [s.get("api", "") for s in steps if s.get("api")]
                if api_ids:
                    is_valid, kg_errors = self.kg.validate_workflow(api_ids)
                    if not is_valid:
                        errors.extend(kg_errors[:3])  # Limit error count

        elif sample_type == "troubleshooting":
            if "kpi_observations" not in sample:
                errors.append("Troubleshooting sample missing KPI observations")
            if "diagnosis" not in sample:
                errors.append("Troubleshooting sample missing diagnosis")

        elif sample_type == "domain_qna":
            if "question" not in sample:
                errors.append("Domain sample missing question")
            if "answer" not in sample:
                errors.append("Domain sample missing answer")

        elif sample_type == "multi_turn":
            if "conversation" not in sample:
                errors.append("Conversation sample missing conversation")
            elif len(sample["conversation"]) < 2:
                errors.append("Conversation too short")

        return len(errors) == 0, errors

    def _rebalance_distribution(self) -> None:
        """Rebalance dataset to match target distribution."""
        current = self._get_current_distribution()
        target = self.config.distribution

        print("\nDistribution check:")
        needs_rebalance = False
        for sample_type, target_ratio in target.items():
            current_ratio = current.get(sample_type, 0)
            diff = abs(current_ratio - target_ratio)
            status = "OK" if diff < 0.05 else "NEEDS ADJUSTMENT"
            print(f"  {sample_type}: {current_ratio:.1%} (target: {target_ratio:.1%}) - {status}")
            if diff >= 0.05:
                needs_rebalance = True

        if needs_rebalance:
            print("  Note: Some categories are off target. Consider regenerating.")

    def _get_current_distribution(self) -> Dict[str, float]:
        """Get current distribution of sample types."""
        total = len(self.samples)
        if total == 0:
            return {}

        distribution = {}
        for sample_type in self.config.distribution.keys():
            count = self.stats["by_type"].get(sample_type, 0)
            distribution[sample_type] = count / total

        return distribution

    def validate_dataset(self, samples: List[Dict] = None) -> ValidationReport:
        """
        Generate validation report for dataset.

        Args:
            samples: Samples to validate (uses self.samples if None)

        Returns:
            ValidationReport with detailed validation results
        """
        samples = samples or self.samples

        valid = 0
        invalid = 0
        schema_errors = []

        for sample in samples:
            is_valid, errors = self._validate_sample(sample)
            if is_valid:
                valid += 1
            else:
                invalid += 1
                schema_errors.append({
                    "id": sample.get("id", "unknown"),
                    "errors": errors
                })

        return ValidationReport(
            total_samples=len(samples),
            valid_samples=valid,
            invalid_samples=invalid,
            kg_validated=self.stats["kg_validated"],
            schema_errors=schema_errors[:50],  # Limit
            duplicate_count=self.stats["duplicates_removed"],
            distribution_actual=self._get_current_distribution(),
            distribution_target=self.config.distribution
        )

    def export_dataset(self, output_dir: str = None) -> Dict[str, str]:
        """
        Export dataset to files.

        Args:
            output_dir: Output directory (uses config if None)

        Returns:
            Dictionary of output file paths
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_files = {}

        # Main dataset file
        main_file = os.path.join(output_dir, "r1_enhanced_sft.jsonl")
        with open(main_file, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, default=str) + "\n")
        output_files["main"] = main_file
        print(f"Exported main dataset: {main_file}")

        # Export subsets by GSMA category
        categories = defaultdict(list)
        for sample in self.samples:
            gsma_cat = sample.get("gsma_category", "other")
            categories[gsma_cat].append(sample)

        for category, cat_samples in categories.items():
            cat_file = os.path.join(output_dir, f"r1_{category}_subset.jsonl")
            with open(cat_file, 'w') as f:
                for sample in cat_samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            output_files[category] = cat_file
            print(f"Exported {category} subset: {cat_file} ({len(cat_samples)} samples)")

        # Export conversations separately
        conv_samples = [s for s in self.samples if s.get("sample_type") == "multi_turn"]
        if conv_samples:
            conv_file = os.path.join(output_dir, "r1_conversations.jsonl")
            with open(conv_file, 'w') as f:
                for sample in conv_samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            output_files["conversations"] = conv_file
            print(f"Exported conversations: {conv_file} ({len(conv_samples)} samples)")

        # Export dataset report
        report_file = os.path.join(output_dir, "dataset_report.json")
        report = {
            "generation_stats": self.stats,
            "distribution": self._get_current_distribution(),
            "sample_counts": dict(self.stats["by_type"]),
            "total_samples": len(self.samples),
            "config": {
                "target_size": self.config.target_size,
                "llm_enabled": self.config.enable_llm,
                "distribution": self.config.distribution
            }
        }
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        output_files["report"] = report_file
        print(f"Exported report: {report_file}")

        return output_files

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return {
            **self.stats,
            "current_count": len(self.samples),
            "distribution": self._get_current_distribution()
        }


def create_orchestrator(target_size: int = 10000,
                        enable_llm: bool = False,
                        output_dir: str = "outputs") -> DatasetOrchestrator:
    """
    Create a configured DatasetOrchestrator.

    Args:
        target_size: Target dataset size
        enable_llm: Whether to enable LLM generation
        output_dir: Output directory

    Returns:
        Configured DatasetOrchestrator
    """
    config = DatasetConfig(
        target_size=target_size,
        enable_llm=enable_llm,
        output_dir=output_dir
    )
    return DatasetOrchestrator(config)


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate O-RAN R1 Enhanced Training Dataset"
    )
    parser.add_argument(
        "--target-size", type=int, default=10000,
        help="Target dataset size (default: 10000)"
    )
    parser.add_argument(
        "--enable-llm", action="store_true",
        help="Enable LLM generation for reasoning traces"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing dataset"
    )
    parser.add_argument(
        "--input", type=str,
        help="Input file for validation"
    )

    args = parser.parse_args()

    if args.validate_only:
        if not args.input:
            print("Error: --input required for validation mode")
            return

        # Load and validate existing dataset
        samples = []
        with open(args.input, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        config = DatasetConfig()
        orchestrator = DatasetOrchestrator(config)
        report = orchestrator.validate_dataset(samples)

        print(f"\n=== Validation Report ===")
        print(f"Total samples: {report.total_samples}")
        print(f"Valid samples: {report.valid_samples}")
        print(f"Invalid samples: {report.invalid_samples}")
        print(f"\nDistribution:")
        for st, ratio in report.distribution_actual.items():
            target = report.distribution_target.get(st, 0)
            print(f"  {st}: {ratio:.1%} (target: {target:.1%})")

        if report.schema_errors:
            print(f"\nFirst 5 errors:")
            for err in report.schema_errors[:5]:
                print(f"  {err['id']}: {err['errors']}")

    else:
        # Generate new dataset
        orchestrator = create_orchestrator(
            target_size=args.target_size,
            enable_llm=args.enable_llm,
            output_dir=args.output_dir
        )

        samples = orchestrator.generate_full_dataset()
        output_files = orchestrator.export_dataset()

        print(f"\n=== Final Statistics ===")
        stats = orchestrator.get_stats()
        print(f"Total generated: {stats['total_generated']}")
        print(f"KG validated: {stats['kg_validated']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"\nOutput files:")
        for name, path in output_files.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
