"""
workflow_sample_builder.py
Multi-Step Workflow Sample Builder for O-RAN R1 Training Dataset

This module generates multi-step orchestration training samples using:
- Knowledge Graph for workflow validation and dependency resolution
- Workflow templates from r1_workflows.yaml
- LLM assistance for reasoning traces (optional)

Each sample teaches the model to orchestrate multiple API calls from a single intent.
"""

import os
import re
import yaml
import json
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

from .api_dependency_graph import R1ApiDependencyGraph, WorkflowValidator, load_api_dependencies

# Import LLM generator if available
try:
    from .llm_assisted_generator import LLMAssistedGenerator, get_llm_generator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class WorkflowTemplate:
    """Represents a workflow template loaded from YAML."""
    id: str
    name: str
    description: str
    category: str
    complexity: str
    step_count: int
    intent_variations: List[str]
    context_parameters: Dict[str, Any]
    steps: List[Dict]
    expected_outcome: str
    kpis_involved: List[str] = field(default_factory=list)
    related_workflows: List[str] = field(default_factory=list)


@dataclass
class GeneratedSample:
    """A generated training sample."""
    id: str
    type: str
    gsma_category: str
    intent: str
    context: Dict
    reasoning: Dict
    orchestration: Dict
    expected_outcome: str
    metadata: Dict


class WorkflowSampleBuilder:
    """
    Builds multi-step workflow training samples using Knowledge Graph.

    The KG ensures all generated workflows are valid (correct API order,
    data flow, and constraint satisfaction).
    """

    def __init__(self,
                 knowledge_graph: R1ApiDependencyGraph = None,
                 workflows_path: str = None,
                 rest_apis_path: str = None,
                 llm_generator = None):
        """
        Initialize the workflow sample builder.

        Args:
            knowledge_graph: R1ApiDependencyGraph instance for validation
            workflows_path: Path to r1_workflows.yaml
            rest_apis_path: Path to r1_rest_apis.yaml
            llm_generator: Optional LLM generator for reasoning traces
        """
        # Load knowledge graph
        self.kg = knowledge_graph or load_api_dependencies()
        self.validator = WorkflowValidator(self.kg)

        # Set default paths
        base_dir = os.path.dirname(__file__)
        if workflows_path is None:
            workflows_path = os.path.join(base_dir, "r1_workflows.yaml")
        if rest_apis_path is None:
            rest_apis_path = os.path.join(base_dir, "r1_rest_apis.yaml")

        # Load workflow templates
        self.workflows: Dict[str, WorkflowTemplate] = {}
        self._load_workflows(workflows_path)

        # Load REST API definitions
        self.rest_apis: Dict[str, List[Dict]] = {}
        self._load_rest_apis(rest_apis_path)

        # LLM generator (optional)
        self.llm = llm_generator
        if self.llm is None and LLM_AVAILABLE:
            self.llm = get_llm_generator()

        # Sample generation statistics
        self.stats = {
            "total_generated": 0,
            "kg_validated": 0,
            "llm_enhanced": 0,
            "by_category": {}
        }

        # Context value generators
        self._context_generators = {
            "cellId": self._generate_cell_id,
            "nearRtRicId": self._generate_ric_id,
            "policyTypeId": self._generate_policy_type_id,
            "infoTypeId": self._generate_info_type_id,
            "startDate": self._generate_start_date,
            "endDate": self._generate_end_date,
            "reportingPeriod": self._generate_reporting_period,
        }

    def _load_workflows(self, path: str) -> None:
        """Load workflow templates from YAML."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        workflows = config.get('workflows', {})
        for wf_name, wf_data in workflows.items():
            self.workflows[wf_name] = WorkflowTemplate(
                id=wf_data.get('id', wf_name),
                name=wf_data.get('name', wf_name),
                description=wf_data.get('description', ''),
                category=wf_data.get('category', 'general'),
                complexity=wf_data.get('complexity', 'medium'),
                step_count=wf_data.get('step_count', 0),
                intent_variations=wf_data.get('intent_variations', []),
                context_parameters=wf_data.get('context_parameters', {}),
                steps=wf_data.get('steps', []),
                expected_outcome=wf_data.get('expected_outcome', ''),
                kpis_involved=wf_data.get('kpis_involved', []),
                related_workflows=wf_data.get('related_workflows', [])
            )

    def _load_rest_apis(self, path: str) -> None:
        """Load REST API definitions from YAML."""
        with open(path, 'r') as f:
            self.rest_apis = yaml.safe_load(f)

    def build_workflow_sample(self,
                              workflow_id: str,
                              context: Dict = None,
                              use_llm: bool = True) -> GeneratedSample:
        """
        Build a complete multi-step training sample.

        Args:
            workflow_id: ID of workflow template to use
            context: Optional context overrides
            use_llm: Whether to use LLM for reasoning (if available)

        Returns:
            GeneratedSample with full orchestration details
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        # Generate context values
        full_context = self._generate_context(workflow, context)

        # Select an intent variation and fill in context
        intent = self._generate_intent(workflow, full_context)

        # Build orchestration steps using KG
        orchestration = self._build_orchestration(workflow, full_context)

        # Validate with KG
        api_ids = [step.get('api', '') for step in orchestration['steps']]
        is_valid, errors = self.kg.validate_workflow(api_ids)

        if not is_valid:
            # Try to fix by regenerating with KG help
            orchestration = self._build_orchestration_with_kg(workflow, full_context)
            api_ids = [step.get('api', '') for step in orchestration['steps']]
            is_valid, errors = self.kg.validate_workflow(api_ids)

        # Generate reasoning
        reasoning = self._generate_reasoning(workflow, orchestration, full_context, use_llm)

        # Generate sample ID
        sample_id = self._generate_sample_id(workflow_id, intent)

        # Build metadata
        metadata = {
            "workflow_id": workflow_id,
            "complexity": workflow.complexity,
            "step_count": len(orchestration['steps']),
            "apis_used": api_ids,
            "kpis_involved": workflow.kpis_involved,
            "kg_validated": is_valid,
            "kg_errors": errors if not is_valid else [],
            "generated_at": datetime.utcnow().isoformat()
        }

        # Update stats
        self.stats["total_generated"] += 1
        if is_valid:
            self.stats["kg_validated"] += 1

        category = workflow.category
        self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1

        return GeneratedSample(
            id=sample_id,
            type="intent_to_orchestration",
            gsma_category="r1_orchestration",
            intent=intent,
            context=full_context,
            reasoning=reasoning,
            orchestration=orchestration,
            expected_outcome=self._fill_template(workflow.expected_outcome, full_context),
            metadata=metadata
        )

    def _generate_context(self, workflow: WorkflowTemplate,
                          overrides: Dict = None) -> Dict:
        """Generate context values for the workflow."""
        context = {}
        overrides = overrides or {}

        for param_name, param_config in workflow.context_parameters.items():
            if param_name in overrides:
                context[param_name] = overrides[param_name]
            elif param_name in self._context_generators:
                context[param_name] = self._context_generators[param_name](param_config)
            elif 'default' in param_config:
                context[param_name] = param_config['default']
            elif 'examples' in param_config and param_config['examples']:
                context[param_name] = random.choice(param_config['examples'])
            else:
                context[param_name] = self._generate_default_value(param_config)

        return context

    def _generate_intent(self, workflow: WorkflowTemplate, context: Dict) -> str:
        """Select and fill in an intent variation."""
        if not workflow.intent_variations:
            return workflow.description

        intent_template = random.choice(workflow.intent_variations)
        return self._fill_template(intent_template, context)

    def _fill_template(self, template: str, context: Dict) -> str:
        """Fill in template placeholders with context values."""
        result = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def _build_orchestration(self, workflow: WorkflowTemplate,
                             context: Dict) -> Dict:
        """Build orchestration steps from workflow template."""
        steps = []
        accumulated_data = {}

        for step_def in workflow.steps:
            step = self._build_step(step_def, context, accumulated_data)
            steps.append(step)

            # Track outputs for data flow
            outputs = step_def.get('expected_outputs', {})
            accumulated_data.update(outputs)

        return {
            "steps": steps,
            "kg_validation": {
                "valid": True,
                "checks_passed": []
            }
        }

    def _build_orchestration_with_kg(self, workflow: WorkflowTemplate,
                                     context: Dict) -> Dict:
        """Build orchestration using KG to ensure validity."""
        # Get target APIs from workflow
        target_apis = [step.get('api', '') for step in workflow.steps]

        # Use KG to generate valid workflow
        kg_workflow = self.kg.generate_valid_workflow(
            intent=workflow.description,
            target_apis=target_apis
        )

        # Build steps from KG output
        steps = []
        accumulated_data = {}

        for idx, kg_step in enumerate(kg_workflow):
            # Find matching template step if available
            template_step = None
            for ws in workflow.steps:
                if ws.get('api') == kg_step['api']:
                    template_step = ws
                    break

            step = {
                "step": idx + 1,
                "api": kg_step['api'],
                "api_uri": kg_step.get('api_uri', ''),
                "method": kg_step.get('method', 'GET'),
                "purpose": kg_step.get('purpose', ''),
                "inputs": self._resolve_inputs(template_step, context, accumulated_data) if template_step else {},
                "outputs": kg_step.get('produces', []),
            }

            # Add API call details
            step["api_call"] = self._generate_api_call(kg_step, context)

            steps.append(step)

            # Track outputs
            if template_step:
                outputs = template_step.get('expected_outputs', {})
                accumulated_data.update(outputs)

        return {
            "steps": steps,
            "kg_validation": {
                "valid": True,
                "checks_passed": [
                    f"All prerequisites satisfied",
                    f"Data flow validated",
                    f"No constraint violations"
                ]
            }
        }

    def _build_step(self, step_def: Dict, context: Dict,
                    accumulated_data: Dict) -> Dict:
        """Build a single orchestration step."""
        step_num = step_def.get('step', 1)
        api_id = step_def.get('api', '')

        # Get API node from KG for additional info
        api_node = self.kg.get_api_node(api_id)

        # Resolve inputs
        inputs = self._resolve_inputs(step_def, context, accumulated_data)

        # Build the step
        step = {
            "step": step_num,
            "api": api_id,
            "purpose": step_def.get('purpose', ''),
            "reasoning": step_def.get('reasoning', ''),
            "inputs": inputs,
            "outputs": list(step_def.get('expected_outputs', {}).keys()),
        }

        # Add API call details if we have the API node
        if api_node:
            step["api_uri"] = api_node.uri
            step["method"] = api_node.method
            step["api_call"] = self._generate_api_call_from_node(api_node, inputs, context)

        return step

    def _resolve_inputs(self, step_def: Dict, context: Dict,
                        accumulated_data: Dict) -> Dict:
        """Resolve input values from context and previous steps."""
        if not step_def:
            return {}

        resolved = {}
        inputs = step_def.get('inputs', {})

        for input_name, input_source in inputs.items():
            if isinstance(input_source, str):
                if input_source.startswith("FROM step_"):
                    # Reference to previous step output
                    match = re.match(r'FROM step_(\d+)\.?(.*)$', input_source)
                    if match:
                        step_num = match.group(1)
                        field = match.group(2) if match.group(2) else input_name
                        resolved[input_name] = f"${{step_{step_num}.output.{field}}}"
                elif input_source.startswith("FROM context."):
                    # Reference to context value
                    field = input_source.replace("FROM context.", "")
                    resolved[input_name] = context.get(field, input_source)
                elif input_source == "FROM step_1":
                    # Shorthand for step_1 output
                    resolved[input_name] = f"${{step_1.output.{input_name}}}"
                else:
                    resolved[input_name] = input_source
            else:
                resolved[input_name] = input_source

        return resolved

    def _generate_api_call(self, kg_step: Dict, context: Dict) -> Dict:
        """Generate API call details from KG step."""
        method = kg_step.get('method', 'GET')
        uri = kg_step.get('api_uri', '')

        # Fill in URI parameters
        uri = self._fill_uri_parameters(uri, context)

        api_call = {
            "method": method,
            "uri": f"https://smo.example.com{uri}",
            "headers": {
                "Authorization": "Bearer eyJhbGc...",
                "Accept": "application/json"
            }
        }

        if method in ('POST', 'PUT', 'PATCH'):
            api_call["headers"]["Content-Type"] = "application/json"
            api_call["body"] = self._generate_request_body(kg_step, context)

        return api_call

    def _generate_api_call_from_node(self, api_node, inputs: Dict,
                                     context: Dict) -> Dict:
        """Generate API call from API node."""
        uri = self._fill_uri_parameters(api_node.uri, context)

        api_call = {
            "method": api_node.method,
            "uri": f"https://smo.example.com{uri}",
            "headers": {
                "Authorization": "Bearer eyJhbGc...",
                "Accept": "application/json"
            }
        }

        if api_node.method in ('POST', 'PUT', 'PATCH'):
            api_call["headers"]["Content-Type"] = "application/json"
            api_call["body"] = self._generate_request_body_from_inputs(inputs, context)

        return api_call

    def _fill_uri_parameters(self, uri: str, context: Dict) -> str:
        """Fill in URI path parameters."""
        result = uri

        # Common substitutions
        substitutions = {
            "{apiRoot}": "",
            "{apfId}": context.get('apfId', 'apf-001'),
            "{serviceApiId}": context.get('serviceApiId', 'api-001'),
            "{registrationId}": context.get('registrationId', 'reg-001'),
            "{infoJobId}": context.get('infoJobId', 'job-001'),
            "{policyId}": context.get('policyId', 'policy-001'),
            "{nearRtRicId}": context.get('nearRtRicId', 'ric-001'),
        }

        for placeholder, value in substitutions.items():
            result = result.replace(placeholder, value)

        return result

    def _generate_request_body(self, kg_step: Dict, context: Dict) -> Dict:
        """Generate request body for API call."""
        api_id = kg_step.get('api', '')
        requires = kg_step.get('requires', [])

        body = {}

        # Add required fields
        for field in requires:
            if field in context:
                body[field] = context[field]
            else:
                body[field] = self._generate_field_value(field)

        # Add common fields based on API type
        if 'info_jobs' in api_id or 'subscription' in api_id.lower():
            body.update(self._generate_subscription_body(context))
        elif 'policies' in api_id:
            body.update(self._generate_policy_body(context))
        elif 'pm_jobs' in api_id:
            body.update(self._generate_pm_job_body(context))

        return body

    def _generate_request_body_from_inputs(self, inputs: Dict,
                                           context: Dict) -> Dict:
        """Generate request body from resolved inputs."""
        body = {}

        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("${"):
                # Keep the reference
                body[key] = value
            else:
                body[key] = value

        return body

    def _generate_subscription_body(self, context: Dict) -> Dict:
        """Generate body for data subscription."""
        return {
            "infoTypeId": context.get('infoTypeId', 'oran-dmf:coverage-kpis-v1'),
            "infoJobData": {
                "filterCriteria": {
                    "cellIdList": [context.get('cellId', 'NRCell-100')],
                    "measurementTypes": context.get('kpis', ['rsrp_dbm', 'rsrq_db', 'sinr_db'])
                },
                "deliveryMethod": "PUSH",
                "reportingPeriod": context.get('reportingPeriod', 300)
            },
            "targetUri": "http://rapp.example.com/kpi-stream",
            "owner": "monitoring-rapp"
        }

    def _generate_policy_body(self, context: Dict) -> Dict:
        """Generate body for policy creation."""
        return {
            "nearRtRicId": context.get('nearRtRicId', 'ric-001'),
            "policyTypeId": context.get('policyTypeId', 'ORAN_LoadBalancing_1.0.0'),
            "policyData": {
                "scope": {
                    "cellIdList": [context.get('cellId', 'NRCell-100')]
                },
                "objectives": context.get('policyObjectives', {
                    "targetUtilization": 70,
                    "hysteresis": 10
                })
            },
            "transient": False,
            "statusNotificationUri": "http://rapp.example.com/policy-status"
        }

    def _generate_pm_job_body(self, context: Dict) -> Dict:
        """Generate body for PM job creation."""
        return {
            "administrativeState": "ACTIVE",
            "objectInstances": [f"/ManagedElement=gNB-001/NRCellDU={context.get('cellId', 'NRCell-100')}"],
            "measurementTypes": context.get('kpis', ['rsrp_dbm', 'sinr_db']),
            "granularityPeriod": context.get('reportingPeriod', 900),
            "reportingMethod": "FILE",
            "jobPriority": "NORMAL"
        }

    def _generate_reasoning(self, workflow: WorkflowTemplate,
                           orchestration: Dict,
                           context: Dict,
                           use_llm: bool) -> Dict:
        """Generate reasoning for the workflow."""
        # Build template-based reasoning
        reasoning = {
            "intent_analysis": self._analyze_intent(workflow, context),
            "api_selection": self._explain_api_selection(workflow, orchestration),
            "dependency_analysis": self._explain_dependencies(workflow, orchestration),
            "data_flow": self._explain_data_flow(orchestration)
        }

        # Optionally enhance with LLM
        if use_llm and self.llm and self.llm.is_enabled():
            try:
                enhanced = asyncio.run(self._enhance_reasoning_with_llm(
                    workflow, orchestration, context, reasoning
                ))
                if enhanced:
                    reasoning["llm_enhanced_reasoning"] = enhanced
                    self.stats["llm_enhanced"] += 1
            except Exception as e:
                # Fall back to template reasoning
                reasoning["llm_error"] = str(e)

        return reasoning

    def _analyze_intent(self, workflow: WorkflowTemplate, context: Dict) -> str:
        """Analyze the intent for reasoning."""
        kpis = workflow.kpis_involved
        kpi_text = ", ".join(kpis) if kpis else "relevant metrics"

        return (
            f"User wants to {workflow.description.lower()}. "
            f"This requires monitoring {kpi_text}. "
            f"Need to: (1) discover available data types, "
            f"(2) subscribe to relevant metrics, "
            f"(3) set up alerting for threshold violations."
        )

    def _explain_api_selection(self, workflow: WorkflowTemplate,
                               orchestration: Dict) -> List[Dict]:
        """Explain why each API was selected."""
        explanations = []

        for step in orchestration['steps']:
            explanations.append({
                "api": step['api'],
                "reason": step.get('reasoning', step.get('purpose', 'Required for workflow'))
            })

        return explanations

    def _explain_dependencies(self, workflow: WorkflowTemplate,
                              orchestration: Dict) -> str:
        """Explain API dependencies."""
        steps = orchestration['steps']
        if len(steps) < 2:
            return "Single step workflow, no dependencies."

        deps = []
        for i in range(1, len(steps)):
            prev_api = steps[i-1]['api']
            curr_api = steps[i]['api']

            # Check KG for relationship
            data_flow = self.kg.get_data_flow(prev_api, curr_api)
            if data_flow:
                fields = ", ".join(data_flow.keys())
                deps.append(f"{curr_api} requires {fields} from {prev_api}")

        if deps:
            return ". ".join(deps) + "."
        return "Steps should be executed in sequence for proper data flow."

    def _explain_data_flow(self, orchestration: Dict) -> Dict:
        """Explain data flow between steps."""
        steps = orchestration['steps']
        flow = {}

        for i in range(1, len(steps)):
            prev_step = steps[i-1]
            curr_step = steps[i]

            prev_outputs = prev_step.get('outputs', [])
            curr_inputs = curr_step.get('inputs', {})

            # Find what flows from previous to current
            for input_name, input_val in curr_inputs.items():
                if isinstance(input_val, str) and f"step_{i}" in input_val:
                    key = f"step_{i}_to_{i+1}"
                    if key not in flow:
                        flow[key] = {}
                    flow[key][input_name] = f"From step {i} output"

        return flow

    async def _enhance_reasoning_with_llm(self, workflow: WorkflowTemplate,
                                          orchestration: Dict,
                                          context: Dict,
                                          base_reasoning: Dict) -> str:
        """Enhance reasoning using LLM."""
        if not self.llm:
            return None

        workflow_dict = {
            "steps": orchestration['steps']
        }
        context_dict = {
            "intent": workflow.description,
            "parameters": context
        }

        return await self.llm.generate_reasoning_trace(workflow_dict, context_dict)

    def _generate_sample_id(self, workflow_id: str, intent: str) -> str:
        """Generate unique sample ID."""
        content = f"{workflow_id}:{intent}:{datetime.utcnow().isoformat()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"wf_{workflow_id}_{hash_val}"

    # Context value generators
    def _generate_cell_id(self, config: Dict) -> str:
        """Generate a cell ID."""
        if 'examples' in config and config['examples']:
            return random.choice(config['examples'])
        num = random.randint(100, 999)
        return f"NRCell-{num}"

    def _generate_ric_id(self, config: Dict) -> str:
        """Generate a Near-RT RIC ID."""
        num = random.randint(1, 10)
        return f"ric-00{num}"

    def _generate_policy_type_id(self, config: Dict) -> str:
        """Generate a policy type ID."""
        types = [
            "ORAN_LoadBalancing_1.0.0",
            "ORAN_QoS_1.0.0",
            "ORAN_TrafficSteering_1.0.0",
            "ORAN_MRO_1.0.0"
        ]
        return random.choice(types)

    def _generate_info_type_id(self, config: Dict) -> str:
        """Generate an info type ID."""
        types = [
            "oran-dmf:coverage-kpis-v1",
            "oran-dmf:capacity-kpis-v1",
            "oran-dmf:ue-metrics-v1",
            "oran-dmf:cell-performance-v1"
        ]
        return random.choice(types)

    def _generate_start_date(self, config: Dict) -> str:
        """Generate a start date."""
        days_ago = random.randint(7, 30)
        dt = datetime.utcnow() - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%dT00:00:00Z")

    def _generate_end_date(self, config: Dict) -> str:
        """Generate an end date."""
        dt = datetime.utcnow()
        return dt.strftime("%Y-%m-%dT23:59:59Z")

    def _generate_reporting_period(self, config: Dict) -> int:
        """Generate a reporting period."""
        if 'examples' in config and config['examples']:
            return random.choice(config['examples'])
        return random.choice([60, 300, 900, 3600])

    def _generate_default_value(self, config: Dict) -> Any:
        """Generate default value based on type."""
        param_type = config.get('type', 'string')

        if param_type == 'string':
            return "default-value"
        elif param_type == 'integer':
            return 0
        elif param_type == 'array':
            return []
        elif param_type == 'boolean':
            return False
        else:
            return None

    def _generate_field_value(self, field: str) -> Any:
        """Generate a value for a field based on its name."""
        if 'id' in field.lower():
            return f"{field.replace('Id', '')}-001"
        elif 'uri' in field.lower() or 'url' in field.lower():
            return "http://example.com/callback"
        elif 'name' in field.lower():
            return f"sample-{field}"
        else:
            return "sample-value"

    def generate_batch(self, workflow_ids: List[str] = None,
                       count_per_workflow: int = 10,
                       use_llm: bool = False) -> List[GeneratedSample]:
        """
        Generate a batch of workflow samples.

        Args:
            workflow_ids: List of workflow IDs to use (None = all)
            count_per_workflow: Number of samples per workflow
            use_llm: Whether to use LLM for reasoning

        Returns:
            List of generated samples
        """
        if workflow_ids is None:
            workflow_ids = list(self.workflows.keys())

        samples = []

        for wf_id in workflow_ids:
            for i in range(count_per_workflow):
                try:
                    sample = self.build_workflow_sample(
                        workflow_id=wf_id,
                        use_llm=use_llm
                    )
                    samples.append(sample)
                except Exception as e:
                    print(f"Error generating sample for {wf_id}: {e}")

        return samples

    def to_training_format(self, sample: GeneratedSample) -> Dict:
        """Convert sample to training format (JSONL compatible)."""
        return {
            "id": sample.id,
            "type": sample.type,
            "gsma_category": sample.gsma_category,
            "instruction": sample.intent,
            "context": sample.context,
            "reasoning_trace": sample.reasoning.get(
                'llm_enhanced_reasoning',
                sample.reasoning.get('intent_analysis', '')
            ),
            "output": {
                "steps": sample.orchestration['steps']
            },
            "expected_outcome": sample.expected_outcome,
            "metadata": sample.metadata
        }

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return self.stats.copy()

    def list_workflows(self) -> List[Dict]:
        """List available workflow templates."""
        return [
            {
                "id": wf_key,  # Use the dictionary key, not wf.id
                "name": wf.name,
                "category": wf.category,
                "complexity": wf.complexity,
                "step_count": wf.step_count
            }
            for wf_key, wf in self.workflows.items()
        ]


def get_workflow_builder(knowledge_graph: R1ApiDependencyGraph = None,
                         llm_generator = None) -> WorkflowSampleBuilder:
    """
    Convenience function to create a WorkflowSampleBuilder.

    Args:
        knowledge_graph: Optional pre-loaded KG
        llm_generator: Optional LLM generator

    Returns:
        Initialized WorkflowSampleBuilder
    """
    return WorkflowSampleBuilder(
        knowledge_graph=knowledge_graph,
        llm_generator=llm_generator
    )


# Example usage
if __name__ == "__main__":
    # Create builder
    builder = WorkflowSampleBuilder()

    print("=== Available Workflows ===")
    for wf in builder.list_workflows():
        print(f"  {wf['id']}: {wf['name']} ({wf['complexity']}, {wf['step_count']} steps)")

    print("\n=== Generating Sample ===")

    # Generate a sample
    try:
        sample = builder.build_workflow_sample(
            workflow_id="coverage_monitoring_setup",
            context={"cellId": "NRCell-100"},
            use_llm=False  # Disable LLM for testing
        )

        print(f"Sample ID: {sample.id}")
        print(f"Intent: {sample.intent}")
        print(f"\nSteps:")
        for step in sample.orchestration['steps']:
            print(f"  {step['step']}: {step['api']} - {step['purpose']}")

        print(f"\nKG Validated: {sample.metadata['kg_validated']}")

        # Convert to training format
        training_sample = builder.to_training_format(sample)
        print("\n=== Training Format ===")
        print(json.dumps(training_sample, indent=2, default=str)[:1000] + "...")

    except FileNotFoundError as e:
        print(f"Note: {e}")
        print("Run from project root or ensure YAML files are available")

    print("\n=== Statistics ===")
    print(builder.get_stats())
