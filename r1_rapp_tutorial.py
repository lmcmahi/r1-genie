#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "requests",
#     "pandas",
#     "plotly",
# ]
# ///
"""
O-RAN R1 Interface & rApp Development Tutorial
"""

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import requests
    import json
    import pandas as pd
    import time
    import uuid
    return json, mo, pd, requests, uuid


@app.cell
def _(mo):
    mo.md("""
    # O-RAN R1 Interface & rApp Development Tutorial

    Welcome to this hands-on tutorial on the **O-RAN R1 Interface** and **rApp Development**.

    ## What You'll Learn

    1. **O-RAN Architecture** - SMO, Near-RT RIC, and rApps
    2. **R1 Interface** - The standardized API between rApps and SMO
    3. **R1 API Categories** - Bootstrap, Data Management, A1 Policy, AI/ML
    4. **Building rApps** - Step-by-step workflows with **proper R1 flows**

    ## Environment

    | Component | URL | Purpose |
    |-----------|-----|---------|
    | OSC ICS | `192.168.86.180:9082` | Information Coordinator Service (Broker) |
    | RAN Simulation | `192.168.86.180:5555` | Data Producer (Simulated RAN) |
    """)
    return


@app.cell
def _(mo, requests):
    # Configuration - Change these for your environment
    ICS_URL = "http://192.168.86.180:9082"      # OSC ICS (the R1 broker)
    PRODUCER_URL = "http://192.168.86.180:5555" # RAN Simulator (data producer)

    # Check environment status
    def check_service(name, url):
        try:
            resp = requests.get(url, timeout=5)
            return f"✅ {name}: Running"
        except:
            return f"❌ {name}: Not available"

    ics_status = check_service("OSC ICS", f"{ICS_URL}/data-consumer/v1/info-types")
    sim_status = check_service("RAN Simulation", f"{PRODUCER_URL}/bootstrap/v1/bootstrap-info")

    mo.md(f"""
    ## Environment Status

    ```
    {ics_status}
    {sim_status}
    ```

    **Configuration:**
    - `ICS_URL = "{ICS_URL}"` (R1 Broker - use this for all R1 operations)
    - `PRODUCER_URL = "{PRODUCER_URL}"` (Data Producer - registered with ICS)
    """)
    return ICS_URL, PRODUCER_URL


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 1: O-RAN Architecture

    ## Key Components

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                Service Management & Orchestration (SMO)            │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                      R1 Interface                           │   │
    │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐                 │   │
    │  │   │  rApp 1  │  │  rApp 2  │  │  rApp N  │                 │   │
    │  │   └──────────┘  └──────────┘  └──────────┘                 │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
    │  │     ICS     │  │     PMS     │  │   AI/ML     │               │
    │  └─────────────┘  └─────────────┘  └─────────────┘               │
    └─────────────────────────────────────────────────────────────────────┘
                                    │ A1
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        Near-RT RIC                                  │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  (10ms - 1s)            │
    │  │  xApp 1  │  │  xApp 2  │  │  xApp N  │                         │
    │  └──────────┘  └──────────┘  └──────────┘                         │
    └─────────────────────────────────────────────────────────────────────┘
                                    │ E2
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  RAN (gNodeBs / Cells)                              │
    └─────────────────────────────────────────────────────────────────────┘
    ```

    ## What is an rApp?

    An **rApp** runs in the SMO and uses the **R1 interface** to:
    - **Consume Data** - Subscribe to network KPIs via ICS
    - **Create Policies** - Deploy A1 policies to Near-RT RIC
    - **Use AI/ML** - Train and deploy ML models
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 2: Understanding the R1 Data Flow

    ## CRITICAL: The Role of ICS (Information Coordinator Service)

    ICS is the **broker** between data producers and consumers. You should **ALWAYS** go through ICS for R1-compliant data access.

    ```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           R1 DATA FLOW                                      │
    └─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │     rApp     │         │   OSC ICS    │         │  RAN Sim     │
    │  (Consumer)  │         │   (Broker)   │         │  (Producer)  │
    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
           │                        │                        │
           │ 1. GET /info-types     │                        │
           │───────────────────────>│                        │
           │    "What data exists?" │                        │
           │                        │                        │
           │ 2. PUT /info-jobs/{id} │                        │
           │───────────────────────>│                        │
           │    "Subscribe to       │ 3. Notify producer     │
           │     coverage-kpis"     │───────────────────────>│
           │                        │    "New consumer for   │
           │                        │     coverage-kpis"     │
           │                        │                        │
           │                        │                        │
           │  ═══════════════ DATA DELIVERY ═══════════════  │
           │                        │                        │
           │  OPTION A: PUSH MODE   │                        │
           │<───────────────────────────────────────────────│
           │    Producer sends to rApp's callback URL        │
           │                        │                        │
           │  OPTION B: PULL MODE   │                        │
           │───────────────────────────────────────────────>│
           │    rApp pulls from producer's delivery endpoint │
           │                        │                        │
    ```

    ## Key Points

    | What | Where | Why |
    |------|-------|-----|
    | **Discover data types** | ICS (`/data-consumer/v1/info-types`) | ICS knows all registered types |
    | **Create subscriptions** | ICS (`/data-consumer/v1/info-jobs`) | ICS coordinates with producers |
    | **Receive data (PUSH)** | Your callback URL | Producer sends to you |
    | **Fetch data (PULL)** | Producer's delivery endpoint | You pull when ready |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 3: The Bootstrap Pattern

    An rApp starts knowing ONLY:
    - `api_root` (e.g., `https://smo.example.com`) - **vendor-specific**
    - `bootstrap_path` (`/bootstrap/v1/bootstrap-info`) - **standard**

    Everything else is **discovered dynamically**.
    """)
    return


@app.cell
def _(PRODUCER_URL, json, mo, requests):
    # Live Bootstrap Demo
    try:
        bootstrap_resp = requests.get(f"{PRODUCER_URL}/bootstrap/v1/bootstrap-info")
        bootstrap_data = bootstrap_resp.json()
        bootstrap_json = json.dumps(bootstrap_data, indent=2)
        bootstrap_status = "✅ Success"
    except Exception as e:
        bootstrap_json = f"Error: {e}"
        bootstrap_status = "❌ Failed"
        bootstrap_data = {}

    mo.md(f"""
    ## Live Bootstrap Demo

    **Request:** `GET {PRODUCER_URL}/bootstrap/v1/bootstrap-info`

    **Status:** {bootstrap_status}

    ```json
    {bootstrap_json}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 4: Data Management & Exposure (DME)

    DME follows a **producer-consumer** model coordinated by **ICS**:

    | DME Type | Description | Measurements |
    |----------|-------------|--------------|
    | `coverage-kpis` | RF signal quality | RSRP, RSRQ, SINR |
    | `capacity-kpis` | Cell load metrics | PRB utilization, throughput |
    | `pm-data` | Full PM counters | All performance measurements |
    | `cell-config` | Cell configuration | txPower, tiltAngle |
    | `alarm-data` | Network faults | Alarms, severity |
    """)
    return


@app.cell
def _(ICS_URL, mo, pd, requests):
    # DME Discovery - ALWAYS via ICS
    try:
        types_resp = requests.get(f"{ICS_URL}/data-consumer/v1/info-types")
        dme_types = types_resp.json()
        type_details = []
        for t in dme_types[:6]:
            try:
                detail = requests.get(f"{ICS_URL}/data-consumer/v1/info-types/{t}").json()
                type_details.append({
                    "DME Type": t,
                    "Status": detail.get("type_status", "N/A"),
                    "Producers": detail.get("no_of_producers", 0),
                })
            except:
                type_details.append({"DME Type": t, "Status": "Unknown", "Producers": 0})

        dme_df = pd.DataFrame(type_details)
        dme_status = f"Found {len(dme_types)} DME types"
    except Exception as e:
        dme_df = pd.DataFrame({"Error": [str(e)]})
        dme_status = f"Error: {e}"
        dme_types = []

    mo.md(f"""
    ## DME Types Discovery (via ICS)

    **This is the R1-compliant way to discover available data!**

    **API:** `GET {ICS_URL}/data-consumer/v1/info-types`

    **Result:** {dme_status}
    """)
    return (dme_df,)


@app.cell
def _(dme_df, mo):
    mo.ui.table(dme_df)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 5: Building Your First rApp (The R1 Way)

    ## Coverage Monitoring rApp

    **Goal:** Monitor cell coverage KPIs using the **proper R1 flow via ICS**.

    ### Step 1: Bootstrap - Discover API Endpoints
    """)
    return


@app.cell
def _(PRODUCER_URL, json, mo, requests):
    # Step 1: Bootstrap
    def do_bootstrap():
        resp = requests.get(f"{PRODUCER_URL}/bootstrap/v1/bootstrap-info")
        return resp.json()

    bootstrap_info = do_bootstrap()
    endpoints = {ep["apiName"]: ep["uri"] for ep in bootstrap_info.get("apiEndpoints", [])}

    mo.md(f"""
    ### Step 1 Result: Discovered Endpoints

    ```json
    {json.dumps(endpoints, indent=2)}
    ```

    Now we know what APIs are available. Next, we'll use **ICS** to discover data types.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 2: Discover DME Types (via ICS)
    """)
    return


@app.cell
def _(ICS_URL, mo, requests):
    # Step 2: Discover DME Types - MUST use ICS
    def discover_types():
        """Query ICS for available data types - this is the R1-compliant way"""
        resp = requests.get(f"{ICS_URL}/data-consumer/v1/info-types")
        return resp.json()

    available_types = discover_types()
    coverage_type = "coverage-kpis" if "coverage-kpis" in available_types else available_types[0] if available_types else None

    mo.md(f"""
    ### Step 2 Result: Available DME Types

    **API:** `GET {ICS_URL}/data-consumer/v1/info-types`

    **Available Types:** `{available_types}`

    **Selected for subscription:** `{coverage_type}`

    This query goes to **ICS** (the broker), NOT directly to the producer!
    """)
    return (coverage_type,)


@app.cell
def _(mo):
    mo.md("""
    ### Step 3: Create Data Job (Subscribe via ICS)
    """)
    return


@app.cell
def _(ICS_URL, coverage_type, json, mo, requests, uuid):
    # Step 3: Create Data Job - MUST use ICS
    def create_data_job(info_type_id, job_owner, job_params=None):
        """
        Create a data subscription via ICS.
        ICS will coordinate with the producer.
        """
        job_id = f"tutorial-{uuid.uuid4().hex[:8]}"
        job_definition = {
            "info_type_id": info_type_id,
            "job_owner": job_owner,
            "job_definition": job_params or {}
        }
        resp = requests.put(
            f"{ICS_URL}/data-consumer/v1/info-jobs/{job_id}",
            json=job_definition
        )
        return job_id, resp.status_code, job_definition

    # Create the job
    created_job_id, job_status_code, job_def = create_data_job(
        info_type_id=coverage_type,
        job_owner="tutorial-rapp",
        job_params={"cells": ["cell_1", "cell_2", "cell_3"]}
    )

    mo.md(f"""
    ### Step 3 Result: Data Job Created

    **API:** `PUT {ICS_URL}/data-consumer/v1/info-jobs/{{job_id}}`

    **Job ID:** `{created_job_id}`

    **Status Code:** `{job_status_code}` {"✅" if job_status_code in [200, 201] else "❌"}

    **Job Definition:**
    ```json
    {json.dumps(job_def, indent=2)}
    ```

    **What happened behind the scenes:**
    1. ICS received our subscription request
    2. ICS notified the RAN Simulator (producer) about our job
    3. Producer is now ready to deliver data for this job
    """)
    return (created_job_id,)


@app.cell
def _(mo):
    mo.md("""
    ### Step 4: Get Data (Two Options)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Understanding Data Delivery Modes

    After creating a data job, there are **two ways** to receive data:

    | Mode | How It Works | When to Use |
    |------|-------------|-------------|
    | **PUSH** | Producer sends data to your `targetUri` callback | Real-time streaming, events |
    | **PULL** | You fetch from producer's delivery endpoint | On-demand queries, batch |

    ### PUSH Mode Setup
    ```python
    # When creating the job, specify your callback URL
    job_definition = {
        "info_type_id": "coverage-kpis",
        "job_owner": "my-rapp",
        "targetUri": "http://my-rapp:8080/data-callback",  # Producer sends here
        "job_definition": {"cells": ["cell_1"]}
    }
    ```

    ### PULL Mode
    After job creation, query the producer's delivery endpoint.
    """)
    return


@app.cell
def _(ICS_URL, PRODUCER_URL, created_job_id, json, mo, requests):
    # Step 4: Fetch data - In a real R1 implementation, this would be via:
    # - PUSH: Data sent to your callback URL
    # - PULL: Query the producer's delivery endpoint

    # First, verify our job exists in ICS
    def verify_job(job_id):
        """Check that ICS has our job registered"""
        try:
            resp = requests.get(f"{ICS_URL}/data-consumer/v1/info-jobs/{job_id}")
            return resp.status_code == 200, resp.json() if resp.status_code == 200 else None
        except:
            return False, None

    job_exists, job_info = verify_job(created_job_id)

    # Now fetch data from the producer's delivery endpoint
    # In OSC NONRTRIC, the producer provides data at its registered callback
    def fetch_coverage_data():
        """
        Fetch data from producer.
        In production, this would be coordinated by ICS or via PUSH.
        """
        resp = requests.get(f"{PRODUCER_URL}/kpis/coverage")
        return resp.json()

    coverage_data = fetch_coverage_data()

    mo.md(f"""
    ### Step 4 Result: Data Retrieved

    **Job Verification via ICS:**
    - Job `{created_job_id}` exists: {"✅ Yes" if job_exists else "❌ No"}

    **Data Fetch:**
    - Source: Producer's delivery endpoint
    - Endpoint: `{PRODUCER_URL}/kpis/coverage`

    **Sample Data (first 2 cells):**
    ```json
    {json.dumps(coverage_data[:2] if isinstance(coverage_data, list) else coverage_data, indent=2)}
    ```

    **Note:** In a production R1 setup:
    - **PUSH mode**: Data would arrive at your registered callback URL
    - **PULL mode**: You'd query the producer's delivery endpoint (as shown here)
    """)
    return (coverage_data,)


@app.cell
def _(mo):
    mo.md("""
    ### Step 5: Analyze Coverage Data
    """)
    return


@app.cell
def _(coverage_data, mo, pd):
    # Step 5: Analyze the data
    RSRP_THRESHOLD = -100
    SINR_THRESHOLD = 0

    def analyze_coverage(data):
        results = []
        for cell in data:
            rsrp = cell.get("rsrp_dbm", 0)
            sinr = cell.get("sinr_db", 0)
            status = "🔴 DEGRADED" if rsrp < RSRP_THRESHOLD or sinr < SINR_THRESHOLD else "🟢 HEALTHY"
            results.append({
                "Cell": cell.get("cellId"),
                "RSRP (dBm)": round(rsrp, 1),
                "RSRQ (dB)": round(cell.get("rsrq_db", 0), 1),
                "SINR (dB)": round(sinr, 1),
                "Status": status
            })
        return pd.DataFrame(results)

    coverage_analysis = analyze_coverage(coverage_data)

    mo.md(f"""
    ### Step 5 Result: Coverage Analysis

    **Thresholds:**
    - RSRP < {RSRP_THRESHOLD} dBm → Degraded
    - SINR < {SINR_THRESHOLD} dB → Degraded
    """)
    return (coverage_analysis,)


@app.cell
def _(coverage_analysis, mo):
    mo.ui.table(coverage_analysis)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 6: Cleanup - Delete Data Job
    """)
    return


@app.cell
def _(ICS_URL, created_job_id, mo, requests):
    # Step 6: Cleanup - delete the job when done
    def delete_data_job(job_id):
        """Delete data job via ICS"""
        resp = requests.delete(f"{ICS_URL}/data-consumer/v1/info-jobs/{job_id}")
        return resp.status_code

    # Uncomment to actually delete:
    # delete_status = delete_data_job(created_job_id)

    mo.md(f"""
    ### Step 6: Cleanup

    When your rApp no longer needs data, **delete the job via ICS**:

    ```python
    requests.delete(f"{{ICS_URL}}/data-consumer/v1/info-jobs/{created_job_id}")
    ```

    This tells ICS to:
    1. Remove the subscription
    2. Notify the producer to stop sending data
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 6: Complete R1 rApp Pattern

    Here's the complete pattern for an R1-compliant rApp:

    ```python
    class R1CompliantRApp:
        def __init__(self, ics_url, producer_url):
            self.ics_url = ics_url          # Always use ICS for R1 operations
            self.producer_url = producer_url # Only for bootstrap & data delivery

        # STEP 1: Bootstrap (call producer directly - it's the entry point)
        def bootstrap(self):
            resp = requests.get(f"{self.producer_url}/bootstrap/v1/bootstrap-info")
            return resp.json()

        # STEP 2: Discover types (ALWAYS via ICS)
        def discover_types(self):
            resp = requests.get(f"{self.ics_url}/data-consumer/v1/info-types")
            return resp.json()

        # STEP 3: Subscribe to data (ALWAYS via ICS)
        def subscribe(self, info_type, callback_url=None):
            job = {
                "info_type_id": info_type,
                "job_owner": "my-rapp",
                "job_definition": {}
            }
            if callback_url:
                job["targetUri"] = callback_url  # For PUSH mode
            resp = requests.put(
                f"{self.ics_url}/data-consumer/v1/info-jobs/job-001",
                json=job
            )
            return resp.status_code

        # STEP 4a: Receive data (PUSH - data arrives at your callback)
        def data_callback(self, data):
            # Your callback endpoint receives this
            self.process_data(data)

        # STEP 4b: Fetch data (PULL - query producer's delivery endpoint)
        def fetch_data(self, endpoint):
            resp = requests.get(f"{self.producer_url}{endpoint}")
            return resp.json()

        # STEP 5: Cleanup (ALWAYS via ICS)
        def unsubscribe(self, job_id):
            requests.delete(f"{self.ics_url}/data-consumer/v1/info-jobs/{job_id}")
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 7: A1 Policy Management

    ## Policy Types

    | Type | Purpose |
    |------|---------|
    | `ORAN_QoSTarget_1.0.0` | Guarantee service levels |
    | `ORAN_LoadBalancing_1.0.0` | Distribute load across cells |
    | `ORAN_TrafficSteering_1.0.0` | Direct traffic to preferred cells |
    """)
    return


@app.cell
def _(PRODUCER_URL, json, mo, requests):
    # Get RICs
    def get_rics():
        resp = requests.get(f"{PRODUCER_URL}/a1-policy-management/v1/rics")
        return resp.json()

    rics = get_rics()
    mo.md(f"""
    ## Available RICs

    **API:** `GET {PRODUCER_URL}/a1-policy-management/v1/rics`

    ```json
    {json.dumps(rics, indent=2)}
    ```
    """)
    return


@app.cell
def _(PRODUCER_URL, json, mo, requests):
    # Get Policy Types
    def get_policy_types():
        resp = requests.get(f"{PRODUCER_URL}/a1-policy-management/v1/policy-types")
        return resp.json()

    policy_types = get_policy_types()
    mo.md(f"""
    ## Policy Types

    ```json
    {json.dumps(policy_types, indent=2)}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive: Create a Policy
    """)
    return


@app.cell
def _(mo):
    # Policy form
    policy_type_select = mo.ui.dropdown(
        options=["ORAN_QoSTarget_1.0.0", "ORAN_LoadBalancing_1.0.0"],
        value="ORAN_QoSTarget_1.0.0",
        label="Policy Type"
    )
    target_cell_select = mo.ui.dropdown(
        options=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
        value="cell_1",
        label="Target Cell"
    )
    bitrate_slider = mo.ui.slider(start=1000, stop=100000, step=1000, value=10000, label="Guaranteed Bitrate (kbps)")

    mo.vstack([policy_type_select, target_cell_select, bitrate_slider])
    return bitrate_slider, policy_type_select, target_cell_select


@app.cell
def _(bitrate_slider, mo, policy_type_select, target_cell_select):
    create_policy_btn = mo.ui.button(label="Create Policy", kind="success")
    mo.hstack([create_policy_btn, mo.md(f"Type: **{policy_type_select.value}** | Cell: **{target_cell_select.value}** | Bitrate: **{bitrate_slider.value}** kbps")])
    return (create_policy_btn,)


@app.cell
def _(
    PRODUCER_URL,
    bitrate_slider,
    create_policy_btn,
    mo,
    policy_type_select,
    requests,
    target_cell_select,
):
    policy_result = ""
    if create_policy_btn.value:
        try:
            policy_body = {
                "nearRtRicId": "ric-001",
                "policyTypeId": policy_type_select.value,
                "policyObject": {
                    "scope": {"cellIdList": [target_cell_select.value]},
                    "qosObjectives": {"guaranteedBitRate": bitrate_slider.value, "packetDelayBudget": 20}
                }
            }
            resp = requests.post(f"{PRODUCER_URL}/a1-policy-management/v1/policies", json=policy_body)
            result = resp.json()
            policy_result = f"""
    ### Policy Created!
    - **Policy ID:** `{result.get('policyId')}`
    - **Status:** `{result.get('status')}`
    """
        except Exception as e:
            policy_result = f"Error: {e}"

    mo.md(policy_result if policy_result else "*Click 'Create Policy' to create*")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 8: AI/ML Management

    ## Available Models
    """)
    return


@app.cell
def _(PRODUCER_URL, mo, pd, requests):
    # Get ML Models
    def get_models():
        resp = requests.get(f"{PRODUCER_URL}/r1-aiml/v1/mlModels")
        return resp.json()

    models = get_models()
    models_df = pd.DataFrame([{"Model": m["modelName"], "Version": m["modelVersion"], "Status": m["status"]} for m in models])

    mo.md(f"""
    **API:** `GET {PRODUCER_URL}/r1-aiml/v1/mlModels`
    """)
    return (models_df,)


@app.cell
def _(mo, models_df):
    mo.ui.table(models_df)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive: Run Inference
    """)
    return


@app.cell
def _(mo):
    model_select = mo.ui.dropdown(
        options=["TrafficPredictor", "AnomalyDetector", "HandoverOptimizer"],
        value="TrafficPredictor",
        label="Model"
    )
    prb_input = mo.ui.slider(start=0, stop=100, value=75, label="PRB Utilization (%)")

    mo.vstack([model_select, prb_input])
    return model_select, prb_input


@app.cell
def _(mo, model_select, prb_input):
    infer_btn = mo.ui.button(label="Get Prediction", kind="success")
    mo.hstack([infer_btn, mo.md(f"Model: **{model_select.value}** | PRB: **{prb_input.value}%**")])
    return (infer_btn,)


@app.cell
def _(PRODUCER_URL, infer_btn, json, mo, model_select, prb_input, requests):
    infer_result = ""
    if infer_btn.value:
        try:
            model_id = f"model-{'traffic' if 'Traffic' in model_select.value else 'anomaly' if 'Anomaly' in model_select.value else 'handover'}-001"
            resp = requests.post(
                f"{PRODUCER_URL}/r1-aiml/v1/mlInferenceJobs",
                json={"modelId": model_id, "inputData": {"cellMetrics": {"prbUtilization": prb_input.value}}, "inferenceMode": "SYNC"}
            )
            result = resp.json()
            infer_result = f"""
    ### Prediction Result
    ```json
    {json.dumps(result.get('prediction', {}), indent=2)}
    ```
    **Latency:** {result.get('latencyMs', 0):.2f} ms
    """
        except Exception as e:
            infer_result = f"Error: {e}"

    mo.md(infer_result if infer_result else "*Click 'Get Prediction' to run inference*")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Section 9: Summary - R1 API Cheatsheet

    ## What Goes Where?

    | Operation | Use ICS? | Endpoint |
    |-----------|----------|----------|
    | **Bootstrap** | No | `GET {PRODUCER}/bootstrap/v1/bootstrap-info` |
    | **Discover DME Types** | ✅ Yes | `GET {ICS}/data-consumer/v1/info-types` |
    | **Get Type Details** | ✅ Yes | `GET {ICS}/data-consumer/v1/info-types/{type}` |
    | **Create Data Job** | ✅ Yes | `PUT {ICS}/data-consumer/v1/info-jobs/{id}` |
    | **Delete Data Job** | ✅ Yes | `DELETE {ICS}/data-consumer/v1/info-jobs/{id}` |
    | **Receive Data (PUSH)** | No | Data arrives at your callback URL |
    | **Fetch Data (PULL)** | No | `GET {PRODUCER}/kpis/...` |
    | **A1 Policies** | No | `{PRODUCER}/a1-policy-management/v1/...` |
    | **AI/ML** | No | `{PRODUCER}/r1-aiml/v1/...` |

    ## Key Takeaways

    1. **ICS is the broker** - Use it for all data discovery and subscriptions
    2. **Bootstrap is the entry point** - Discover available APIs dynamically
    3. **PUSH vs PULL** - Choose based on your use case
    4. **Always cleanup** - Delete jobs when no longer needed

    ---

    # Exercises

    ## Exercise 1: Coverage Alert rApp
    Build an rApp that subscribes to `coverage-kpis` via ICS and alerts when RSRP < -95 dBm.

    ## Exercise 2: Multi-Cell Monitor
    Subscribe to data for all 5 cells and create a dashboard showing cell health.

    ## Exercise 3: Policy Automation
    Build an rApp that automatically creates a QoS policy when coverage degrades.

    ## Exercise 4: Anomaly Detection Pipeline
    Subscribe to `pm-data`, run it through the AnomalyDetector model, and log results.
    """)
    return


if __name__ == "__main__":
    app.run()
