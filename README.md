# Agentic Financial Fraud Detector

An **agentic AI system for financial fraud detection and analyst triage**, built with **Streamlit**, **Isolation Forests**, and **local LLM reasoning (Ollama)**.  
The app simulates, detects, and explains suspicious financial transactions — blending machine learning detection with natural-language interpretability.

---

## Overview

| Tab | Function |
|-----|-----------|
| **Run** | Simulates transactions and injects synthetic “attack” bursts (e.g., \$49.99 repeated transactions). Runs the detection pipeline using Isolation Forest + heuristic rule-based scoring. |
| **Inspect** | Lets analysts explore flagged transactions. Compares each transaction to a user’s historical profile (z-score, velocity, fan-out, merchant familiarity). |
| **Report** | Summarizes top-K alerts, recall, precision, and generates **LLM-based incident summaries** and **triage recommendations** for investigation. |

---

---

## Tech Stack

- **Python 3.11+**
- **Streamlit** — Interactive visualization and UI  
- **scikit-learn** — Isolation Forest anomaly detection  
- **Altair** — Dynamic charts for user behavior visualization  
- **Ollama + phi-3-mini** — Local LLM for analyst narratives  
- **NumPy / Pandas** — Data wrangling and feature generation  

---

## How It Works

1. **Data Loading** — Loads demo or synthetic transaction data.  
2. **Feature Engineering** — Builds velocity, fan-out, and amount z-score features per user.  
3. **Detection Engine** — Combines Isolation Forest anomaly scores and burst-rule scores into a `combined_score`.  
4. **Attack Injection (Optional)** — Simulates adversarial bursts (\$49.99 repeated payments) to evaluate detection performance.  
5. **LLM Narrative** — Generates analyst summaries explaining anomalies and recommending next steps.  

---

## Running Locally

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Run the Streamlit app

streamlit run app/app_simple.py

### 3️⃣ Start Ollama for local LLM explanations (Optional) 

ollama run phi3:mini

---

## Example Outputs

### Run Tab

- Inject synthetic fraud bursts

- Detect anomalies using rule + model scoring

- Visualize scores over time

### Inspect Tab

- Compare suspicious transactions against user patterns

- Visualize amount distributions and hour-of-day anomalies

### Report Tab

- Generate metrics (precision@K, recall@K)

- Produce LLM-based analyst summaries + triage recommendations

---

## Branches

- main	: Documentation and overview (this README, architecture, and setup)
- master : Full implementation of the simplified fraud detection system




