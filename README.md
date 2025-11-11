# Agentic Financial Fraud Detector

An **agentic AI system for financial fraud detection and analyst triage**, built with **Streamlit**, **Isolation Forests**, and **local LLM reasoning (Ollama)**.  
The app simulates, detects, and explains suspicious financial transactions — blending machine learning detection with natural-language interpretability.

---

## Overview

| Tab | Function |
|-----|-----------|
| **Run** | Simulates one of the 3 types of attacks that the user chooses. Runs the detection pipeline using Isolation Forest or XGBoost based on user choice |
| **Inspect** | Lets analysts explore flagged transactions. Does per transaction analysis and reasons via LLM why that row was flagged as an alert|
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
3. **Detection Engine** — Uses Isolation Forest (or XGBoost) anomaly scores.
4. **Attack Injection** — Simulates adversarial bursts (like cashout, repeated small transactions, stealth escalation) to evaluate detection performance.  
5. **LLM Narrative** — Generates analyst summaries explaining anomalies and recommending next steps.  

---

## Running Locally

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Run the Streamlit app

streamlit run app/app_simple.py

### 3️⃣ Start Ollama for local LLM explanations (Optional) 

ollama run phi3:mini

Choose your attack and model on the app - and feel free to experiment different scenarios!

---

## Example Outputs

### Run Tab

- Inject synthetic fraud bursts
  
- Choose which model you want to use

- Choose what feature set do you want to consider


### Inspect Tab

- Analyse each transaction separately

- Understand why a particular row was flagged as an alert

### Report Tab

- Generate metrics (precision@K, recall@K)

- Produce LLM-based analyst summaries + investigation strategies

---

## Branches

- main	: Documentation and overview (README for setup and description)
- master : Full implementation of the simplified fraud detection system




