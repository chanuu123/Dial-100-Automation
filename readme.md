# Towards Autonomous Emergency Response
## A Multi-Agent LLM System for End-to-End PSAP Automation

This repository contains the implementation and research artifacts for the **Master’s Thesis Project** titled **“Towards Autonomous Emergency Response: A Multi-Agent LLM System for End-to-End PSAP Automation.”**  
The project explores how Large Language Models (LLMs) and speech-processing pipelines can be combined into an agentic architecture to partially automate workflows in **Public Safety Answering Points (PSAPs)**.

---

## 📌 Project Overview

Public Safety Answering Points are the first line of contact during emergencies, where operators must quickly understand distressed speech, classify incidents, and dispatch resources. This project proposes a **dual-agent AI system** that mirrors real PSAP task flows:

1. **Call Agent** – Handles live emergency calls using an ASR → LLM → TTS loop.
2. **Resource Allocation Agent** – Analyzes structured call information and recommends emergency resources using a fine-tuned LLM.

The system is designed to reduce cognitive load on human operators, minimize dispatch latency, and improve consistency in emergency response decisions.

---

## 🧠 System Architecture

### 1. Call Agent
- **Automatic Speech Recognition (ASR):** OpenAI Whisper
- **Reasoning LLM:** Gemini Flash 2.5 Pro (low-latency, multilingual)
- **Text-to-Speech (TTS):** PyAudio-based speech synthesis
- **Function:**
  - Transcribes emergency calls in real time
  - Conducts structured, multi-turn dialogue
  - Extracts critical incident details (location, injuries, fire risk, etc.)

### 2. Resource Allocation Agent
- **Core Model:** Gemma3 7B (QLoRA fine-tuned)
- **Training Data:** Re-expressed FIR (First Information Report) data from real road accident cases
- **Tasks:**
  - Incident type classification
  - Resource requirement prediction (police, ambulance, fire)
  - Estimation of number of vehicles required

---

## 🏗️ Methodology

1. Model real PSAP task flows based on emergency control room operations.
2. Capture and structure emergency call data via the Call Agent.
3. Prepare domain-specific training data from FIR documents.
4. Fine-tune multiple open-source LLMs using **QLoRA**.
5. Evaluate models on accuracy, latency, and prediction error.
6. Select the best-performing model for deployment.

---

## 📊 Results

Comparative evaluation across multiple LLMs showed:

| Model        | Accident Type Accuracy | Resource Prediction | Vehicle Count Error | Avg. Latency |
|--------------|-----------------------|--------------------|--------------------|--------------|
| **Gemma3 7B** | **98.5%**              | **98%**             | **0.24**           | **~5.2 sec** |
| Llama2 7B    | 96.5%                  | 94.5%              | 0.33               | ~10.3 sec    |
| Mixtral 7B   | 91.5%                  | 87.5%              | 2.2                | ~11.5 sec    |
| DeepSeek R1  | 88.5%                  | 71.5%              | 2.32               | ~2.5 min     |

**Gemma3 7B** achieved the best balance of accuracy, stability, and low latency, making it suitable for time-critical PSAP environments.

---

## ⚙️ Tech Stack

- **Programming Language:** Python
- **ASR:** Whisper
- **LLMs:** Gemini Flash 2.5 Pro, Gemma3 7B, Llama2, Mixtral
- **Fine-Tuning:** QLoRA
- **Audio Processing:** PyAudio
- **Data Processing:** Pandas, Excel-based datasets

---

## 🚧 Limitations

- Limited multilingual reasoning performance in current LLMs
- High computational demand for handling large volumes of concurrent calls
- Potential vulnerability to adversarial or poisoning-style interactions

---



## ⭐ Repository Status

This repository is intended for **academic and research use**. Production deployment in real PSAPs would require extensive validation, compliance checks, and regulatory approval.

