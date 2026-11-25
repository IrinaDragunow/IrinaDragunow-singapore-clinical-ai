# Singapore Clinical AI (educational simulation)

A learning project (2025) exploring semantic search and simple multimodal input in a healthcare setting.

> **Not for medical use.** All patient cases, guidelines and figures in this app are fictional sample data written for the exercise.

**Live demo:** https://irinadragunow-singapore-clinical-ai.streamlit.app

## What it does

- **Semantic search:** embeds a small, hand-written set of sample guidelines with `all-MiniLM-L6-v2` and ranks them by cosine similarity to a clinical note; falls back to keyword search if embeddings are unavailable.
- **Entity extraction:** rule-based (regex and dictionaries) detection of medications, conditions and vital signs in the note.
- **Image input:** basic OpenCV image features with template responses; no trained diagnostic model.

## Stack

Python · Streamlit · sentence-transformers · scikit-learn · OpenCV (optional) · Plotly · Pandas

## Run locally

```bash
git clone https://github.com/IrinaDragunow/IrinaDragunow-singapore-clinical-ai.git
cd IrinaDragunow-singapore-clinical-ai
pip install -r requirements.txt
streamlit run singapore_clinical_ai_production.py
```

## Limitations

- Knowledge base and cases are invented samples, not clinical guidelines.
- Entity extraction is pattern-based; image "analysis" is illustrative only.
- Prototype for learning, not a production system.
