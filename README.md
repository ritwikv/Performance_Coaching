# Performance_Coaching

A Streamlit-based dashboard for analyzing call transcripts and providing performance coaching feedback using AI evaluation.

## Features

- 📊 Interactive dashboard for transcript analysis
- 🤖 Automated Mistral AI evaluation (no manual command line required!)
- 📁 File upload support for JSON transcripts
- 📈 Performance metrics and coaching recommendations
- 💾 Results storage and history

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit dashboard:
```bash
streamlit run streamlit_feedback_dashboard.py
```

3. Upload a transcript and click "Run Mistral Evaluation" - the system will automatically execute the evaluation script!

## Files

- `streamlit_feedback_dashboard.py` - Main dashboard application
- `mistral_transcript_evaluator.py` - AI evaluation script (runs automatically)
- `Call Transcript Sample 1.json` - Sample transcript data
- `requirements.txt` - Python dependencies

## Usage

The dashboard automatically handles running the Mistral evaluation when you click the "Run Mistral Evaluation" button. No need to manually run commands from the terminal!
