# Call Center Transcript Evaluator

A comprehensive Python tool for evaluating call center transcripts using the Mistral 7B model. This tool analyzes customer service interactions to provide coaching feedback on language correctness, communication effectiveness, and overall performance.

## Features

### 🎯 Core Functionality
1. **Data Extraction**: Automatically extracts Q&A pairs from JSON transcript files
2. **English Language Evaluation**: Analyzes grammar, spelling, and language correctness
3. **Sentence Analysis**: Identifies long sentences and recommends crisp alternatives
4. **Word Pattern Detection**: Finds repetitive words and crutch words usage
5. **Knowledge Document Creation**: Generates knowledge base from transcripts
6. **Sentiment Analysis**: Evaluates emotional tone and provides feedback
7. **Topic Summarization**: Identifies themes and topics in conversations
8. **RAGAS Evaluation**: Framework ready for advanced evaluation metrics (commented for later use)

### 📊 Output Features
- Structured DataFrame with Q&A pairs and metadata
- Comprehensive evaluation reports
- Coaching feedback in natural language
- Performance metrics and recommendations
- Knowledge documents for training purposes

## Installation

### Prerequisites
- Python 3.8 or higher
- CPU-based system (no GPU required)
- Mistral 7B model file: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`

### Step 1: Install Dependencies
```bash
# Install basic requirements
pip install -r requirements.txt

# Install llama-cpp-python for CPU usage
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install additional NLP tools
pip install textblob
python -m textblob.download_corpora
```

### Step 2: Download Mistral Model
Download the Mistral 7B GGUF model file and place it in your project directory:
- Model: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- Source: Hugging Face or official Mistral repositories

### Step 3: Verify Installation
```python
from call_center_transcript_evaluator import CallCenterTranscriptEvaluator

# Test initialization
evaluator = CallCenterTranscriptEvaluator("path/to/your/mistral-model.gguf")
```

## Usage

### Basic Usage
```python
from call_center_transcript_evaluator import CallCenterTranscriptEvaluator

# Initialize evaluator with your model path
evaluator = CallCenterTranscriptEvaluator("mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Evaluate a transcript
results = evaluator.evaluate_transcript("Call Transcript Sample 1.json")

# Generate report
report = evaluator.generate_report(results, "evaluation_report.txt")
print(report)
```

### Command Line Usage
```bash
python call_center_transcript_evaluator.py
```

### Advanced Usage
```python
# Load and structure data manually
transcript_data = evaluator.load_transcript_json("transcript.json")
df = evaluator.extract_qa_pairs(transcript_data)

# Individual evaluations
english_eval = evaluator.evaluate_english_correctness("Sample text")
sentiment = evaluator.analyze_sentiment("Sample response")
topic = evaluator.summarize_topic_theme("Question", "Answer")

# Create knowledge documents
knowledge_docs = evaluator.create_knowledge_documents(df)
```

## JSON Input Format

Your transcript JSON files should follow this structure:

```json
{
  "call_ID": "12345",
  "CSR_ID": "JaneDoe123", 
  "call_date": "2024-02-01",
  "call_time": "02:16:43",
  "call_transcript": [
    "CSR: Thank you for calling ABC Travel, this is Jane. How may I assist you today?",
    "Customer: Yes, I need help with a reservation I made last week.",
    "CSR: I apologize for the trouble. May I have your name and reservation number?",
    "Customer: It's John Smith. My reservation number is 012345."
  ]
}
```

## Output Structure

### DataFrame Columns
- `call_ID`: Unique call identifier
- `CSR_ID`: Customer service representative ID
- `call_date`: Date of the call
- `call_time`: Time of the call
- `Questions`: Customer questions/statements
- `Answers`: CSR/Supervisor responses

### Evaluation Results
Each Q&A pair receives:
- **English Correctness**: Grammar and spelling evaluation with coaching
- **Sentence Analysis**: Length analysis and recommendations for clarity
- **Repetition Analysis**: Word repetition and crutch word detection
- **Sentiment Analysis**: Emotional tone evaluation with feedback
- **Topic Summary**: Theme identification and categorization

## RAGAS Integration (Future Use)

The code includes commented sections for RAGAS evaluation framework:

```python
# Uncomment these sections when ready to use RAGAS
# from ragas import evaluate
# from ragas.metrics import (
#     context_precision,
#     context_recall, 
#     context_entity_recall,
#     answer_relevancy,
#     faithfulness
# )
```

### RAGAS Metrics Included
- Context Precision
- Context Recall  
- Context Entities Recall
- Response Relevancy
- Faithfulness

## Configuration

### Model Configuration
```python
# Adjust these parameters in the CallCenterTranscriptEvaluator class
evaluator = CallCenterTranscriptEvaluator(
    model_path="your-model-path.gguf"
)

# Model parameters (in _load_model method)
self.model = Llama(
    model_path=self.model_path,
    n_ctx=4096,      # Context window size
    n_threads=4,     # CPU threads to use
    verbose=False    # Disable verbose output
)
```

### Analysis Parameters
```python
# Sentence length threshold (words)
long_sentence_threshold = 20

# Crutch words list (customizable)
crutch_words = [
    'um', 'uh', 'like', 'you know', 'actually', 
    'basically', 'literally', 'sort of', 'kind of'
]
```

## Performance Tips

### For CPU-Only Systems
1. **Optimize Thread Count**: Adjust `n_threads` based on your CPU cores
2. **Context Window**: Reduce `n_ctx` if memory is limited
3. **Batch Processing**: Process multiple files in sequence rather than parallel
4. **Model Size**: Use Q4_K_M quantization for balance of speed and quality

### Memory Management
```python
# For large transcript files, process in chunks
def process_large_transcript(file_path, chunk_size=10):
    # Implementation for chunked processing
    pass
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```
   Error: Failed to load model
   ```
   - Verify model file path and permissions
   - Check available system memory
   - Ensure llama-cpp-python is properly installed

2. **Out of Memory**
   ```
   Error: Not enough memory
   ```
   - Reduce `n_ctx` parameter
   - Use smaller model quantization
   - Process fewer transcripts simultaneously

3. **Slow Performance**
   - Increase `n_threads` (up to CPU core count)
   - Use SSD storage for model file
   - Close other memory-intensive applications

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
```

## Examples

### Example 1: Basic Evaluation
```python
evaluator = CallCenterTranscriptEvaluator("mistral-model.gguf")
results = evaluator.evaluate_transcript("sample_call.json")

# Access specific evaluations
for eval_data in results['evaluations']:
    print(f"Question: {eval_data['question']}")
    print(f"Sentiment: {eval_data['sentiment_analysis']['sentiment_label']}")
    print(f"Topic: {eval_data['topic_summary']}")
```

### Example 2: Batch Processing
```python
import glob

evaluator = CallCenterTranscriptEvaluator("mistral-model.gguf")

# Process all JSON files in directory
for json_file in glob.glob("transcripts/*.json"):
    print(f"Processing {json_file}...")
    results = evaluator.evaluate_transcript(json_file)
    
    # Save individual reports
    output_file = f"reports/{Path(json_file).stem}_report.txt"
    evaluator.generate_report(results, output_file)
```

### Example 3: Custom Analysis
```python
# Analyze specific aspects
text = "Um, well, I think that, you know, we can definitely help you with that issue."

# Individual analyses
repetition = evaluator.detect_repetition_and_crutch_words(text)
print(f"Crutch words found: {repetition['crutch_words']}")

sentiment = evaluator.analyze_sentiment(text)
print(f"Sentiment coaching: {sentiment['coaching_feedback']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Create an issue in the repository

## Changelog

### Version 1.0.0
- Initial release with core functionality
- Mistral 7B integration
- Comprehensive evaluation features
- RAGAS framework preparation
- Detailed reporting system

