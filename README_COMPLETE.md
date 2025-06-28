# Call Center Transcript Evaluation System

A comprehensive Python solution for evaluating call center transcripts using Mistral 7B model with RAG pipeline, DeepEval framework, and Streamlit dashboard.

## 🌟 Features

### Core Capabilities
- **📊 Data Processing**: JSON to DataFrame conversion with Q&A mapping
- **🤖 Mistral 7B Integration**: CPU-optimized local model inference
- **🔍 RAG Pipeline**: Knowledge base creation and expert answer generation
- **📈 Quality Analysis**: Sentence structure, repetition, hold/transfer detection
- **🎯 DeepEval Metrics**: Answer relevancy and correctness evaluation
- **💭 Sentiment Analysis**: Emotional tone and coaching feedback
- **📋 Comprehensive Reporting**: 200-word summaries and detailed feedback
- **🖥️ Streamlit Dashboard**: Interactive web interface

### Analysis Components
1. **Sentence Analysis**: Identifies long sentences and recommends crisp alternatives
2. **Repetition Detection**: Finds unnecessary repetition and crutch words
3. **Hold/Transfer Analysis**: Detects hold requests and call transfers with reasons
4. **AHT Correlation**: Links feedback to Average Handling Time impact
5. **Knowledge Generation**: Creates expert-level responses from transcripts
6. **Performance Evaluation**: Compares CSR answers to expert responses
7. **Sentiment Coaching**: Provides personalized feedback on communication style
8. **Topic Summarization**: Identifies main themes and discussion points

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- CPU with multiple cores (GPU not required)
- Mistral 7B model file (4.1GB)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Performance_Coaching
```

2. **Run setup script**
```bash
python setup.py
```

3. **Download Mistral model**
```bash
# Option 1: Using huggingface-hub
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Option 2: Manual download from Hugging Face
# Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# Download: mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

4. **Launch dashboard**
```bash
streamlit run enhanced_streamlit_dashboard.py
# OR
./launch_dashboard.sh
```

## 📁 Project Structure

```
Performance_Coaching/
├── data_processor.py              # JSON to DataFrame conversion
├── mistral_model.py              # Mistral 7B model integration
├── rag_pipeline.py               # RAG implementation
├── quality_analyzer.py           # Call quality analysis
├── deepeval_mistral.py           # DeepEval integration
├── sentiment_analyzer.py         # Sentiment and topic analysis
├── evaluation_orchestrator.py    # Main coordination engine
├── enhanced_streamlit_dashboard.py # Web interface
├── config.py                     # Configuration management
├── setup.py                      # Installation script
├── requirements_complete.txt     # All dependencies
├── Call Transcript Sample 1.json # Sample data
└── README_COMPLETE.md           # This file
```

## 🔧 Configuration

### System Configuration
The system uses `config.py` for centralized configuration:

```python
from config import SystemConfig, ConfigManager

# Load configuration
config = ConfigManager.load_config()

# Modify settings
config.model.temperature = 0.1
config.enable_rag = True
config.enable_deepeval = True

# Save configuration
ConfigManager.save_config(config)
```

### Environment Variables
```bash
export MISTRAL_MODEL_PATH="path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
export OUTPUT_DIRECTORY="./results"
export LOG_LEVEL="INFO"
```

## 📊 Usage Examples

### 1. Streamlit Dashboard (Recommended)
```bash
streamlit run enhanced_streamlit_dashboard.py
```
- Upload JSON transcript files
- Configure analysis settings
- Click "Run Mistral Evaluation"
- View comprehensive results and visualizations

### 2. Command Line Interface
```python
from evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    mistral_model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    enable_rag=True,
    enable_deepeval=True
)

# Initialize and run
orchestrator = EvaluationOrchestrator(config)
orchestrator.initialize()
results = orchestrator.evaluate_transcript_file("transcript.json")

# Save results
orchestrator.save_results("results.json")
```

### 3. Individual Component Usage
```python
# Data processing only
from data_processor import CallTranscriptProcessor
processor = CallTranscriptProcessor()
df = processor.process_multiple_transcripts(["file1.json", "file2.json"])

# Quality analysis only
from quality_analyzer import CallQualityAnalyzer
analyzer = CallQualityAnalyzer()
analysis = analyzer.analyze_single_response("CSR response text")

# Sentiment analysis only
from sentiment_analyzer import SentimentTopicAnalyzer
sentiment_analyzer = SentimentTopicAnalyzer()
result = sentiment_analyzer.analyze_conversation("question", "answer")
```

## 📋 Input Format

### JSON Transcript Structure
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

### Required Fields
- `call_ID`: Unique identifier for the call
- `CSR_ID`: Customer service representative identifier
- `call_date`: Date in YYYY-MM-DD format
- `call_time`: Time in HH:MM:SS format
- `call_transcript`: Array of conversation turns with "CSR:" or "Customer:" prefixes

## 📈 Output Examples

### Evaluation Results
```json
{
  "call_id": "12345",
  "csr_id": "JaneDoe123",
  "quality_scores": {
    "overall_score": 8.2,
    "clarity_score": 7.8,
    "conciseness_score": 8.5
  },
  "deepeval_scores": {
    "relevancy": {"score": 0.85, "passed": true},
    "correctness": {"score": 0.78, "passed": true},
    "overall": {"score": 0.815, "passed": true}
  },
  "sentiment_analysis": {
    "sentiment_label": "Positive",
    "emotional_tone": "Professional",
    "coaching_feedback": "Excellent professional tone with empathetic approach."
  },
  "aht_impact": {
    "aht_impact_level": "Low",
    "estimated_time_increase": "5%"
  },
  "concise_summary": "Excellent performance with 8.2/10 quality score. Professional tone with empathetic approach. Successfully handled reservation inquiry with minimal AHT impact."
}
```

### Coaching Feedback Examples
- **Quality**: "Your clarity score is 7.8/10. Consider breaking down long sentences for better customer understanding."
- **Sentiment**: "Excellent positive sentiment with professional tone. Your empathetic approach builds customer trust."
- **Relevancy**: "Your relevancy score is 0.9/1.0 - you directly addressed all customer concerns."
- **AHT Impact**: "Low AHT impact (5% increase). Good efficiency with minimal hold requests."

## 🔍 Analysis Details

### 1. Data Processing Pipeline
- Extracts metadata (call_ID, CSR_ID, call_date, call_time)
- Maps CSR/Customer interactions to Questions/Answers
- Handles greeting identification when CSR starts conversation
- Creates one-to-one Q&A mapping in DataFrame format

### 2. Quality Analysis Engine
- **Sentence Analysis**: Identifies sentences >25 words, recommends 15-20 word alternatives
- **Repetition Detection**: Finds crutch words (um, uh, like) and repeated phrases
- **Hold Request Analysis**: Detects "please hold", "one moment", etc.
- **Transfer Detection**: Identifies call transfers and reasons
- **AHT Correlation**: Links quality issues to handling time impact

### 3. RAG Pipeline
- **Document Chunking**: Splits conversations into 512-token chunks with 50-token overlap
- **Embedding Generation**: Uses sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: ChromaDB for efficient similarity search
- **Expert Answer Generation**: Creates ground truth responses using Mistral 7B
- **Retrieval**: Top-5 relevant chunks for context-aware answers

### 4. DeepEval Integration
- **Custom LLM Wrapper**: Adapts Mistral 7B for DeepEval framework
- **Relevancy Metric**: Measures how well CSR answers address customer questions
- **Correctness Metric**: Compares CSR answers to expert responses
- **Scoring**: 0.0-1.0 scale with configurable thresholds (default: 0.7)

### 5. Sentiment Analysis
- **Traditional Methods**: TextBlob for baseline sentiment scoring
- **Mistral Enhancement**: AI-powered emotional tone detection
- **Coaching Generation**: Personalized feedback based on communication style
- **Topic Identification**: Main themes and concern categorization

## ⚙️ Performance Optimization

### CPU Optimization
- **Model Quantization**: Uses Q4_K_M quantized model (4.1GB vs 13GB)
- **Thread Configuration**: Automatically uses all available CPU cores
- **Memory Management**: Efficient model loading and caching
- **Batch Processing**: Optimized for single-conversation evaluation

### Caching Strategy
- **Model Responses**: Caches similar prompts to avoid re-inference
- **Embeddings**: Stores computed embeddings for reuse
- **Configuration**: Persistent settings across sessions

### Memory Usage
- **Recommended**: 8GB+ RAM
- **Minimum**: 6GB RAM
- **Model Loading**: ~4.5GB for Mistral 7B Q4_K_M
- **Processing**: Additional 1-2GB during evaluation

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Fails**
```
Error: Model file not found
Solution: Ensure mistral-7b-instruct-v0.2.Q4_K_M.gguf is in project directory
```

2. **Memory Error**
```
Error: Out of memory
Solution: Close other applications, use smaller batch sizes
```

3. **Import Errors**
```
Error: Module not found
Solution: Run pip install -r requirements_complete.txt
```

4. **Slow Performance**
```
Issue: Evaluation takes too long
Solution: Reduce n_ctx, disable RAG for faster processing
```

### Debug Mode
```python
from config import SystemConfig
config = SystemConfig()
config.debug_mode = True
config.log_level = "DEBUG"
```

### Log Files
- Application logs: `logs/evaluation.log`
- Error logs: `logs/errors.log`
- Performance logs: `logs/performance.log`

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test Components
```python
# Test data processor
python -c "from data_processor import CallTranscriptProcessor; print('✅ Data processor OK')"

# Test Mistral model
python -c "from mistral_model import MistralEvaluator; print('✅ Mistral model OK')"

# Test configuration
python -c "from config import SystemConfig; print('✅ Configuration OK')"
```

## 📚 API Reference

### Main Classes

#### EvaluationOrchestrator
```python
class EvaluationOrchestrator:
    def __init__(self, config: EvaluationConfig)
    def initialize(self) -> bool
    def evaluate_transcript_file(self, file_path: str) -> List[EvaluationResult]
    def save_results(self, output_path: str, format: str) -> bool
```

#### CallTranscriptProcessor
```python
class CallTranscriptProcessor:
    def load_json_transcript(self, file_path: str) -> Optional[Dict]
    def process_single_transcript(self, transcript_data: Dict) -> List[Dict]
    def process_multiple_transcripts(self, file_paths: List[str]) -> pd.DataFrame
```

#### MistralEvaluator
```python
class MistralEvaluator:
    def __init__(self, config: MistralConfig)
    def load_model(self) -> bool
    def analyze_sentence_structure(self, text: str) -> Dict[str, Any]
    def analyze_sentiment(self, text: str) -> Dict[str, Any]
    def generate_expert_answer(self, question: str) -> str
```

## 🤝 Contributing

### Development Setup
```bash
git clone <repository-url>
cd Performance_Coaching
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_complete.txt
pip install -e .
```

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings for all functions
- Include unit tests for new features

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Mistral 7B model
- **DeepEval** framework for evaluation metrics
- **ChromaDB** for vector storage
- **Streamlit** for the web interface
- **Sentence Transformers** for embeddings

## 📞 Support

For questions, issues, or feature requests:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description
4. Include system information and error logs

---

**Built with ❤️ for call center performance improvement**

