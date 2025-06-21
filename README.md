# Call Transcript Data Extractor

This repository contains Python scripts to extract and analyze data from JSON files containing call transcripts. The scripts separate Customer questions from CSR (Customer Service Representative) answers and provide both JSON and DataFrame outputs for analysis.

## 🚀 Available Scripts

### 1. `extract_call_data.py` - JSON Output
Basic extraction script that outputs to JSON format.

### 2. `extract_call_data_dataframe.py` - DataFrame Output ⭐ **NEW**
Advanced script that creates pandas DataFrame with one-to-one Question-Answer mapping.

### 3. `demo_dataframe_usage.py` - Analysis Examples
Demonstrates various ways to analyze the DataFrame data.

### 4. `mistral_transcript_evaluator.py` - AI-Powered Evaluation ⭐ **NEW**
Advanced AI evaluation system using Mistral 7B for comprehensive call center performance coaching.

### 5. `setup_mistral_evaluator.py` - Setup Assistant
Automated setup script for Mistral evaluator dependencies and configuration.

### 6. `mistral_usage_guide.py` - Usage Examples
Comprehensive examples and usage patterns for the Mistral evaluator.

## ✨ Features

### Basic Features (Both Scripts)
- ✅ Extracts data from the `call_transcript` tag in JSON files
- ✅ Separates Customer data into "Questions" 
- ✅ Separates CSR/Supervisor data into "Answers"
- ✅ Preserves call metadata (Call ID, CSR ID, Date, Time)
- ✅ Comprehensive error handling

### DataFrame-Specific Features ⭐
- ✅ **One-to-one Question-Answer mapping** in pandas DataFrame
- ✅ **Metadata as separate columns** (call_ID, CSR_ID, call_date, call_time)
- ✅ **Interaction sequence numbering**
- ✅ **CSV and Excel export** capabilities
- ✅ **Advanced analysis functions**
- ✅ **Text length analysis**
- ✅ **Pattern detection** (unanswered questions, CSR-initiated responses)

### AI-Powered Evaluation Features 🤖 **NEW**
- ✅ **Mistral 7B Integration** - Local AI model for comprehensive evaluation
- ✅ **English Language Coaching** - Grammar, spelling, and professional language feedback
- ✅ **Sentence Structure Analysis** - Identifies long sentences and recommends crisp communication
- ✅ **Crutch Word Detection** - Identifies and provides coaching on filler words
- ✅ **Knowledge Document Creation** - Automatically generates knowledge base from transcripts
- ✅ **RAGAS Framework Evaluation** - Context Precision, Recall, Relevancy, and Faithfulness metrics
- ✅ **Sentiment Analysis** - Emotional tone assessment with coaching feedback
- ✅ **Topic Summarization** - Automatic categorization of customer questions
- ✅ **Comprehensive Coaching Reports** - Personalized feedback in conversational format

## 📖 Usage

### DataFrame Version (Recommended) ⭐

```python
# Basic usage - creates CSV and Excel files
python extract_call_data_dataframe.py

# Run analysis demo
python demo_dataframe_usage.py
```

### JSON Version

```python
# Basic usage - creates JSON output
python extract_call_data.py
```

### Using as Modules

```python
# DataFrame version
from extract_call_data_dataframe import extract_call_transcript_to_dataframe
df = extract_call_transcript_to_dataframe("your_file.json")

# JSON version  
from extract_call_data import extract_call_transcript_data
data = extract_call_transcript_data("your_file.json")
```

### AI-Powered Evaluation (Mistral 7B) 🤖 **NEW**

```bash
# Setup (automated)
python setup_mistral_evaluator.py

# Manual setup
pip install -r requirements_mistral.txt

# Download Mistral model (4.1GB)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Run evaluation
python mistral_transcript_evaluator.py

# View usage examples
python mistral_usage_guide.py
```

### Requirements

```bash
# For DataFrame version
pip install pandas openpyxl

# For AI evaluation
pip install -r requirements_mistral.txt

# JSON version has no external dependencies
```

## Input Format

The script expects a JSON file with the following structure:

```json
{
  "call_ID": "12345",
  "CSR_ID": "JaneDoe123", 
  "call_date": "2024-02-01",
  "call_time": "02:16:43",
  "call_transcript": [
    "CSR: Hello, how can I help you?",
    "Customer: I need help with my booking.",
    "CSR: I'd be happy to assist you.",
    ...
  ]
}
```

## 📊 Output Formats

### DataFrame Output (CSV/Excel) ⭐

The DataFrame version creates a structured table with one-to-one Question-Answer mapping:

| interaction_id | Questions | Answers | call_ID | CSR_ID | call_date | call_time |
|----------------|-----------|---------|---------|--------|-----------|-----------|
| 1 | Customer question 1 | CSR answer 1 | 12345 | JaneDoe123 | 2024-02-01 | 02:16:43 |
| 2 | Customer question 2 | CSR answer 2 | 12345 | JaneDoe123 | 2024-02-01 | 02:16:43 |
| ... | ... | ... | ... | ... | ... | ... |

**Key Benefits:**
- ✅ Perfect for data analysis and machine learning
- ✅ Easy to filter, sort, and analyze
- ✅ Metadata in separate columns for easy grouping
- ✅ Handles mismatched question/answer counts gracefully

### JSON Output

The JSON version generates a file with separated questions and answers:

```json
{
  "Questions": [
    "I need help with my booking.",
    "Can you check my reservation?",
    ...
  ],
  "Answers": [
    "Hello, how can I help you?",
    "I'd be happy to assist you.",
    ...
  ],
  "metadata": {
    "call_ID": "12345",
    "CSR_ID": "JaneDoe123",
    "call_date": "2024-02-01", 
    "call_time": "02:16:43",
    "total_questions": 8,
    "total_answers": 10
  }
}
```

## Functions

### `extract_call_transcript_data(json_file_path)`
- **Purpose**: Main extraction function
- **Parameters**: `json_file_path` (str) - Path to the input JSON file
- **Returns**: Dictionary with Questions, Answers, and metadata
- **Error Handling**: Returns `None` if file not found or invalid JSON

### `save_extracted_data(extracted_data, output_file_path)`
- **Purpose**: Save extracted data to JSON file
- **Parameters**: 
  - `extracted_data` (dict) - Data returned from extraction function
  - `output_file_path` (str) - Path for output file

### `print_extracted_data(extracted_data)`
- **Purpose**: Display extracted data in readable format
- **Parameters**: `extracted_data` (dict) - Data to display

## Requirements

- Python 3.6+
- No external dependencies (uses only built-in modules)

## Error Handling

The script handles common errors:
- ❌ File not found
- ❌ Invalid JSON format  
- ❌ Missing required fields
- ❌ File write permissions

## Example Output

```
============================================================
CALL TRANSCRIPT DATA EXTRACTION
============================================================
Call ID: 12345
CSR ID: JaneDoe123
Date: 2024-02-01
Time: 02:16:43
Total Questions: 8
Total Answers: 10

QUESTIONS (Customer):
----------------------------------------
1. Yes, I need help with a reservation I made last week...

ANSWERS (CSR/Supervisor):
----------------------------------------
1. Thank you for calling ABC Travel, this is Jane...

============================================================
EXTRACTION COMPLETE!
✅ Questions extracted: 8
✅ Answers extracted: 10
✅ Output saved to: extracted_call_data.json
```

## Notes

- The script treats both "CSR:" and "Supervisor:" entries as answers
- Customer entries are treated as questions
- All prefixes ("Customer:", "CSR:", "Supervisor:") are automatically removed
- Metadata from the original JSON is preserved in the output

## 🤖 AI-Powered Evaluation with Mistral 7B

### Overview
The Mistral 7B evaluator provides comprehensive AI-powered analysis of call center transcripts for performance coaching. It runs locally on your machine for privacy and security.

### Key Evaluation Areas

#### 1. **English Language Correctness** 📝
- **Grammar Analysis**: Identifies and corrects grammatical errors
- **Spelling Check**: Detects and suggests corrections for misspelled words
- **Professional Language**: Recommends more professional alternatives
- **Coaching Format**: *"Your grammar score is 8.5/10. Consider using 'may I' instead of 'can I' for more formal communication."*

#### 2. **Sentence Structure & Clarity** 🎯
- **Length Analysis**: Identifies overly long sentences (>20 words)
- **Clarity Assessment**: Evaluates communication effectiveness
- **Crisp Recommendations**: Suggests breaking complex sentences
- **Coaching Format**: *"Your average sentence length is 25 words. Try breaking this into 2 shorter sentences for better clarity."*

#### 3. **Crutch Word Detection** 🚫
- **Filler Words**: Detects "um", "uh", "like", "you know", etc.
- **Repetitive Language**: Identifies overused words and phrases
- **Professional Alternatives**: Suggests replacement vocabulary
- **Coaching Format**: *"You used 'actually' 4 times. Try varying your language with 'in fact' or 'specifically'."*

#### 4. **Knowledge Document Creation** 📚
- **Automatic Extraction**: Creates knowledge base from all transcripts
- **Topic Categorization**: Groups information by themes
- **Best Practices**: Identifies successful resolution patterns
- **Reference Material**: Generates searchable knowledge articles

#### 5. **RAGAS Framework Evaluation** 📊
Evaluates responses using industry-standard metrics:

- **Context Precision** (0.0-1.0): How relevant is the provided information?
- **Context Recall** (0.0-1.0): How complete is the information coverage?
- **Context Entity Recall** (0.0-1.0): Are all important entities mentioned?
- **Answer Relevancy** (0.0-1.0): How well does the answer address the question?
- **Faithfulness** (0.0-1.0): How accurate is the information provided?

**Coaching Format**: *"Your faithfulness score is 0.9 - excellent! You covered all the facts except the refund timeline. Consider mentioning specific timeframes for complete accuracy."*

#### 6. **Sentiment Analysis** 😊
- **Emotional Tone**: Positive, negative, or neutral assessment
- **Professional Impact**: How sentiment affects customer experience
- **Coaching Feedback**: Personalized emotional intelligence coaching
- **Coaching Format**: *"You maintained a positive and empathetic tone throughout the interaction. Your customer likely felt heard and valued."*

#### 7. **Topic Summarization** 🏷️
- **Question Categorization**: Billing, technical, complaint, inquiry, etc.
- **Issue Classification**: Urgency level and complexity assessment
- **Theme Identification**: Main topics and customer needs
- **Trend Analysis**: Common issues across multiple calls

### Sample Evaluation Output

```json
{
  "interaction_id": 1,
  "english_evaluation": {
    "english_score": 8.5,
    "feedback": "Good grammar overall. Consider using 'may I' instead of 'can I'..."
  },
  "sentence_analysis": {
    "clarity_score": 7.8,
    "avg_sentence_length": 18.5,
    "feedback": "Communication is clear. One sentence could be shortened..."
  },
  "ragas_scores": {
    "context_precision": 0.85,
    "faithfulness": 0.92,
    "answer_relevancy": 0.88
  },
  "ragas_coaching": "Your faithfulness score of 0.92 is excellent! You provided accurate information...",
  "sentiment_analysis": {
    "sentiment_category": "positive",
    "feedback": "You maintained a professional and empathetic tone..."
  },
  "question_topic": "Flight cancellation - High urgency customer complaint"
}
```

### Generated Reports

#### 1. **Individual Coaching Reports**
- Personalized feedback for each interaction
- Specific improvement recommendations
- Strengths and areas for development
- Action items for skill building

#### 2. **Summary Performance Reports**
- Overall performance metrics
- Team-wide trends and patterns
- Training recommendations
- Coaching priorities

#### 3. **Knowledge Base Documents**
- Best practice examples
- Common issue resolutions
- Standard operating procedures
- Training materials

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space (for model and data)
- **CPU**: Multi-core processor recommended
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher

### Privacy & Security

- ✅ **Fully Local**: All processing happens on your machine
- ✅ **No Data Upload**: Transcripts never leave your environment
- ✅ **GDPR Compliant**: Complete data privacy control
- ✅ **Offline Capable**: Works without internet connection
