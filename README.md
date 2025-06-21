# Call Transcript Data Extractor

This repository contains Python scripts to extract and analyze data from JSON files containing call transcripts. The scripts separate Customer questions from CSR (Customer Service Representative) answers and provide both JSON and DataFrame outputs for analysis.

## 🚀 Available Scripts

### 1. `extract_call_data.py` - JSON Output
Basic extraction script that outputs to JSON format.

### 2. `extract_call_data_dataframe.py` - DataFrame Output ⭐ **NEW**
Advanced script that creates pandas DataFrame with one-to-one Question-Answer mapping.

### 3. `demo_dataframe_usage.py` - Analysis Examples
Demonstrates various ways to analyze the DataFrame data.

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

### Requirements

```bash
# For DataFrame version
pip install pandas openpyxl

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
