# Call Transcript Data Extractor

This Python script extracts data from JSON files containing call transcripts, specifically designed to separate Customer questions from CSR (Customer Service Representative) answers.

## Features

- ✅ Extracts data from the `call_transcript` tag in JSON files
- ✅ Separates Customer data into "Questions" 
- ✅ Separates CSR/Supervisor data into "Answers"
- ✅ Preserves call metadata (Call ID, CSR ID, Date, Time)
- ✅ Provides detailed console output
- ✅ Saves extracted data to a new JSON file
- ✅ Error handling for file operations

## Usage

### Basic Usage

```python
python extract_call_data.py
```

This will process the default file `"Call Transcript Sample 1.json"` and create `"extracted_call_data.json"`.

### Using as a Module

```python
from extract_call_data import extract_call_transcript_data, save_extracted_data

# Extract data from your JSON file
data = extract_call_transcript_data("your_file.json")

# Save to output file
save_extracted_data(data, "output.json")
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

## Output Format

The script generates a JSON file with separated questions and answers:

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

