import json

def extract_call_transcript_data(json_file_path):
    """
    Extract data from JSON file's call_transcript tag.
    CSR data goes to 'Answers' and Customer data goes to 'Questions'.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary containing Questions and Answers
    """
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract call_transcript data
        call_transcript = data.get('call_transcript', [])
        
        # Initialize lists for questions and answers
        questions = []
        answers = []
        
        # Process each entry in call_transcript
        for entry in call_transcript:
            if entry.startswith('Customer:'):
                # Remove 'Customer:' prefix and add to questions
                question = entry.replace('Customer:', '').strip()
                questions.append(question)
            elif entry.startswith('CSR:'):
                # Remove 'CSR:' prefix and add to answers
                answer = entry.replace('CSR:', '').strip()
                answers.append(answer)
            elif entry.startswith('Supervisor:'):
                # Treat supervisor responses as CSR answers
                answer = entry.replace('Supervisor:', '').strip()
                answers.append(answer)
        
        # Create result dictionary
        result = {
            'Questions': questions,
            'Answers': answers,
            'metadata': {
                'call_ID': data.get('call_ID'),
                'CSR_ID': data.get('CSR_ID'),
                'call_date': data.get('call_date'),
                'call_time': data.get('call_time'),
                'total_questions': len(questions),
                'total_answers': len(answers)
            }
        }
        
        return result
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def save_extracted_data(extracted_data, output_file_path):
    """
    Save extracted data to a JSON file.
    
    Args:
        extracted_data (dict): The extracted questions and answers
        output_file_path (str): Path to save the output file
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(extracted_data, file, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def print_extracted_data(extracted_data):
    """
    Print the extracted data in a readable format.
    
    Args:
        extracted_data (dict): The extracted questions and answers
    """
    if not extracted_data:
        return
    
    print("=" * 60)
    print("CALL TRANSCRIPT DATA EXTRACTION")
    print("=" * 60)
    
    # Print metadata
    metadata = extracted_data.get('metadata', {})
    print(f"Call ID: {metadata.get('call_ID')}")
    print(f"CSR ID: {metadata.get('CSR_ID')}")
    print(f"Date: {metadata.get('call_date')}")
    print(f"Time: {metadata.get('call_time')}")
    print(f"Total Questions: {metadata.get('total_questions')}")
    print(f"Total Answers: {metadata.get('total_answers')}")
    print()
    
    # Print Questions (Customer data)
    print("QUESTIONS (Customer):")
    print("-" * 40)
    for i, question in enumerate(extracted_data.get('Questions', []), 1):
        print(f"{i}. {question}")
        print()
    
    # Print Answers (CSR data)
    print("ANSWERS (CSR/Supervisor):")
    print("-" * 40)
    for i, answer in enumerate(extracted_data.get('Answers', []), 1):
        print(f"{i}. {answer}")
        print()

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = "Call Transcript Sample 1.json"
    output_file = "extracted_call_data.json"
    
    print("Extracting data from call transcript...")
    
    # Extract data from JSON file
    extracted_data = extract_call_transcript_data(input_file)
    
    if extracted_data:
        # Print the extracted data
        print_extracted_data(extracted_data)
        
        # Save to output file
        save_extracted_data(extracted_data, output_file)
        
        print("=" * 60)
        print("EXTRACTION COMPLETE!")
        print(f"✅ Questions extracted: {len(extracted_data['Questions'])}")
        print(f"✅ Answers extracted: {len(extracted_data['Answers'])}")
        print(f"✅ Output saved to: {output_file}")
    else:
        print("❌ Failed to extract data from the JSON file.")

