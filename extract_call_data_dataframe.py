import json
import pandas as pd
from typing import Optional, Dict, Any

def extract_call_transcript_to_dataframe(json_file_path: str) -> Optional[pd.DataFrame]:
    """
    Extract data from JSON file's call_transcript tag and create a DataFrame.
    CSR data goes to 'Answers' and Customer data goes to 'Questions' with one-to-one mapping.
    Metadata is added as separate columns.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        pd.DataFrame: DataFrame with Questions, Answers, and metadata columns
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
        
        # Create one-to-one mapping by padding shorter list with None
        max_length = max(len(questions), len(answers))
        
        # Pad questions list if shorter
        while len(questions) < max_length:
            questions.append(None)
        
        # Pad answers list if shorter
        while len(answers) < max_length:
            answers.append(None)
        
        # Extract metadata
        metadata = {
            'call_ID': data.get('call_ID'),
            'CSR_ID': data.get('CSR_ID'),
            'call_date': data.get('call_date'),
            'call_time': data.get('call_time')
        }
        
        # Create DataFrame
        df_data = {
            'Questions': questions,
            'Answers': answers,
            'call_ID': [metadata['call_ID']] * max_length,
            'CSR_ID': [metadata['CSR_ID']] * max_length,
            'call_date': [metadata['call_date']] * max_length,
            'call_time': [metadata['call_time']] * max_length
        }
        
        df = pd.DataFrame(df_data)
        
        # Add interaction sequence number
        df.insert(0, 'interaction_id', range(1, len(df) + 1))
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def save_dataframe_to_csv(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_file_path (str): Path to save the CSV file
    """
    try:
        df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"DataFrame successfully saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")

def save_dataframe_to_excel(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Save DataFrame to Excel file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_file_path (str): Path to save the Excel file
    """
    try:
        df.to_excel(output_file_path, index=False, engine='openpyxl')
        print(f"DataFrame successfully saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving Excel file: {str(e)}")
        print("Note: Install openpyxl with 'pip install openpyxl' for Excel support")

def print_dataframe_summary(df: pd.DataFrame) -> None:
    """
    Print summary information about the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
    """
    if df is None or df.empty:
        print("No data to display.")
        return
    
    print("=" * 80)
    print("CALL TRANSCRIPT DATAFRAME SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"Total Interactions: {len(df)}")
    print(f"Questions (non-null): {df['Questions'].notna().sum()}")
    print(f"Answers (non-null): {df['Answers'].notna().sum()}")
    print()
    
    # Metadata info
    metadata_cols = ['call_ID', 'CSR_ID', 'call_date', 'call_time']
    print("METADATA:")
    print("-" * 40)
    for col in metadata_cols:
        if col in df.columns:
            unique_val = df[col].iloc[0] if not df[col].isna().all() else "N/A"
            print(f"{col}: {unique_val}")
    print()
    
    # DataFrame structure
    print("DATAFRAME STRUCTURE:")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Sample data
    print("SAMPLE DATA (First 5 rows):")
    print("-" * 40)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.head())
    print()

def analyze_interaction_patterns(df: pd.DataFrame) -> None:
    """
    Analyze interaction patterns in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    if df is None or df.empty:
        return
    
    print("INTERACTION ANALYSIS:")
    print("-" * 40)
    
    # Count non-null values
    questions_count = df['Questions'].notna().sum()
    answers_count = df['Answers'].notna().sum()
    
    print(f"Total Customer Questions: {questions_count}")
    print(f"Total CSR/Supervisor Answers: {answers_count}")
    
    # Check for missing mappings
    missing_questions = df[df['Questions'].isna() & df['Answers'].notna()]
    missing_answers = df[df['Answers'].isna() & df['Questions'].notna()]
    
    if len(missing_questions) > 0:
        print(f"⚠️  Answers without corresponding Questions: {len(missing_questions)}")
    
    if len(missing_answers) > 0:
        print(f"⚠️  Questions without corresponding Answers: {len(missing_answers)}")
    
    # Perfect mappings
    perfect_mappings = df[df['Questions'].notna() & df['Answers'].notna()]
    print(f"✅ Perfect Question-Answer pairs: {len(perfect_mappings)}")
    print()

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = "Call Transcript Sample 1.json"
    csv_output_file = "call_transcript_dataframe.csv"
    excel_output_file = "call_transcript_dataframe.xlsx"
    
    print("Extracting call transcript data to DataFrame...")
    print("=" * 60)
    
    # Extract data to DataFrame
    df = extract_call_transcript_to_dataframe(input_file)
    
    if df is not None:
        # Print summary
        print_dataframe_summary(df)
        
        # Analyze patterns
        analyze_interaction_patterns(df)
        
        # Save to CSV
        save_dataframe_to_csv(df, csv_output_file)
        
        # Try to save to Excel (optional)
        try:
            save_dataframe_to_excel(df, excel_output_file)
        except ImportError:
            print("Excel export skipped - install openpyxl for Excel support")
        
        print("=" * 60)
        print("EXTRACTION COMPLETE!")
        print(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        print(f"✅ CSV saved to: {csv_output_file}")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Display detailed view of first few interactions
        print("\nDETAILED VIEW (First 3 interactions):")
        print("-" * 60)
        for idx, row in df.head(3).iterrows():
            print(f"Interaction {row['interaction_id']}:")
            print(f"  Question: {row['Questions'] if pd.notna(row['Questions']) else 'N/A'}")
            print(f"  Answer: {row['Answers'] if pd.notna(row['Answers']) else 'N/A'}")
            print(f"  Call ID: {row['call_ID']}")
            print()
            
    else:
        print("❌ Failed to extract data from the JSON file.")

