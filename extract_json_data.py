"""
JSON Call Transcript Data Extractor
Extracts data from 'call_transcript' tag and maps CSR -> Answers, Customer -> Questions
"""

import json
import pandas as pd
from typing import Dict, List, Any

def extract_call_transcript_data(json_file_path: str) -> pd.DataFrame:
    """
    Extract data from JSON file and create DataFrame with Questions and Answers
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        pd.DataFrame: DataFrame with Questions (Customer) and Answers (CSR) columns
    """
    
    print(f"🔄 Loading JSON file: {json_file_path}")
    
    try:
        # Load JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print("✅ JSON file loaded successfully")
        
        # Extract call_transcript data
        if 'call_transcript' not in data:
            raise ValueError("❌ 'call_transcript' tag not found in JSON file")
        
        call_transcript = data['call_transcript']
        print(f"📋 Found call_transcript with {len(call_transcript)} entries")
        
        # Initialize lists for Questions and Answers
        questions = []
        answers = []
        
        # Process each entry in call_transcript
        for i, entry in enumerate(call_transcript):
            if isinstance(entry, dict):
                # Check for Customer data (goes to Questions)
                if 'Customer' in entry:
                    questions.append(entry['Customer'])
                    answers.append(None)  # No corresponding answer yet
                    print(f"📝 Entry {i+1}: Customer -> Questions")
                
                # Check for CSR data (goes to Answers)
                elif 'CSR' in entry:
                    # If we have more CSRs than Customers, add empty question
                    if len(answers) < len(questions):
                        # Match with previous question
                        answers[-1] = entry['CSR']
                    else:
                        # Add new entry with empty question
                        questions.append(None)
                        answers.append(entry['CSR'])
                    print(f"📝 Entry {i+1}: CSR -> Answers")
                
                else:
                    print(f"⚠️ Entry {i+1}: Unknown format - {list(entry.keys())}")
            else:
                print(f"⚠️ Entry {i+1}: Not a dictionary - {type(entry)}")
        
        # Ensure both lists have the same length
        max_length = max(len(questions), len(answers))
        
        # Pad shorter list with None values
        while len(questions) < max_length:
            questions.append(None)
        while len(answers) < max_length:
            answers.append(None)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Questions': questions,
            'Answers': answers
        })
        
        # Add metadata if available
        metadata_columns = {}
        for key, value in data.items():
            if key != 'call_transcript' and not isinstance(value, (list, dict)):
                metadata_columns[key] = value
        
        # Add metadata columns to DataFrame
        for col_name, col_value in metadata_columns.items():
            df[col_name] = col_value
        
        print(f"✅ DataFrame created successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Display summary
        non_null_questions = df['Questions'].notna().sum()
        non_null_answers = df['Answers'].notna().sum()
        print(f"📈 Non-null Questions: {non_null_questions}")
        print(f"📈 Non-null Answers: {non_null_answers}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file_path}' not found")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def save_extracted_data(df: pd.DataFrame, output_prefix: str = "extracted_call_data"):
    """
    Save extracted data to CSV and Excel files
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_prefix (str): Prefix for output files
    """
    
    if df.empty:
        print("❌ No data to save")
        return
    
    try:
        # Save to CSV
        csv_filename = f"{output_prefix}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✅ Data saved to: {csv_filename}")
        
        # Save to Excel
        excel_filename = f"{output_prefix}.xlsx"
        df.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"✅ Data saved to: {excel_filename}")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")

def display_sample_data(df: pd.DataFrame, num_rows: int = 5):
    """
    Display sample data from the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to display
        num_rows (int): Number of rows to display
    """
    
    if df.empty:
        print("❌ No data to display")
        return
    
    print(f"\n📋 Sample Data (first {num_rows} rows):")
    print("=" * 80)
    
    for i in range(min(num_rows, len(df))):
        row = df.iloc[i]
        print(f"\n🔹 Row {i+1}:")
        print(f"   Question: {row['Questions']}")
        print(f"   Answer:   {row['Answers']}")
        
        # Display metadata if available
        metadata_cols = [col for col in df.columns if col not in ['Questions', 'Answers']]
        if metadata_cols:
            print("   Metadata:")
            for col in metadata_cols:
                print(f"     {col}: {row[col]}")
    
    print("=" * 80)

def main():
    """
    Main function to extract and process call transcript data
    """
    
    print("🎯 Call Transcript Data Extractor")
    print("=" * 50)
    
    # JSON file path
    json_file = "Call Transcript Sample 1.json"
    
    # Extract data
    df = extract_call_transcript_data(json_file)
    
    if not df.empty:
        # Display sample data
        display_sample_data(df)
        
        # Save extracted data
        save_extracted_data(df)
        
        print(f"\n🎉 Extraction completed successfully!")
        print(f"📊 Total interactions extracted: {len(df)}")
        
    else:
        print("❌ No data extracted. Please check your JSON file.")

if __name__ == "__main__":
    main()
