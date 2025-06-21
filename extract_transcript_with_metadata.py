"""
Call Transcript Data Extractor with One-to-One Mapping and Metadata
Extracts Questions, Answers with proper alignment and metadata columns
"""

import json
import pandas as pd
from typing import Dict, List, Any, Tuple

def extract_transcript_with_metadata(json_file_path: str) -> pd.DataFrame:
    """
    Extract call transcript data with one-to-one Q&A mapping and metadata
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        pd.DataFrame: DataFrame with Questions, Answers, and metadata columns
    """
    
    print(f"🔄 Loading JSON file: {json_file_path}")
    
    try:
        # Load JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print("✅ JSON file loaded successfully")
        
        # Extract metadata
        metadata = {
            'call_ID': data.get('call_ID', 'Unknown'),
            'CSR_ID': data.get('CSR_ID', 'Unknown'),
            'call_date': data.get('call_date', 'Unknown'),
            'call_time': data.get('call_time', 'Unknown')
        }
        
        print(f"📋 Extracted metadata: {metadata}")
        
        # Extract call_transcript data
        if 'call_transcript' not in data:
            raise ValueError("❌ 'call_transcript' tag not found in JSON file")
        
        call_transcript = data['call_transcript']
        print(f"📋 Found call_transcript with {len(call_transcript)} entries")
        
        # Process transcript to create one-to-one mapping
        qa_pairs = create_one_to_one_mapping(call_transcript)
        
        if not qa_pairs:
            print("⚠️ No valid Q&A pairs found")
            return pd.DataFrame()
        
        # Create DataFrame with proper structure
        df_data = []
        
        for i, (question, answer) in enumerate(qa_pairs):
            row = {
                'Questions': question,
                'Answers': answer,
                'call_ID': metadata['call_ID'],
                'CSR_ID': metadata['CSR_ID'],
                'call_date': metadata['call_date'],
                'call_time': metadata['call_time'],
                'interaction_sequence': i + 1  # Track the order of interactions
            }
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        print(f"✅ DataFrame created successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Total Q&A pairs: {len(df)}")
        
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

def create_one_to_one_mapping(call_transcript: List[Dict]) -> List[Tuple[str, str]]:
    """
    Create one-to-one mapping between Customer questions and CSR answers
    
    Args:
        call_transcript (List[Dict]): List of transcript entries
        
    Returns:
        List[Tuple[str, str]]: List of (question, answer) pairs
    """
    
    print("🔄 Creating one-to-one Q&A mapping...")
    
    qa_pairs = []
    pending_question = None
    
    for i, entry in enumerate(call_transcript):
        if not isinstance(entry, dict):
            print(f"⚠️ Entry {i+1}: Not a dictionary - skipping")
            continue
        
        # Process Customer entry (Question)
        if 'Customer' in entry:
            customer_text = entry['Customer'].strip() if entry['Customer'] else ""
            
            if customer_text:
                # If we have a pending question without answer, pair it with empty answer
                if pending_question is not None:
                    qa_pairs.append((pending_question, ""))
                    print(f"📝 Paired pending question with empty answer")
                
                # Set new pending question
                pending_question = customer_text
                print(f"📝 Entry {i+1}: Customer question captured")
            
        # Process CSR entry (Answer)
        elif 'CSR' in entry:
            csr_text = entry['CSR'].strip() if entry['CSR'] else ""
            
            if csr_text:
                if pending_question is not None:
                    # Perfect match - pair question with answer
                    qa_pairs.append((pending_question, csr_text))
                    print(f"✅ Entry {i+1}: Q&A pair created")
                    pending_question = None
                else:
                    # CSR response without preceding question
                    qa_pairs.append("", csr_text)
                    print(f"📝 Entry {i+1}: CSR answer without question")
        
        else:
            print(f"⚠️ Entry {i+1}: Unknown format - {list(entry.keys())}")
    
    # Handle any remaining pending question
    if pending_question is not None:
        qa_pairs.append((pending_question, ""))
        print(f"📝 Final pending question paired with empty answer")
    
    print(f"✅ Created {len(qa_pairs)} Q&A pairs")
    return qa_pairs

def display_dataframe_summary(df: pd.DataFrame):
    """
    Display comprehensive summary of the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
    """
    
    if df.empty:
        print("❌ No data to display")
        return
    
    print(f"\n📊 DataFrame Summary:")
    print("=" * 60)
    print(f"📋 Total Rows: {len(df)}")
    print(f"📋 Total Columns: {len(df.columns)}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Data quality metrics
    print(f"\n📈 Data Quality:")
    print(f"   Non-null Questions: {df['Questions'].notna().sum()} ({df['Questions'].notna().sum()/len(df)*100:.1f}%)")
    print(f"   Non-null Answers: {df['Answers'].notna().sum()} ({df['Answers'].notna().sum()/len(df)*100:.1f}%)")
    print(f"   Complete Q&A pairs: {((df['Questions'].notna()) & (df['Answers'].notna())).sum()}")
    
    # Metadata summary
    print(f"\n📋 Metadata:")
    print(f"   Call ID: {df['call_ID'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"   CSR ID: {df['CSR_ID'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"   Call Date: {df['call_date'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"   Call Time: {df['call_time'].iloc[0] if len(df) > 0 else 'N/A'}")
    
    print("=" * 60)

def display_sample_interactions(df: pd.DataFrame, num_samples: int = 3):
    """
    Display sample Q&A interactions
    
    Args:
        df (pd.DataFrame): DataFrame to display
        num_samples (int): Number of samples to show
    """
    
    if df.empty:
        print("❌ No interactions to display")
        return
    
    print(f"\n💬 Sample Interactions (first {num_samples}):")
    print("=" * 80)
    
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\n🔹 Interaction {row['interaction_sequence']}:")
        print(f"   ❓ Question: {row['Questions'] if pd.notna(row['Questions']) else '[No Question]'}")
        print(f"   💬 Answer:   {row['Answers'] if pd.notna(row['Answers']) else '[No Answer]'}")
    
    print("=" * 80)

def save_dataframe(df: pd.DataFrame, output_prefix: str = "call_transcript_data"):
    """
    Save DataFrame to CSV and Excel files
    
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
        df.to_excel(excel_filename, index=False)
        print(f"✅ Data saved to: {excel_filename}")
        
        # Save summary statistics
        summary_filename = f"{output_prefix}_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write("Call Transcript Data Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total Interactions: {len(df)}\n")
            f.write(f"Questions with content: {df['Questions'].notna().sum()}\n")
            f.write(f"Answers with content: {df['Answers'].notna().sum()}\n")
            f.write(f"Complete Q&A pairs: {((df['Questions'].notna()) & (df['Answers'].notna())).sum()}\n")
            f.write(f"\nMetadata:\n")
            f.write(f"Call ID: {df['call_ID'].iloc[0]}\n")
            f.write(f"CSR ID: {df['CSR_ID'].iloc[0]}\n")
            f.write(f"Call Date: {df['call_date'].iloc[0]}\n")
            f.write(f"Call Time: {df['call_time'].iloc[0]}\n")
        
        print(f"✅ Summary saved to: {summary_filename}")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")

def main():
    """
    Main function to extract and process call transcript data
    """
    
    print("🎯 Call Transcript Extractor with One-to-One Mapping")
    print("=" * 60)
    
    # JSON file path
    json_file = "Call Transcript Sample 1.json"
    
    # Extract data
    df = extract_transcript_with_metadata(json_file)
    
    if not df.empty:
        # Display summary
        display_dataframe_summary(df)
        
        # Display sample interactions
        display_sample_interactions(df)
        
        # Save data
        print(f"\n💾 Saving extracted data...")
        save_dataframe(df)
        
        print(f"\n🎉 Extraction completed successfully!")
        print(f"📊 Total interactions processed: {len(df)}")
        print(f"📋 Metadata columns: call_ID, CSR_ID, call_date, call_time")
        print(f"💬 One-to-one Q&A mapping maintained")
        
    else:
        print("❌ No data extracted. Please check your JSON file.")

if __name__ == "__main__":
    main()
