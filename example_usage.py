"""
Example usage of the JSON Call Transcript Data Extractor
"""

from extract_json_data import extract_call_transcript_data, save_extracted_data, display_sample_data

def main():
    """
    Example of how to use the data extraction functions
    """
    
    print("🎯 Example: Extracting Call Transcript Data")
    print("=" * 50)
    
    # 1. Extract data from JSON file
    json_file = "Call Transcript Sample 1.json"
    df = extract_call_transcript_data(json_file)
    
    if not df.empty:
        print(f"\n📊 Successfully extracted {len(df)} interactions")
        
        # 2. Display sample data
        print("\n📋 Sample extracted data:")
        display_sample_data(df, num_rows=3)
        
        # 3. Save to files
        print("\n💾 Saving extracted data...")
        save_extracted_data(df, "my_extracted_data")
        
        # 4. Access specific data
        print("\n🔍 Data Analysis:")
        print(f"   Total Questions: {df['Questions'].notna().sum()}")
        print(f"   Total Answers: {df['Answers'].notna().sum()}")
        print(f"   Available columns: {list(df.columns)}")
        
        # 5. Filter non-null pairs
        complete_pairs = df.dropna(subset=['Questions', 'Answers'])
        print(f"   Complete Q&A pairs: {len(complete_pairs)}")
        
        print("\n✅ Example completed successfully!")
        
    else:
        print("❌ No data extracted. Please check if 'Call Transcript Sample 1.json' exists.")

if __name__ == "__main__":
    main()
