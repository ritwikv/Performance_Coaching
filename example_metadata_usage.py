"""
Example usage of the Call Transcript Extractor with Metadata and One-to-One Mapping
"""

from extract_transcript_with_metadata import extract_transcript_with_metadata, display_dataframe_summary, display_sample_interactions
import pandas as pd

def demonstrate_extraction():
    """
    Demonstrate the extraction functionality with detailed examples
    """
    
    print("🎯 Demonstration: Call Transcript Extraction with Metadata")
    print("=" * 70)
    
    # Extract data from JSON file
    json_file = "Call Transcript Sample 1.json"
    df = extract_transcript_with_metadata(json_file)
    
    if df.empty:
        print("❌ No data extracted. Please ensure 'Call Transcript Sample 1.json' exists.")
        return
    
    # Show the DataFrame structure
    print(f"\n📊 DataFrame Structure:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display first few rows
    print(f"\n📋 First 3 rows of extracted data:")
    print(df.head(3).to_string(index=False))
    
    # Show metadata consistency
    print(f"\n📋 Metadata Verification:")
    print(f"Unique Call IDs: {df['call_ID'].nunique()} (should be 1)")
    print(f"Unique CSR IDs: {df['CSR_ID'].nunique()} (should be 1)")
    print(f"Unique Call Dates: {df['call_date'].nunique()} (should be 1)")
    print(f"Unique Call Times: {df['call_time'].nunique()} (should be 1)")
    
    # Analyze Q&A mapping quality
    print(f"\n💬 Q&A Mapping Analysis:")
    total_rows = len(df)
    questions_with_content = df['Questions'].notna().sum()
    answers_with_content = df['Answers'].notna().sum()
    complete_pairs = ((df['Questions'].notna()) & (df['Answers'].notna())).sum()
    
    print(f"Total interactions: {total_rows}")
    print(f"Questions with content: {questions_with_content} ({questions_with_content/total_rows*100:.1f}%)")
    print(f"Answers with content: {answers_with_content} ({answers_with_content/total_rows*100:.1f}%)")
    print(f"Complete Q&A pairs: {complete_pairs} ({complete_pairs/total_rows*100:.1f}%)")
    
    # Show interaction sequence
    print(f"\n🔢 Interaction Sequence:")
    for i, row in df.iterrows():
        status = "✅ Complete" if (pd.notna(row['Questions']) and pd.notna(row['Answers'])) else "⚠️ Incomplete"
        print(f"   {row['interaction_sequence']}: {status}")
    
    return df

def analyze_data_quality(df: pd.DataFrame):
    """
    Perform detailed data quality analysis
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    
    if df.empty:
        return
    
    print(f"\n🔍 Data Quality Analysis:")
    print("=" * 50)
    
    # Check for empty strings vs null values
    questions_empty = (df['Questions'] == '').sum()
    answers_empty = (df['Answers'] == '').sum()
    questions_null = df['Questions'].isna().sum()
    answers_null = df['Answers'].isna().sum()
    
    print(f"Questions - Empty strings: {questions_empty}, Null values: {questions_null}")
    print(f"Answers - Empty strings: {answers_empty}, Null values: {answers_null}")
    
    # Check metadata completeness
    print(f"\n📋 Metadata Completeness:")
    for col in ['call_ID', 'CSR_ID', 'call_date', 'call_time']:
        missing = df[col].isna().sum()
        unknown = (df[col] == 'Unknown').sum()
        print(f"   {col}: Missing={missing}, Unknown={unknown}")
    
    # Text length analysis
    print(f"\n📏 Text Length Analysis:")
    if df['Questions'].notna().any():
        q_lengths = df['Questions'].dropna().str.len()
        print(f"   Questions - Min: {q_lengths.min()}, Max: {q_lengths.max()}, Avg: {q_lengths.mean():.1f}")
    
    if df['Answers'].notna().any():
        a_lengths = df['Answers'].dropna().str.len()
        print(f"   Answers - Min: {a_lengths.min()}, Max: {a_lengths.max()}, Avg: {a_lengths.mean():.1f}")

def demonstrate_data_access(df: pd.DataFrame):
    """
    Demonstrate different ways to access and filter the data
    
    Args:
        df (pd.DataFrame): DataFrame to demonstrate with
    """
    
    if df.empty:
        return
    
    print(f"\n🔧 Data Access Examples:")
    print("=" * 40)
    
    # Access specific metadata
    print(f"1. Accessing metadata:")
    print(f"   Call ID: {df['call_ID'].iloc[0]}")
    print(f"   CSR ID: {df['CSR_ID'].iloc[0]}")
    
    # Filter complete Q&A pairs
    complete_pairs = df.dropna(subset=['Questions', 'Answers'])
    print(f"\n2. Complete Q&A pairs: {len(complete_pairs)} out of {len(df)}")
    
    # Filter by interaction sequence
    first_three = df[df['interaction_sequence'] <= 3]
    print(f"\n3. First 3 interactions: {len(first_three)} rows")
    
    # Access questions only
    questions_only = df[df['Questions'].notna() & df['Answers'].isna()]
    print(f"\n4. Questions without answers: {len(questions_only)}")
    
    # Access answers only
    answers_only = df[df['Answers'].notna() & df['Questions'].isna()]
    print(f"\n5. Answers without questions: {len(answers_only)}")

def export_examples(df: pd.DataFrame):
    """
    Demonstrate different export options
    
    Args:
        df (pd.DataFrame): DataFrame to export
    """
    
    if df.empty:
        return
    
    print(f"\n💾 Export Examples:")
    print("=" * 30)
    
    try:
        # Export complete pairs only
        complete_pairs = df.dropna(subset=['Questions', 'Answers'])
        if not complete_pairs.empty:
            complete_pairs.to_csv("complete_qa_pairs.csv", index=False)
            print(f"✅ Complete Q&A pairs exported: complete_qa_pairs.csv")
        
        # Export metadata summary
        metadata_summary = df[['call_ID', 'CSR_ID', 'call_date', 'call_time']].drop_duplicates()
        metadata_summary.to_csv("call_metadata.csv", index=False)
        print(f"✅ Metadata summary exported: call_metadata.csv")
        
        # Export questions only
        questions_df = df[['Questions', 'interaction_sequence', 'call_ID']].dropna(subset=['Questions'])
        if not questions_df.empty:
            questions_df.to_csv("questions_only.csv", index=False)
            print(f"✅ Questions only exported: questions_only.csv")
        
        # Export answers only
        answers_df = df[['Answers', 'interaction_sequence', 'CSR_ID']].dropna(subset=['Answers'])
        if not answers_df.empty:
            answers_df.to_csv("answers_only.csv", index=False)
            print(f"✅ Answers only exported: answers_only.csv")
            
    except Exception as e:
        print(f"❌ Export error: {e}")

def main():
    """
    Main demonstration function
    """
    
    # Run the demonstration
    df = demonstrate_extraction()
    
    if not df.empty:
        # Perform quality analysis
        analyze_data_quality(df)
        
        # Show data access examples
        demonstrate_data_access(df)
        
        # Show export examples
        export_examples(df)
        
        print(f"\n🎉 Demonstration completed!")
        print(f"📊 Successfully processed {len(df)} interactions with proper one-to-one mapping")
        print(f"📋 All metadata columns included: call_ID, CSR_ID, call_date, call_time")
        
    else:
        print("❌ Demonstration failed. Please check your JSON file.")

if __name__ == "__main__":
    main()
