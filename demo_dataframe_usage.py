"""
Demo script showing how to use the call transcript DataFrame for analysis
"""

import pandas as pd
from extract_call_data_dataframe import extract_call_transcript_to_dataframe

def demo_dataframe_analysis():
    """
    Demonstrate various ways to analyze the call transcript DataFrame
    """
    
    # Load the data
    print("🔄 Loading call transcript data...")
    df = extract_call_transcript_to_dataframe("Call Transcript Sample 1.json")
    
    if df is None:
        print("❌ Failed to load data")
        return
    
    print("✅ Data loaded successfully!")
    print("=" * 60)
    
    # Basic DataFrame info
    print("📊 BASIC DATAFRAME INFO:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Show metadata
    print("📋 CALL METADATA:")
    metadata_cols = ['call_ID', 'CSR_ID', 'call_date', 'call_time']
    for col in metadata_cols:
        print(f"{col}: {df[col].iloc[0]}")
    print()
    
    # Analysis examples
    print("🔍 ANALYSIS EXAMPLES:")
    print("-" * 40)
    
    # 1. Count interactions
    total_interactions = len(df)
    questions_count = df['Questions'].notna().sum()
    answers_count = df['Answers'].notna().sum()
    
    print(f"1. Total interactions: {total_interactions}")
    print(f"   - Customer questions: {questions_count}")
    print(f"   - CSR/Supervisor answers: {answers_count}")
    print()
    
    # 2. Find perfect question-answer pairs
    perfect_pairs = df[df['Questions'].notna() & df['Answers'].notna()]
    print(f"2. Perfect Question-Answer pairs: {len(perfect_pairs)}")
    print()
    
    # 3. Find unanswered questions
    unanswered = df[df['Questions'].notna() & df['Answers'].isna()]
    if len(unanswered) > 0:
        print(f"3. Unanswered questions: {len(unanswered)}")
        for idx, row in unanswered.iterrows():
            print(f"   - Interaction {row['interaction_id']}: {row['Questions'][:50]}...")
    else:
        print("3. All questions have corresponding answers ✅")
    print()
    
    # 4. Find answers without questions (CSR initiated)
    csr_initiated = df[df['Answers'].notna() & df['Questions'].isna()]
    if len(csr_initiated) > 0:
        print(f"4. CSR-initiated responses: {len(csr_initiated)}")
        for idx, row in csr_initiated.iterrows():
            print(f"   - Interaction {row['interaction_id']}: {row['Answers'][:50]}...")
    else:
        print("4. No CSR-initiated responses")
    print()
    
    # 5. Text analysis examples
    print("📝 TEXT ANALYSIS EXAMPLES:")
    print("-" * 40)
    
    # Average text length
    df['question_length'] = df['Questions'].str.len()
    df['answer_length'] = df['Answers'].str.len()
    
    avg_q_length = df['question_length'].mean()
    avg_a_length = df['answer_length'].mean()
    
    print(f"Average question length: {avg_q_length:.1f} characters")
    print(f"Average answer length: {avg_a_length:.1f} characters")
    print()
    
    # 6. Export specific subsets
    print("💾 EXPORT EXAMPLES:")
    print("-" * 40)
    
    # Export only perfect pairs
    perfect_pairs_only = df[df['Questions'].notna() & df['Answers'].notna()].copy()
    perfect_pairs_only.to_csv('perfect_pairs_only.csv', index=False)
    print(f"✅ Exported {len(perfect_pairs_only)} perfect pairs to 'perfect_pairs_only.csv'")
    
    # Export questions only
    questions_only = df[df['Questions'].notna()][['interaction_id', 'Questions'] + metadata_cols].copy()
    questions_only.to_csv('questions_only.csv', index=False)
    print(f"✅ Exported {len(questions_only)} questions to 'questions_only.csv'")
    
    # Export answers only
    answers_only = df[df['Answers'].notna()][['interaction_id', 'Answers'] + metadata_cols].copy()
    answers_only.to_csv('answers_only.csv', index=False)
    print(f"✅ Exported {len(answers_only)} answers to 'answers_only.csv'")
    print()
    
    # 7. Show sample interactions
    print("💬 SAMPLE INTERACTIONS:")
    print("-" * 40)
    
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"Interaction {row['interaction_id']}:")
        
        if pd.notna(row['Questions']):
            print(f"  🙋 Customer: {row['Questions'][:100]}{'...' if len(row['Questions']) > 100 else ''}")
        else:
            print("  🙋 Customer: [No question]")
            
        if pd.notna(row['Answers']):
            print(f"  👩‍💼 CSR: {row['Answers'][:100]}{'...' if len(row['Answers']) > 100 else ''}")
        else:
            print("  👩‍💼 CSR: [No answer]")
        print()
    
    print("=" * 60)
    print("🎉 Demo completed! Check the generated CSV files for exported data.")

if __name__ == "__main__":
    demo_dataframe_analysis()

