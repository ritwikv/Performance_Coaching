"""
Mistral Transcript Evaluator - Usage Guide and Examples
Demonstrates how to use the evaluator with different configurations
"""

import pandas as pd
import json
from mistral_transcript_evaluator import MistralTranscriptEvaluator
from extract_call_data_dataframe import extract_call_transcript_to_dataframe

def example_basic_usage():
    """
    Basic usage example
    """
    print("📖 EXAMPLE 1: Basic Usage")
    print("-" * 40)
    
    # Initialize evaluator
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    evaluator = MistralTranscriptEvaluator(model_path)
    
    # Load data
    df = extract_call_transcript_to_dataframe("Call Transcript Sample 1.json")
    
    # Evaluate
    results = evaluator.evaluate_dataframe(df)
    
    # Save results
    results.to_csv('basic_evaluation_results.csv', index=False)
    print("✅ Basic evaluation completed!")

def example_single_interaction():
    """
    Example of evaluating a single interaction
    """
    print("\n📖 EXAMPLE 2: Single Interaction Evaluation")
    print("-" * 40)
    
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    evaluator = MistralTranscriptEvaluator(model_path)
    
    # Sample interaction
    question = "I need help with my booking. My flight was canceled!"
    answer = "I apologize for the trouble. Let me look up your booking and see what options we have available for you."
    
    # Evaluate single interaction
    result = evaluator.evaluate_single_interaction(question, answer, 1)
    
    print("📊 Evaluation Results:")
    print(f"English Score: {result.get('english_evaluation', {}).get('english_score', 'N/A')}")
    print(f"Clarity Score: {result.get('sentence_analysis', {}).get('clarity_score', 'N/A')}")
    print(f"Speech Score: {result.get('repetition_analysis', {}).get('speech_score', 'N/A')}")
    
    print("✅ Single interaction evaluation completed!")

def example_custom_analysis():
    """
    Example of custom analysis functions
    """
    print("\n📖 EXAMPLE 3: Custom Analysis Functions")
    print("-" * 40)
    
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    evaluator = MistralTranscriptEvaluator(model_path)
    
    sample_text = "Um, well, I think, you know, we can definitely help you with that issue, absolutely."
    
    # Test individual analysis functions
    print("🔍 English Correctness Analysis:")
    english_result = evaluator.evaluate_english_correctness(sample_text)
    print(f"Score: {english_result['english_score']}")
    print(f"Feedback: {english_result['feedback'][:100]}...")
    
    print("\n🔍 Repetition and Crutch Words Analysis:")
    repetition_result = evaluator.detect_repetition_and_crutch_words(sample_text)
    print(f"Crutch words found: {repetition_result['crutch_words']}")
    print(f"Speech score: {repetition_result['speech_score']}")
    
    print("\n🔍 Sentiment Analysis:")
    sentiment_result = evaluator.analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment_result['sentiment_category']}")
    print(f"Polarity: {sentiment_result['polarity']:.2f}")

def example_batch_processing():
    """
    Example of processing multiple transcript files
    """
    print("\n📖 EXAMPLE 4: Batch Processing Multiple Files")
    print("-" * 40)
    
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    evaluator = MistralTranscriptEvaluator(model_path)
    
    # List of transcript files (example)
    transcript_files = [
        "Call Transcript Sample 1.json",
        # Add more files as needed
    ]
    
    all_results = []
    
    for i, file_path in enumerate(transcript_files):
        print(f"Processing file {i+1}/{len(transcript_files)}: {file_path}")
        
        try:
            df = extract_call_transcript_to_dataframe(file_path)
            if df is not None:
                results = evaluator.evaluate_dataframe(df)
                results['source_file'] = file_path
                all_results.append(results)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv('batch_evaluation_results.csv', index=False)
        print("✅ Batch processing completed!")

def example_generate_coaching_report():
    """
    Example of generating detailed coaching reports
    """
    print("\n📖 EXAMPLE 5: Detailed Coaching Report Generation")
    print("-" * 40)
    
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    evaluator = MistralTranscriptEvaluator(model_path)
    
    # Load and evaluate data
    df = extract_call_transcript_to_dataframe("Call Transcript Sample 1.json")
    results_df = evaluator.evaluate_dataframe(df)
    
    # Generate comprehensive coaching report
    coaching_report = evaluator.generate_summary_report(results_df)
    
    # Save detailed coaching report
    with open('detailed_coaching_report.txt', 'w') as f:
        f.write("COMPREHENSIVE COACHING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(coaching_report)
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("INDIVIDUAL INTERACTION DETAILS\n")
        f.write("=" * 50 + "\n\n")
        
        # Add individual interaction details
        for idx, row in results_df.iterrows():
            if row.get('evaluation_status') == 'completed':
                f.write(f"INTERACTION {row.get('interaction_id', idx+1)}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Question: {row.get('question', 'N/A')[:100]}...\n")
                f.write(f"Answer: {row.get('answer', 'N/A')[:100]}...\n\n")
                
                # Add evaluation details
                if 'english_evaluation' in row and row['english_evaluation']:
                    f.write(f"English Score: {row['english_evaluation'].get('english_score', 'N/A')}\n")
                
                if 'ragas_coaching' in row and row['ragas_coaching']:
                    f.write(f"RAGAS Coaching: {row['ragas_coaching'][:200]}...\n")
                
                if 'sentiment_analysis' in row and row['sentiment_analysis']:
                    f.write(f"Sentiment: {row['sentiment_analysis'].get('sentiment_category', 'N/A')}\n")
                
                f.write("\n" + "-" * 30 + "\n\n")
    
    print("✅ Detailed coaching report generated: detailed_coaching_report.txt")

def example_configuration_options():
    """
    Example showing different configuration options
    """
    print("\n📖 EXAMPLE 6: Configuration Options")
    print("-" * 40)
    
    # Different model configurations
    configs = [
        {
            "name": "Fast Processing",
            "n_ctx": 2048,
            "n_threads": 8,
            "description": "Faster processing with smaller context"
        },
        {
            "name": "High Quality",
            "n_ctx": 8192,
            "n_threads": 4,
            "description": "Higher quality analysis with larger context"
        },
        {
            "name": "Balanced",
            "n_ctx": 4096,
            "n_threads": 6,
            "description": "Balanced speed and quality"
        }
    ]
    
    print("Available configurations:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config['name']}: {config['description']}")
        print(f"   Context: {config['n_ctx']}, Threads: {config['n_threads']}")
    
    # Example of using custom configuration
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    custom_evaluator = MistralTranscriptEvaluator(
        model_path=model_path,
        n_ctx=2048,  # Smaller context for faster processing
        n_threads=8   # More threads for speed
    )
    
    print("✅ Custom configuration example shown!")

def main():
    """
    Main function to run all examples
    """
    print("🎯 Mistral Transcript Evaluator - Usage Examples")
    print("=" * 60)
    
    print("\n🔧 Available Examples:")
    print("1. Basic Usage")
    print("2. Single Interaction Evaluation") 
    print("3. Custom Analysis Functions")
    print("4. Batch Processing")
    print("5. Detailed Coaching Reports")
    print("6. Configuration Options")
    
    print("\n" + "=" * 60)
    print("📚 USAGE INSTRUCTIONS:")
    print("=" * 60)
    
    print("""
1. SETUP:
   - Run: python setup_mistral_evaluator.py
   - Ensure Mistral model file is downloaded
   - Install requirements: pip install -r requirements_mistral.txt

2. BASIC USAGE:
   - python mistral_transcript_evaluator.py

3. CUSTOM USAGE:
   - Import: from mistral_transcript_evaluator import MistralTranscriptEvaluator
   - Initialize with your model path
   - Call evaluation methods

4. OUTPUT FILES:
   - transcript_evaluation_results.csv: Detailed results
   - transcript_evaluation_results.json: JSON format
   - evaluation_summary_report.txt: Summary report

5. FEATURES:
   ✅ English language correctness evaluation
   ✅ Sentence structure analysis
   ✅ Repetition and crutch word detection
   ✅ Knowledge document creation
   ✅ RAGAS framework evaluation
   ✅ Sentiment analysis
   ✅ Question topic summarization
   ✅ Comprehensive coaching feedback
    """)
    
    print("\n" + "=" * 60)
    print("🚀 Ready to start evaluating call center transcripts!")
    print("Run individual examples by calling their functions.")

if __name__ == "__main__":
    main()

