#!/usr/bin/env python3
"""
Example Usage of Call Center Transcript Evaluator
=================================================

This script demonstrates how to use the CallCenterTranscriptEvaluator
with the provided sample transcript.

Make sure you have:
1. Installed all requirements: pip install -r requirements.txt
2. Downloaded the Mistral 7B model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
3. Updated the MODEL_PATH variable below to point to your model file
"""

import os
from pathlib import Path
from call_center_transcript_evaluator import CallCenterTranscriptEvaluator

# Configuration
MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Update this path!
TRANSCRIPT_FILE = "Call Transcript Sample 1.json"
OUTPUT_REPORT = "sample_evaluation_report.txt"

def main():
    """Main example function."""
    print("🚀 Call Center Transcript Evaluator - Example Usage")
    print("=" * 60)
    
    # Check if model file exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("Please download the Mistral 7B model and update MODEL_PATH")
        print("Model needed: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        return
    
    # Check if transcript file exists
    if not Path(TRANSCRIPT_FILE).exists():
        print(f"❌ Error: Transcript file not found: {TRANSCRIPT_FILE}")
        return
    
    print(f"✅ Model file found: {MODEL_PATH}")
    print(f"✅ Transcript file found: {TRANSCRIPT_FILE}")
    print()
    
    # Initialize the evaluator
    print("🔄 Initializing Mistral 7B model...")
    try:
        evaluator = CallCenterTranscriptEvaluator(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure llama-cpp-python is installed correctly")
        return
    
    print()
    
    # Load and preview the transcript data
    print("📄 Loading transcript data...")
    transcript_data = evaluator.load_transcript_json(TRANSCRIPT_FILE)
    
    if transcript_data:
        print(f"✅ Loaded transcript for Call ID: {transcript_data.get('call_ID')}")
        print(f"   CSR ID: {transcript_data.get('CSR_ID')}")
        print(f"   Date: {transcript_data.get('call_date')}")
        print(f"   Lines in transcript: {len(transcript_data.get('call_transcript', []))}")
    else:
        print("❌ Failed to load transcript data")
        return
    
    print()
    
    # Extract Q&A pairs
    print("🔍 Extracting Questions and Answers...")
    df = evaluator.extract_qa_pairs(transcript_data)
    print(f"✅ Extracted {len(df)} Q&A pairs")
    
    # Show sample Q&A pairs
    print("\n📋 Sample Q&A Pairs:")
    print("-" * 40)
    for i, row in df.head(3).iterrows():
        if row['Questions'] and row['Answers']:
            print(f"Q{i+1}: {row['Questions'][:100]}...")
            print(f"A{i+1}: {row['Answers'][:100]}...")
            print()
    
    # Perform complete evaluation
    print("🔬 Starting comprehensive evaluation...")
    print("This may take a few minutes depending on your CPU...")
    print()
    
    try:
        results = evaluator.evaluate_transcript(TRANSCRIPT_FILE)
        
        if 'error' in results:
            print(f"❌ Evaluation failed: {results['error']}")
            return
        
        print("✅ Evaluation completed successfully!")
        print()
        
        # Display summary statistics
        stats = results['summary_stats']
        print("📊 Evaluation Summary:")
        print(f"   • Total Q&A pairs evaluated: {stats['total_qa_pairs']}")
        print(f"   • Knowledge documents created: {len(results['knowledge_documents'])}")
        print(f"   • Evaluation timestamp: {stats['evaluation_timestamp']}")
        print()
        
        # Show sample evaluation results
        print("🎯 Sample Evaluation Results:")
        print("-" * 50)
        
        if results['evaluations']:
            sample_eval = results['evaluations'][0]
            
            # Sentiment analysis sample
            sentiment = sample_eval['sentiment_analysis']
            print(f"Sentiment Analysis:")
            print(f"   • Classification: {sentiment.get('sentiment_label', 'N/A')}")
            print(f"   • Coaching: {sentiment.get('coaching_feedback', 'N/A')[:150]}...")
            print()
            
            # Sentence analysis sample
            sentence_analysis = sample_eval['sentence_analysis']
            print(f"Sentence Analysis:")
            print(f"   • Average words per sentence: {sentence_analysis['average_words_per_sentence']}")
            print(f"   • Long sentences found: {sentence_analysis['long_sentences_count']}")
            print()
            
            # Word repetition sample
            repetition = sample_eval['repetition_analysis']
            print(f"Word Analysis:")
            print(f"   • Vocabulary diversity: {repetition['vocabulary_diversity']}")
            print(f"   • Crutch words found: {repetition['crutch_words']}")
            print()
        
        # Generate and save full report
        print("📝 Generating comprehensive report...")
        report = evaluator.generate_report(results, OUTPUT_REPORT)
        print(f"✅ Report saved to: {OUTPUT_REPORT}")
        print()
        
        # Show knowledge documents sample
        print("📚 Knowledge Documents Created:")
        print("-" * 40)
        for i, doc in enumerate(results['knowledge_documents'][:2], 1):
            print(f"Document {i}:")
            print(doc[:200] + "..." if len(doc) > 200 else doc)
            print()
        
        print("🎉 Example completed successfully!")
        print(f"📄 Check the full report in: {OUTPUT_REPORT}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Check the logs for more details")

def demonstrate_individual_features():
    """Demonstrate individual features of the evaluator."""
    print("\n" + "=" * 60)
    print("🔧 Individual Feature Demonstrations")
    print("=" * 60)
    
    # Sample text for demonstration
    sample_text = "Um, well, I think that, you know, we can definitely help you with that issue, and, uh, we really want to make sure that you're completely satisfied with our service."
    
    try:
        evaluator = CallCenterTranscriptEvaluator(MODEL_PATH)
        
        print(f"\n📝 Sample Text: {sample_text}")
        print()
        
        # Demonstrate sentiment analysis
        print("😊 Sentiment Analysis:")
        sentiment = evaluator.analyze_sentiment(sample_text)
        print(f"   • Sentiment: {sentiment.get('sentiment_label', 'N/A')}")
        print(f"   • Polarity: {sentiment.get('polarity', 'N/A')}")
        print()
        
        # Demonstrate repetition detection
        print("🔄 Repetition & Crutch Word Detection:")
        repetition = evaluator.detect_repetition_and_crutch_words(sample_text)
        print(f"   • Crutch words found: {repetition['crutch_words']}")
        print(f"   • Vocabulary diversity: {repetition['vocabulary_diversity']}")
        print()
        
        # Demonstrate sentence analysis
        print("📏 Sentence Length Analysis:")
        sentence_analysis = evaluator.analyze_sentence_length(sample_text)
        print(f"   • Average words per sentence: {sentence_analysis['average_words_per_sentence']}")
        print(f"   • Long sentences: {sentence_analysis['long_sentences_count']}")
        print()
        
    except Exception as e:
        print(f"❌ Error in individual demonstrations: {e}")

if __name__ == "__main__":
    main()
    
    # Uncomment the line below to see individual feature demonstrations
    # demonstrate_individual_features()
    
    print("\n" + "=" * 60)
    print("💡 Next Steps:")
    print("1. Review the generated report file")
    print("2. Uncomment RAGAS evaluation sections when ready")
    print("3. Customize the evaluation parameters as needed")
    print("4. Process your own transcript files")
    print("=" * 60)

