"""
Demo script for the Streamlit Dashboard
Creates sample evaluation data for testing the dashboard
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_evaluation_data():
    """
    Create sample evaluation data for dashboard testing
    """
    
    # Sample interactions data
    sample_interactions = [
        {
            "interaction_id": 1,
            "question": "I need help with my booking. My flight was canceled!",
            "answer": "I apologize for the trouble. Let me look up your booking and see what options we have available for you.",
            "csr_id": "JaneDoe123"
        },
        {
            "interaction_id": 2,
            "question": "What's my reservation number? I can't find my email.",
            "answer": "I can help you locate that information. May I have your name and the email address you used for the booking?",
            "csr_id": "JaneDoe123"
        },
        {
            "interaction_id": 3,
            "question": "This is unacceptable! I booked this trip months ago!",
            "answer": "I completely understand your frustration. Let me see what I can do to resolve this situation for you right away.",
            "csr_id": "JaneDoe123"
        },
        {
            "interaction_id": 4,
            "question": "Can you help me change my flight date?",
            "answer": "Absolutely! I'd be happy to help you change your flight date. Let me check the available options for you.",
            "csr_id": "MikeSmith456"
        },
        {
            "interaction_id": 5,
            "question": "What are the fees for changing my ticket?",
            "answer": "The change fees depend on your ticket type. Let me review your specific booking to give you accurate information.",
            "csr_id": "MikeSmith456"
        }
    ]
    
    # Generate evaluation results
    evaluation_results = []
    
    for interaction in sample_interactions:
        # Generate realistic scores with some variation
        base_english_score = random.uniform(7.0, 9.5)
        base_clarity_score = random.uniform(6.5, 9.0)
        base_speech_score = random.uniform(7.5, 9.2)
        
        # Generate RAGAS scores (0.0 to 1.0)
        ragas_scores = {
            "context_precision": random.uniform(0.7, 0.95),
            "context_recall": random.uniform(0.65, 0.9),
            "context_entity_recall": random.uniform(0.6, 0.85),
            "answer_relevancy": random.uniform(0.75, 0.95),
            "faithfulness": random.uniform(0.8, 0.98)
        }
        
        # Generate sentiment
        sentiments = ["positive", "neutral", "positive", "positive", "neutral"]
        sentiment = random.choice(sentiments)
        
        # Create evaluation result
        result = {
            "interaction_id": interaction["interaction_id"],
            "question": interaction["question"],
            "answer": interaction["answer"],
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
            "evaluation_status": "completed",
            
            "english_evaluation": {
                "english_score": round(base_english_score, 1),
                "feedback": f"Your English proficiency score is {base_english_score:.1f}/10. " + 
                          random.choice([
                              "Excellent grammar and professional language usage.",
                              "Good communication with minor areas for improvement.",
                              "Consider using more formal language in professional settings.",
                              "Strong vocabulary and clear expression."
                          ]),
                "has_errors": base_english_score < 8.0
            },
            
            "sentence_analysis": {
                "clarity_score": round(base_clarity_score, 1),
                "avg_sentence_length": random.uniform(12, 25),
                "long_sentences_count": random.randint(0, 2),
                "feedback": f"Your communication clarity score is {base_clarity_score:.1f}/10. " +
                          random.choice([
                              "Your sentences are well-structured and easy to understand.",
                              "Consider breaking longer sentences for better clarity.",
                              "Excellent use of concise and clear communication.",
                              "Your message structure effectively conveys information."
                          ])
            },
            
            "repetition_analysis": {
                "speech_score": round(base_speech_score, 1),
                "repeated_words": {} if random.random() > 0.3 else {"really": 2, "actually": 3},
                "crutch_words": {} if random.random() > 0.4 else {"um": 1, "you know": 2},
                "feedback": f"Your speech quality score is {base_speech_score:.1f}/10. " +
                          random.choice([
                              "Excellent speech clarity with minimal filler words.",
                              "Consider reducing the use of crutch words like 'um' and 'you know'.",
                              "Professional and articulate communication style.",
                              "Good vocabulary variety and speech flow."
                          ])
            },
            
            "ragas_scores": ragas_scores,
            
            "ragas_coaching": f"Your performance metrics show strong results! " +
                            f"Faithfulness score of {ragas_scores['faithfulness']:.2f} indicates excellent accuracy. " +
                            f"Answer relevancy of {ragas_scores['answer_relevancy']:.2f} shows you're addressing customer needs well. " +
                            random.choice([
                                "Keep up the excellent work!",
                                "Consider providing more specific details to improve context precision.",
                                "Your responses demonstrate good knowledge of procedures.",
                                "Focus on including all relevant information in your responses."
                            ]),
            
            "sentiment_analysis": {
                "sentiment_category": sentiment,
                "polarity": random.uniform(-0.2, 0.8) if sentiment == "positive" else random.uniform(-0.1, 0.1),
                "subjectivity": random.uniform(0.3, 0.7),
                "feedback": f"Your emotional tone was {sentiment}. " +
                          random.choice([
                              "You maintained a professional and empathetic approach.",
                              "Your positive attitude likely made the customer feel valued.",
                              "Consider adding more warmth to your communication style.",
                              "Excellent emotional intelligence in handling the situation."
                          ])
            },
            
            "question_topic": random.choice([
                "Flight booking assistance - Medium priority inquiry",
                "Reservation lookup - Low priority request", 
                "Flight cancellation complaint - High priority issue",
                "Flight change request - Medium priority service",
                "Fee inquiry - Low priority information request"
            ])
        }
        
        evaluation_results.append(result)
    
    return evaluation_results

def create_sample_transcript_dataframe():
    """
    Create sample transcript DataFrame
    """
    
    data = {
        'interaction_id': [1, 2, 3, 4, 5],
        'Questions': [
            "I need help with my booking. My flight was canceled!",
            "What's my reservation number? I can't find my email.",
            "This is unacceptable! I booked this trip months ago!",
            "Can you help me change my flight date?",
            "What are the fees for changing my ticket?"
        ],
        'Answers': [
            "I apologize for the trouble. Let me look up your booking and see what options we have available for you.",
            "I can help you locate that information. May I have your name and the email address you used for the booking?",
            "I completely understand your frustration. Let me see what I can do to resolve this situation for you right away.",
            "Absolutely! I'd be happy to help you change your flight date. Let me check the available options for you.",
            "The change fees depend on your ticket type. Let me review your specific booking to give you accurate information."
        ],
        'call_ID': ['12345'] * 5,
        'CSR_ID': ['JaneDoe123', 'JaneDoe123', 'JaneDoe123', 'MikeSmith456', 'MikeSmith456'],
        'call_date': ['2024-02-01'] * 5,
        'call_time': ['02:16:43', '02:18:15', '02:22:30', '03:45:12', '03:47:55']
    }
    
    return pd.DataFrame(data)

def main():
    """
    Main function to create demo data
    """
    print("🎯 Creating Demo Data for Dashboard")
    print("=" * 40)
    
    # Create sample evaluation results
    print("📊 Generating sample evaluation results...")
    evaluation_results = create_sample_evaluation_data()
    
    # Save to JSON
    with open('demo_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"✅ Saved {len(evaluation_results)} evaluation results to demo_evaluation_results.json")
    
    # Create sample transcript DataFrame
    print("📋 Generating sample transcript data...")
    transcript_df = create_sample_transcript_dataframe()
    
    # Save to CSV
    transcript_df.to_csv('demo_transcript_data.csv', index=False)
    print(f"✅ Saved {len(transcript_df)} transcript records to demo_transcript_data.csv")
    
    # Create sample transcript JSON (matching expected format)
    sample_transcript = {
        "call_ID": "12345",
        "CSR_ID": "JaneDoe123",
        "call_date": "2024-02-01",
        "call_time": "02:16:43",
        "call_transcript": [
            "CSR: Thank you for calling ABC Travel, this is Jane. How may I assist you today?",
            "Customer: I need help with my booking. My flight was canceled!",
            "CSR: I apologize for the trouble. Let me look up your booking and see what options we have available for you.",
            "Customer: What's my reservation number? I can't find my email.",
            "CSR: I can help you locate that information. May I have your name and the email address you used for the booking?",
            "Customer: This is unacceptable! I booked this trip months ago!",
            "CSR: I completely understand your frustration. Let me see what I can do to resolve this situation for you right away."
        ]
    }
    
    with open('demo_call_transcript.json', 'w') as f:
        json.dump(sample_transcript, f, indent=2)
    print("✅ Saved sample call transcript to demo_call_transcript.json")
    
    print("\n" + "=" * 40)
    print("🎉 Demo data created successfully!")
    print("\n📁 Files created:")
    print("   - demo_evaluation_results.json")
    print("   - demo_transcript_data.csv") 
    print("   - demo_call_transcript.json")
    
    print("\n🚀 To test the dashboard:")
    print("1. Copy demo_evaluation_results.json to transcript_evaluation_results.json")
    print("2. Copy demo_call_transcript.json to Call Transcript Sample 1.json")
    print("3. Run: python launch_dashboard.py")
    
    print("\n💡 Or run the dashboard directly:")
    print("streamlit run streamlit_feedback_dashboard.py")

if __name__ == "__main__":
    main()

