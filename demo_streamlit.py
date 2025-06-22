#!/usr/bin/env python3
"""
Demo Script for Streamlit Call Center Transcript Evaluator
==========================================================

This script demonstrates how to use the Streamlit application
and provides sample interactions for testing.

Usage:
    python demo_streamlit.py
"""

import json
import os
from pathlib import Path

def create_sample_transcripts():
    """Create additional sample transcript files for testing."""
    
    # Sample transcript 2 - Shorter conversation
    sample_2 = {
        "call_ID": "67890",
        "CSR_ID": "MikeSmith456",
        "call_date": "2024-02-15",
        "call_time": "14:30:22",
        "call_transcript": [
            "CSR: Good afternoon, thank you for calling TechSupport Pro. This is Mike, how can I help you today?",
            "Customer: Hi, I'm having trouble with my internet connection. It keeps dropping out every few minutes.",
            "CSR: I'm sorry to hear about the connection issues. Let me help you troubleshoot this. Can you tell me what type of modem you're using?",
            "Customer: It's a Netgear router, about 2 years old. The lights keep blinking red.",
            "CSR: Okay, the red blinking light usually indicates a connection problem. Let's try restarting your modem first. Can you unplug it for 30 seconds and then plug it back in?",
            "Customer: Alright, I've unplugged it. Should I wait exactly 30 seconds?",
            "CSR: Yes, waiting 30 seconds allows the device to fully reset. Now you can plug it back in and wait for the lights to stabilize.",
            "Customer: Okay, it's plugged back in. The lights are cycling through different colors now.",
            "CSR: Perfect! That's normal during the startup process. It should take about 2-3 minutes for all lights to turn solid green. This usually resolves most connection issues.",
            "Customer: Great! The lights are now solid green and my internet seems to be working. Thank you so much for your help!",
            "CSR: You're very welcome! I'm glad we could resolve this quickly. If you experience any more issues, please don't hesitate to call us back. Have a great day!"
        ]
    }
    
    # Sample transcript 3 - Complex billing issue
    sample_3 = {
        "call_ID": "11111",
        "CSR_ID": "SarahJohnson789",
        "call_date": "2024-02-20",
        "call_time": "09:15:33",
        "call_transcript": [
            "CSR: Thank you for calling BillCorp Customer Service. This is Sarah speaking. How may I assist you today?",
            "Customer: I'm calling about my bill. There are charges on here that I don't understand and I'm really frustrated.",
            "CSR: I completely understand your frustration, and I'm here to help clarify those charges for you. May I have your account number please?",
            "Customer: It's 555-123-4567. I've been a customer for 5 years and I've never seen charges like this before.",
            "CSR: Thank you for providing that. Let me pull up your account right now. I can see you've been a valued customer since 2019. I'm looking at your recent bill now. Can you tell me which specific charges you're concerned about?",
            "Customer: There's a $25 service fee and a $15 equipment charge. I never requested any service or new equipment.",
            "CSR: I see those charges on your account. The $25 service fee was applied when a technician visited your location on February 5th to resolve a signal issue. The $15 equipment charge is for a new cable box that was installed during that visit.",
            "Customer: But I never called for a technician! Someone just showed up at my door saying there was a problem in the neighborhood.",
            "CSR: I apologize for the confusion. Let me investigate this further. I can see in our system that this was actually a proactive service call due to area-wide signal issues. You shouldn't have been charged for this since it was our initiative to fix a network problem.",
            "Customer: Exactly! So why am I being charged for something that was your company's problem?",
            "CSR: You're absolutely right, and I sincerely apologize for this billing error. I'm going to remove both charges from your account immediately. The $25 service fee and $15 equipment charge will be credited back to your account within 2-3 business days.",
            "Customer: Thank you, that's what I was hoping to hear. Will this affect my service in any way?",
            "CSR: Not at all. Your service will continue normally, and you'll see the $40 credit on your next bill. I'm also adding a note to your account about this situation. Is there anything else I can help you with today?",
            "Customer: No, that resolves my issue. Thank you for taking care of this so quickly.",
            "CSR: You're very welcome! Thank you for being a loyal customer, and again, I apologize for the billing confusion. Have a wonderful day!"
        ]
    }
    
    # Save sample files
    samples_dir = Path("sample_transcripts")
    samples_dir.mkdir(exist_ok=True)
    
    with open(samples_dir / "sample_transcript_2.json", "w") as f:
        json.dump(sample_2, f, indent=2)
    
    with open(samples_dir / "sample_transcript_3.json", "w") as f:
        json.dump(sample_3, f, indent=2)
    
    return [
        samples_dir / "sample_transcript_2.json",
        samples_dir / "sample_transcript_3.json"
    ]

def print_demo_instructions():
    """Print instructions for using the demo."""
    print("🎯 Call Center Transcript Evaluator - Streamlit Demo")
    print("=" * 60)
    print()
    print("📋 Demo Instructions:")
    print("1. Launch the Streamlit app:")
    print("   python launch_streamlit.py")
    print()
    print("2. Load the Mistral model:")
    print("   - Use the sidebar to configure model path")
    print("   - Click 'Load Model' button")
    print("   - Wait for 'Model Ready' confirmation")
    print()
    print("3. Test with sample files:")
    print("   - Upload 'Call Transcript Sample 1.json' (original)")
    print("   - Try the additional samples created by this demo")
    print("   - Click 'Run Mistral Evaluation' for each")
    print()
    print("4. Explore the results:")
    print("   - View detailed feedback for each Q&A pair")
    print("   - Check sentiment analysis and coaching")
    print("   - Review topic summaries and themes")
    print("   - Download comprehensive reports")
    print()
    print("🎨 UI Features to Test:")
    print("- Expandable Q&A sections")
    print("- Tabbed analysis views (English, Sentences, Words, Sentiment, Topic)")
    print("- Interactive visualizations")
    print("- Knowledge document generation")
    print("- Report download functionality")
    print()

def print_sample_outputs():
    """Print examples of expected outputs."""
    print("📊 Expected Output Examples:")
    print("-" * 40)
    print()
    print("🎯 CSR Performance Summary:")
    print("CSR ID: JaneDoe123")
    print("Call ID: 12345")
    print("Overall Sentiment: Professional and Empathetic")
    print("Key Strengths: Excellent problem-solving, clear communication")
    print("Areas for Improvement: Reduce sentence length, avoid filler words")
    print()
    print("😊 Sentiment Analysis Example:")
    print("'You maintained a positive and helpful tone throughout the interaction.'")
    print("'Your empathy score is excellent - you clearly understood the customer's frustration.'")
    print("'Consider being more confident in your responses to build customer trust.'")
    print()
    print("🎯 Topic Summary Example:")
    print("Main Topic: Flight Cancellation and Refund Request")
    print("Issue Complexity: Moderate")
    print("Skills Demonstrated: Empathy, Policy Knowledge, Problem Resolution")
    print("Coaching Focus: Continue excellent customer service approach")
    print()
    print("📝 English Correctness Example:")
    print("Grammar Score: 8/10")
    print("Suggestions: Use more active voice, reduce run-on sentences")
    print("Coaching: 'Great professional language! Try shorter, more direct responses.'")
    print()

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking Prerequisites:")
    print("-" * 30)
    
    # Check main files
    required_files = [
        "streamlit_app.py",
        "call_center_transcript_evaluator.py",
        "Call Transcript Sample 1.json"
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_good = False
    
    # Check for model file
    model_files = [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.q4_k_m.gguf"
    ]
    
    model_found = False
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ {model_file}")
            model_found = True
            break
    
    if not model_found:
        print("⚠️  Mistral model file not found")
        print("   Download and place in project directory")
    
    print()
    return all_good and model_found

def main():
    """Main demo function."""
    print_demo_instructions()
    print()
    
    # Check prerequisites
    prereqs_ok = check_prerequisites()
    
    if not prereqs_ok:
        print("❌ Some prerequisites are missing. Please install required files first.")
        print()
    
    # Create additional sample files
    print("📁 Creating Additional Sample Files:")
    try:
        sample_files = create_sample_transcripts()
        print("✅ Created sample transcript files:")
        for file in sample_files:
            print(f"   - {file}")
        print()
    except Exception as e:
        print(f"❌ Error creating sample files: {e}")
        print()
    
    # Print expected outputs
    print_sample_outputs()
    
    print("🚀 Ready to Start Demo!")
    print("=" * 60)
    print("Run: python launch_streamlit.py")
    print("Then open your browser to: http://localhost:8501")
    print()
    print("💡 Pro Tips:")
    print("- Start with the original 'Call Transcript Sample 1.json'")
    print("- Try different sample files to see various scenarios")
    print("- Explore all tabs and features in the interface")
    print("- Download reports to see the full evaluation output")
    print("- Check the sidebar for configuration options")

if __name__ == "__main__":
    main()

