#!/usr/bin/env python3
"""
Basic Streamlit Dashboard for Call Center Transcript Evaluation
Lightweight version without heavy ML dependencies for initial testing
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure Streamlit page
st.set_page_config(
    page_title="Call Center Performance Coaching - Basic",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check which dependencies are available"""
    deps = {}
    
    # Core dependencies
    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        deps['pandas'] = False
    
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        deps['numpy'] = False
    
    # ML dependencies (optional)
    try:
        import llama_cpp
        deps['llama_cpp'] = True
    except ImportError:
        deps['llama_cpp'] = False
    
    try:
        import sentence_transformers
        deps['sentence_transformers'] = True
    except ImportError:
        deps['sentence_transformers'] = False
    
    try:
        import chromadb
        deps['chromadb'] = True
    except ImportError:
        deps['chromadb'] = False
    
    try:
        import textblob
        deps['textblob'] = True
    except ImportError:
        deps['textblob'] = False
    
    return deps

def load_and_process_transcript(file_path: str) -> Optional[pd.DataFrame]:
    """Load and process transcript using basic data processor"""
    try:
        from data_processor import CallTranscriptProcessor
        
        processor = CallTranscriptProcessor()
        transcript_data = processor.load_json_transcript(file_path)
        
        if transcript_data:
            records = processor.process_single_transcript(transcript_data)
            return pd.DataFrame(records)
        return None
        
    except Exception as e:
        st.error(f"Error processing transcript: {e}")
        return None

def basic_quality_analysis(text: str) -> Dict[str, Any]:
    """Basic quality analysis without ML dependencies"""
    words = text.split()
    sentences = text.split('.')
    
    # Basic metrics
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Simple crutch word detection
    crutch_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 'kind of']
    crutch_count = sum(text.lower().count(word) for word in crutch_words)
    
    # Hold request detection
    hold_patterns = ['please hold', 'hold on', 'one moment', 'just a moment', 'bear with me']
    hold_requests = sum(text.lower().count(pattern) for pattern in hold_patterns)
    
    # Transfer detection
    transfer_patterns = ['transfer you to', 'speak to a supervisor', 'escalate this']
    transfer_mentions = sum(text.lower().count(pattern) for pattern in transfer_patterns)
    
    # Basic scoring
    clarity_score = max(0, min(10, 10 - (avg_sentence_length - 15) * 0.3))
    repetition_score = max(0, min(10, 10 - crutch_count * 2))
    efficiency_score = max(0, min(10, 10 - hold_requests * 2 - transfer_mentions * 3))
    
    overall_score = (clarity_score + repetition_score + efficiency_score) / 3
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': round(avg_sentence_length, 2),
        'crutch_count': crutch_count,
        'hold_requests': hold_requests,
        'transfer_mentions': transfer_mentions,
        'clarity_score': round(clarity_score, 2),
        'repetition_score': round(repetition_score, 2),
        'efficiency_score': round(efficiency_score, 2),
        'overall_score': round(overall_score, 2)
    }

def basic_sentiment_analysis(text: str) -> Dict[str, Any]:
    """Basic sentiment analysis without TextBlob"""
    text_lower = text.lower()
    
    # Positive words
    positive_words = ['thank', 'please', 'happy', 'excellent', 'great', 'wonderful', 'appreciate']
    positive_count = sum(text_lower.count(word) for word in positive_words)
    
    # Negative words
    negative_words = ['sorry', 'apologize', 'problem', 'issue', 'trouble', 'difficult', 'frustrated']
    negative_count = sum(text_lower.count(word) for word in negative_words)
    
    # Professional words
    professional_words = ['may i', 'could you', 'i would be happy', 'certainly', 'absolutely']
    professional_count = sum(text_lower.count(phrase) for phrase in professional_words)
    
    # Simple sentiment calculation
    if positive_count > negative_count:
        sentiment = "Positive"
    elif negative_count > positive_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Tone analysis
    if professional_count > 0:
        tone = "Professional"
    elif positive_count > 0:
        tone = "Friendly"
    elif negative_count > 0:
        tone = "Apologetic"
    else:
        tone = "Neutral"
    
    return {
        'sentiment': sentiment,
        'tone': tone,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'professional_count': professional_count
    }

def generate_basic_coaching_feedback(quality_analysis: Dict, sentiment_analysis: Dict) -> str:
    """Generate basic coaching feedback"""
    feedback_parts = []
    
    # Quality feedback
    overall_score = quality_analysis['overall_score']
    if overall_score >= 8:
        feedback_parts.append(f"Excellent performance with an overall score of {overall_score}/10.")
    elif overall_score >= 6:
        feedback_parts.append(f"Good performance with a score of {overall_score}/10, with room for improvement.")
    else:
        feedback_parts.append(f"Performance needs attention with a score of {overall_score}/10.")
    
    # Specific recommendations
    if quality_analysis['avg_sentence_length'] > 25:
        feedback_parts.append("Consider using shorter, clearer sentences for better customer understanding.")
    
    if quality_analysis['crutch_count'] > 3:
        feedback_parts.append("Reduce use of filler words like 'um', 'uh', and 'like' for more professional communication.")
    
    if quality_analysis['hold_requests'] > 2:
        feedback_parts.append("Minimize hold requests to improve customer experience and reduce call time.")
    
    # Sentiment feedback
    sentiment = sentiment_analysis['sentiment']
    tone = sentiment_analysis['tone']
    
    if sentiment == "Positive":
        feedback_parts.append(f"Great job maintaining a {sentiment.lower()} tone with {tone.lower()} communication style.")
    elif sentiment == "Negative":
        feedback_parts.append(f"Consider using more positive language while maintaining your {tone.lower()} approach.")
    else:
        feedback_parts.append(f"Your {tone.lower()} communication style is appropriate for customer service.")
    
    return " ".join(feedback_parts)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Call Center Performance Coaching - Basic Mode</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Dependency check
    st.sidebar.header("🔍 System Status")
    deps = check_dependencies()
    
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        st.sidebar.write(f"{status} {dep}")
    
    # Check if model file exists
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    model_exists = os.path.exists(model_path)
    st.sidebar.write(f"{'✅' if model_exists else '❌'} Mistral Model")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyze", "📊 Results", "ℹ️ About"])
    
    with tab1:
        st.header("📁 Upload Call Transcripts")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose JSON transcript files",
            type=['json'],
            accept_multiple_files=True,
            help="Upload one or more JSON files containing call transcripts"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            # Process files
            all_results = []
            
            for file in uploaded_files:
                st.subheader(f"Processing: {file.name}")
                
                # Save file temporarily
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Process transcript
                df = load_and_process_transcript(temp_path)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Processed {len(df)} conversation pairs")
                    
                    # Analyze each conversation
                    for idx, row in df.iterrows():
                        question = row['question']
                        answer = row['answer']
                        
                        # Basic analysis
                        quality_analysis = basic_quality_analysis(answer)
                        sentiment_analysis = basic_sentiment_analysis(answer)
                        coaching_feedback = generate_basic_coaching_feedback(quality_analysis, sentiment_analysis)
                        
                        result = {
                            'file_name': file.name,
                            'call_id': row.get('call_ID', ''),
                            'csr_id': row.get('CSR_ID', ''),
                            'interaction_sequence': row.get('interaction_sequence', 0),
                            'question': question,
                            'answer': answer,
                            'quality_analysis': quality_analysis,
                            'sentiment_analysis': sentiment_analysis,
                            'coaching_feedback': coaching_feedback
                        }
                        all_results.append(result)
                    
                    # Show sample analysis
                    if len(df) > 0:
                        sample_row = df.iloc[0]
                        sample_quality = basic_quality_analysis(sample_row['answer'])
                        sample_sentiment = basic_sentiment_analysis(sample_row['answer'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Score", f"{sample_quality['overall_score']}/10")
                        with col2:
                            st.metric("Sentiment", sample_sentiment['sentiment'])
                        with col3:
                            st.metric("Tone", sample_sentiment['tone'])
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Store results in session state
            st.session_state['analysis_results'] = all_results
            
            if all_results:
                st.success(f"🎉 Analysis completed! Processed {len(all_results)} conversations.")
                st.info("View detailed results in the 'Results' tab.")
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
            results = st.session_state['analysis_results']
            
            # Summary metrics
            st.subheader("📈 Summary Metrics")
            
            total_conversations = len(results)
            avg_quality = sum(r['quality_analysis']['overall_score'] for r in results) / total_conversations
            
            sentiment_counts = {}
            for r in results:
                sentiment = r['sentiment_analysis']['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", total_conversations)
            with col2:
                st.metric("Average Quality Score", f"{avg_quality:.1f}/10")
            with col3:
                most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                st.metric("Most Common Sentiment", most_common_sentiment)
            
            # Detailed results
            st.subheader("📋 Detailed Results")
            
            # Create DataFrame for display
            display_data = []
            for result in results:
                row = {
                    'CSR_ID': result['csr_id'],
                    'Call_ID': result['call_id'],
                    'Quality_Score': result['quality_analysis']['overall_score'],
                    'Sentiment': result['sentiment_analysis']['sentiment'],
                    'Tone': result['sentiment_analysis']['tone'],
                    'Avg_Sentence_Length': result['quality_analysis']['avg_sentence_length'],
                    'Crutch_Words': result['quality_analysis']['crutch_count'],
                    'Hold_Requests': result['quality_analysis']['hold_requests']
                }
                display_data.append(row)
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
            
            # Individual feedback
            st.subheader("💬 Individual Coaching Feedback")
            
            selected_idx = st.selectbox(
                "Select conversation for detailed feedback:",
                range(len(results)),
                format_func=lambda x: f"Conversation {x+1} - {results[x]['csr_id']}"
            )
            
            if selected_idx is not None:
                selected_result = results[selected_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Customer Question:**")
                    st.write(selected_result['question'])
                    st.write("**CSR Response:**")
                    st.write(selected_result['answer'])
                
                with col2:
                    st.write("**Quality Analysis:**")
                    quality = selected_result['quality_analysis']
                    st.write(f"- Overall Score: {quality['overall_score']}/10")
                    st.write(f"- Clarity Score: {quality['clarity_score']}/10")
                    st.write(f"- Avg Sentence Length: {quality['avg_sentence_length']} words")
                    st.write(f"- Crutch Words: {quality['crutch_count']}")
                    
                    st.write("**Sentiment Analysis:**")
                    sentiment = selected_result['sentiment_analysis']
                    st.write(f"- Sentiment: {sentiment['sentiment']}")
                    st.write(f"- Tone: {sentiment['tone']}")
                
                st.write("**Coaching Feedback:**")
                st.info(selected_result['coaching_feedback'])
        
        else:
            st.info("📊 No analysis results available. Please upload and analyze transcripts first.")
    
    with tab3:
        st.header("ℹ️ About This Dashboard")
        st.markdown("""
        This is a **basic mode** version of the Call Center Performance Coaching Dashboard that works without heavy ML dependencies.
        
        ### Features Available:
        - **📁 File Upload**: Upload JSON transcript files for analysis
        - **📊 Basic Quality Analysis**: Sentence structure, word count, crutch word detection
        - **💭 Simple Sentiment Analysis**: Positive/negative/neutral sentiment detection
        - **📈 Performance Metrics**: Overall scoring and recommendations
        - **💬 Coaching Feedback**: Personalized improvement suggestions
        
        ### Limitations (Basic Mode):
        - No Mistral 7B model inference
        - No RAG pipeline or expert answer generation
        - No DeepEval metrics
        - Simplified sentiment analysis (no TextBlob)
        - Basic quality scoring algorithms
        
        ### To Enable Full Features:
        1. Install all dependencies: `pip install -r requirements_complete.txt`
        2. Download Mistral model: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
        3. Use the full dashboard: `streamlit run enhanced_streamlit_dashboard.py`
        
        ### System Requirements:
        - Python 3.8+
        - 4GB+ RAM (basic mode)
        - JSON transcript files in the specified format
        """)
        
        # System status
        st.subheader("🔧 Current System Status")
        deps = check_dependencies()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Dependencies:**")
            for dep in ['pandas', 'numpy']:
                status = "✅ Available" if deps.get(dep, False) else "❌ Missing"
                st.write(f"- {dep}: {status}")
        
        with col2:
            st.write("**ML Dependencies (Optional):**")
            for dep in ['llama_cpp', 'sentence_transformers', 'chromadb', 'textblob']:
                status = "✅ Available" if deps.get(dep, False) else "❌ Missing"
                st.write(f"- {dep}: {status}")

if __name__ == "__main__":
    main()

