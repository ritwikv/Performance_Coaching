#!/usr/bin/env python3
"""
Streamlit Frontend for Call Center Transcript Evaluator
======================================================

A comprehensive web interface for the Call Center Transcript Evaluator
using Mistral 7B model. This frontend provides an intuitive way to:
- Upload transcript JSON files
- Configure evaluation parameters
- Run Mistral evaluations
- View detailed feedback and coaching
- Download evaluation reports

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Import the main evaluator
try:
    from call_center_transcript_evaluator import CallCenterTranscriptEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing evaluator: {e}")
    EVALUATOR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Call Center Transcript Evaluator",
    page_icon="🎯",
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
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .feedback-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'uploaded_file_content' not in st.session_state:
        st.session_state.uploaded_file_content = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

def load_model(model_path):
    """Load the Mistral model."""
    try:
        with st.spinner("🤖 Loading Mistral 7B model... This may take a few minutes..."):
            evaluator = CallCenterTranscriptEvaluator(model_path)
            if evaluator.model is not None:
                st.session_state.evaluator = evaluator
                st.session_state.model_loaded = True
                return True
            else:
                return False
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return False

def validate_json_structure(json_data):
    """Validate the structure of uploaded JSON file."""
    required_fields = ['call_ID', 'CSR_ID', 'call_date', 'call_time', 'call_transcript']
    missing_fields = [field for field in required_fields if field not in json_data]
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    if not isinstance(json_data['call_transcript'], list):
        return False, "call_transcript must be a list"
    
    if len(json_data['call_transcript']) == 0:
        return False, "call_transcript cannot be empty"
    
    return True, "Valid JSON structure"

def display_transcript_preview(json_data):
    """Display a preview of the uploaded transcript."""
    st.markdown('<div class="section-header">📄 Transcript Preview</div>', unsafe_allow_html=True)
    
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Call ID", json_data.get('call_ID', 'N/A'))
    with col2:
        st.metric("CSR ID", json_data.get('CSR_ID', 'N/A'))
    with col3:
        st.metric("Date", json_data.get('call_date', 'N/A'))
    with col4:
        st.metric("Time", json_data.get('call_time', 'N/A'))
    
    # Transcript lines
    st.markdown("**Transcript Lines:**")
    transcript_lines = json_data.get('call_transcript', [])
    
    # Show first few lines
    preview_lines = min(5, len(transcript_lines))
    for i, line in enumerate(transcript_lines[:preview_lines]):
        if line.startswith('Customer:'):
            st.markdown(f"🗣️ **Customer:** {line.replace('Customer:', '').strip()}")
        elif line.startswith('CSR:'):
            st.markdown(f"👨‍💼 **CSR:** {line.replace('CSR:', '').strip()}")
        elif line.startswith('Supervisor:'):
            st.markdown(f"👩‍💼 **Supervisor:** {line.replace('Supervisor:', '').strip()}")
        else:
            st.markdown(f"💬 {line}")
    
    if len(transcript_lines) > preview_lines:
        st.markdown(f"... and {len(transcript_lines) - preview_lines} more lines")
    
    st.markdown(f"**Total Lines:** {len(transcript_lines)}")

def run_evaluation(json_data, temp_file_path):
    """Run the Mistral evaluation on the uploaded transcript."""
    if not st.session_state.model_loaded or st.session_state.evaluator is None:
        st.error("Model not loaded. Please load the model first.")
        return None
    
    try:
        with st.spinner("🔬 Running comprehensive evaluation... This may take several minutes..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Extracting Q&A pairs...")
            progress_bar.progress(20)
            
            # Run evaluation
            results = st.session_state.evaluator.evaluate_transcript(temp_file_path)
            progress_bar.progress(100)
            status_text.text("Evaluation completed!")
            
            return results
            
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return None

def display_evaluation_results(results):
    """Display comprehensive evaluation results."""
    if not results or 'error' in results:
        st.error(f"Evaluation error: {results.get('error', 'Unknown error')}")
        return
    
    st.markdown('<div class="section-header">📊 Evaluation Results</div>', unsafe_allow_html=True)
    
    # Summary metrics
    metadata = results['transcript_metadata']
    stats = results['summary_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Call ID", metadata['call_ID'])
    with col2:
        st.metric("CSR ID", metadata['CSR_ID'])
    with col3:
        st.metric("Q&A Pairs", stats['total_qa_pairs'])
    with col4:
        st.metric("Knowledge Docs", len(results['knowledge_documents']))
    
    # Detailed evaluations
    st.markdown('<div class="section-header">🎯 Detailed Feedback by Q&A Pair</div>', unsafe_allow_html=True)
    
    for i, eval_data in enumerate(results['evaluations']):
        with st.expander(f"Q&A Pair {i+1} - {eval_data.get('topic_summary', 'Analysis')[:50]}..."):
            
            # Question and Answer
            st.markdown("**🗣️ Customer Question:**")
            st.markdown(f'<div class="feedback-box">{eval_data["question"]}</div>', unsafe_allow_html=True)
            
            st.markdown("**👨‍💼 Agent Response:**")
            st.markdown(f'<div class="feedback-box">{eval_data["answer"]}</div>', unsafe_allow_html=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📝 English", "📏 Sentences", "🔄 Words", "😊 Sentiment", "🎯 Topic"
            ])
            
            with tab1:
                st.markdown("**English Correctness Evaluation:**")
                english_eval = eval_data['english_correctness']
                st.markdown(f'<div class="feedback-box">{english_eval["evaluation"]}</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("**Sentence Length Analysis:**")
                sent_analysis = eval_data['sentence_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Words/Sentence", f"{sent_analysis['average_words_per_sentence']:.1f}")
                with col2:
                    st.metric("Long Sentences", sent_analysis['long_sentences_count'])
                
                if sent_analysis['long_sentences']:
                    st.markdown("**Long Sentences Found:**")
                    for long_sent in sent_analysis['long_sentences']:
                        st.markdown(f"- {long_sent['sentence']} ({long_sent['word_count']} words)")
                
                st.markdown("**Recommendations:**")
                st.markdown(f'<div class="feedback-box">{sent_analysis["recommendations"]}</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown("**Word Repetition & Crutch Words:**")
                rep_analysis = eval_data['repetition_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vocabulary Diversity", f"{rep_analysis['vocabulary_diversity']:.3f}")
                with col2:
                    st.metric("Total Words", rep_analysis['total_words'])
                
                if rep_analysis['repeated_words']:
                    st.markdown("**Repeated Words:**")
                    for word, count in rep_analysis['repeated_words'].items():
                        st.markdown(f"- '{word}': {count} times")
                
                if rep_analysis['crutch_words']:
                    st.markdown("**Crutch Words Found:**")
                    for word, count in rep_analysis['crutch_words'].items():
                        st.markdown(f"- '{word}': {count} times")
                
                st.markdown("**Coaching Feedback:**")
                st.markdown(f'<div class="feedback-box">{rep_analysis["coaching_feedback"]}</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown("**Sentiment Analysis:**")
                sentiment = eval_data['sentiment_analysis']
                
                if 'error' not in sentiment:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment.get('sentiment_label', 'N/A'))
                    with col2:
                        st.metric("Polarity", f"{sentiment.get('polarity', 0):.3f}")
                    with col3:
                        st.metric("Subjectivity", f"{sentiment.get('subjectivity', 0):.3f}")
                    
                    # Sentiment coaching
                    st.markdown("**Sentiment Coaching:**")
                    coaching = sentiment.get('coaching_feedback', 'No coaching available')
                    st.markdown(f'<div class="feedback-box">{coaching}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Sentiment analysis error: {sentiment['error']}")
            
            with tab5:
                st.markdown("**Topic & Theme Summary:**")
                topic_summary = eval_data['topic_summary']
                st.markdown(f'<div class="feedback-box">{topic_summary}</div>', unsafe_allow_html=True)

def display_knowledge_documents(knowledge_docs):
    """Display created knowledge documents."""
    st.markdown('<div class="section-header">📚 Knowledge Documents Created</div>', unsafe_allow_html=True)
    
    for i, doc in enumerate(knowledge_docs, 1):
        with st.expander(f"Knowledge Document {i}"):
            st.markdown(doc)

def create_summary_dashboard(results):
    """Create a summary dashboard with visualizations."""
    if not results or 'evaluations' not in results:
        return
    
    st.markdown('<div class="section-header">📈 Summary Dashboard</div>', unsafe_allow_html=True)
    
    evaluations = results['evaluations']
    
    # Collect data for visualizations
    sentiment_data = []
    sentence_lengths = []
    crutch_word_counts = []
    
    for eval_data in evaluations:
        # Sentiment data
        sentiment = eval_data.get('sentiment_analysis', {})
        if 'sentiment_label' in sentiment:
            sentiment_data.append(sentiment['sentiment_label'])
        
        # Sentence length data
        sent_analysis = eval_data.get('sentence_analysis', {})
        if 'average_words_per_sentence' in sent_analysis:
            sentence_lengths.append(sent_analysis['average_words_per_sentence'])
        
        # Crutch word data
        rep_analysis = eval_data.get('repetition_analysis', {})
        if 'crutch_words' in rep_analysis:
            crutch_word_counts.append(len(rep_analysis['crutch_words']))
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if sentiment_data:
            # Sentiment distribution
            sentiment_df = pd.DataFrame({'Sentiment': sentiment_data})
            sentiment_counts = sentiment_df['Sentiment'].value_counts()
            
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        if sentence_lengths:
            # Sentence length distribution
            fig_sentences = px.histogram(
                x=sentence_lengths,
                title="Average Words per Sentence Distribution",
                labels={'x': 'Average Words per Sentence', 'y': 'Count'}
            )
            st.plotly_chart(fig_sentences, use_container_width=True)
    
    # Summary statistics
    if sentence_lengths:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Sentence Length", f"{sum(sentence_lengths)/len(sentence_lengths):.1f} words")
        with col2:
            st.metric("Max Sentence Length", f"{max(sentence_lengths):.1f} words")
        with col3:
            st.metric("Min Sentence Length", f"{min(sentence_lengths):.1f} words")

def generate_download_report(results):
    """Generate downloadable report."""
    if not results:
        return None
    
    try:
        # Generate report using the evaluator
        report = st.session_state.evaluator.generate_report(results)
        return report
    except Exception as e:
        st.error(f"Failed to generate report: {e}")
        return None

def create_download_link(content, filename, link_text):
    """Create a download link for content."""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎯 Call Center Transcript Evaluator</div>', unsafe_allow_html=True)
    st.markdown("**Powered by Mistral 7B Model for Comprehensive Performance Analysis**")
    
    # Check if evaluator is available
    if not EVALUATOR_AVAILABLE:
        st.markdown('<div class="error-box">❌ Evaluator module not available. Please ensure call_center_transcript_evaluator.py is in the same directory.</div>', unsafe_allow_html=True)
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model configuration
        st.subheader("🤖 Model Settings")
        model_path = st.text_input(
            "Mistral Model Path",
            value="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            help="Path to your Mistral 7B GGUF model file"
        )
        
        # Model loading
        if st.button("🔄 Load Model", type="primary"):
            if Path(model_path).exists():
                if load_model(model_path):
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model")
            else:
                st.error(f"❌ Model file not found: {model_path}")
        
        # Model status
        if st.session_state.model_loaded:
            st.success("🤖 Model Ready")
        else:
            st.warning("⚠️ Model Not Loaded")
        
        st.divider()
        
        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        show_detailed_feedback = st.checkbox("Show Detailed Feedback", value=True)
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        show_knowledge_docs = st.checkbox("Show Knowledge Documents", value=True)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Evaluate", "📊 Results", "📚 Knowledge Base"])
    
    with tab1:
        st.markdown("### 📤 Upload Transcript File")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a JSON transcript file",
            type=['json'],
            help="Upload a JSON file with the required transcript structure"
        )
        
        if uploaded_file is not None:
            try:
                # Read and parse JSON
                json_content = json.loads(uploaded_file.read().decode('utf-8'))
                st.session_state.uploaded_file_content = json_content
                
                # Validate JSON structure
                is_valid, message = validate_json_structure(json_content)
                
                if is_valid:
                    st.markdown('<div class="success-box">✅ Valid transcript file uploaded!</div>', unsafe_allow_html=True)
                    
                    # Display preview
                    display_transcript_preview(json_content)
                    
                    # Evaluation button
                    st.markdown("### 🚀 Run Evaluation")
                    
                    if st.button("🔬 Run Mistral Evaluation", type="primary", disabled=not st.session_state.model_loaded):
                        if st.session_state.model_loaded:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                                json.dump(json_content, tmp_file)
                                temp_file_path = tmp_file.name
                            
                            try:
                                # Run evaluation
                                results = run_evaluation(json_content, temp_file_path)
                                
                                if results:
                                    st.session_state.evaluation_results = results
                                    st.success("✅ Evaluation completed successfully!")
                                    st.balloons()
                                else:
                                    st.error("❌ Evaluation failed")
                                
                            finally:
                                # Clean up temporary file
                                if os.path.exists(temp_file_path):
                                    os.unlink(temp_file_path)
                        else:
                            st.error("❌ Please load the Mistral model first")
                    
                    if not st.session_state.model_loaded:
                        st.markdown('<div class="warning-box">⚠️ Please load the Mistral model in the sidebar before running evaluation.</div>', unsafe_allow_html=True)
                
                else:
                    st.markdown(f'<div class="error-box">❌ Invalid file structure: {message}</div>', unsafe_allow_html=True)
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {e}")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        
        else:
            st.info("👆 Please upload a JSON transcript file to begin evaluation")
            
            # Show example JSON structure
            with st.expander("📋 Example JSON Structure"):
                example_json = {
                    "call_ID": "12345",
                    "CSR_ID": "JaneDoe123",
                    "call_date": "2024-02-01",
                    "call_time": "02:16:43",
                    "call_transcript": [
                        "CSR: Thank you for calling ABC Travel, this is Jane. How may I assist you today?",
                        "Customer: Yes, I need help with a reservation I made last week.",
                        "CSR: I apologize for the trouble. May I have your name and reservation number?"
                    ]
                }
                st.json(example_json)
    
    with tab2:
        st.markdown("### 📊 Evaluation Results")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            # Display results
            if show_detailed_feedback:
                display_evaluation_results(results)
            
            # Show visualizations
            if show_visualizations:
                create_summary_dashboard(results)
            
            # Download report
            st.markdown("### 📥 Download Report")
            if st.button("📄 Generate Full Report"):
                report_content = generate_download_report(results)
                if report_content:
                    st.download_button(
                        label="📥 Download Evaluation Report",
                        data=report_content,
                        file_name=f"evaluation_report_{results['transcript_metadata']['call_ID']}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("📊 No evaluation results available. Please upload and evaluate a transcript first.")
    
    with tab3:
        st.markdown("### 📚 Knowledge Documents")
        
        if st.session_state.evaluation_results and show_knowledge_docs:
            knowledge_docs = st.session_state.evaluation_results.get('knowledge_documents', [])
            if knowledge_docs:
                display_knowledge_documents(knowledge_docs)
            else:
                st.info("📚 No knowledge documents available.")
        else:
            st.info("📚 Knowledge documents will appear here after evaluation.")
    
    # Footer
    st.markdown("---")
    st.markdown("**🎯 Call Center Transcript Evaluator** | Powered by Mistral 7B | Built with Streamlit")

if __name__ == "__main__":
    main()

