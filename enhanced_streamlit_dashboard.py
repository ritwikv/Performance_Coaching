#!/usr/bin/env python3
"""
Enhanced Streamlit Dashboard for Call Center Transcript Evaluation
Complete frontend with 'Run Mistral Evaluation' functionality and feedback display
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our evaluation components
from evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig
from data_processor import CallTranscriptProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="Call Center Performance Coaching Dashboard",
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
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class DashboardState:
    """Manages dashboard state and session variables."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = []
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = None
        if 'last_evaluation_time' not in st.session_state:
            st.session_state.last_evaluation_time = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'evaluation_in_progress' not in st.session_state:
            st.session_state.evaluation_in_progress = False

class ConfigurationManager:
    """Manages evaluation configuration through the UI."""
    
    @staticmethod
    def render_configuration_sidebar():
        """Render configuration options in sidebar."""
        st.sidebar.header("🔧 Configuration")
        
        # Model configuration
        st.sidebar.subheader("Model Settings")
        model_path = st.sidebar.text_input(
            "Mistral Model Path",
            value="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            help="Path to the Mistral model file"
        )
        
        # Feature toggles
        st.sidebar.subheader("Analysis Features")
        enable_rag = st.sidebar.checkbox("Enable RAG Pipeline", value=True)
        enable_deepeval = st.sidebar.checkbox("Enable DeepEval Metrics", value=True)
        enable_quality = st.sidebar.checkbox("Enable Quality Analysis", value=True)
        enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
        
        # Output settings
        st.sidebar.subheader("Output Settings")
        output_format = st.sidebar.selectbox(
            "Output Format",
            ["json", "csv", "excel"],
            index=0
        )
        
        save_intermediate = st.sidebar.checkbox("Save Intermediate Results", value=True)
        
        # Create configuration
        config = EvaluationConfig(
            mistral_model_path=model_path,
            enable_rag=enable_rag,
            enable_deepeval=enable_deepeval,
            enable_quality_analysis=enable_quality,
            enable_sentiment_analysis=enable_sentiment,
            output_format=output_format,
            save_intermediate_results=save_intermediate
        )
        
        return config

class FileUploadManager:
    """Manages file upload functionality."""
    
    @staticmethod
    def render_file_upload():
        """Render file upload interface."""
        st.header("📁 Upload Call Transcripts")
        
        uploaded_files = st.file_uploader(
            "Choose JSON transcript files",
            type=['json'],
            accept_multiple_files=True,
            help="Upload one or more JSON files containing call transcripts"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            # Show file details
            with st.expander("📋 File Details"):
                for file in uploaded_files:
                    st.write(f"**{file.name}** - {file.size} bytes")
            
            # Save uploaded files temporarily
            saved_paths = []
            for file in uploaded_files:
                file_path = f"temp_{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(file_path)
            
            st.session_state.uploaded_files = saved_paths
            return saved_paths
        
        return []

class EvaluationRunner:
    """Handles the evaluation process."""
    
    @staticmethod
    def render_evaluation_controls(config: EvaluationConfig, file_paths: List[str]):
        """Render evaluation controls and run button."""
        st.header("🚀 Run Mistral Evaluation")
        
        if not file_paths:
            st.warning("⚠️ Please upload transcript files first")
            return
        
        # Show configuration summary
        with st.expander("⚙️ Current Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Path:**", config.mistral_model_path)
                st.write("**RAG Enabled:**", "✅" if config.enable_rag else "❌")
                st.write("**DeepEval Enabled:**", "✅" if config.enable_deepeval else "❌")
            with col2:
                st.write("**Quality Analysis:**", "✅" if config.enable_quality_analysis else "❌")
                st.write("**Sentiment Analysis:**", "✅" if config.enable_sentiment_analysis else "❌")
                st.write("**Output Format:**", config.output_format.upper())
        
        # Evaluation button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🎯 Run Mistral Evaluation",
                type="primary",
                disabled=st.session_state.evaluation_in_progress,
                use_container_width=True
            ):
                EvaluationRunner.run_evaluation(config, file_paths)
    
    @staticmethod
    def run_evaluation(config: EvaluationConfig, file_paths: List[str]):
        """Run the complete evaluation process."""
        st.session_state.evaluation_in_progress = True
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize orchestrator
            status_text.text("🔄 Initializing evaluation engine...")
            progress_bar.progress(10)
            
            orchestrator = EvaluationOrchestrator(config)
            
            if not orchestrator.initialize():
                st.error("❌ Failed to initialize evaluation engine. Please check your configuration.")
                return
            
            st.session_state.orchestrator = orchestrator
            progress_bar.progress(20)
            
            # Process each file
            all_results = []
            total_files = len(file_paths)
            
            for i, file_path in enumerate(file_paths):
                status_text.text(f"📊 Evaluating file {i+1}/{total_files}: {os.path.basename(file_path)}")
                
                # Evaluate transcript
                results = orchestrator.evaluate_transcript_file(file_path)
                all_results.extend(results)
                
                # Update progress
                progress = 20 + (70 * (i + 1) / total_files)
                progress_bar.progress(int(progress))
            
            # Save results
            status_text.text("💾 Saving evaluation results...")
            progress_bar.progress(95)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.{config.output_format}"
            
            if orchestrator.save_results(output_file, config.output_format):
                st.session_state.evaluation_results = all_results
                st.session_state.last_evaluation_time = datetime.now()
                
                progress_bar.progress(100)
                status_text.text("✅ Evaluation completed successfully!")
                
                # Show success message
                st.success(f"""
                🎉 **Evaluation Completed Successfully!**
                
                - **Files Processed:** {total_files}
                - **Conversations Evaluated:** {len(all_results)}
                - **Results Saved:** {output_file}
                - **Processing Time:** {sum(r.processing_time_seconds for r in all_results):.1f} seconds
                """)
                
                # Clean up temporary files
                for file_path in file_paths:
                    if os.path.exists(file_path) and file_path.startswith("temp_"):
                        os.remove(file_path)
                
            else:
                st.error("❌ Failed to save evaluation results")
                
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
            
        finally:
            st.session_state.evaluation_in_progress = False
            progress_bar.empty()
            status_text.empty()

class ResultsViewer:
    """Handles display of evaluation results."""
    
    @staticmethod
    def render_results_dashboard():
        """Render the main results dashboard."""
        if not st.session_state.evaluation_results:
            st.info("📊 No evaluation results available. Please run an evaluation first.")
            return
        
        st.header("📈 Evaluation Results Dashboard")
        
        results = st.session_state.evaluation_results
        
        # Summary metrics
        ResultsViewer.render_summary_metrics(results)
        
        # Detailed results
        ResultsViewer.render_detailed_results(results)
        
        # Visualizations
        ResultsViewer.render_visualizations(results)
    
    @staticmethod
    def render_summary_metrics(results: List):
        """Render summary metrics cards."""
        st.subheader("📊 Summary Metrics")
        
        # Calculate summary statistics
        total_conversations = len(results)
        unique_csrs = len(set(r.csr_id for r in results))
        
        # Quality scores
        quality_scores = [r.quality_scores.get('quality_metrics', {}).get('overall_score', 0) 
                         for r in results if r.quality_scores]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # DeepEval scores
        deepeval_scores = [r.deepeval_scores.get('overall', {}).get('score', 0) 
                          for r in results if r.deepeval_scores]
        avg_deepeval = sum(deepeval_scores) / len(deepeval_scores) if deepeval_scores else 0
        
        # Sentiment distribution
        sentiments = [r.sentiment_analysis.get('sentiment_label', 'Unknown') 
                     for r in results if r.sentiment_analysis]
        positive_sentiment = sentiments.count('Positive') / len(sentiments) * 100 if sentiments else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Conversations",
                value=total_conversations,
                delta=f"{unique_csrs} CSRs"
            )
        
        with col2:
            st.metric(
                label="Avg Quality Score",
                value=f"{avg_quality:.1f}/10",
                delta="Quality Analysis"
            )
        
        with col3:
            st.metric(
                label="Avg DeepEval Score",
                value=f"{avg_deepeval:.2f}/1.0",
                delta="AI Evaluation"
            )
        
        with col4:
            st.metric(
                label="Positive Sentiment",
                value=f"{positive_sentiment:.1f}%",
                delta="Communication Style"
            )
    
    @staticmethod
    def render_detailed_results(results: List):
        """Render detailed results table."""
        st.subheader("📋 Detailed Results")
        
        # Create DataFrame for display
        display_data = []
        for result in results:
            row = {
                'CSR_ID': result.csr_id,
                'Call_ID': result.call_id,
                'Date': result.call_date,
                'Topic': result.topic_analysis.get('main_topic', 'N/A') if result.topic_analysis else 'N/A',
                'Quality_Score': result.quality_scores.get('quality_metrics', {}).get('overall_score', 0) if result.quality_scores else 0,
                'DeepEval_Score': result.deepeval_scores.get('overall', {}).get('score', 0) if result.deepeval_scores else 0,
                'Sentiment': result.sentiment_analysis.get('sentiment_label', 'N/A') if result.sentiment_analysis else 'N/A',
                'AHT_Impact': result.aht_impact.get('aht_impact_level', 'N/A') if result.aht_impact else 'N/A'
            }
            display_data.append(row)
        
        df = pd.DataFrame(display_data)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            csr_filter = st.multiselect("Filter by CSR", options=df['CSR_ID'].unique())
        with col2:
            topic_filter = st.multiselect("Filter by Topic", options=df['Topic'].unique())
        with col3:
            sentiment_filter = st.multiselect("Filter by Sentiment", options=df['Sentiment'].unique())
        
        # Apply filters
        filtered_df = df.copy()
        if csr_filter:
            filtered_df = filtered_df[filtered_df['CSR_ID'].isin(csr_filter)]
        if topic_filter:
            filtered_df = filtered_df[filtered_df['Topic'].isin(topic_filter)]
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['Sentiment'].isin(sentiment_filter)]
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Individual result details
        if st.checkbox("Show Individual Coaching Feedback"):
            selected_csr = st.selectbox("Select CSR for detailed feedback", options=df['CSR_ID'].unique())
            
            csr_results = [r for r in results if r.csr_id == selected_csr]
            if csr_results:
                for i, result in enumerate(csr_results):
                    with st.expander(f"Conversation {i+1} - {result.topic_analysis.get('main_topic', 'General') if result.topic_analysis else 'General'}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Customer Question:**")
                            st.write(result.question)
                            st.write("**CSR Response:**")
                            st.write(result.answer)
                        
                        with col2:
                            st.write("**Sentiment Analysis:**")
                            if result.sentiment_analysis:
                                sentiment = result.sentiment_analysis
                                st.write(f"- Sentiment: {sentiment.get('sentiment_label', 'N/A')}")
                                st.write(f"- Emotional Tone: {sentiment.get('emotional_tone', 'N/A')}")
                                st.write(f"- Coaching: {sentiment.get('coaching_feedback', 'N/A')}")
                            
                            st.write("**Topic Summary:**")
                            if result.topic_analysis:
                                topic = result.topic_analysis
                                st.write(f"- Main Topic: {topic.get('main_topic', 'N/A')}")
                                st.write(f"- Category: {topic.get('concern_category', 'N/A')}")
                        
                        st.write("**Concise Summary:**")
                        st.info(result.concise_summary)
    
    @staticmethod
    def render_visualizations(results: List):
        """Render data visualizations."""
        st.subheader("📊 Performance Visualizations")
        
        # Prepare data
        df_viz = pd.DataFrame([
            {
                'CSR_ID': r.csr_id,
                'Quality_Score': r.quality_scores.get('quality_metrics', {}).get('overall_score', 0) if r.quality_scores else 0,
                'DeepEval_Score': r.deepeval_scores.get('overall', {}).get('score', 0) if r.deepeval_scores else 0,
                'Sentiment': r.sentiment_analysis.get('sentiment_label', 'Unknown') if r.sentiment_analysis else 'Unknown',
                'Topic': r.topic_analysis.get('main_topic', 'Unknown') if r.topic_analysis else 'Unknown',
                'AHT_Impact': r.aht_impact.get('aht_impact_level', 'Unknown') if r.aht_impact else 'Unknown'
            }
            for r in results
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality Score Distribution
            fig1 = px.histogram(
                df_viz, 
                x='Quality_Score', 
                title='Quality Score Distribution',
                nbins=10,
                color_discrete_sequence=['#1f77b4']
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Sentiment Distribution
            sentiment_counts = df_viz['Sentiment'].value_counts()
            fig3 = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution'
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # CSR Performance Comparison
            csr_performance = df_viz.groupby('CSR_ID').agg({
                'Quality_Score': 'mean',
                'DeepEval_Score': 'mean'
            }).reset_index()
            
            fig2 = px.scatter(
                csr_performance,
                x='Quality_Score',
                y='DeepEval_Score',
                text='CSR_ID',
                title='CSR Performance Comparison',
                labels={'Quality_Score': 'Quality Score (0-10)', 'DeepEval_Score': 'DeepEval Score (0-1)'}
            )
            fig2.update_traces(textposition="top center")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Topic Distribution
            topic_counts = df_viz['Topic'].value_counts()
            fig4 = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                title='Topic Distribution',
                labels={'x': 'Topic', 'y': 'Count'}
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)

def main():
    """Main Streamlit application."""
    # Initialize session state
    DashboardState.initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Call Center Performance Coaching Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    config = ConfigurationManager.render_configuration_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Evaluation", "📊 Results", "ℹ️ About"])
    
    with tab1:
        # File upload
        file_paths = FileUploadManager.render_file_upload()
        
        st.markdown("---")
        
        # Evaluation controls
        EvaluationRunner.render_evaluation_controls(config, file_paths)
        
        # Show last evaluation info
        if st.session_state.last_evaluation_time:
            st.info(f"🕒 Last evaluation completed: {st.session_state.last_evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with tab2:
        ResultsViewer.render_results_dashboard()
    
    with tab3:
        st.header("ℹ️ About This Dashboard")
        st.markdown("""
        This dashboard provides comprehensive evaluation of call center transcripts using the Mistral 7B language model.
        
        ### Features:
        - **📁 File Upload**: Upload JSON transcript files for analysis
        - **🤖 AI Evaluation**: Uses local Mistral 7B model for CPU-optimized inference
        - **📊 Quality Analysis**: Analyzes sentence structure, repetition, hold requests, and transfers
        - **🎯 DeepEval Metrics**: Measures answer relevancy and correctness
        - **💭 Sentiment Analysis**: Identifies emotional tone and provides coaching feedback
        - **📈 RAG Pipeline**: Creates knowledge base and generates expert-level answers
        - **📋 Comprehensive Reporting**: Detailed feedback and 200-word summaries
        
        ### How to Use:
        1. Configure settings in the sidebar
        2. Upload your JSON transcript files
        3. Click "Run Mistral Evaluation"
        4. View results in the Results tab
        
        ### Requirements:
        - Mistral 7B model file (mistral-7b-instruct-v0.2.Q4_K_M.gguf)
        - JSON transcript files in the specified format
        - Sufficient CPU resources for model inference
        """)
        
        # System status
        st.subheader("🔧 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Path:**", config.mistral_model_path)
            st.write("**Model Exists:**", "✅" if os.path.exists(config.mistral_model_path) else "❌")
        
        with col2:
            st.write("**RAG Enabled:**", "✅" if config.enable_rag else "❌")
            st.write("**DeepEval Enabled:**", "✅" if config.enable_deepeval else "❌")

if __name__ == "__main__":
    main()

