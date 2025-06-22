import streamlit as st
import subprocess
import sys
import os
import json
from datetime import datetime

def run_mistral_evaluation():
    """Execute the mistral_transcript_evaluator.py script and return the result."""
    try:
        # Check if the script exists
        if not os.path.exists('mistral_transcript_evaluator.py'):
            return False, "", "mistral_transcript_evaluator.py not found in current directory"
        
        # Run the mistral_transcript_evaluator.py script
        result = subprocess.run([
            sys.executable, 'mistral_transcript_evaluator.py'
        ], capture_output=True, text=True, timeout=300, cwd=os.getcwd())  # 5 minute timeout
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "", "Evaluation timed out after 5 minutes"
    except FileNotFoundError as e:
        return False, "", f"Python interpreter or script not found: {str(e)}"
    except Exception as e:
        return False, "", f"Error running evaluation: {str(e)}"

def main():
    st.set_page_config(
        page_title="Performance Coaching Feedback Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Performance Coaching Feedback Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Evaluation Results", "Settings"])
    
    if page == "Dashboard":
        st.header("Call Transcript Analysis")
        
        # File upload section
        st.subheader("📁 Upload Transcript")
        uploaded_file = st.file_uploader(
            "Choose a JSON transcript file", 
            type=['json'],
            help="Upload a call transcript in JSON format"
=======
"""
Streamlit Frontend for Call Center Performance Coaching Dashboard
Integrates with Mistral 7B evaluation system to provide feedback by CSR_ID
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import our evaluation modules
try:
    from extract_call_data_dataframe import extract_call_transcript_to_dataframe
    from mistral_transcript_evaluator import MistralTranscriptEvaluator
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all required modules are available in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Call Center Performance Dashboard",
    page_icon="📞",
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
    }
    .feedback-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #17a2b8; font-weight: bold; }
    .score-needs-improvement { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class FeedbackDashboard:
    """
    Main dashboard class for call center performance feedback
    """
    
    def __init__(self):
        self.data_loaded = False
        self.evaluation_results = None
        self.transcript_data = None
        self.available_csrs = []
        
    def load_data(self):
        """Load transcript and evaluation data"""
        try:
            # Load transcript data
            if os.path.exists("Call Transcript Sample 1.json"):
                self.transcript_data = extract_call_transcript_to_dataframe("Call Transcript Sample 1.json")
                if self.transcript_data is not None:
                    self.available_csrs = self.transcript_data['CSR_ID'].unique().tolist()
            
            # Load evaluation results if available
            if os.path.exists("transcript_evaluation_results.json"):
                with open("transcript_evaluation_results.json", 'r') as f:
                    self.evaluation_results = json.load(f)
                self.data_loaded = True
            elif os.path.exists("transcript_evaluation_results.csv"):
                self.evaluation_results = pd.read_csv("transcript_evaluation_results.csv")
                self.data_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def get_score_class(self, score: float) -> str:
        """Get CSS class based on score"""
        if score >= 8.5:
            return "score-excellent"
        elif score >= 7.0:
            return "score-good"
        elif score >= 5.0:
            return "score-needs-improvement"
        else:
            return "score-poor"
    
    def get_score_label(self, score: float) -> str:
        """Get descriptive label for score"""
        if score >= 8.5:
            return "Excellent"
        elif score >= 7.0:
            return "Good"
        elif score >= 5.0:
            return "Needs Improvement"
        else:
            return "Needs Attention"
    
    def display_feedback_card(self, interaction_data: Dict, show_details: bool = True):
        """Display a feedback card for an interaction"""
        
        with st.container():
            st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
            
            # Header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"🗣️ Interaction {interaction_data.get('interaction_id', 'N/A')}")
            with col2:
                if 'timestamp' in interaction_data:
                    st.caption(f"📅 {interaction_data['timestamp'][:10]}")
            with col3:
                status = interaction_data.get('evaluation_status', 'unknown')
                if status == 'completed':
                    st.success("✅ Evaluated")
                else:
                    st.warning("⚠️ Pending")
            
            # Question and Answer
            if show_details:
                if interaction_data.get('question'):
                    st.markdown("**🙋 Customer Question:**")
                    st.info(interaction_data['question'])
                
                if interaction_data.get('answer'):
                    st.markdown("**👩‍💼 CSR Response:**")
                    st.info(interaction_data['answer'])
            
            # Evaluation Scores
            if interaction_data.get('evaluation_status') == 'completed':
                st.markdown("### 📊 Performance Scores")
                
                score_cols = st.columns(4)
                
                # English Score
                if 'english_evaluation' in interaction_data and interaction_data['english_evaluation']:
                    english_score = interaction_data['english_evaluation'].get('english_score', 0)
                    with score_cols[0]:
                        score_class = self.get_score_class(english_score)
                        st.markdown(f'<div class="metric-card"><h4>📝 English</h4><p class="{score_class}">{english_score:.1f}/10</p><small>{self.get_score_label(english_score)}</small></div>', unsafe_allow_html=True)
                
                # Clarity Score
                if 'sentence_analysis' in interaction_data and interaction_data['sentence_analysis']:
                    clarity_score = interaction_data['sentence_analysis'].get('clarity_score', 0)
                    with score_cols[1]:
                        score_class = self.get_score_class(clarity_score)
                        st.markdown(f'<div class="metric-card"><h4>🎯 Clarity</h4><p class="{score_class}">{clarity_score:.1f}/10</p><small>{self.get_score_label(clarity_score)}</small></div>', unsafe_allow_html=True)
                
                # Speech Score
                if 'repetition_analysis' in interaction_data and interaction_data['repetition_analysis']:
                    speech_score = interaction_data['repetition_analysis'].get('speech_score', 0)
                    with score_cols[2]:
                        score_class = self.get_score_class(speech_score)
                        st.markdown(f'<div class="metric-card"><h4>🗣️ Speech</h4><p class="{score_class}">{speech_score:.1f}/10</p><small>{self.get_score_label(speech_score)}</small></div>', unsafe_allow_html=True)
                
                # Sentiment
                if 'sentiment_analysis' in interaction_data and interaction_data['sentiment_analysis']:
                    sentiment = interaction_data['sentiment_analysis'].get('sentiment_category', 'neutral')
                    with score_cols[3]:
                        sentiment_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(sentiment, "😐")
                        st.markdown(f'<div class="metric-card"><h4>😊 Sentiment</h4><p>{sentiment_emoji}</p><small>{sentiment.title()}</small></div>', unsafe_allow_html=True)
                
                # RAGAS Scores
                if 'ragas_scores' in interaction_data and interaction_data['ragas_scores']:
                    st.markdown("### 📈 RAGAS Quality Metrics")
                    ragas_cols = st.columns(5)
                    ragas_scores = interaction_data['ragas_scores']
                    
                    metrics = [
                        ('Context Precision', 'context_precision'),
                        ('Context Recall', 'context_recall'),
                        ('Entity Recall', 'context_entity_recall'),
                        ('Relevancy', 'answer_relevancy'),
                        ('Faithfulness', 'faithfulness')
                    ]
                    
                    for i, (label, key) in enumerate(metrics):
                        if key in ragas_scores:
                            score = ragas_scores[key] * 10  # Convert to 0-10 scale
                            with ragas_cols[i]:
                                score_class = self.get_score_class(score)
                                st.markdown(f'<div class="metric-card"><h5>{label}</h5><p class="{score_class}">{score:.1f}/10</p></div>', unsafe_allow_html=True)
                
                # Coaching Feedback
                st.markdown("### 💬 Coaching Feedback")
                
                feedback_tabs = st.tabs(["🎯 Overall", "📝 English", "🗣️ Communication", "😊 Sentiment", "📚 Knowledge"])
                
                with feedback_tabs[0]:  # Overall
                    if 'ragas_coaching' in interaction_data and interaction_data['ragas_coaching']:
                        st.markdown("**📊 Performance Coaching:**")
                        st.success(interaction_data['ragas_coaching'])
                
                with feedback_tabs[1]:  # English
                    if 'english_evaluation' in interaction_data and interaction_data['english_evaluation']:
                        feedback = interaction_data['english_evaluation'].get('feedback', '')
                        if feedback:
                            st.markdown("**📝 English Language Coaching:**")
                            st.info(feedback)
                
                with feedback_tabs[2]:  # Communication
                    if 'sentence_analysis' in interaction_data and interaction_data['sentence_analysis']:
                        feedback = interaction_data['sentence_analysis'].get('feedback', '')
                        if feedback:
                            st.markdown("**🎯 Communication Clarity:**")
                            st.info(feedback)
                    
                    if 'repetition_analysis' in interaction_data and interaction_data['repetition_analysis']:
                        feedback = interaction_data['repetition_analysis'].get('feedback', '')
                        if feedback:
                            st.markdown("**🗣️ Speech Quality:**")
                            st.info(feedback)
                
                with feedback_tabs[3]:  # Sentiment
                    if 'sentiment_analysis' in interaction_data and interaction_data['sentiment_analysis']:
                        feedback = interaction_data['sentiment_analysis'].get('feedback', '')
                        if feedback:
                            st.markdown("**😊 Emotional Intelligence:**")
                            st.info(feedback)
                
                with feedback_tabs[4]:  # Knowledge
                    if 'question_topic' in interaction_data and interaction_data['question_topic']:
                        st.markdown("**📚 Topic Analysis:**")
                        st.info(interaction_data['question_topic'])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def create_performance_charts(self, csr_data: List[Dict]):
        """Create performance visualization charts"""
        
        if not csr_data:
            return
        
        # Extract scores for visualization
        interactions = []
        for interaction in csr_data:
            if interaction.get('evaluation_status') == 'completed':
                scores = {
                    'interaction_id': interaction.get('interaction_id', 0),
                    'english_score': 0,
                    'clarity_score': 0,
                    'speech_score': 0,
                    'context_precision': 0,
                    'faithfulness': 0,
                    'answer_relevancy': 0
                }
                
                if 'english_evaluation' in interaction and interaction['english_evaluation']:
                    scores['english_score'] = interaction['english_evaluation'].get('english_score', 0)
                
                if 'sentence_analysis' in interaction and interaction['sentence_analysis']:
                    scores['clarity_score'] = interaction['sentence_analysis'].get('clarity_score', 0)
                
                if 'repetition_analysis' in interaction and interaction['repetition_analysis']:
                    scores['speech_score'] = interaction['repetition_analysis'].get('speech_score', 0)
                
                if 'ragas_scores' in interaction and interaction['ragas_scores']:
                    ragas = interaction['ragas_scores']
                    scores['context_precision'] = ragas.get('context_precision', 0) * 10
                    scores['faithfulness'] = ragas.get('faithfulness', 0) * 10
                    scores['answer_relevancy'] = ragas.get('answer_relevancy', 0) * 10
                
                interactions.append(scores)
        
        if not interactions:
            st.warning("No evaluation data available for visualization.")
            return
        
        df_scores = pd.DataFrame(interactions)
        
        # Performance Trend Chart
        st.subheader("📈 Performance Trends")
        
        fig_trend = go.Figure()
        
        metrics = ['english_score', 'clarity_score', 'speech_score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        names = ['English Proficiency', 'Communication Clarity', 'Speech Quality']
        
        for metric, color, name in zip(metrics, colors, names):
            if metric in df_scores.columns:
                fig_trend.add_trace(go.Scatter(
                    x=df_scores['interaction_id'],
                    y=df_scores[metric],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ))
        
        fig_trend.update_layout(
            title="Performance Scores Across Interactions",
            xaxis_title="Interaction ID",
            yaxis_title="Score (0-10)",
            yaxis=dict(range=[0, 10]),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # RAGAS Metrics Radar Chart
        if any(col in df_scores.columns for col in ['context_precision', 'faithfulness', 'answer_relevancy']):
            st.subheader("🎯 RAGAS Quality Assessment")
            
            # Calculate average RAGAS scores
            ragas_metrics = ['context_precision', 'faithfulness', 'answer_relevancy']
            ragas_labels = ['Context Precision', 'Faithfulness', 'Answer Relevancy']
            ragas_values = []
            
            for metric in ragas_metrics:
                if metric in df_scores.columns:
                    avg_score = df_scores[metric].mean()
                    ragas_values.append(avg_score)
                else:
                    ragas_values.append(0)
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=ragas_values,
                theta=ragas_labels,
                fill='toself',
                name='Average Performance',
                line_color='rgb(31, 119, 180)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=True,
                title="Average RAGAS Quality Metrics",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Score Distribution
        st.subheader("📊 Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall performance histogram
            all_scores = []
            for _, row in df_scores.iterrows():
                for metric in ['english_score', 'clarity_score', 'speech_score']:
                    if metric in df_scores.columns and row[metric] > 0:
                        all_scores.append(row[metric])
            
            if all_scores:
                fig_hist = px.histogram(
                    x=all_scores,
                    nbins=20,
                    title="Overall Score Distribution",
                    labels={'x': 'Score', 'y': 'Frequency'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Average scores by category
            avg_scores = {
                'English': df_scores['english_score'].mean() if 'english_score' in df_scores.columns else 0,
                'Clarity': df_scores['clarity_score'].mean() if 'clarity_score' in df_scores.columns else 0,
                'Speech': df_scores['speech_score'].mean() if 'speech_score' in df_scores.columns else 0
            }
            
            fig_bar = px.bar(
                x=list(avg_scores.keys()),
                y=list(avg_scores.values()),
                title="Average Scores by Category",
                labels={'x': 'Category', 'y': 'Average Score'},
                color=list(avg_scores.values()),
                color_continuous_scale='RdYlGn'
            )
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📞 Call Center Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = FeedbackDashboard()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Data loading section
        st.subheader("📂 Data Management")
        
        if st.button("🔄 Load/Refresh Data", type="primary"):
            with st.spinner("Loading data..."):
                if dashboard.load_data():
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load data")
        
        # File upload section
        st.subheader("📤 Upload Transcript")
        uploaded_file = st.file_uploader(
            "Upload JSON transcript file",
            type=['json'],
            help="Upload a call transcript JSON file for analysis"

        )
        
        if uploaded_file is not None:
            try:

                # Load and display transcript info
                transcript_data = json.load(uploaded_file)
                st.success(f"✅ Transcript loaded successfully!")
                
                # Display basic info about the transcript
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("File Name", uploaded_file.name)
                with col2:
                    st.metric("File Size", f"{uploaded_file.size} bytes")
                
                # Show transcript preview
                with st.expander("📄 Transcript Preview"):
                    st.json(transcript_data)
                    
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid transcript file.")
        
        # Evaluation section
        st.subheader("🤖 AI Evaluation")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Run Mistral Evaluation button
            if st.button("🚀 Run Mistral Evaluation", type="primary", use_container_width=True):
                # Show progress message
                progress_placeholder = st.empty()
                progress_placeholder.info("🔄 Starting Mistral evaluation...")
                
                # Show spinner and run evaluation
                with st.spinner('Running Mistral Evaluation... Please wait...'):
                    success, stdout, stderr = run_mistral_evaluation()
                
                # Clear progress message
                progress_placeholder.empty()
                
                if success:
                    st.success("✅ Mistral evaluation completed successfully!")
                    if stdout:
                        st.subheader("📋 Evaluation Output:")
                        st.text_area("Output", stdout, height=200, key="success_output")
                else:
                    st.error("❌ Mistral evaluation failed!")
                    if stderr:
                        st.subheader("🚨 Error Details:")
                        st.text_area("Error", stderr, height=100, key="error_details")
                    if stdout:
                        st.subheader("📋 Output (if any):")
                        st.text_area("Output", stdout, height=100, key="partial_output")
        
        with col2:
            st.info("""
            **About Mistral Evaluation:**
            
            This will automatically run the Mistral AI model to evaluate the uploaded transcript.
            The evaluation includes:
            - Performance analysis
            - Feedback generation
            - Coaching recommendations
            
            No need to run commands manually - just click the button!
            """)
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write("**Current Working Directory:**", os.getcwd())
                st.write("**Python Executable:**", sys.executable)
                script_exists = os.path.exists('mistral_transcript_evaluator.py')
                st.write("**Script Exists:**", "✅ Yes" if script_exists else "❌ No")
                if script_exists:
                    st.write("**Script Path:**", os.path.abspath('mistral_transcript_evaluator.py'))
    
    elif page == "Evaluation Results":
        st.header("📈 Evaluation Results")
        st.info("Evaluation results will be displayed here after running the Mistral evaluation.")
        
        # Check if there are any result files
        result_files = [f for f in os.listdir('.') if f.startswith('evaluation_result') and f.endswith('.json')]
        
        if result_files:
            st.subheader("Recent Evaluations")
            for file in sorted(result_files, reverse=True):
                with st.expander(f"📄 {file}"):
                    try:
                        with open(file, 'r') as f:
                            result_data = json.load(f)
                        st.json(result_data)
                    except Exception as e:
                        st.error(f"Error loading {file}: {str(e)}")
        else:
            st.warning("No evaluation results found. Run an evaluation first.")
    
    elif page == "Settings":
        st.header("⚙️ Settings")
        
        st.subheader("Evaluation Settings")
        timeout_minutes = st.slider("Evaluation Timeout (minutes)", 1, 30, 5)
        
        st.subheader("File Paths")
        evaluator_path = st.text_input("Mistral Evaluator Script Path", "mistral_transcript_evaluator.py")
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Performance Coaching Dashboard | Built with Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

                # Save uploaded file
                with open("uploaded_transcript.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File uploaded successfully!")
                
                # Process uploaded file
                if st.button("🔍 Analyze Uploaded File"):
                    with st.spinner("Processing transcript..."):
                        # This would trigger the Mistral evaluation
                        st.info("Analysis feature requires Mistral model setup. Please run the evaluation script separately.")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
        
        # Evaluation controls
        st.subheader("🤖 AI Evaluation")
        
        if st.button("🚀 Run Mistral Evaluation"):
            if os.path.exists("Model\mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
                st.info("Starting Mistral evaluation... This may take several minutes.")
                st.code("python mistral_transcript_evaluator.py")
                st.info("Please run the above command in your terminal to start evaluation.")
            else:
                st.warning("Mistral model not found. Please download the model first.")
                st.code("python setup_mistral_evaluator.py")
    
    # Main content area
    if not dashboard.load_data():
        st.warning("⚠️ No evaluation data found. Please ensure you have:")
        st.markdown("""
        1. **Transcript data**: `Call Transcript Sample 1.json`
        2. **Evaluation results**: Run `python mistral_transcript_evaluator.py`
        3. **Required files**: `transcript_evaluation_results.json` or `.csv`
        """)
        
        # Show sample data structure
        with st.expander("📋 Expected Data Structure"):
            st.json({
                "interaction_id": 1,
                "question": "Customer question here...",
                "answer": "CSR response here...",
                "english_evaluation": {"english_score": 8.5, "feedback": "..."},
                "ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.85},
                "sentiment_analysis": {"sentiment_category": "positive"}
            })
        
        return
    
    # CSR Selection
    st.header("👤 Select CSR for Feedback")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if dashboard.available_csrs:
            selected_csr = st.selectbox(
                "Choose CSR ID:",
                options=["All CSRs"] + dashboard.available_csrs,
                index=0
            )
        else:
            st.warning("No CSR data available")
            return
    
    with col2:
        show_details = st.checkbox("Show Interaction Details", value=True)
    
    with col3:
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            st.rerun()
    
    # Filter data by CSR
    if selected_csr == "All CSRs":
        # Show summary for all CSRs
        st.header("📊 Overall Performance Summary")
        
        if isinstance(dashboard.evaluation_results, list):
            total_interactions = len(dashboard.evaluation_results)
            completed_evaluations = len([r for r in dashboard.evaluation_results if r.get('evaluation_status') == 'completed'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Interactions", total_interactions)
            with col2:
                st.metric("Evaluated", completed_evaluations)
            with col3:
                completion_rate = (completed_evaluations / total_interactions * 100) if total_interactions > 0 else 0
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            with col4:
                st.metric("Available CSRs", len(dashboard.available_csrs))
            
            # Show all interactions
            st.header("🗂️ All Interactions")
            for interaction in dashboard.evaluation_results:
                dashboard.display_feedback_card(interaction, show_details)
        
    else:
        # Show specific CSR feedback
        st.header(f"👤 Feedback for CSR: {selected_csr}")
        
        # Filter evaluation results for selected CSR
        csr_interactions = []
        if isinstance(dashboard.evaluation_results, list):
            # If evaluation results don't have CSR_ID, match with transcript data
            for interaction in dashboard.evaluation_results:
                interaction_id = interaction.get('interaction_id')
                if dashboard.transcript_data is not None:
                    matching_row = dashboard.transcript_data[dashboard.transcript_data['interaction_id'] == interaction_id]
                    if not matching_row.empty and matching_row.iloc[0]['CSR_ID'] == selected_csr:
                        csr_interactions.append(interaction)
        
        if not csr_interactions:
            st.warning(f"No evaluation data found for CSR: {selected_csr}")
            return
        
        # CSR Performance Summary
        st.subheader("📈 Performance Summary")
        
        completed_interactions = [i for i in csr_interactions if i.get('evaluation_status') == 'completed']
        
        if completed_interactions:
            # Calculate average scores
            english_scores = []
            clarity_scores = []
            speech_scores = []
            
            for interaction in completed_interactions:
                if 'english_evaluation' in interaction and interaction['english_evaluation']:
                    english_scores.append(interaction['english_evaluation'].get('english_score', 0))
                if 'sentence_analysis' in interaction and interaction['sentence_analysis']:
                    clarity_scores.append(interaction['sentence_analysis'].get('clarity_score', 0))
                if 'repetition_analysis' in interaction and interaction['repetition_analysis']:
                    speech_scores.append(interaction['repetition_analysis'].get('speech_score', 0))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_english = np.mean(english_scores) if english_scores else 0
                st.metric("📝 Avg English Score", f"{avg_english:.1f}/10")
            
            with col2:
                avg_clarity = np.mean(clarity_scores) if clarity_scores else 0
                st.metric("🎯 Avg Clarity Score", f"{avg_clarity:.1f}/10")
            
            with col3:
                avg_speech = np.mean(speech_scores) if speech_scores else 0
                st.metric("🗣️ Avg Speech Score", f"{avg_speech:.1f}/10")
            
            with col4:
                st.metric("📊 Interactions", len(completed_interactions))
            
            # Performance Charts
            dashboard.create_performance_charts(csr_interactions)
        
        # Individual Interaction Feedback
        st.subheader("💬 Individual Interaction Feedback")
        
        for interaction in csr_interactions:
            dashboard.display_feedback_card(interaction, show_details)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        📞 Call Center Performance Dashboard | Powered by Mistral 7B AI | 
        <a href='https://github.com/ritwikv/Performance_Coaching'>View Source</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
