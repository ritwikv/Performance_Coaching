"""
Performance Coaching Feedback Dashboard

This Streamlit dashboard provides an interface for running transcript evaluations
and displaying results. 

NOTE: RAGAS evaluation display has been commented out as requested.
The dashboard now shows simulated performance metrics instead of RAGAS metrics.
To re-enable RAGAS display, uncomment the relevant sections in the "Evaluation Results" page.
"""

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
        if not os.path.exists('mistral_transcript_evaluator_simulated_Performance_metrics.py'):
            return False, "", "mistral_transcript_evaluator_simulated_Performance_metrics.py not found in current directory"
        
        # Run the mistral_transcript_evaluator_simulated_Performance_metrics.py script
        result = subprocess.run([
            sys.executable, 'mistral_transcript_evaluator_simulated_Performance_metrics.py'
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
            
            This will automatically run the transcript evaluator to analyze the uploaded transcript.
            The evaluation includes:
            - Performance analysis (simulated metrics)
            - Feedback generation
            - Coaching recommendations
            
            **Note:** RAGAS evaluation has been disabled as requested.
            The script now uses simulated performance metrics instead.
            
            No need to run commands manually - just click the button!
            """)
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write("**Current Working Directory:**", os.getcwd())
                st.write("**Python Executable:**", sys.executable)
                script_exists = os.path.exists('mistral_transcript_evaluator_simulated_Performance_metrics.py')
                st.write("**Script Exists:**", "✅ Yes" if script_exists else "❌ No")
                if script_exists:
                    st.write("**Script Path:**", os.path.abspath('mistral_transcript_evaluator_simulated_Performance_metrics.py'))
    
    elif page == "Evaluation Results":
        st.header("📈 Evaluation Results")
        st.info("Evaluation results will be displayed here after running the transcript evaluation.")
        
        # RAGAS Evaluation Display - COMMENTED OUT
        # =======================================
        # The following section would display RAGAS evaluation metrics
        # This has been commented out as RAGAS evaluation is disabled
        #
        # if 'ragas_metrics' in result_data:
        #     st.subheader("📊 RAGAS Evaluation Metrics")
        #     ragas_data = result_data['ragas_metrics']
        #     
        #     col1, col2, col3, col4 = st.columns(4)
        #     with col1:
        #         st.metric("Answer Relevancy", f"{ragas_data.get('answer_relevancy', 0):.3f}")
        #     with col2:
        #         st.metric("Faithfulness", f"{ragas_data.get('faithfulness', 0):.3f}")
        #     with col3:
        #         st.metric("Context Recall", f"{ragas_data.get('context_recall', 0):.3f}")
        #     with col4:
        #         st.metric("Context Precision", f"{ragas_data.get('context_precision', 0):.3f}")
        # =======================================
        
        # Check if there are any result files
        result_files = [f for f in os.listdir('.') if f.startswith('evaluation_result') and f.endswith('.json')]
        
        if result_files:
            st.subheader("Recent Evaluations")
            for file in sorted(result_files, reverse=True):
                with st.expander(f"📄 {file}"):
                    try:
                        with open(file, 'r') as f:
                            result_data = json.load(f)
                        
                        # Display performance metrics in a user-friendly way
                        if 'performance_metrics' in result_data:
                            st.subheader("📊 Performance Metrics")
                            metrics = result_data['performance_metrics']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Overall Rating", f"{metrics.get('overall_rating', 0)}/100")
                            with col2:
                                st.metric("Communication Clarity", f"{metrics.get('communication_clarity', 0):.1f}/100")
                            with col3:
                                st.metric("Engagement Level", f"{metrics.get('engagement_level', 0):.1f}/100")
                            with col4:
                                st.metric("Professionalism", f"{metrics.get('professionalism_score', 0):.1f}/100")
                        
                        # Display coaching recommendations
                        if 'coaching_recommendations' in result_data:
                            st.subheader("🎯 Coaching Recommendations")
                            for i, rec in enumerate(result_data['coaching_recommendations'], 1):
                                st.write(f"{i}. {rec}")
                        
                        # Display summary
                        if 'summary' in result_data:
                            st.subheader("📋 Summary")
                            st.text(result_data['summary'])
                        
                        # Show raw JSON data in a collapsible section
                        with st.expander("🔍 Raw Data"):
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
        evaluator_path = st.text_input("Mistral Evaluator Script Path", "mistral_transcript_evaluator_simulated_Performance_metrics.py")
        
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