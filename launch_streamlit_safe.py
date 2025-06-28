#!/usr/bin/env python3
"""
Safe Streamlit launcher with proper error handling for Mistral model issues
"""

import streamlit as st
import sys
import os
import multiprocessing
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_system_requirements():
    """Check if system meets requirements"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 6:
            issues.append(f"Low memory: {memory_gb:.1f}GB (8GB+ recommended)")
    except ImportError:
        pass
    
    # Check CPU cores
    cpu_count = multiprocessing.cpu_count()
    if cpu_count < 2:
        issues.append(f"Low CPU cores: {cpu_count} (4+ recommended)")
    
    return issues

def safe_mistral_config():
    """Create a safe Mistral configuration"""
    try:
        from mistral_model import MistralConfig
        
        # Safe thread count
        cpu_count = multiprocessing.cpu_count()
        safe_threads = max(1, min(cpu_count - 1, 8))  # Cap at 8 threads
        
        config = MistralConfig(
            model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_threads=safe_threads,
            n_ctx=2048,  # Reduced context for stability
            temperature=0.1,
            max_tokens=256,  # Reduced for faster inference
            verbose=False
        )
        
        return config
    except Exception as e:
        st.error(f"Error creating Mistral config: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Call Center Performance Coaching - Safe Mode",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Call Center Performance Coaching - Safe Mode")
    st.markdown("---")
    
    # System check
    st.header("🔍 System Check")
    issues = check_system_requirements()
    
    if issues:
        st.warning("⚠️ System Issues Detected:")
        for issue in issues:
            st.write(f"• {issue}")
    else:
        st.success("✅ System requirements met")
    
    # CPU info
    cpu_count = multiprocessing.cpu_count()
    safe_threads = max(1, min(cpu_count - 1, 8))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Cores", cpu_count)
    with col2:
        st.metric("Safe Threads", safe_threads)
    with col3:
        st.metric("Model Context", "2048 tokens")
    
    st.markdown("---")
    
    # Model check
    st.header("🤖 Model Configuration")
    
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        model_size = os.path.getsize(model_path) / (1024**3)
        st.success(f"✅ Model found: {model_path} ({model_size:.1f} GB)")
    else:
        st.error(f"❌ Model not found: {model_path}")
        st.info("""
        **Download the model:**
        ```bash
        pip install huggingface-hub
        huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
        ```
        """)
    
    # Safe configuration display
    st.subheader("🔧 Safe Configuration")
    config = safe_mistral_config()
    if config:
        st.json({
            "model_path": config.model_path,
            "n_threads": config.n_threads,
            "n_ctx": config.n_ctx,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        })
    
    st.markdown("---")
    
    # Launch options
    st.header("🚀 Launch Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🛡️ Launch Safe Mode Dashboard", type="primary"):
            if model_exists:
                st.info("Launching safe mode dashboard...")
                # Import and run the safe version
                try:
                    import subprocess
                    subprocess.Popen([
                        sys.executable, "-m", "streamlit", "run", 
                        "enhanced_streamlit_dashboard.py",
                        "--server.port", "8502"
                    ])
                    st.success("Dashboard launched on port 8502!")
                    st.info("Visit: http://localhost:8502")
                except Exception as e:
                    st.error(f"Launch failed: {e}")
            else:
                st.error("Please download the model first")
    
    with col2:
        if st.button("🧪 Test Model Loading"):
            if model_exists:
                with st.spinner("Testing model loading..."):
                    try:
                        from mistral_model import MistralEvaluator
                        
                        config = safe_mistral_config()
                        evaluator = MistralEvaluator(config)
                        
                        if evaluator.load_model():
                            st.success("✅ Model loaded successfully!")
                            
                            # Test inference
                            test_response = evaluator.generate_response(
                                "Test prompt: Hello", max_tokens=10
                            )
                            st.info(f"Test response: {test_response}")
                        else:
                            st.error("❌ Model loading failed")
                            
                    except Exception as e:
                        st.error(f"❌ Test failed: {e}")
                        st.info("Try the troubleshooting steps below")
            else:
                st.error("Please download the model first")
    
    # Troubleshooting
    st.markdown("---")
    st.header("🔧 Troubleshooting")
    
    with st.expander("Common Issues & Solutions"):
        st.markdown("""
        **1. GGML_ASSERT: n_threads > 0**
        - **Cause**: Thread count configuration issue
        - **Solution**: Use the safe configuration above (automatically fixed)
        
        **2. Out of Memory Error**
        - **Cause**: Insufficient RAM
        - **Solution**: 
          - Close other applications
          - Reduce n_ctx to 1024 or 512
          - Use smaller model if available
        
        **3. Model Loading Timeout**
        - **Cause**: Large model file, slow disk
        - **Solution**: 
          - Wait longer (first load takes time)
          - Use SSD if possible
          - Check available disk space
        
        **4. Import Errors**
        - **Cause**: Missing dependencies
        - **Solution**: 
          ```bash
          pip install -r requirements_complete.txt
          ```
        
        **5. Streamlit Crashes**
        - **Cause**: Various model/memory issues
        - **Solution**: 
          - Use this safe launcher
          - Restart with reduced settings
          - Check system resources
        """)
    
    # System info
    with st.expander("System Information"):
        st.write(f"**Python Version**: {sys.version}")
        st.write(f"**CPU Cores**: {multiprocessing.cpu_count()}")
        st.write(f"**Platform**: {sys.platform}")
        st.write(f"**Working Directory**: {os.getcwd()}")
        
        # Check dependencies
        dependencies = [
            "pandas", "numpy", "streamlit", "plotly", 
            "llama_cpp", "sentence_transformers", "chromadb"
        ]
        
        st.write("**Dependencies**:")
        for dep in dependencies:
            try:
                __import__(dep)
                st.write(f"✅ {dep}")
            except ImportError:
                st.write(f"❌ {dep} (missing)")

if __name__ == "__main__":
    main()

