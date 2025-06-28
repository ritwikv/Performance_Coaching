#!/usr/bin/env python3
"""
Clean Streamlit Launcher with Warning Suppression
Launches the enhanced Streamlit dashboard with minimal warnings
"""

import os
import sys
import warnings
import subprocess
from pathlib import Path

def setup_clean_environment():
    """Setup environment to minimize warnings from ML libraries"""
    print("🔧 Setting up clean environment...")
    
    # HuggingFace Hub settings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # Transformers settings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Torch settings
    os.environ['TORCH_LOGS'] = 'error'
    
    # Python warnings
    os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::UserWarning'
    
    # Filter warnings at the Python level
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    print("✅ Environment configured for minimal warnings")

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'enhanced_streamlit_dashboard.py',
        'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        
        if 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' in missing_files:
            print("\n📥 To download the Mistral model:")
            print("huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False")
        
        return False
    
    print("✅ All required files found")
    return True

def launch_streamlit():
    """Launch Streamlit with clean environment"""
    print("🚀 Launching Streamlit dashboard...")
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "enhanced_streamlit_dashboard.py"]
        
        # Add Streamlit configuration for cleaner output
        env = os.environ.copy()
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main launcher function"""
    print("🎯 Clean Streamlit Launcher for Call Center Evaluation")
    print("=" * 60)
    
    # Setup clean environment
    setup_clean_environment()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot launch - missing required files")
        return
    
    # Launch Streamlit
    launch_streamlit()

if __name__ == "__main__":
    main()

