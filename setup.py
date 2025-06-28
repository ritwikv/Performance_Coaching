#!/usr/bin/env python3
"""
Setup script for Call Center Transcript Evaluation System
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    requirements_files = [
        "requirements_complete.txt",
        "requirements_mistral.txt",
        "requirements_streamlit.txt"
    ]
    
    # Use the most comprehensive requirements file available
    requirements_file = None
    for req_file in requirements_files:
        if os.path.exists(req_file):
            requirements_file = req_file
            break
    
    if not requirements_file:
        print("❌ No requirements file found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print(f"✅ Successfully installed packages from {requirements_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_model_instructions():
    """Provide instructions for downloading the Mistral model."""
    print("\n🤖 Mistral Model Setup")
    print("=" * 50)
    print("You need to download the Mistral 7B model file:")
    print("Model: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    print("\nDownload options:")
    print("1. From Hugging Face:")
    print("   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    print("2. Using huggingface-hub:")
    print("   pip install huggingface-hub")
    print("   huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False")
    print("\nPlace the model file in the project root directory.")
    print("The file should be approximately 4.1 GB in size.")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "results",
        "chroma_db", 
        "cache",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_default_config():
    """Create default configuration file."""
    print("⚙️ Creating default configuration...")
    
    from config import SystemConfig, ConfigManager
    
    config = SystemConfig()
    
    # Check if model file exists and update path
    model_files = [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.q4_k_m.gguf",
        "mistral-7b-instruct.gguf"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            config.model.model_path = model_file
            print(f"✅ Found model file: {model_file}")
            break
    else:
        print("⚠️ Model file not found - using default path")
    
    # Save configuration
    if ConfigManager.save_config(config):
        print("✅ Default configuration saved to system_config.json")
    else:
        print("❌ Failed to save configuration")

def create_sample_data():
    """Create sample data if it doesn't exist."""
    print("📄 Checking sample data...")
    
    if os.path.exists("Call Transcript Sample 1.json"):
        print("✅ Sample transcript file already exists")
        return
    
    # Create a simple sample if the original doesn't exist
    sample_data = {
        "call_ID": "SAMPLE_001",
        "CSR_ID": "SampleCSR",
        "call_date": "2024-01-01",
        "call_time": "10:00:00",
        "call_transcript": [
            "CSR: Thank you for calling Sample Company, this is Sarah. How may I assist you today?",
            "Customer: Hi, I need help with my recent order. It hasn't arrived yet.",
            "CSR: I apologize for the delay. May I have your order number to look this up for you?",
            "Customer: Sure, it's ORDER123456.",
            "CSR: Thank you. I can see your order was shipped yesterday and should arrive tomorrow. I'll send you a tracking number via email.",
            "Customer: That would be great, thank you for your help!",
            "CSR: You're welcome! Is there anything else I can help you with today?",
            "Customer: No, that's all. Thank you!",
            "CSR: Have a wonderful day!"
        ]
    }
    
    try:
        with open("Call Transcript Sample 1.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        print("✅ Created sample transcript file")
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")

def create_launch_scripts():
    """Create convenient launch scripts."""
    print("🚀 Creating launch scripts...")
    
    # Streamlit launch script
    streamlit_script = """#!/bin/bash
# Launch Streamlit Dashboard
echo "🚀 Starting Call Center Performance Coaching Dashboard..."
streamlit run enhanced_streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
"""
    
    # Python launch script
    python_script = """#!/usr/bin/env python3
# Launch evaluation from command line
from evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python launch_evaluation.py <transcript_file.json>")
        return
    
    config = EvaluationConfig()
    orchestrator = EvaluationOrchestrator(config)
    
    if orchestrator.initialize():
        results = orchestrator.evaluate_transcript_file(sys.argv[1])
        print(f"Evaluation completed: {len(results)} conversations processed")
    else:
        print("Failed to initialize evaluation system")

if __name__ == "__main__":
    main()
"""
    
    try:
        # Create bash script for Streamlit
        with open("launch_dashboard.sh", "w") as f:
            f.write(streamlit_script)
        os.chmod("launch_dashboard.sh", 0o755)
        
        # Create Python script for command line
        with open("launch_evaluation.py", "w") as f:
            f.write(python_script)
        
        print("✅ Created launch scripts:")
        print("   - launch_dashboard.sh (Streamlit)")
        print("   - launch_evaluation.py (Command line)")
        
    except Exception as e:
        print(f"❌ Failed to create launch scripts: {e}")

def run_tests():
    """Run basic tests to verify installation."""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        print("Testing imports...")
        import pandas
        import numpy
        import streamlit
        print("✅ Core packages imported successfully")
        
        # Test data processor
        print("Testing data processor...")
        from data_processor import CallTranscriptProcessor
        processor = CallTranscriptProcessor()
        print("✅ Data processor initialized")
        
        # Test configuration
        print("Testing configuration...")
        from config import SystemConfig, ConfigManager
        config = SystemConfig()
        issues = ConfigManager.validate_config(config)
        if not issues:
            print("✅ Configuration is valid")
        else:
            print(f"⚠️ Configuration issues: {len(issues)}")
        
        print("✅ Basic tests passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Download the Mistral model (see instructions above)")
    print("2. Place the model file in the project directory")
    print("3. Launch the dashboard:")
    print("   streamlit run enhanced_streamlit_dashboard.py")
    print("   OR")
    print("   ./launch_dashboard.sh")
    print("4. Upload your JSON transcript files")
    print("5. Click 'Run Mistral Evaluation'")
    print("\nFor command-line usage:")
    print("   python launch_evaluation.py your_transcript.json")
    print("\nFor help and documentation:")
    print("   Check the README.md file")
    print("   View the About tab in the dashboard")

def main():
    """Main setup function."""
    print("🔧 Call Center Transcript Evaluation System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("⚠️ Continuing with setup despite installation issues...")
    
    # Create directories
    create_directories()
    
    # Create configuration
    create_default_config()
    
    # Create sample data
    create_sample_data()
    
    # Create launch scripts
    create_launch_scripts()
    
    # Provide model download instructions
    download_model_instructions()
    
    # Run tests
    if run_tests():
        print_next_steps()
        return True
    else:
        print("❌ Setup completed with issues. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

