"""
Setup script for Mistral Transcript Evaluator
Helps with installation and configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔄 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_mistral.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def download_textblob_corpora():
    """Download TextBlob corpora for sentiment analysis"""
    print("🔄 Downloading TextBlob corpora...")
    
    try:
        import textblob
        textblob.download_corpora()
        print("✅ TextBlob corpora downloaded!")
        return True
    except Exception as e:
        print(f"⚠️ TextBlob corpora download failed: {e}")
        return False

def check_model_file():
    """Check if Mistral model file exists"""
    model_files = [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.q4_k_m.gguf",
        "Mistral-7B-Instruct-v0.2.Q4_K_M.gguf"
    ]
    
    print("🔍 Checking for Mistral model file...")
    
    current_dir = Path(".")
    found_model = None
    
    for model_file in model_files:
        if (current_dir / model_file).exists():
            found_model = model_file
            break
    
    if found_model:
        print(f"✅ Found Mistral model: {found_model}")
        return found_model
    else:
        print("❌ Mistral model file not found!")
        print("\n📥 To download the Mistral model:")
        print("1. Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        print("2. Download: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        print("3. Place it in the current directory")
        print("\nOr use wget/curl:")
        print("wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        return None

def create_config_file(model_path: str):
    """Create configuration file"""
    config = {
        "model_path": model_path,
        "n_ctx": 4096,
        "n_threads": 4,
        "max_tokens": 512,
        "temperature": 0.3
    }
    
    import json
    with open("mistral_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: mistral_config.json")

def test_installation():
    """Test if everything is working"""
    print("🧪 Testing installation...")
    
    try:
        # Test llama-cpp-python
        from llama_cpp import Llama
        print("✅ llama-cpp-python working")
        
        # Test RAGAS
        from ragas import evaluate
        print("✅ RAGAS working")
        
        # Test TextBlob
        from textblob import TextBlob
        blob = TextBlob("This is a test.")
        sentiment = blob.sentiment
        print("✅ TextBlob working")
        
        # Test pandas
        import pandas as pd
        print("✅ Pandas working")
        
        print("🎉 All components working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Mistral Transcript Evaluator Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 2: Download TextBlob corpora
    download_textblob_corpora()
    
    # Step 3: Check for model file
    model_path = check_model_file()
    
    # Step 4: Create config file
    if model_path:
        create_config_file(model_path)
    
    # Step 5: Test installation
    if test_installation():
        print("\n" + "=" * 50)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        if model_path:
            print("✅ Ready to run: python mistral_transcript_evaluator.py")
        else:
            print("⚠️  Please download the Mistral model file to complete setup")
            
        print("\n📚 Next steps:")
        print("1. Ensure your transcript data is available")
        print("2. Run: python mistral_transcript_evaluator.py")
        print("3. Check the generated evaluation reports")
        
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and retry")

if __name__ == "__main__":
    main()

