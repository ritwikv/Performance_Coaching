#!/usr/bin/env python3
"""
Streamlit App Launcher for Call Center Transcript Evaluator
==========================================================

This script provides an easy way to launch the Streamlit application
with proper configuration and error handling.

Usage:
    python launch_streamlit.py
    
Or with custom options:
    python launch_streamlit.py --port 8502 --host 0.0.0.0
"""

import sys
import subprocess
import argparse
from pathlib import Path
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'textblob'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    return missing_packages

def check_files():
    """Check if required files exist."""
    required_files = [
        'streamlit_app.py',
        'call_center_transcript_evaluator.py',
        'Call Transcript Sample 1.json'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    return missing_files

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='Launch Call Center Transcript Evaluator Streamlit App')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the app on (default: 8501)')
    parser.add_argument('--host', type=str, default='localhost', help='Host to run the app on (default: localhost)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Call Center Transcript Evaluator - Streamlit Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\n📥 To install missing packages:")
        print("   pip install -r requirements_streamlit.txt")
        return 1
    else:
        print("✅ All dependencies found")
    
    # Check required files
    print("\n📄 Checking required files...")
    missing_files = check_files()
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("\n📋 Required files:")
        print("   - streamlit_app.py (main application)")
        print("   - call_center_transcript_evaluator.py (evaluator engine)")
        print("   - Call Transcript Sample 1.json (sample data)")
        return 1
    else:
        print("✅ All required files found")
    
    # Check for Mistral model
    print("\n🤖 Checking for Mistral model...")
    model_files = [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.q4_k_m.gguf",
        "mistral-7b-instruct.gguf"
    ]
    
    model_found = False
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Model found: {model_file}")
            model_found = True
            break
    
    if not model_found:
        print("⚠️  Mistral model not found")
        print("   You'll need to load the model through the app interface")
        print("   Download: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    
    # Launch Streamlit
    print(f"\n🌐 Launching Streamlit app on {args.host}:{args.port}")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    try:
        # Launch the Streamlit app
        print("🔄 Starting Streamlit server...")
        print(f"📱 Open your browser to: http://{args.host}:{args.port}")
        print("\n💡 Tips:")
        print("   - Load the Mistral model in the sidebar first")
        print("   - Upload the sample JSON file to test")
        print("   - Press Ctrl+C to stop the server")
        print("\n" + "=" * 60)
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit app stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start Streamlit: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

