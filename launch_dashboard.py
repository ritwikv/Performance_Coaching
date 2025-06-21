"""
Dashboard Launcher Script
Easy way to start the Streamlit dashboard with proper configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("pip install -r requirements_streamlit.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "Call Transcript Sample 1.json",
        "extract_call_data_dataframe.py",
        "streamlit_feedback_dashboard.py"
    ]
    
    optional_files = [
        "transcript_evaluation_results.json",
        "transcript_evaluation_results.csv",
        "mistral_transcript_evaluator.py"
    ]
    
    missing_required = []
    missing_optional = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_required.append(file)
    
    for file in optional_files:
        if not os.path.exists(file):
            missing_optional.append(file)
    
    if missing_required:
        print("❌ Missing required files:")
        for file in missing_required:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    
    if missing_optional:
        print("⚠️  Optional files not found (dashboard will have limited functionality):")
        for file in missing_optional:
            print(f"   - {file}")
        print("\n💡 To enable full functionality:")
        print("   1. Run: python mistral_transcript_evaluator.py")
        print("   2. This will generate evaluation results for the dashboard")
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Call Center Performance Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⚠️  To stop the dashboard, press Ctrl+C in this terminal")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_feedback_dashboard.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("📞 Call Center Performance Dashboard Launcher")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    # Check data files
    print("\n🔍 Checking data files...")
    if not check_data_files():
        return
    
    print("\n" + "=" * 50)
    print("🎉 Ready to launch dashboard!")
    print("=" * 50)
    
    # Ask user if they want to proceed
    response = input("\n🚀 Launch dashboard now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        launch_dashboard()
    else:
        print("👋 Launch cancelled. Run this script again when ready!")
        print("\n💡 Manual launch command:")
        print("streamlit run streamlit_feedback_dashboard.py")

if __name__ == "__main__":
    main()

