#!/usr/bin/env python3
"""
Launch script for basic mode dashboard
Avoids heavy ML dependencies that cause loading issues
"""

import subprocess
import sys
import os

def check_basic_requirements():
    """Check if basic requirements are met"""
    try:
        import pandas
        import numpy
        import streamlit
        print("✅ Basic requirements met")
        return True
    except ImportError as e:
        print(f"❌ Missing basic requirement: {e}")
        print("Install with: pip install pandas numpy streamlit")
        return False

def main():
    print("🚀 Launching Call Center Performance Coaching - Basic Mode")
    print("=" * 60)
    
    if not check_basic_requirements():
        return False
    
    print("📊 Starting basic dashboard (no heavy ML dependencies)...")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        # Launch basic dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_basic_dashboard.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

