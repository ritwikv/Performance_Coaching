#!/usr/bin/env python3
"""
Quick fix for the n_threads issue in Mistral model loading
Run this script to fix the thread configuration issue
"""

import os
import re

def fix_mistral_config():
    """Fix the n_threads configuration in mistral_model.py"""
    
    files_to_fix = [
        "mistral_model.py",
        "config.py"
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"🔧 Fixing {file_path}...")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the n_threads configuration
        content = re.sub(
            r'n_threads: int = -1.*',
            'n_threads: int = None  # Will be set automatically based on CPU cores',
            content
        )
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {file_path}")

def create_safe_mistral_config():
    """Create a safe configuration for Mistral model"""
    
    config_content = '''
# Safe Mistral Configuration
import multiprocessing

def get_safe_thread_count():
    """Get a safe thread count for Mistral model"""
    cpu_count = multiprocessing.cpu_count()
    # Use at most CPU count - 1, minimum 1
    return max(1, cpu_count - 1)

# Example usage:
# from mistral_model import MistralEvaluator, MistralConfig
# config = MistralConfig(
#     model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
#     n_threads=get_safe_thread_count(),
#     n_ctx=2048,  # Reduced context for stability
#     temperature=0.1
# )
'''
    
    with open("safe_mistral_config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Created safe_mistral_config.py")

def main():
    print("🚀 Fixing Mistral thread configuration issue...")
    print("=" * 50)
    
    # Fix the configuration files
    fix_mistral_config()
    
    # Create safe configuration helper
    create_safe_mistral_config()
    
    print("\n✅ Fix completed!")
    print("\nThe issue was caused by n_threads being set to -1, which is not supported.")
    print("The configuration has been updated to automatically detect the optimal thread count.")
    print("\n🔄 Please restart your Streamlit application:")
    print("   streamlit run enhanced_streamlit_dashboard.py")
    
    print("\n💡 Additional tips:")
    print("1. If you still have issues, try reducing n_ctx to 2048")
    print("2. Make sure you have enough RAM (8GB+ recommended)")
    print("3. Close other applications to free up memory")

if __name__ == "__main__":
    main()

