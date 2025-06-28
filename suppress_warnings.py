#!/usr/bin/env python3
"""
Warning Suppression Module for ML Dependencies
Suppresses common warnings from huggingface_hub, transformers, and torch
"""

import os
import warnings
import logging

def suppress_ml_warnings():
    """Suppress common ML library warnings for cleaner output"""
    
    # Suppress HuggingFace Hub warnings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # Suppress transformers warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Suppress torch warnings
    os.environ['TORCH_LOGS'] = 'error'
    
    # Filter specific warning categories
    warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
    warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.utils._pytree.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*resume_download.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*symlinks.*')
    
    # Set logging levels for noisy libraries
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)

def setup_clean_environment():
    """Setup a clean environment with minimal warnings"""
    print("🔧 Setting up clean ML environment...")
    
    # Apply warning suppressions
    suppress_ml_warnings()
    
    # Additional environment optimizations
    os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::UserWarning'
    
    print("✅ Warning suppression applied")
    print("✅ Environment optimized for clean output")

if __name__ == "__main__":
    setup_clean_environment()
    print("\n🧪 Testing warning suppression...")
    
    try:
        # Test sentence-transformers loading
        from sentence_transformers import SentenceTransformer
        print("📥 Loading sentence-transformers model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded with minimal warnings!")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✅ Test encoding successful: shape {embedding.shape}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

