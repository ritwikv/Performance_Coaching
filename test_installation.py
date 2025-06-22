#!/usr/bin/env python3
"""
Installation Test Script for Call Center Transcript Evaluator
=============================================================

This script tests if all dependencies are properly installed and
the system is ready to run the evaluator.

Run this script before using the main evaluator to ensure everything is set up correctly.
"""

import sys
import importlib
from pathlib import Path
import json

def test_python_version():
    """Test if Python version is compatible."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_required_packages():
    """Test if all required packages are installed."""
    print("\n📦 Testing required packages...")
    
    required_packages = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("textblob", "Sentiment analysis"),
        ("pathlib", "File path handling"),
        ("json", "JSON processing"),
        ("re", "Regular expressions"),
        ("logging", "Logging functionality"),
        ("datetime", "Date/time handling"),
    ]
    
    optional_packages = [
        ("llama_cpp", "Mistral model support"),
        ("ragas", "RAGAS evaluation framework"),
        ("spacy", "Advanced NLP"),
        ("transformers", "Transformer models"),
        ("torch", "PyTorch backend"),
    ]
    
    all_good = True
    
    # Test required packages
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package:<15} - {description}")
        except ImportError:
            print(f"   ❌ {package:<15} - {description} (REQUIRED)")
            all_good = False
    
    # Test optional packages
    print("\n📦 Testing optional packages...")
    for package, description in optional_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package:<15} - {description}")
        except ImportError:
            print(f"   ⚠️  {package:<15} - {description} (Optional)")
    
    return all_good

def test_sample_data():
    """Test if sample data file exists and is valid."""
    print("\n📄 Testing sample data...")
    
    sample_file = "Call Transcript Sample 1.json"
    
    if not Path(sample_file).exists():
        print(f"   ❌ Sample file not found: {sample_file}")
        return False
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['call_ID', 'CSR_ID', 'call_date', 'call_time', 'call_transcript']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"   ❌ Missing required fields: {missing_fields}")
            return False
        
        transcript_lines = len(data.get('call_transcript', []))
        print(f"   ✅ Sample file valid - {transcript_lines} transcript lines")
        return True
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

def test_model_file():
    """Test if Mistral model file exists."""
    print("\n🤖 Testing Mistral model file...")
    
    # Common model file names
    model_files = [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.q4_k_m.gguf",
        "mistral-7b-instruct.gguf",
    ]
    
    found_model = None
    for model_file in model_files:
        if Path(model_file).exists():
            found_model = model_file
            break
    
    if found_model:
        size_mb = Path(found_model).stat().st_size / (1024 * 1024)
        print(f"   ✅ Model file found: {found_model} ({size_mb:.1f} MB)")
        return True, found_model
    else:
        print("   ❌ Mistral model file not found")
        print("   📥 Please download: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        print("   🔗 From: Hugging Face or official Mistral repositories")
        return False, None

def test_basic_functionality():
    """Test basic functionality without the model."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test JSON loading
        sample_data = {
            "call_ID": "test123",
            "CSR_ID": "testCSR",
            "call_date": "2024-01-01",
            "call_time": "12:00:00",
            "call_transcript": [
                "CSR: Hello, how can I help you?",
                "Customer: I need help with my booking.",
                "CSR: I'd be happy to assist you with that."
            ]
        }
        
        # Test data extraction logic
        questions = []
        answers = []
        
        for line in sample_data['call_transcript']:
            line = line.strip()
            if line.startswith('Customer:'):
                questions.append(line.replace('Customer:', '').strip())
            elif line.startswith('CSR:'):
                answers.append(line.replace('CSR:', '').strip())
        
        if len(questions) > 0 and len(answers) > 0:
            print("   ✅ Data extraction logic working")
        else:
            print("   ❌ Data extraction logic failed")
            return False
        
        # Test text analysis functions
        import re
        from textblob import TextBlob
        
        sample_text = "This is a test sentence for analysis."
        
        # Test sentence splitting
        sentences = re.split(r'[.!?]+', sample_text)
        if len(sentences) > 0:
            print("   ✅ Sentence analysis working")
        else:
            print("   ❌ Sentence analysis failed")
            return False
        
        # Test sentiment analysis
        blob = TextBlob(sample_text)
        polarity = blob.sentiment.polarity
        if isinstance(polarity, (int, float)):
            print("   ✅ Sentiment analysis working")
        else:
            print("   ❌ Sentiment analysis failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def test_output_directory():
    """Test if output directory can be created."""
    print("\n📁 Testing output directory...")
    
    output_dir = Path("evaluation_results")
    
    try:
        output_dir.mkdir(exist_ok=True)
        
        # Test write permissions
        test_file = output_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()  # Delete test file
        
        print(f"   ✅ Output directory ready: {output_dir.absolute()}")
        return True
        
    except Exception as e:
        print(f"   ❌ Cannot create/write to output directory: {e}")
        return False

def main():
    """Run all installation tests."""
    print("🚀 Call Center Transcript Evaluator - Installation Test")
    print("=" * 70)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Packages", test_required_packages),
        ("Sample Data", test_sample_data),
        ("Basic Functionality", test_basic_functionality),
        ("Output Directory", test_output_directory),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Test model file separately (not required for basic functionality)
    model_available, model_path = test_model_file()
    results["Model File"] = model_available
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        if model_available:
            print(f"🤖 Model ready: {model_path}")
        print("\n🚀 You can now run: python example_usage.py")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        
        if not results.get("Required Packages", True):
            print("\n📦 To install required packages:")
            print("   pip install -r requirements.txt")
        
        if not results.get("Model File", True):
            print("\n🤖 To get the Mistral model:")
            print("   1. Download mistral-7b-instruct-v0.2.Q4_K_M.gguf")
            print("   2. Place it in the current directory")
            print("   3. Update MODEL_PATH in config.py if needed")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

