#!/usr/bin/env python3
"""
Integration test script for Call Center Transcript Evaluation System
Tests all components without requiring the Mistral model
"""

import os
import sys
import json
from datetime import datetime

def test_data_processor():
    """Test data processing functionality."""
    print("🧪 Testing Data Processor...")
    
    try:
        from data_processor import CallTranscriptProcessor
        
        processor = CallTranscriptProcessor()
        
        # Test with sample file
        if os.path.exists("Call Transcript Sample 1.json"):
            transcript_data = processor.load_json_transcript("Call Transcript Sample 1.json")
            if transcript_data:
                records = processor.process_single_transcript(transcript_data)
                print(f"✅ Processed {len(records)} conversation pairs")
                
                # Test DataFrame creation
                import pandas as pd
                df = pd.DataFrame(records)
                print(f"✅ Created DataFrame with shape {df.shape}")
                
                # Test analysis
                analysis = processor.analyze_conversation_patterns(df)
                print(f"✅ Generated conversation analysis: {len(analysis)} metrics")
                
                return True
            else:
                print("❌ Failed to load sample transcript")
                return False
        else:
            print("⚠️ Sample transcript file not found")
            return False
            
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("🧪 Testing Configuration...")
    
    try:
        from config import SystemConfig, ConfigManager
        
        # Test default config creation
        config = SystemConfig()
        print("✅ Created default configuration")
        
        # Test validation
        issues = ConfigManager.validate_config(config)
        if len(issues) <= 1:  # Allow model file not found
            print("✅ Configuration validation passed")
        else:
            print(f"⚠️ Configuration has {len(issues)} issues (expected if model not downloaded)")
        
        # Test save/load
        test_config_file = "test_config.json"
        if ConfigManager.save_config(config, test_config_file):
            print("✅ Configuration saved successfully")
            
            loaded_config = ConfigManager.load_config(test_config_file)
            print("✅ Configuration loaded successfully")
            
            # Cleanup
            if os.path.exists(test_config_file):
                os.remove(test_config_file)
            
            return True
        else:
            print("❌ Failed to save configuration")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_quality_analyzer():
    """Test quality analysis without Mistral model."""
    print("🧪 Testing Quality Analyzer...")
    
    try:
        from quality_analyzer import CallQualityAnalyzer
        
        analyzer = CallQualityAnalyzer()
        
        # Test with sample text
        sample_text = "I apologize for the trouble. May I have your name and reservation number to look up your booking?"
        
        analysis = analyzer.analyze_single_response(sample_text)
        
        if 'quality_metrics' in analysis:
            print("✅ Quality analysis completed")
            print(f"✅ Generated {len(analysis)} analysis components")
            return True
        else:
            print("❌ Quality analysis incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Quality analyzer test failed: {e}")
        return False

def test_sentiment_analyzer():
    """Test sentiment analysis without Mistral model."""
    print("🧪 Testing Sentiment Analyzer...")
    
    try:
        from sentiment_analyzer import SentimentTopicAnalyzer
        
        # Test without Mistral model (traditional methods only)
        analyzer = SentimentTopicAnalyzer(mistral_evaluator=None)
        
        question = "I need help with a reservation I made last week"
        answer = "I apologize for the trouble. May I have your name and reservation number to look up your booking?"
        
        result = analyzer.analyze_conversation(question, answer)
        
        if 'sentiment' in result and 'topic' in result:
            print("✅ Sentiment analysis completed")
            print(f"✅ Sentiment: {result['sentiment'].get('sentiment_label', 'Unknown')}")
            print(f"✅ Topic: {result['topic'].get('main_topic', 'Unknown')}")
            return True
        else:
            print("❌ Sentiment analysis incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Sentiment analyzer test failed: {e}")
        return False

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing Imports...")
    
    imports_to_test = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("json", "JSON handling"),
        ("datetime", "Date/time operations"),
        ("pathlib", "Path operations"),
        ("logging", "Logging"),
        ("dataclasses", "Data structures"),
        ("typing", "Type hints")
    ]
    
    failed_imports = []
    
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
        except ImportError:
            print(f"❌ {module_name} - {description}")
            failed_imports.append(module_name)
    
    # Test optional imports
    optional_imports = [
        ("textblob", "Sentiment analysis"),
        ("streamlit", "Web interface"),
        ("plotly", "Visualizations"),
        ("sentence_transformers", "Embeddings"),
        ("chromadb", "Vector database"),
        ("deepeval", "Evaluation framework"),
        ("llama_cpp", "Mistral model")
    ]
    
    print("\n📦 Optional Dependencies:")
    for module_name, description in optional_imports:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
        except ImportError:
            print(f"⚠️ {module_name} - {description} (install for full functionality)")
    
    return len(failed_imports) == 0

def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing File Structure...")
    
    required_files = [
        "data_processor.py",
        "mistral_model.py", 
        "rag_pipeline.py",
        "quality_analyzer.py",
        "deepeval_mistral.py",
        "sentiment_analyzer.py",
        "evaluation_orchestrator.py",
        "enhanced_streamlit_dashboard.py",
        "config.py",
        "requirements_complete.txt",
        "Call Transcript Sample 1.json"
    ]
    
    missing_files = []
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0

def generate_test_report(results):
    """Generate a test report."""
    print("\n" + "="*60)
    print("📊 INTEGRATION TEST REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Download Mistral model: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        print("2. Install optional dependencies: pip install -r requirements_complete.txt")
        print("3. Launch dashboard: streamlit run enhanced_streamlit_dashboard.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("Install missing dependencies with: pip install -r requirements_complete.txt")
    
    return passed_tests == total_tests

def main():
    """Run all integration tests."""
    print("🚀 Call Center Transcript Evaluation System - Integration Tests")
    print("="*70)
    
    # Run all tests
    test_results = {
        "File Structure": test_file_structure(),
        "Imports": test_imports(),
        "Configuration": test_configuration(),
        "Data Processor": test_data_processor(),
        "Quality Analyzer": test_quality_analyzer(),
        "Sentiment Analyzer": test_sentiment_analyzer()
    }
    
    # Generate report
    success = generate_test_report(test_results)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

