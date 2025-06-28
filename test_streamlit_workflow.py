#!/usr/bin/env python3
"""
Test script that simulates the exact Streamlit workflow to reproduce the error
"""

import os
import sys
from datetime import datetime

def test_streamlit_workflow():
    """Test the exact workflow that Streamlit follows."""
    
    print("🔍 Testing Streamlit workflow...")
    
    try:
        # Step 1: Import all required modules (like Streamlit does)
        print("📦 Importing modules...")
        from evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig
        print("✅ Modules imported successfully")
        
        # Step 2: Create configuration (like Streamlit sidebar)
        print("⚙️ Creating configuration...")
        config = EvaluationConfig()
        
        # Simulate user selections from Streamlit sidebar
        config.enable_rag = True
        config.enable_quality_analysis = True
        config.enable_deepeval = False  # Disable to focus on RAG
        config.enable_sentiment_analysis = True
        config.mistral_model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        config.output_format = "json"
        
        print("✅ Configuration created")
        print(f"   - RAG enabled: {config.enable_rag}")
        print(f"   - Quality analysis: {config.enable_quality_analysis}")
        print(f"   - DeepEval: {config.enable_deepeval}")
        print(f"   - Sentiment analysis: {config.enable_sentiment_analysis}")
        
        # Step 3: Create orchestrator (like process_files function)
        print("🔧 Creating EvaluationOrchestrator...")
        orchestrator = EvaluationOrchestrator(config)
        print("✅ EvaluationOrchestrator created")
        
        # Step 4: Initialize orchestrator (this is where the error should occur)
        print("🚀 Initializing orchestrator (this may fail due to missing dependencies)...")
        
        # Note: This will likely fail due to missing llama-cpp-python, but we want to see
        # if we get the RAGPipeline error or a different error
        init_success = orchestrator.initialize()
        
        if init_success:
            print("✅ Orchestrator initialized successfully!")
            
            # Step 5: Test file processing if we have a sample file
            sample_file = "Call Transcript Sample 1.json"
            if os.path.exists(sample_file):
                print(f"📁 Testing file processing with: {sample_file}")
                results = orchestrator.evaluate_transcript_file(sample_file)
                print(f"✅ File processed successfully, {len(results)} results")
            else:
                print("⚠️ No sample file found, skipping file processing test")
                
        else:
            print("❌ Orchestrator initialization failed (expected due to missing dependencies)")
            print("   This is normal if llama-cpp-python is not installed")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error occurred: {error_msg}")
        
        # Check if this is the specific RAGPipeline error we're looking for
        if "RAGPipeline.__init__()" in error_msg and "mistral_evaluator" in error_msg:
            print("🎯 FOUND THE RAGPIPELINE ERROR!")
            print("   This is the error the user reported")
        else:
            print("🔍 This is a different error than the RAGPipeline issue")
        
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Streamlit Workflow Test")
    print("=" * 50)
    
    success = test_streamlit_workflow()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test encountered errors!")
        
    print("\n📝 Summary:")
    print("   - If you see 'FOUND THE RAGPIPELINE ERROR!', that's the issue we need to fix")
    print("   - If you see initialization failures due to missing dependencies, that's expected")
    print("   - If everything passes, the original error may have been resolved")
