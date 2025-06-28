#!/usr/bin/env python3
"""
Test script to reproduce the RAGPipeline error
"""

import os
import sys
from evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig

def test_rag_pipeline_error():
    """Test to reproduce the RAGPipeline mistral_evaluator error."""
    
    print("🔍 Testing RAGPipeline error reproduction...")
    
    try:
        # Create config with RAG enabled (same as Streamlit)
        config = EvaluationConfig()
        config.enable_rag = True
        config.enable_deepeval = False  # Disable to isolate RAG issue
        
        print("✅ Config created")
        
        # Create orchestrator (this should trigger RAG pipeline creation)
        print("🔧 Creating EvaluationOrchestrator...")
        orchestrator = EvaluationOrchestrator(config)
        print("✅ EvaluationOrchestrator created successfully")
        
        # Initialize orchestrator (this is where the RAG pipeline error should occur)
        print("🔧 Initializing EvaluationOrchestrator...")
        if not orchestrator.initialize():
            print("❌ Failed to initialize orchestrator")
            return False
        print("✅ EvaluationOrchestrator initialized successfully")
        
        # Test with a sample file if it exists
        sample_file = "Call Transcript Sample 1.json"
        if os.path.exists(sample_file):
            print(f"📁 Testing with sample file: {sample_file}")
            results = orchestrator.evaluate_transcript_file(sample_file)
            print(f"✅ Evaluation completed with {len(results)} results")
        else:
            print("⚠️ Sample file not found, skipping file evaluation test")
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_rag_pipeline_error()
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
        sys.exit(1)
