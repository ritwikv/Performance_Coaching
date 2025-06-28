#!/usr/bin/env python3
"""
Minimal test to reproduce RAGPipeline initialization issue
"""

def test_rag_pipeline_signature():
    """Test RAGPipeline signature and creation."""
    
    print("🔍 Testing RAGPipeline signature...")
    
    try:
        from rag_pipeline import RAGPipeline, RAGConfig
        from mistral_model import MistralEvaluator, MistralConfig
        import inspect
        
        # Check signature
        sig = inspect.signature(RAGPipeline.__init__)
        print(f"✅ RAGPipeline.__init__ signature: {sig}")
        print(f"✅ Parameters: {list(sig.parameters.keys())}")
        
        # Test creation with positional args
        print("\n🧪 Testing positional arguments...")
        rag_config = RAGConfig()
        mistral_config = MistralConfig()
        mistral_evaluator = MistralEvaluator(mistral_config)
        
        rag1 = RAGPipeline(rag_config, mistral_evaluator)
        print("✅ RAGPipeline created with positional args")
        
        # Test creation with keyword args
        print("\n🧪 Testing keyword arguments...")
        rag2 = RAGPipeline(config=rag_config, mistral_evaluator=mistral_evaluator)
        print("✅ RAGPipeline created with keyword args")
        
        # Test creation with mixed args (like in the orchestrator)
        print("\n🧪 Testing mixed arguments (orchestrator style)...")
        rag3 = RAGPipeline(rag_config, mistral_evaluator=mistral_evaluator)
        print("✅ RAGPipeline created with mixed args")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_pipeline_signature()
    if success:
        print("\n🎉 All RAGPipeline tests passed!")
    else:
        print("\n💥 RAGPipeline tests failed!")

