#!/usr/bin/env python3
"""
Mistral 7B Model Integration for Call Center Transcript Evaluation
Optimized for CPU inference using llama-cpp-python
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    Llama = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MistralConfig:
    """Configuration for Mistral model."""
    model_path: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    n_ctx: int = 4096  # Context window
    n_threads: int = None  # Will be set automatically based on CPU cores
    n_gpu_layers: int = 0  # CPU only
    temperature: float = 0.1  # Low temperature for consistent evaluation
    max_tokens: int = 512
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    verbose: bool = False

class MistralEvaluator:
    """
    Mistral 7B model wrapper optimized for call center transcript evaluation.
    """
    
    def __init__(self, config: MistralConfig = None):
        self.config = config or MistralConfig()
        self.model = None
        self.is_loaded = False
        
        # Evaluation prompts
        self.prompts = {
            'sentence_analysis': """
Analyze the following customer service response for sentence structure and clarity:

Response: "{text}"

Evaluate:
1. Are sentences too long or complex?
2. Recommend more concise alternatives
3. Rate clarity (1-10)
4. Suggest improvements

Provide feedback in this format:
CLARITY_SCORE: [1-10]
LONG_SENTENCES: [Yes/No]
RECOMMENDATIONS: [specific suggestions]
CRISP_ALTERNATIVES: [suggested rewrites]
""",
            
            'repetition_analysis': """
Analyze the following customer service response for repetition and crutch words:

Response: "{text}"

Identify:
1. Repeated phrases or words
2. Crutch words (um, uh, like, you know, etc.)
3. Unnecessary repetition
4. Impact on professionalism

Provide feedback in this format:
REPETITION_FOUND: [Yes/No]
REPEATED_PHRASES: [list of repeated elements]
CRUTCH_WORDS: [list of crutch words found]
PROFESSIONALISM_IMPACT: [High/Medium/Low]
RECOMMENDATIONS: [specific improvements]
""",
            
            'hold_transfer_analysis': """
Analyze the following customer service response for hold requests and call transfers:

Response: "{text}"

Identify:
1. Requests to hold or wait
2. Call transfer mentions
3. Reason for transfer (if any)
4. Professional handling

Provide feedback in this format:
HOLD_REQUEST: [Yes/No]
TRANSFER_MENTIONED: [Yes/No]
TRANSFER_REASON: [reason if found]
PROFESSIONAL_HANDLING: [Yes/No]
RECOMMENDATIONS: [improvements if needed]
""",
            
            'sentiment_analysis': """
Analyze the sentiment and tone of this customer service response:

Response: "{text}"

Evaluate:
1. Overall sentiment (positive/negative/neutral)
2. Emotional tone
3. Professionalism level
4. Customer impact

Provide feedback in this format:
SENTIMENT: [Positive/Negative/Neutral]
TONE_DESCRIPTION: [brief description]
PROFESSIONALISM: [High/Medium/Low]
CUSTOMER_IMPACT: [Positive/Negative/Neutral]
COACHING_FEEDBACK: [personalized coaching message]
""",
            
            'topic_summary': """
Summarize the main topic and theme of this customer interaction:

Customer Question: "{question}"
CSR Response: "{answer}"

Provide:
1. Main topic/theme
2. Customer concern category
3. Resolution approach
4. Key discussion points

Provide summary in this format:
MAIN_TOPIC: [primary topic]
CONCERN_CATEGORY: [category of customer concern]
RESOLUTION_APPROACH: [how CSR addressed it]
KEY_POINTS: [main discussion elements]
""",
            
            'rag_answer_generation': """
You are an expert customer service representative with years of experience. Based on the following customer question, provide the best possible professional response:

Customer Question: "{question}"
Context: Call center for {company_context}

Provide a comprehensive, professional response that:
1. Acknowledges the customer's concern
2. Provides clear, accurate information
3. Offers specific solutions or next steps
4. Maintains a helpful, empathetic tone
5. Follows best practices for customer service

Response:
""",
            
            'answer_evaluation': """
Evaluate the quality of this customer service response:

Customer Question: "{question}"
CSR Response: "{csr_answer}"
Expert Response: "{expert_answer}"

Compare the CSR response to the expert response and evaluate:
1. Relevance to the question
2. Completeness of information
3. Professional tone
4. Problem-solving approach
5. Areas for improvement

Provide evaluation in this format:
RELEVANCE_SCORE: [1-10]
COMPLETENESS_SCORE: [1-10]
PROFESSIONALISM_SCORE: [1-10]
OVERALL_SCORE: [1-10]
STRENGTHS: [what was done well]
IMPROVEMENTS: [specific areas to improve]
COACHING_MESSAGE: [personalized feedback for the CSR]
"""
        }
    
    def load_model(self) -> bool:
        """Load the Mistral model."""
        if Llama is None:
            logger.error("llama-cpp-python not available. Please install it.")
            return False
        
        if not os.path.exists(self.config.model_path):
            logger.error(f"Model file not found: {self.config.model_path}")
            logger.info("Please ensure the Mistral model file is in the correct location.")
            return False
        
        try:
            logger.info(f"Loading Mistral model from {self.config.model_path}")
            start_time = time.time()
            
            # Determine optimal thread count
            import multiprocessing
            if self.config.n_threads is None or self.config.n_threads <= 0:
                n_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
            else:
                n_threads = self.config.n_threads
            
            logger.info(f"Using {n_threads} threads for model inference")
            
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_threads=n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=self.config.verbose
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response from the model."""
        if not self.is_loaded:
            if not self.load_model():
                return "Error: Model not loaded"
        
        try:
            max_tokens = max_tokens or self.config.max_tokens
            
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                stop=["</s>", "[INST]", "[/INST]"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure and clarity."""
        prompt = self.prompts['sentence_analysis'].format(text=text)
        response = self.generate_response(prompt)
        return self._parse_structured_response(response)
    
    def analyze_repetition(self, text: str) -> Dict[str, Any]:
        """Analyze repetition and crutch words."""
        prompt = self.prompts['repetition_analysis'].format(text=text)
        response = self.generate_response(prompt)
        return self._parse_structured_response(response)
    
    def analyze_hold_transfer(self, text: str) -> Dict[str, Any]:
        """Analyze hold requests and transfers."""
        prompt = self.prompts['hold_transfer_analysis'].format(text=text)
        response = self.generate_response(prompt)
        return self._parse_structured_response(response)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and tone."""
        prompt = self.prompts['sentiment_analysis'].format(text=text)
        response = self.generate_response(prompt)
        return self._parse_structured_response(response)
    
    def summarize_topic(self, question: str, answer: str) -> Dict[str, Any]:
        """Summarize topic and theme."""
        prompt = self.prompts['topic_summary'].format(question=question, answer=answer)
        response = self.generate_response(prompt)
        return self._parse_structured_response(response)
    
    def generate_expert_answer(self, question: str, company_context: str = "travel services") -> str:
        """Generate expert-level answer for RAG pipeline."""
        prompt = self.prompts['rag_answer_generation'].format(
            question=question, 
            company_context=company_context
        )
        return self.generate_response(prompt, max_tokens=1024)
    
    def evaluate_answer_quality(self, question: str, csr_answer: str, expert_answer: str) -> Dict[str, Any]:
        """Evaluate CSR answer quality against expert answer."""
        prompt = self.prompts['answer_evaluation'].format(
            question=question,
            csr_answer=csr_answer,
            expert_answer=expert_answer
        )
        response = self.generate_response(prompt, max_tokens=1024)
        return self._parse_structured_response(response)
    
    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response into dictionary."""
        result = {'raw_response': response}
        
        lines = response.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                result[key] = value
        
        return result
    
    def batch_analyze(self, texts: List[str], analysis_type: str) -> List[Dict[str, Any]]:
        """Perform batch analysis on multiple texts."""
        results = []
        
        analysis_methods = {
            'sentence': self.analyze_sentence_structure,
            'repetition': self.analyze_repetition,
            'hold_transfer': self.analyze_hold_transfer,
            'sentiment': self.analyze_sentiment
        }
        
        if analysis_type not in analysis_methods:
            logger.error(f"Unknown analysis type: {analysis_type}")
            return results
        
        method = analysis_methods[analysis_type]
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)} for {analysis_type} analysis")
            result = method(text)
            result['text_index'] = i
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            'model_path': self.config.model_path,
            'is_loaded': self.is_loaded,
            'config': {
                'n_ctx': self.config.n_ctx,
                'n_threads': self.config.n_threads,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }

def main():
    """Example usage of MistralEvaluator."""
    # Initialize with default config
    config = MistralConfig(
        model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_tokens=512
    )
    
    evaluator = MistralEvaluator(config)
    
    # Test text
    test_text = "I apologize for the trouble. May I have your name and reservation number to look up your booking?"
    
    print("Mistral Model Information:")
    print(json.dumps(evaluator.get_model_info(), indent=2))
    
    if evaluator.load_model():
        print("\n=== Sentence Analysis ===")
        result = evaluator.analyze_sentence_structure(test_text)
        print(json.dumps(result, indent=2))
        
        print("\n=== Sentiment Analysis ===")
        result = evaluator.analyze_sentiment(test_text)
        print(json.dumps(result, indent=2))
        
        print("\n=== Expert Answer Generation ===")
        question = "I need help with a reservation I made last week."
        expert_answer = evaluator.generate_expert_answer(question)
        print(f"Question: {question}")
        print(f"Expert Answer: {expert_answer}")
    else:
        print("Failed to load model. Please check the model path and installation.")

if __name__ == "__main__":
    main()
