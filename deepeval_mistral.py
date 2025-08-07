#!/usr/bin/env python3
"""
DeepEval Integration with Local Mistral Model
Custom implementation to use local Mistral 7B instead of OpenAI API
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import re
from abc import ABC, abstractmethod

try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import BaseMetric
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    print("Warning: DeepEval not installed. Install with: pip install deepeval")
    evaluate = None
    LLMTestCase = None
    BaseMetric = None
    DeepEvalBaseLLM = None

from mistral_model import MistralEvaluator, MistralConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralLLMWrapper(DeepEvalBaseLLM if DeepEvalBaseLLM else object):
    """Wrapper to make Mistral model compatible with DeepEval."""
    
    def __init__(self, mistral_evaluator: MistralEvaluator):
        self.mistral_evaluator = mistral_evaluator
        self.model_name = "mistral-7b-instruct-v0.2"
    
    def load_model(self):
        """Load the Mistral model."""
        return self.mistral_evaluator.load_model()
    
    def generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Generate response using Mistral model."""
        return self.mistral_evaluator.generate_response(prompt, max_tokens=1024)
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    async def a_generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Async generate (fallback to sync)."""
        return self.generate(prompt, schema)

class MistralAnswerRelevancyMetric(BaseMetric if BaseMetric else object):
    """Custom Answer Relevancy Metric using local Mistral model."""
    
    def __init__(self, mistral_llm: MistralLLMWrapper, threshold: float = 0.7):
        self.mistral_llm = mistral_llm
        self.threshold = threshold
        self.evaluation_model = mistral_llm
        
    def measure(self, test_case: 'LLMTestCase') -> float:
        """Measure answer relevancy."""
        prompt = f"""
Evaluate how relevant the given answer is to the question asked. Consider:
1. Does the answer directly address the question?
2. Is the information provided pertinent to what was asked?
3. Are there any irrelevant details or tangents?

Question: {test_case.input}
Answer: {test_case.actual_output}

Rate the relevancy on a scale of 0.0 to 1.0 where:
- 1.0 = Perfectly relevant, directly answers the question
- 0.8 = Highly relevant with minor irrelevant details
- 0.6 = Moderately relevant but missing some key points
- 0.4 = Somewhat relevant but significant gaps
- 0.2 = Barely relevant, mostly off-topic
- 0.0 = Completely irrelevant

Provide your evaluation in this exact format:
RELEVANCY_SCORE: [score between 0.0 and 1.0]
REASONING: [brief explanation of the score]
"""
        
        response = self.mistral_llm.generate(prompt)
        score = self._extract_score(response)
        
        self.score = score
        self.reason = self._extract_reasoning(response)
        self.success = score >= self.threshold
        
        return score
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from response."""
        # Look for RELEVANCY_SCORE pattern
        score_match = re.search(r'RELEVANCY_SCORE:\s*([0-9]*\.?[0-9]+)', response)
        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                pass
        
        # Fallback: look for any decimal number between 0 and 1
        decimal_match = re.search(r'\b(0\.[0-9]+|1\.0+|0\.0+)\b', response)
        if decimal_match:
            try:
                return float(decimal_match.group(1))
            except ValueError:
                pass
        
        # Default fallback
        logger.warning(f"Could not extract score from response: {response[:100]}...")
        return 0.5
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return "No reasoning provided"
    
    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return hasattr(self, 'success') and self.success
    
    @property
    def __name__(self):
        return "Answer Relevancy"

class MistralCorrectnessMetric(BaseMetric if BaseMetric else object):
    """Custom Correctness Metric using local Mistral model."""
    
    def __init__(self, mistral_llm: MistralLLMWrapper, threshold: float = 0.7):
        self.mistral_llm = mistral_llm
        self.threshold = threshold
        self.evaluation_model = mistral_llm
    
    def measure(self, test_case: 'LLMTestCase') -> float:
        """Measure answer correctness."""
        prompt = f"""
Evaluate the correctness of the given answer compared to the expected answer. Consider:
1. Factual accuracy of the information provided
2. Completeness of the response
3. Alignment with the expected answer
4. Absence of incorrect or misleading information

Question: {test_case.input}
Actual Answer: {test_case.actual_output}
Expected Answer: {test_case.expected_output}

Rate the correctness on a scale of 0.0 to 1.0 where:
- 1.0 = Completely correct and comprehensive
- 0.8 = Mostly correct with minor omissions
- 0.6 = Generally correct but missing important details
- 0.4 = Partially correct with some errors
- 0.2 = Mostly incorrect with few correct elements
- 0.0 = Completely incorrect or misleading

Provide your evaluation in this exact format:
CORRECTNESS_SCORE: [score between 0.0 and 1.0]
REASONING: [detailed explanation of the score]
MISSING_ELEMENTS: [what was missing from the actual answer]
INCORRECT_ELEMENTS: [what was incorrect in the actual answer]
"""
        
        response = self.mistral_llm.generate(prompt)
        score = self._extract_score(response)
        
        self.score = score
        self.reason = self._extract_reasoning(response)
        self.missing_elements = self._extract_missing_elements(response)
        self.incorrect_elements = self._extract_incorrect_elements(response)
        self.success = score >= self.threshold
        
        return score
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from response."""
        score_match = re.search(r'CORRECTNESS_SCORE:\s*([0-9]*\.?[0-9]+)', response)
        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Fallback
        decimal_match = re.search(r'\b(0\.[0-9]+|1\.0+|0\.0+)\b', response)
        if decimal_match:
            try:
                return float(decimal_match.group(1))
            except ValueError:
                pass
        
        logger.warning(f"Could not extract correctness score from response: {response[:100]}...")
        return 0.5
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=MISSING_ELEMENTS:|INCORRECT_ELEMENTS:|$)', response, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return "No reasoning provided"
    
    def _extract_missing_elements(self, response: str) -> str:
        """Extract missing elements from response."""
        missing_match = re.search(r'MISSING_ELEMENTS:\s*(.+?)(?=INCORRECT_ELEMENTS:|$)', response, re.DOTALL)
        if missing_match:
            return missing_match.group(1).strip()
        return "None identified"
    
    def _extract_incorrect_elements(self, response: str) -> str:
        """Extract incorrect elements from response."""
        incorrect_match = re.search(r'INCORRECT_ELEMENTS:\s*(.+)', response, re.DOTALL)
        if incorrect_match:
            return incorrect_match.group(1).strip()
        return "None identified"
    
    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return hasattr(self, 'success') and self.success
    
    @property
    def __name__(self):
        return "Correctness"

class MistralEvaluationEngine:
    """Main evaluation engine using Mistral model with DeepEval-style interface."""
    
    def __init__(self, mistral_config: MistralConfig = None):
        self.mistral_evaluator = MistralEvaluator(mistral_config)
        self.mistral_llm = MistralLLMWrapper(self.mistral_evaluator)
        
        # Initialize metrics
        self.relevancy_metric = MistralAnswerRelevancyMetric(self.mistral_llm)
        self.correctness_metric = MistralCorrectnessMetric(self.mistral_llm)
    
    def initialize(self) -> bool:
        """Initialize the evaluation engine."""
        return self.mistral_llm.load_model()
    
    def create_test_case(self, question: str, csr_answer: str, expert_answer: str, 
                        metadata: Dict = None) -> Dict[str, Any]:
        """Create a test case for evaluation."""
        if LLMTestCase is None:
            # Fallback implementation if DeepEval not available
            return {
                'input': question,
                'actual_output': csr_answer,
                'expected_output': expert_answer,
                'metadata': metadata or {}
            }
        
        return LLMTestCase(
            input=question,
            actual_output=csr_answer,
            expected_output=expert_answer,
            context=metadata.get('context', []) if metadata else []
        )
    
    def evaluate_single_case(self, test_case: Union[Dict, 'LLMTestCase']) -> Dict[str, Any]:
        """Evaluate a single test case."""
        # Convert dict to test case if needed
        if isinstance(test_case, dict):
            if LLMTestCase is not None:
                test_case = LLMTestCase(
                    input=test_case['input'],
                    actual_output=test_case['actual_output'],
                    expected_output=test_case['expected_output']
                )
        
        results = {}
        
        # Evaluate relevancy
        try:
            relevancy_score = self.relevancy_metric.measure(test_case)
            results['relevancy'] = {
                'score': relevancy_score,
                'passed': self.relevancy_metric.is_successful(),
                'reasoning': getattr(self.relevancy_metric, 'reason', 'No reasoning available'),
                'threshold': self.relevancy_metric.threshold
            }
        except Exception as e:
            logger.error(f"Error in relevancy evaluation: {e}")
            results['relevancy'] = {
                'score': 0.0,
                'passed': False,
                'reasoning': f"Evaluation error: {str(e)}",
                'threshold': self.relevancy_metric.threshold
            }
        
        # Evaluate correctness
        try:
            correctness_score = self.correctness_metric.measure(test_case)
            results['correctness'] = {
                'score': correctness_score,
                'passed': self.correctness_metric.is_successful(),
                'reasoning': getattr(self.correctness_metric, 'reason', 'No reasoning available'),
                'missing_elements': getattr(self.correctness_metric, 'missing_elements', 'None identified'),
                'incorrect_elements': getattr(self.correctness_metric, 'incorrect_elements', 'None identified'),
                'threshold': self.correctness_metric.threshold
            }
        except Exception as e:
            logger.error(f"Error in correctness evaluation: {e}")
            results['correctness'] = {
                'score': 0.0,
                'passed': False,
                'reasoning': f"Evaluation error: {str(e)}",
                'missing_elements': 'Could not evaluate',
                'incorrect_elements': 'Could not evaluate',
                'threshold': self.correctness_metric.threshold
            }
        
        # Calculate overall score
        overall_score = (results['relevancy']['score'] + results['correctness']['score']) / 2
        results['overall'] = {
            'score': round(overall_score, 3),
            'passed': results['relevancy']['passed'] and results['correctness']['passed']
        }
        
        return results
    
    def evaluate_batch(self, test_cases: List[Union[Dict, 'LLMTestCase']]) -> List[Dict[str, Any]]:
        """Evaluate multiple test cases."""
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            result = self.evaluate_single_case(test_case)
            result['test_case_index'] = i
            results.append(result)
        
        return results
    
    def generate_coaching_feedback(self, evaluation_result: Dict[str, Any], 
                                 csr_id: str = "") -> str:
        """Generate personalized coaching feedback based on evaluation results."""
        relevancy = evaluation_result['relevancy']
        correctness = evaluation_result['correctness']
        overall = evaluation_result['overall']
        
        feedback_parts = []
        
        # Overall performance
        if overall['score'] >= 0.8:
            feedback_parts.append(f"Excellent work! Your overall performance score is {overall['score']:.1f}/1.0.")
        elif overall['score'] >= 0.6:
            feedback_parts.append(f"Good performance with room for improvement. Your overall score is {overall['score']:.1f}/1.0.")
        else:
            feedback_parts.append(f"Your performance needs attention. Your overall score is {overall['score']:.1f}/1.0.")
        
        # Relevancy feedback
        if relevancy['score'] >= 0.8:
            feedback_parts.append(f"Your relevancy score is excellent at {relevancy['score']:.1f}/1.0 - you directly addressed the customer's question.")
        elif relevancy['score'] >= 0.6:
            feedback_parts.append(f"Your relevancy score is {relevancy['score']:.1f}/1.0. {relevancy['reasoning']}")
        else:
            feedback_parts.append(f"Your relevancy score is low at {relevancy['score']:.1f}/1.0. Focus on directly addressing what the customer is asking. {relevancy['reasoning']}")
        
        # Correctness feedback
        if correctness['score'] >= 0.8:
            feedback_parts.append(f"Your correctness score is excellent at {correctness['score']:.1f}/1.0 - your information was accurate and complete.")
        elif correctness['score'] >= 0.6:
            feedback_parts.append(f"Your correctness score is {correctness['score']:.1f}/1.0. {correctness['reasoning']}")
            if correctness['missing_elements'] != 'None identified':
                feedback_parts.append(f"Missing elements: {correctness['missing_elements']}")
        else:
            feedback_parts.append(f"Your correctness score needs improvement at {correctness['score']:.1f}/1.0. {correctness['reasoning']}")
            if correctness['missing_elements'] != 'None identified':
                feedback_parts.append(f"You need to include: {correctness['missing_elements']}")
            if correctness['incorrect_elements'] != 'None identified':
                feedback_parts.append(f"Please correct: {correctness['incorrect_elements']}")
        
        # Improvement suggestions
        if not overall['passed']:
            feedback_parts.append("Focus on providing complete, accurate responses that directly address customer questions.")
        
        return " ".join(feedback_parts)

def main():
    """Example usage of MistralEvaluationEngine."""
    # Initialize evaluation engine
    mistral_config = MistralConfig(model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    engine = MistralEvaluationEngine(mistral_config)
    
    if not engine.initialize():
        print("Failed to initialize evaluation engine")
        return
    
    # Create test case
    question = "I need help with a reservation I made last week"
    csr_answer = "I apologize for the trouble. May I have your name and reservation number to look up your booking?"
    expert_answer = "I'd be happy to help you with your reservation. To locate your booking quickly, could you please provide your name and reservation number? This will allow me to access your details and address any concerns you may have."
    
    test_case = engine.create_test_case(question, csr_answer, expert_answer)
    
    print("Evaluating test case...")
    result = engine.evaluate_single_case(test_case)
    
    print(f"\nEvaluation Results:")
    print(f"Overall Score: {result['overall']['score']:.3f}")
    print(f"Relevancy Score: {result['relevancy']['score']:.3f}")
    print(f"Correctness Score: {result['correctness']['score']:.3f}")
    
    print(f"\nCoaching Feedback:")
    feedback = engine.generate_coaching_feedback(result, "CSR123")
    print(feedback)

if __name__ == "__main__":
    main()

