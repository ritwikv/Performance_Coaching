#!/usr/bin/env python3
"""
Evaluation Orchestrator - Main coordination engine for call center transcript evaluation
Coordinates all analysis components and generates unified feedback
"""

import logging
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from data_processor import CallTranscriptProcessor
from mistral_model import MistralEvaluator, MistralConfig
from rag_pipeline import RAGPipeline, RAGConfig
from quality_analyzer import CallQualityAnalyzer, QualityMetrics
from deepeval_mistral import MistralEvaluationEngine
from sentiment_analyzer import SentimentTopicAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for the evaluation orchestrator."""
    mistral_model_path: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    enable_rag: bool = True
    enable_deepeval: bool = True
    enable_quality_analysis: bool = True
    enable_sentiment_analysis: bool = True
    output_format: str = "json"  # json, csv, excel
    save_intermediate_results: bool = True
    max_concurrent_evaluations: int = 1  # CPU optimization

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single conversation."""
    # Basic information
    call_id: str = ""
    csr_id: str = ""
    call_date: str = ""
    call_time: str = ""
    interaction_sequence: int = 0
    
    # Input data
    question: str = ""
    answer: str = ""
    expert_answer: str = ""
    
    # Quality analysis
    quality_scores: Dict[str, Any] = None
    aht_impact: Dict[str, Any] = None
    
    # DeepEval metrics
    deepeval_scores: Dict[str, Any] = None
    
    # Sentiment and topic analysis
    sentiment_analysis: Dict[str, Any] = None
    topic_analysis: Dict[str, Any] = None
    
    # Coaching feedback
    coaching_feedback: str = ""
    concise_summary: str = ""
    
    # Metadata
    evaluation_timestamp: str = ""
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = {}
        if self.aht_impact is None:
            self.aht_impact = {}
        if self.deepeval_scores is None:
            self.deepeval_scores = {}
        if self.sentiment_analysis is None:
            self.sentiment_analysis = {}
        if self.topic_analysis is None:
            self.topic_analysis = {}

class FeedbackGenerator:
    """Generates comprehensive coaching feedback and summaries."""
    
    def __init__(self):
        self.feedback_templates = {
            'excellent': "Excellent performance! Your {metric} score of {score} demonstrates strong customer service skills.",
            'good': "Good work on {metric} with a score of {score}. {improvement_suggestion}",
            'needs_improvement': "Your {metric} score of {score} indicates room for improvement. {specific_guidance}",
            'poor': "Your {metric} score of {score} needs immediate attention. {corrective_action}"
        }
    
    def generate_comprehensive_feedback(self, result: EvaluationResult) -> Tuple[str, str]:
        """Generate comprehensive coaching feedback and concise summary."""
        feedback_sections = []
        summary_points = []
        
        # Quality Analysis Feedback
        if result.quality_scores:
            quality_feedback, quality_summary = self._generate_quality_feedback(result.quality_scores)
            feedback_sections.append(f"**Quality Analysis:**\n{quality_feedback}")
            summary_points.extend(quality_summary)
        
        # DeepEval Feedback
        if result.deepeval_scores:
            deepeval_feedback, deepeval_summary = self._generate_deepeval_feedback(result.deepeval_scores)
            feedback_sections.append(f"**Performance Evaluation:**\n{deepeval_feedback}")
            summary_points.extend(deepeval_summary)
        
        # Sentiment Feedback
        if result.sentiment_analysis:
            sentiment_feedback, sentiment_summary = self._generate_sentiment_feedback(result.sentiment_analysis)
            feedback_sections.append(f"**Communication Style:**\n{sentiment_feedback}")
            summary_points.extend(sentiment_summary)
        
        # Topic Analysis Feedback
        if result.topic_analysis:
            topic_feedback, topic_summary = self._generate_topic_feedback(result.topic_analysis)
            feedback_sections.append(f"**Topic Handling:**\n{topic_feedback}")
            summary_points.extend(topic_summary)
        
        # AHT Impact Feedback
        if result.aht_impact:
            aht_feedback, aht_summary = self._generate_aht_feedback(result.aht_impact)
            feedback_sections.append(f"**Efficiency Impact:**\n{aht_feedback}")
            summary_points.extend(aht_summary)
        
        # Combine feedback
        comprehensive_feedback = "\n\n".join(feedback_sections)
        
        # Generate concise summary (200 words max)
        concise_summary = self._create_concise_summary(summary_points, result)
        
        return comprehensive_feedback, concise_summary
    
    def _generate_quality_feedback(self, quality_scores: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate quality analysis feedback."""
        feedback_parts = []
        summary_points = []
        
        # Overall quality score
        quality_metrics = quality_scores.get('quality_metrics')
        overall_score = quality_metrics.overall_score if quality_metrics else 0
        if overall_score >= 8:
            feedback_parts.append(f"Outstanding quality with an overall score of {overall_score}/10.")
            summary_points.append("Excellent quality performance")
        elif overall_score >= 6:
            feedback_parts.append(f"Good quality performance with a score of {overall_score}/10.")
            summary_points.append("Good quality with room for improvement")
        else:
            feedback_parts.append(f"Quality score of {overall_score}/10 needs improvement.")
            summary_points.append("Quality requires attention")
        
        # Specific quality aspects
        sentence_analysis = quality_scores.get('sentence_analysis', {})
        if sentence_analysis.get('clarity_score', 0) < 6:
            feedback_parts.append("Focus on clearer, more concise communication.")
            summary_points.append("Improve clarity")
        
        repetition_analysis = quality_scores.get('repetition_analysis', {})
        if repetition_analysis.get('total_crutch_count', 0) > 3:
            feedback_parts.append("Reduce use of filler words and repetitive phrases.")
            summary_points.append("Eliminate filler words")
        
        return " ".join(feedback_parts), summary_points
    
    def _generate_deepeval_feedback(self, deepeval_scores: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate DeepEval feedback."""
        feedback_parts = []
        summary_points = []
        
        relevancy = deepeval_scores.get('relevancy', {})
        correctness = deepeval_scores.get('correctness', {})
        overall = deepeval_scores.get('overall', {})
        
        # Overall performance
        overall_score = overall.get('score', 0)
        if overall_score >= 0.8:
            feedback_parts.append(f"Excellent evaluation performance with {overall_score:.1f}/1.0 overall score.")
            summary_points.append("Excellent evaluation scores")
        elif overall_score >= 0.6:
            feedback_parts.append(f"Good performance with {overall_score:.1f}/1.0 overall score.")
            summary_points.append("Good evaluation performance")
        else:
            feedback_parts.append(f"Performance score of {overall_score:.1f}/1.0 needs improvement.")
            summary_points.append("Evaluation scores need improvement")
        
        # Relevancy feedback
        rel_score = relevancy.get('score', 0)
        if rel_score < 0.7:
            feedback_parts.append(f"Your relevancy score is {rel_score:.1f}/1.0. Focus on directly addressing customer questions.")
            summary_points.append("Improve answer relevancy")
        
        # Correctness feedback
        corr_score = correctness.get('score', 0)
        if corr_score < 0.7:
            feedback_parts.append(f"Your correctness score is {corr_score:.1f}/1.0. Ensure information accuracy and completeness.")
            summary_points.append("Improve answer accuracy")
        
        return " ".join(feedback_parts), summary_points
    
    def _generate_sentiment_feedback(self, sentiment_analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate sentiment feedback."""
        feedback_parts = []
        summary_points = []
        
        sentiment_label = sentiment_analysis.get('sentiment_label', 'Neutral')
        emotional_tone = sentiment_analysis.get('emotional_tone', 'Professional')
        polarity = sentiment_analysis.get('polarity', 0)
        
        if sentiment_label == 'Positive':
            feedback_parts.append(f"Excellent positive sentiment with {emotional_tone.lower()} tone.")
            summary_points.append("Positive communication style")
        elif sentiment_label == 'Negative':
            feedback_parts.append(f"Consider more positive language. Current tone appears {emotional_tone.lower()}.")
            summary_points.append("Improve positive language use")
        else:
            feedback_parts.append(f"Neutral tone with {emotional_tone.lower()} approach.")
            summary_points.append("Maintain professional tone")
        
        # Add coaching feedback if available
        coaching = sentiment_analysis.get('coaching_feedback', '')
        if coaching:
            feedback_parts.append(coaching)
        
        return " ".join(feedback_parts), summary_points
    
    def _generate_topic_feedback(self, topic_analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate topic analysis feedback."""
        feedback_parts = []
        summary_points = []
        
        main_topic = topic_analysis.get('main_topic', 'General')
        concern_category = topic_analysis.get('concern_category', 'General')
        
        feedback_parts.append(f"Topic: {main_topic} ({concern_category})")
        summary_points.append(f"Handled {main_topic.lower()} inquiry")
        
        # Topic-specific guidance
        if main_topic == 'Billing':
            feedback_parts.append("For billing issues, ensure clear explanation of charges and next steps.")
        elif main_topic == 'Reservation':
            feedback_parts.append("For reservations, provide specific details and alternative options.")
        elif main_topic == 'Technical Support':
            feedback_parts.append("For technical issues, offer step-by-step guidance.")
        
        return " ".join(feedback_parts), summary_points
    
    def _generate_aht_feedback(self, aht_impact: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate AHT impact feedback."""
        feedback_parts = []
        summary_points = []
        
        impact_level = aht_impact.get('aht_impact_level', 'Unknown')
        impact_score = aht_impact.get('aht_impact_score', 0)
        estimated_increase = aht_impact.get('estimated_time_increase', '0%')
        
        if impact_level == 'High':
            feedback_parts.append(f"High AHT impact ({estimated_increase} increase). Focus on efficiency.")
            summary_points.append("High AHT impact - improve efficiency")
        elif impact_level == 'Medium':
            feedback_parts.append(f"Moderate AHT impact ({estimated_increase} increase). Some efficiency improvements needed.")
            summary_points.append("Moderate AHT impact")
        else:
            feedback_parts.append(f"Low AHT impact ({estimated_increase} increase). Good efficiency.")
            summary_points.append("Good efficiency")
        
        # Specific impact factors
        impact_factors = aht_impact.get('impact_factors', [])
        if impact_factors:
            factor_descriptions = [f['description'] for f in impact_factors[:2]]
            feedback_parts.extend(factor_descriptions)
        
        return " ".join(feedback_parts), summary_points
    
    def _create_concise_summary(self, summary_points: List[str], result: EvaluationResult) -> str:
        """Create a concise 200-word summary."""
        summary_parts = []
        
        # Header
        summary_parts.append(f"Performance Summary for {result.csr_id}:")
        
        # Key points (limit to most important)
        key_points = summary_points[:6]  # Limit to 6 key points
        for point in key_points:
            summary_parts.append(f"• {point}")
        
        # Overall assessment
        overall_score = 0
        score_count = 0
        
        if result.quality_scores and 'quality_metrics' in result.quality_scores:
            quality_metrics = result.quality_scores.get('quality_metrics')
            if quality_metrics:
                overall_score += quality_metrics.overall_score
            score_count += 1
        
        if result.deepeval_scores and 'overall' in result.deepeval_scores:
            overall_score += result.deepeval_scores['overall'].get('score', 0) * 10
            score_count += 1
        
        if score_count > 0:
            avg_score = overall_score / score_count
            if avg_score >= 8:
                summary_parts.append("Overall: Excellent performance meeting high standards.")
            elif avg_score >= 6:
                summary_parts.append("Overall: Good performance with opportunities for improvement.")
            else:
                summary_parts.append("Overall: Performance requires focused improvement efforts.")
        
        # Topic and sentiment
        if result.topic_analysis:
            topic = result.topic_analysis.get('main_topic', 'General')
            summary_parts.append(f"Successfully handled {topic.lower()} inquiry.")
        
        if result.sentiment_analysis:
            sentiment = result.sentiment_analysis.get('sentiment_label', 'Neutral')
            tone = result.sentiment_analysis.get('emotional_tone', 'Professional')
            summary_parts.append(f"Communication style: {sentiment.lower()} sentiment with {tone.lower()} tone.")
        
        # Join and limit to ~200 words
        full_summary = " ".join(summary_parts)
        words = full_summary.split()
        if len(words) > 200:
            full_summary = " ".join(words[:200]) + "..."
        
        return full_summary

class EvaluationOrchestrator:
    """Main orchestrator for comprehensive call center transcript evaluation."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
        # Initialize components
        self.data_processor = CallTranscriptProcessor()
        self.mistral_evaluator = None
        self.rag_pipeline = None
        self.quality_analyzer = None
        self.deepeval_engine = None
        self.sentiment_analyzer = None
        self.feedback_generator = FeedbackGenerator()
        
        # Results storage
        self.evaluation_results: List[EvaluationResult] = []
        
    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Evaluation Orchestrator...")
        
        # Initialize Mistral model
        mistral_config = MistralConfig(model_path=self.config.mistral_model_path)
        self.mistral_evaluator = MistralEvaluator(mistral_config)
        
        if not self.mistral_evaluator.load_model():
            logger.error("Failed to load Mistral model")
            return False
        
        # Initialize RAG pipeline
        if self.config.enable_rag:
            rag_config = RAGConfig()
            # Pass the already-loaded Mistral evaluator to avoid loading model twice
            self.rag_pipeline = RAGPipeline(rag_config, mistral_evaluator=self.mistral_evaluator)
            if not self.rag_pipeline.initialize():
                logger.warning("RAG pipeline initialization failed - continuing without RAG")
                self.config.enable_rag = False
        
        # Initialize quality analyzer
        if self.config.enable_quality_analysis:
            self.quality_analyzer = CallQualityAnalyzer()
        
        # Initialize DeepEval engine
        if self.config.enable_deepeval:
            self.deepeval_engine = MistralEvaluationEngine(mistral_config)
            if not self.deepeval_engine.initialize():
                logger.warning("DeepEval engine initialization failed - continuing without DeepEval")
                self.config.enable_deepeval = False
        
        # Initialize sentiment analyzer
        if self.config.enable_sentiment_analysis:
            self.sentiment_analyzer = SentimentTopicAnalyzer(self.mistral_evaluator)
        
        logger.info("Evaluation Orchestrator initialized successfully")
        return True
    
    def evaluate_transcript_file(self, file_path: str) -> List[EvaluationResult]:
        """Evaluate a single transcript file."""
        logger.info(f"Evaluating transcript file: {file_path}")
        
        # Load and process transcript
        transcript_data = self.data_processor.load_json_transcript(file_path)
        if not transcript_data:
            logger.error(f"Failed to load transcript: {file_path}")
            return []
        
        # Process into DataFrame
        records = self.data_processor.process_single_transcript(transcript_data)
        df = pd.DataFrame(records)
        
        if df.empty:
            logger.warning("No conversation pairs found in transcript")
            return []
        
        # Build RAG knowledge base if enabled
        if self.config.enable_rag and self.rag_pipeline:
            logger.info("Building RAG knowledge base...")
            self.rag_pipeline.build_knowledge_base(df)
        
        # Evaluate each conversation
        results = []
        for idx, row in df.iterrows():
            logger.info(f"Evaluating conversation {idx + 1}/{len(df)}")
            result = self._evaluate_single_conversation(row)
            results.append(result)
        
        self.evaluation_results.extend(results)
        return results
    
    def _evaluate_single_conversation(self, row: pd.Series) -> EvaluationResult:
        """Evaluate a single conversation."""
        start_time = datetime.now()
        
        # Initialize result
        result = EvaluationResult(
            call_id=row.get('call_ID', ''),
            csr_id=row.get('CSR_ID', ''),
            call_date=row.get('call_date', ''),
            call_time=row.get('call_time', ''),
            interaction_sequence=row.get('interaction_sequence', 0),
            question=row.get('question', ''),
            answer=row.get('answer', ''),
            evaluation_timestamp=datetime.now().isoformat()
        )
        
        # Get expert answer from RAG
        if self.config.enable_rag and self.rag_pipeline:
            rag_result = self.rag_pipeline.get_expert_answer(result.question)
            result.expert_answer = rag_result['expert_answer']
        else:
            # Fallback: generate expert answer directly
            result.expert_answer = self.mistral_evaluator.generate_expert_answer(result.question)
        
        # Quality analysis
        if self.config.enable_quality_analysis and self.quality_analyzer:
            quality_analysis = self.quality_analyzer.analyze_single_response(result.answer)
            result.quality_scores = quality_analysis
            result.aht_impact = quality_analysis.get('aht_impact', {})
        
        # DeepEval evaluation
        if self.config.enable_deepeval and self.deepeval_engine:
            test_case = self.deepeval_engine.create_test_case(
                result.question, result.answer, result.expert_answer
            )
            deepeval_result = self.deepeval_engine.evaluate_single_case(test_case)
            result.deepeval_scores = deepeval_result
        
        # Sentiment and topic analysis
        if self.config.enable_sentiment_analysis and self.sentiment_analyzer:
            sentiment_topic_result = self.sentiment_analyzer.analyze_conversation(
                result.question, result.answer
            )
            result.sentiment_analysis = sentiment_topic_result.get('sentiment', {})
            result.topic_analysis = sentiment_topic_result.get('topic', {})
        
        # Generate comprehensive feedback
        comprehensive_feedback, concise_summary = self.feedback_generator.generate_comprehensive_feedback(result)
        result.coaching_feedback = comprehensive_feedback
        result.concise_summary = concise_summary
        
        # Calculate processing time
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def save_results(self, output_path: str, format: str = None) -> bool:
        """Save evaluation results to file."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to save")
            return False
        
        format = format or self.config.output_format
        
        try:
            if format.lower() == 'json':
                results_dict = [asdict(result) for result in self.evaluation_results]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                # Flatten results for CSV
                flattened_results = []
                for result in self.evaluation_results:
                    flat_result = self._flatten_result(result)
                    flattened_results.append(flat_result)
                
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_path, index=False)
            
            elif format.lower() == 'excel':
                # Create multiple sheets
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    for result in self.evaluation_results:
                        # Get quality score safely
                        quality_score = 0
                        if result.quality_scores and result.quality_scores.get('quality_metrics'):
                            quality_metrics = result.quality_scores.get('quality_metrics')
                            if hasattr(quality_metrics, 'overall_score'):
                                quality_score = quality_metrics.overall_score
                        
                        summary_data.append({
                            'Call_ID': result.call_id,
                            'CSR_ID': result.csr_id,
                            'Overall_Quality_Score': quality_score,
                            'DeepEval_Overall_Score': result.deepeval_scores.get('overall', {}).get('score', 0) if result.deepeval_scores else 0,
                            'Sentiment': result.sentiment_analysis.get('sentiment_label', '') if result.sentiment_analysis else '',
                            'Topic': result.topic_analysis.get('main_topic', '') if result.topic_analysis else '',
                            'AHT_Impact': result.aht_impact.get('aht_impact_level', '') if result.aht_impact else '',
                            'Concise_Summary': result.concise_summary
                        })
                    
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Detailed results
                    flattened_results = [self._flatten_result(result) for result in self.evaluation_results]
                    pd.DataFrame(flattened_results).to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            logger.info(f"Results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def _flatten_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """Flatten evaluation result for tabular output."""
        flat = {
            'call_id': result.call_id,
            'csr_id': result.csr_id,
            'call_date': result.call_date,
            'call_time': result.call_time,
            'question': result.question,
            'answer': result.answer,
            'expert_answer': result.expert_answer,
            'concise_summary': result.concise_summary,
            'processing_time_seconds': result.processing_time_seconds
        }
        
        # Flatten quality scores
        if result.quality_scores:
            quality_metrics = result.quality_scores.get('quality_metrics')
            if quality_metrics and hasattr(quality_metrics, 'overall_score'):
                flat.update({
                    'quality_overall_score': getattr(quality_metrics, 'overall_score', 0),
                    'quality_clarity_score': getattr(quality_metrics, 'clarity_score', 0),
                    'quality_conciseness_score': getattr(quality_metrics, 'conciseness_score', 0)
                })
            else:
                flat.update({
                    'quality_overall_score': 0,
                    'quality_clarity_score': 0,
                    'quality_conciseness_score': 0
                })
        
        # Flatten DeepEval scores
        if result.deepeval_scores:
            flat.update({
                'deepeval_overall_score': result.deepeval_scores.get('overall', {}).get('score', 0),
                'deepeval_relevancy_score': result.deepeval_scores.get('relevancy', {}).get('score', 0),
                'deepeval_correctness_score': result.deepeval_scores.get('correctness', {}).get('score', 0)
            })
        
        # Flatten sentiment analysis
        if result.sentiment_analysis:
            flat.update({
                'sentiment_label': result.sentiment_analysis.get('sentiment_label', ''),
                'sentiment_polarity': result.sentiment_analysis.get('polarity', 0),
                'emotional_tone': result.sentiment_analysis.get('emotional_tone', '')
            })
        
        # Flatten topic analysis
        if result.topic_analysis:
            flat.update({
                'main_topic': result.topic_analysis.get('main_topic', ''),
                'concern_category': result.topic_analysis.get('concern_category', '')
            })
        
        # Flatten AHT impact
        if result.aht_impact:
            flat.update({
                'aht_impact_level': result.aht_impact.get('aht_impact_level', ''),
                'aht_impact_score': result.aht_impact.get('aht_impact_score', 0)
            })
        
        return flat
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations."""
        if not self.evaluation_results:
            return {}
        
        summary = {
            'total_evaluations': len(self.evaluation_results),
            'unique_csrs': len(set(r.csr_id for r in self.evaluation_results)),
            'unique_calls': len(set(r.call_id for r in self.evaluation_results)),
            'avg_processing_time': sum(r.processing_time_seconds for r in self.evaluation_results) / len(self.evaluation_results),
            'evaluation_period': {
                'start': min(r.evaluation_timestamp for r in self.evaluation_results),
                'end': max(r.evaluation_timestamp for r in self.evaluation_results)
            }
        }
        
        # Quality score statistics
        quality_scores = []
        for r in self.evaluation_results:
            if r.quality_scores and r.quality_scores.get('quality_metrics'):
                quality_metrics = r.quality_scores.get('quality_metrics')
                if hasattr(quality_metrics, 'overall_score'):
                    quality_scores.append(quality_metrics.overall_score)
        if quality_scores:
            summary['quality_stats'] = {
                'avg_score': sum(quality_scores) / len(quality_scores),
                'min_score': min(quality_scores),
                'max_score': max(quality_scores)
            }
        
        # DeepEval score statistics
        deepeval_scores = [r.deepeval_scores.get('overall', {}).get('score', 0) 
                          for r in self.evaluation_results if r.deepeval_scores]
        if deepeval_scores:
            summary['deepeval_stats'] = {
                'avg_score': sum(deepeval_scores) / len(deepeval_scores),
                'min_score': min(deepeval_scores),
                'max_score': max(deepeval_scores)
            }
        
        return summary

def main():
    """Example usage of EvaluationOrchestrator."""
    # Configure evaluation
    config = EvaluationConfig(
        mistral_model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        enable_rag=True,
        enable_deepeval=True,
        enable_quality_analysis=True,
        enable_sentiment_analysis=True,
        output_format="json"
    )
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(config)
    
    if not orchestrator.initialize():
        print("Failed to initialize orchestrator")
        return
    
    # Evaluate sample transcript
    results = orchestrator.evaluate_transcript_file("Call Transcript Sample 1.json")
    
    if results:
        print(f"Evaluated {len(results)} conversations")
        
        # Show first result
        first_result = results[0]
        print(f"\nSample Result for {first_result.csr_id}:")
        print(f"Question: {first_result.question[:100]}...")
        print(f"Answer: {first_result.answer[:100]}...")
        print(f"\nConcise Summary:\n{first_result.concise_summary}")
        
        # Save results
        orchestrator.save_results("evaluation_results.json", "json")
        orchestrator.save_results("evaluation_results.xlsx", "excel")
        
        # Show summary
        summary = orchestrator.get_evaluation_summary()
        print(f"\nEvaluation Summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
