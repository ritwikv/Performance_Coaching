#!/usr/bin/env python3
"""
Call Center Transcript Evaluator using Mistral 7B Model
========================================================

This script evaluates call center transcripts using a local Mistral 7B model to:
1. Extract and structure data from JSON transcripts
2. Evaluate English language correctness
3. Identify long sentences and recommend improvements
4. Detect word repetition and crutch words
5. Create knowledge documents
6. Perform RAGAS evaluation (commented for later use)
7. Analyze sentiment
8. Summarize topics/themes

Author: AI Assistant
Date: 2024
"""

import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Any
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For local Mistral model
try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")

# For sentiment analysis
try:
    from textblob import TextBlob
except ImportError:
    print("Warning: textblob not installed. Install with: pip install textblob")

# For RAGAS evaluation (commented out for later use)
# from ragas import evaluate
# from ragas.metrics import (
#     context_precision,
#     context_recall, 
#     context_entity_recall,
#     answer_relevancy,
#     faithfulness
# )

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CallCenterTranscriptEvaluator:
    """Main class for evaluating call center transcripts using Mistral 7B model."""
    
    def __init__(self, model_path: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        """
        Initialize the evaluator with Mistral model.
        
        Args:
            model_path (str): Path to the Mistral GGUF model file
        """
        self.model_path = model_path
        self.model = None
        self.knowledge_documents = []
        self.crutch_words = [
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'I mean', 'right?', 'okay?', 'so', 'well'
        ]
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the Mistral 7B model."""
        try:
            logger.info(f"Loading Mistral model from {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window
                n_threads=4,  # Number of CPU threads
                verbose=False
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def load_transcript_json(self, json_file_path: str) -> Dict:
        """
        Load transcript data from JSON file.
        
        Args:
            json_file_path (str): Path to the JSON file
            
        Returns:
            Dict: Loaded JSON data
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            logger.info(f"Successfully loaded transcript from {json_file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return {}
    
    def extract_qa_pairs(self, transcript_data: Dict) -> pd.DataFrame:
        """
        Extract Questions and Answers from transcript data and create DataFrame.
        
        Args:
            transcript_data (Dict): Raw transcript data from JSON
            
        Returns:
            pd.DataFrame: Structured data with Questions, Answers, and metadata
        """
        if not transcript_data or 'call_transcript' not in transcript_data:
            logger.error("Invalid transcript data structure")
            return pd.DataFrame()
        
        # Extract metadata
        metadata = {
            'call_ID': transcript_data.get('call_ID', ''),
            'CSR_ID': transcript_data.get('CSR_ID', ''),
            'call_date': transcript_data.get('call_date', ''),
            'call_time': transcript_data.get('call_time', '')
        }
        
        # Extract transcript lines
        transcript_lines = transcript_data['call_transcript']
        
        questions = []
        answers = []
        
        for line in transcript_lines:
            line = line.strip()
            if line.startswith('Customer:'):
                questions.append(line.replace('Customer:', '').strip())
            elif line.startswith('CSR:') or line.startswith('Supervisor:'):
                # Handle both CSR and Supervisor responses as answers
                prefix = 'CSR:' if line.startswith('CSR:') else 'Supervisor:'
                answers.append(line.replace(prefix, '').strip())
        
        # Ensure equal length by padding shorter list
        max_len = max(len(questions), len(answers))
        questions.extend([''] * (max_len - len(questions)))
        answers.extend([''] * (max_len - len(answers)))
        
        # Create DataFrame
        df_data = {
            'call_ID': [metadata['call_ID']] * max_len,
            'CSR_ID': [metadata['CSR_ID']] * max_len,
            'call_date': [metadata['call_date']] * max_len,
            'call_time': [metadata['call_time']] * max_len,
            'Questions': questions,
            'Answers': answers
        }
        
        df = pd.DataFrame(df_data)
        logger.info(f"Extracted {len(df)} Q&A pairs from transcript")
        return df
    
    def query_mistral(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Query the Mistral model with a prompt.
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Model response
        """
        if not self.model:
            return "Model not loaded"
        
        try:
            # Format prompt for Mistral Instruct
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return f"Error: {e}"
    
    def evaluate_english_correctness(self, text: str) -> Dict[str, Any]:
        """
        Evaluate English language correctness and provide coaching.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            Dict: Evaluation results and coaching feedback
        """
        prompt = f"""
        Please evaluate the following text for English language correctness and provide coaching feedback:
        
        Text: "{text}"
        
        Please provide:
        1. Grammar errors (if any)
        2. Spelling mistakes (if any)
        3. Suggestions for improvement
        4. Overall correctness score (1-10)
        5. Coaching advice in a friendly, constructive manner
        
        Format your response as a structured evaluation.
        """
        
        response = self.query_mistral(prompt, max_tokens=400)
        
        return {
            'original_text': text,
            'evaluation': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_sentence_length(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentence length and recommend crisp sentences.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Analysis results and recommendations
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        long_sentences = []
        word_counts = []
        
        for sentence in sentences:
            word_count = len(sentence.split())
            word_counts.append(word_count)
            if word_count > 20:  # Consider sentences with >20 words as long
                long_sentences.append({
                    'sentence': sentence,
                    'word_count': word_count
                })
        
        avg_words = np.mean(word_counts) if word_counts else 0
        
        # Generate recommendations using Mistral
        if long_sentences:
            long_text = "\n".join([f"- {s['sentence']} ({s['word_count']} words)" 
                                 for s in long_sentences])
            prompt = f"""
            The following sentences are too long and need to be made more crisp and concise:
            
            {long_text}
            
            Please provide:
            1. Shorter, clearer alternatives for each sentence
            2. General tips for writing crisp sentences
            3. Coaching advice for the agent
            """
            
            recommendations = self.query_mistral(prompt, max_tokens=500)
        else:
            recommendations = "Great job! All sentences are appropriately concise."
        
        return {
            'total_sentences': len(sentences),
            'average_words_per_sentence': round(avg_words, 2),
            'long_sentences_count': len(long_sentences),
            'long_sentences': long_sentences,
            'recommendations': recommendations
        }
    
    def detect_repetition_and_crutch_words(self, text: str) -> Dict[str, Any]:
        """
        Detect word repetition and crutch words usage.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Detection results and feedback
        """
        # Convert to lowercase for analysis
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find repeated words (appearing more than 3 times)
        repeated_words = {word: count for word, count in word_freq.items() 
                         if count > 3 and len(word) > 3}
        
        # Find crutch words
        found_crutch_words = {}
        for crutch in self.crutch_words:
            crutch_lower = crutch.lower()
            if crutch_lower in text_lower:
                count = text_lower.count(crutch_lower)
                if count > 0:
                    found_crutch_words[crutch] = count
        
        # Generate coaching feedback
        prompt = f"""
        Analyze this text for word repetition and filler words:
        
        Text: "{text}"
        
        Repeated words found: {repeated_words}
        Crutch/filler words found: {found_crutch_words}
        
        Please provide coaching feedback on:
        1. How to reduce word repetition
        2. Alternatives to crutch words
        3. Tips for more confident speech
        4. Specific suggestions for improvement
        """
        
        coaching = self.query_mistral(prompt, max_tokens=400)
        
        return {
            'repeated_words': repeated_words,
            'crutch_words': found_crutch_words,
            'total_words': len(words),
            'unique_words': len(set(words)),
            'vocabulary_diversity': round(len(set(words)) / len(words), 3) if words else 0,
            'coaching_feedback': coaching
        }
    
    def create_knowledge_documents(self, df: pd.DataFrame) -> List[str]:
        """
        Create knowledge documents from the transcripts.
        
        Args:
            df (pd.DataFrame): DataFrame with Q&A pairs
            
        Returns:
            List[str]: Generated knowledge documents
        """
        knowledge_docs = []
        
        # Combine all answers to create comprehensive knowledge
        all_answers = df['Answers'].dropna().tolist()
        combined_text = " ".join(all_answers)
        
        prompt = f"""
        Based on the following call center conversation responses, create comprehensive knowledge documents that capture:
        1. Key policies and procedures mentioned
        2. Common customer issues and their solutions
        3. Best practices for customer service
        4. Standard responses and protocols
        
        Call center responses:
        {combined_text}
        
        Please create 3-5 distinct knowledge documents, each focusing on a specific aspect of customer service.
        Format each document clearly with a title and content.
        """
        
        response = self.query_mistral(prompt, max_tokens=800)
        
        # Split response into individual documents
        docs = re.split(r'\n(?=\d+\.|\w+:)', response)
        knowledge_docs = [doc.strip() for doc in docs if doc.strip()]
        
        self.knowledge_documents = knowledge_docs
        logger.info(f"Created {len(knowledge_docs)} knowledge documents")
        
        return knowledge_docs
    
    # RAGAS EVALUATION SECTION - COMMENTED FOR LATER USE
    """
    def evaluate_with_ragas(self, df: pd.DataFrame) -> Dict[str, Any]:
        '''
        Evaluate answers using RAGAS framework with local Mistral model.
        
        Args:
            df (pd.DataFrame): DataFrame with Q&A pairs
            
        Returns:
            Dict: RAGAS evaluation results
        '''
        if not self.knowledge_documents:
            logger.warning("No knowledge documents available for RAGAS evaluation")
            return {}
        
        # Prepare data for RAGAS evaluation
        questions = df['Questions'].dropna().tolist()
        answers = df['Answers'].dropna().tolist()
        contexts = [self.knowledge_documents] * len(questions)  # Use knowledge docs as context
        
        # Create dataset for RAGAS
        dataset = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truths': answers  # Using answers as ground truth for this example
        }
        
        # Configure RAGAS to use local Mistral model
        # Note: This would require custom configuration to use local model instead of OpenAI
        # The exact implementation depends on RAGAS version and local model integration
        
        try:
            # Evaluate using RAGAS metrics
            result = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    context_entity_recall,
                    answer_relevancy,
                    faithfulness
                ]
            )
            
            # Generate coaching feedback for each metric
            coaching_feedback = self._generate_ragas_coaching(result)
            
            return {
                'ragas_scores': result,
                'coaching_feedback': coaching_feedback
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {'error': str(e)}
    
    def _generate_ragas_coaching(self, ragas_results: Dict) -> Dict[str, str]:
        '''
        Generate coaching feedback based on RAGAS scores.
        
        Args:
            ragas_results (Dict): RAGAS evaluation results
            
        Returns:
            Dict: Coaching feedback for each metric
        '''
        coaching = {}
        
        for metric, score in ragas_results.items():
            if isinstance(score, (int, float)):
                prompt = f'''
                Based on a {metric} score of {score:.2f} (scale 0-1), provide coaching feedback.
                Explain what this score means and give specific advice for improvement.
                Make it encouraging and actionable.
                '''
                
                feedback = self.query_mistral(prompt, max_tokens=200)
                coaching[metric] = feedback
        
        return coaching
    """
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of the text and provide feedback.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results and feedback
        """
        try:
            # Use TextBlob for basic sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = "Positive"
            elif polarity < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            # Generate coaching feedback using Mistral
            prompt = f"""
            Analyze the sentiment of this customer service response:
            
            Text: "{text}"
            
            Sentiment Analysis:
            - Polarity: {polarity:.2f} (range: -1 to 1)
            - Subjectivity: {subjectivity:.2f} (range: 0 to 1)
            - Classification: {sentiment_label}
            
            Please provide coaching feedback in a sentence format about the agent's emotional tone and demeanor.
            For example: "You were very happy and positive in your response" or "Your tone seemed frustrated, try to maintain calm professionalism"
            """
            
            coaching = self.query_mistral(prompt, max_tokens=200)
            
            return {
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'sentiment_label': sentiment_label,
                'coaching_feedback': coaching
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'error': str(e),
                'coaching_feedback': "Unable to analyze sentiment at this time."
            }
    
    def summarize_topic_theme(self, question: str, answer: str) -> str:
        """
        Summarize the topic or theme of the question-answer pair.
        
        Args:
            question (str): Customer question
            answer (str): Agent answer
            
        Returns:
            str: Topic/theme summary
        """
        prompt = f"""
        Analyze the following customer service interaction and summarize the main topic or theme:
        
        Customer Question: "{question}"
        Agent Response: "{answer}"
        
        Please provide:
        1. The main topic/theme (e.g., "Flight Cancellation", "Refund Request", "Booking Issue")
        2. A brief summary of what the interaction was about
        3. The type of customer service issue this represents
        
        Keep the response concise and focused.
        """
        
        return self.query_mistral(prompt, max_tokens=150)
    
    def evaluate_transcript(self, json_file_path: str) -> Dict[str, Any]:
        """
        Main method to evaluate a complete transcript.
        
        Args:
            json_file_path (str): Path to the JSON transcript file
            
        Returns:
            Dict: Complete evaluation results
        """
        logger.info(f"Starting evaluation of {json_file_path}")
        
        # Load and structure data
        transcript_data = self.load_transcript_json(json_file_path)
        if not transcript_data:
            return {'error': 'Failed to load transcript data'}
        
        df = self.extract_qa_pairs(transcript_data)
        if df.empty:
            return {'error': 'Failed to extract Q&A pairs'}
        
        # Create knowledge documents
        knowledge_docs = self.create_knowledge_documents(df)
        
        # Evaluate each answer
        evaluations = []
        
        for idx, row in df.iterrows():
            if not row['Answers']:  # Skip empty answers
                continue
                
            question = row['Questions']
            answer = row['Answers']
            
            logger.info(f"Evaluating Q&A pair {idx + 1}")
            
            # Perform all evaluations
            english_eval = self.evaluate_english_correctness(answer)
            sentence_analysis = self.analyze_sentence_length(answer)
            repetition_analysis = self.detect_repetition_and_crutch_words(answer)
            sentiment_analysis = self.analyze_sentiment(answer)
            topic_summary = self.summarize_topic_theme(question, answer)
            
            evaluation = {
                'qa_pair_index': idx,
                'call_ID': row['call_ID'],
                'CSR_ID': row['CSR_ID'],
                'question': question,
                'answer': answer,
                'english_correctness': english_eval,
                'sentence_analysis': sentence_analysis,
                'repetition_analysis': repetition_analysis,
                'sentiment_analysis': sentiment_analysis,
                'topic_summary': topic_summary
            }
            
            evaluations.append(evaluation)
        
        # Compile final results
        results = {
            'transcript_metadata': {
                'call_ID': transcript_data.get('call_ID'),
                'CSR_ID': transcript_data.get('CSR_ID'),
                'call_date': transcript_data.get('call_date'),
                'call_time': transcript_data.get('call_time')
            },
            'qa_dataframe': df,
            'knowledge_documents': knowledge_docs,
            'evaluations': evaluations,
            'summary_stats': {
                'total_qa_pairs': len(evaluations),
                'evaluation_timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info("Evaluation completed successfully")
        return results
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results (Dict): Results from evaluate_transcript
            output_file (str): Optional file path to save the report
            
        Returns:
            str: Formatted report
        """
        if 'error' in evaluation_results:
            return f"Error in evaluation: {evaluation_results['error']}"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CALL CENTER TRANSCRIPT EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        # Metadata
        metadata = evaluation_results['transcript_metadata']
        report_lines.append(f"\nCall ID: {metadata['call_ID']}")
        report_lines.append(f"CSR ID: {metadata['CSR_ID']}")
        report_lines.append(f"Date: {metadata['call_date']}")
        report_lines.append(f"Time: {metadata['call_time']}")
        
        # Summary
        stats = evaluation_results['summary_stats']
        report_lines.append(f"\nTotal Q&A Pairs Evaluated: {stats['total_qa_pairs']}")
        report_lines.append(f"Evaluation Timestamp: {stats['evaluation_timestamp']}")
        
        # Knowledge Documents
        report_lines.append(f"\n\nKNOWLEDGE DOCUMENTS CREATED:")
        report_lines.append("-" * 40)
        for i, doc in enumerate(evaluation_results['knowledge_documents'], 1):
            report_lines.append(f"\nDocument {i}:")
            report_lines.append(doc[:500] + "..." if len(doc) > 500 else doc)
        
        # Individual Evaluations
        report_lines.append(f"\n\nDETAILED EVALUATIONS:")
        report_lines.append("=" * 50)
        
        for eval_data in evaluation_results['evaluations']:
            report_lines.append(f"\n--- Q&A Pair {eval_data['qa_pair_index'] + 1} ---")
            report_lines.append(f"Question: {eval_data['question'][:200]}...")
            report_lines.append(f"Answer: {eval_data['answer'][:200]}...")
            
            # English Correctness
            report_lines.append(f"\n📝 English Correctness Evaluation:")
            report_lines.append(eval_data['english_correctness']['evaluation'])
            
            # Sentence Analysis
            sent_analysis = eval_data['sentence_analysis']
            report_lines.append(f"\n📏 Sentence Length Analysis:")
            report_lines.append(f"- Average words per sentence: {sent_analysis['average_words_per_sentence']}")
            report_lines.append(f"- Long sentences found: {sent_analysis['long_sentences_count']}")
            report_lines.append(f"- Recommendations: {sent_analysis['recommendations'][:300]}...")
            
            # Repetition Analysis
            rep_analysis = eval_data['repetition_analysis']
            report_lines.append(f"\n🔄 Word Repetition & Crutch Words:")
            report_lines.append(f"- Vocabulary diversity: {rep_analysis['vocabulary_diversity']}")
            report_lines.append(f"- Repeated words: {rep_analysis['repeated_words']}")
            report_lines.append(f"- Crutch words: {rep_analysis['crutch_words']}")
            report_lines.append(f"- Coaching: {rep_analysis['coaching_feedback'][:300]}...")
            
            # Sentiment Analysis
            sentiment = eval_data['sentiment_analysis']
            report_lines.append(f"\n😊 Sentiment Analysis:")
            report_lines.append(f"- Sentiment: {sentiment.get('sentiment_label', 'N/A')}")
            report_lines.append(f"- Polarity: {sentiment.get('polarity', 'N/A')}")
            report_lines.append(f"- Coaching: {sentiment.get('coaching_feedback', 'N/A')}")
            
            # Topic Summary
            report_lines.append(f"\n🎯 Topic Summary:")
            report_lines.append(eval_data['topic_summary'])
            
            report_lines.append("\n" + "-" * 50)
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report


def main():
    """Main function to demonstrate the evaluator."""
    # Initialize the evaluator
    # Note: Update the model path to match your local installation
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    evaluator = CallCenterTranscriptEvaluator(model_path)
    
    # Evaluate the sample transcript
    json_file = "Call Transcript Sample 1.json"
    
    if not Path(json_file).exists():
        print(f"Error: {json_file} not found!")
        return
    
    print("Starting transcript evaluation...")
    results = evaluator.evaluate_transcript(json_file)
    
    if 'error' in results:
        print(f"Evaluation failed: {results['error']}")
        return
    
    # Generate and display report
    report = evaluator.generate_report(results, "evaluation_report.txt")
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Evaluated {results['summary_stats']['total_qa_pairs']} Q&A pairs")
    print(f"Created {len(results['knowledge_documents'])} knowledge documents")
    print("Report saved to: evaluation_report.txt")
    
    # Display first few lines of report
    print("\nReport Preview:")
    print("-" * 40)
    print(report[:1000] + "..." if len(report) > 1000 else report)


if __name__ == "__main__":
    main()

