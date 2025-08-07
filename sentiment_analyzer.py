#!/usr/bin/env python3
"""
Sentiment Analysis and Topic Summarization for Call Center Transcripts
Analyzes sentiment and identifies topics/themes with coaching feedback
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import json

try:
    from textblob import TextBlob
except ImportError:
    print("Warning: TextBlob not installed. Install with: pip install textblob")
    TextBlob = None

from mistral_model import MistralEvaluator, MistralConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results."""
    polarity: float = 0.0  # -1 (negative) to 1 (positive)
    subjectivity: float = 0.0  # 0 (objective) to 1 (subjective)
    sentiment_label: str = "Neutral"
    confidence: float = 0.0
    emotional_tone: str = "Professional"
    coaching_feedback: str = ""

@dataclass
class TopicResult:
    """Data class for topic analysis results."""
    main_topic: str = ""
    concern_category: str = ""
    resolution_approach: str = ""
    key_points: List[str] = None
    topic_confidence: float = 0.0
    
    def __post_init__(self):
        if self.key_points is None:
            self.key_points = []

class TraditionalSentimentAnalyzer:
    """Traditional sentiment analysis using TextBlob and rule-based approaches."""
    
    def __init__(self):
        # Positive sentiment indicators
        self.positive_indicators = {
            'words': ['happy', 'pleased', 'satisfied', 'excellent', 'great', 'wonderful', 
                     'fantastic', 'amazing', 'perfect', 'outstanding', 'delighted'],
            'phrases': ['thank you', 'appreciate', 'glad to help', 'my pleasure', 
                       'absolutely', 'certainly', 'of course', 'no problem']
        }
        
        # Negative sentiment indicators
        self.negative_indicators = {
            'words': ['sorry', 'apologize', 'unfortunately', 'problem', 'issue', 'trouble',
                     'difficult', 'frustrated', 'upset', 'angry', 'disappointed'],
            'phrases': ['i apologize', 'i understand your frustration', 'this is unacceptable',
                       'i see the problem', 'let me fix this']
        }
        
        # Professional tone indicators
        self.professional_indicators = [
            'may i', 'could you please', 'i would be happy to', 'let me help you',
            'i understand', 'certainly', 'absolutely', 'of course'
        ]
        
        # Emotional tone patterns
        self.emotional_patterns = {
            'empathetic': ['understand', 'feel', 'appreciate', 'realize'],
            'apologetic': ['sorry', 'apologize', 'regret', 'unfortunate'],
            'confident': ['certainly', 'absolutely', 'definitely', 'sure'],
            'helpful': ['help', 'assist', 'support', 'resolve'],
            'patient': ['take your time', 'no rush', 'whenever you\'re ready']
        }
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using TextBlob and rule-based approach."""
        text_lower = text.lower()
        
        # TextBlob analysis
        if TextBlob is not None:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            # Fallback rule-based analysis
            polarity = self._calculate_rule_based_polarity(text_lower)
            subjectivity = self._calculate_subjectivity(text_lower)
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment_label = "Positive"
        elif polarity < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Calculate confidence
        confidence = min(abs(polarity) * 2, 1.0)
        
        # Determine emotional tone
        emotional_tone = self._analyze_emotional_tone(text_lower)
        
        # Generate coaching feedback
        coaching_feedback = self._generate_sentiment_coaching(
            polarity, sentiment_label, emotional_tone, text_lower
        )
        
        return SentimentResult(
            polarity=round(polarity, 3),
            subjectivity=round(subjectivity, 3),
            sentiment_label=sentiment_label,
            confidence=round(confidence, 3),
            emotional_tone=emotional_tone,
            coaching_feedback=coaching_feedback
        )
    
    def _calculate_rule_based_polarity(self, text: str) -> float:
        """Calculate polarity using rule-based approach."""
        positive_score = 0
        negative_score = 0
        
        # Count positive indicators
        for word in self.positive_indicators['words']:
            positive_score += text.count(word)
        for phrase in self.positive_indicators['phrases']:
            positive_score += text.count(phrase) * 2  # Phrases weighted more
        
        # Count negative indicators
        for word in self.negative_indicators['words']:
            negative_score += text.count(word)
        for phrase in self.negative_indicators['phrases']:
            negative_score += text.count(phrase) * 2
        
        # Calculate polarity
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        polarity = (positive_score - negative_score) / max(total_words, 1)
        return max(-1.0, min(1.0, polarity * 10))  # Scale and clamp
    
    def _calculate_subjectivity(self, text: str) -> float:
        """Calculate subjectivity score."""
        subjective_words = ['feel', 'think', 'believe', 'opinion', 'personally', 
                           'seems', 'appears', 'probably', 'maybe']
        
        subjective_count = sum(text.count(word) for word in subjective_words)
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        return min(subjective_count / total_words * 5, 1.0)
    
    def _analyze_emotional_tone(self, text: str) -> str:
        """Analyze emotional tone of the text."""
        tone_scores = {}
        
        for tone, words in self.emotional_patterns.items():
            score = sum(text.count(word) for word in words)
            if score > 0:
                tone_scores[tone] = score
        
        # Check for professional indicators
        professional_score = sum(text.count(phrase) for phrase in self.professional_indicators)
        if professional_score > 0:
            tone_scores['professional'] = professional_score
        
        if not tone_scores:
            return "Neutral"
        
        # Return the tone with highest score
        dominant_tone = max(tone_scores, key=tone_scores.get)
        return dominant_tone.capitalize()
    
    def _generate_sentiment_coaching(self, polarity: float, sentiment_label: str, 
                                   emotional_tone: str, text: str) -> str:
        """Generate coaching feedback based on sentiment analysis."""
        feedback_parts = []
        
        # Sentiment feedback
        if sentiment_label == "Positive":
            if polarity > 0.5:
                feedback_parts.append("Excellent! You maintained a very positive and upbeat tone throughout your response.")
            else:
                feedback_parts.append("Good job maintaining a positive tone in your response.")
        elif sentiment_label == "Negative":
            if polarity < -0.3:
                feedback_parts.append("Your response had a notably negative tone. Consider using more positive language even when addressing problems.")
            else:
                feedback_parts.append("Your response had a slightly negative tone. Try to balance problem acknowledgment with positive solutions.")
        else:
            feedback_parts.append("You maintained a neutral, professional tone in your response.")
        
        # Emotional tone feedback
        tone_feedback = {
            'Empathetic': "Great job showing empathy and understanding for the customer's situation.",
            'Apologetic': "You appropriately acknowledged the issue. Balance apologies with solution-focused language.",
            'Confident': "Your confident tone helps build customer trust in your ability to help.",
            'Helpful': "Your helpful attitude comes through clearly in your response.",
            'Patient': "Your patient approach helps create a positive customer experience.",
            'Professional': "You maintained excellent professional standards in your communication.",
            'Neutral': "Consider adding more warmth and personality to better connect with customers."
        }
        
        if emotional_tone in tone_feedback:
            feedback_parts.append(tone_feedback[emotional_tone])
        
        # Specific improvement suggestions
        if 'sorry' in text and text.count('sorry') > 2:
            feedback_parts.append("Avoid over-apologizing - focus on solutions after acknowledging the issue.")
        
        if any(phrase in text for phrase in self.professional_indicators):
            feedback_parts.append("Excellent use of polite, professional language.")
        
        return " ".join(feedback_parts)

class MistralTopicAnalyzer:
    """Topic analysis using Mistral model."""
    
    def __init__(self, mistral_evaluator: MistralEvaluator):
        self.mistral_evaluator = mistral_evaluator
        
        # Common call center categories
        self.topic_categories = {
            'billing': ['payment', 'bill', 'charge', 'refund', 'cost', 'price', 'invoice'],
            'technical': ['system', 'website', 'app', 'login', 'password', 'error', 'bug'],
            'reservation': ['booking', 'reservation', 'trip', 'flight', 'hotel', 'cancel'],
            'account': ['account', 'profile', 'information', 'update', 'change'],
            'complaint': ['complaint', 'dissatisfied', 'unhappy', 'problem', 'issue'],
            'inquiry': ['question', 'information', 'help', 'assistance', 'clarification']
        }
    
    def analyze_topic(self, question: str, answer: str) -> TopicResult:
        """Analyze topic and theme of the conversation."""
        # Use Mistral for detailed topic analysis
        mistral_result = self.mistral_evaluator.summarize_topic(question, answer)
        
        # Extract structured information
        main_topic = mistral_result.get('main_topic', self._extract_basic_topic(question))
        concern_category = mistral_result.get('concern_category', self._categorize_concern(question))
        resolution_approach = mistral_result.get('resolution_approach', 'Standard response')
        key_points_str = mistral_result.get('key_points', '')
        
        # Parse key points
        key_points = []
        if key_points_str:
            # Split by common delimiters
            points = re.split(r'[,;•\-\n]', key_points_str)
            key_points = [point.strip() for point in points if point.strip()]
        
        # Calculate confidence based on keyword matching
        topic_confidence = self._calculate_topic_confidence(question, main_topic)
        
        return TopicResult(
            main_topic=main_topic,
            concern_category=concern_category,
            resolution_approach=resolution_approach,
            key_points=key_points,
            topic_confidence=topic_confidence
        )
    
    def _extract_basic_topic(self, question: str) -> str:
        """Extract basic topic using keyword matching."""
        question_lower = question.lower()
        
        for category, keywords in self.topic_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category.capitalize()
        
        return "General Inquiry"
    
    def _categorize_concern(self, question: str) -> str:
        """Categorize the type of customer concern."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['cancel', 'refund', 'money back']):
            return "Cancellation/Refund"
        elif any(word in question_lower for word in ['problem', 'issue', 'wrong', 'error']):
            return "Problem Resolution"
        elif any(word in question_lower for word in ['how', 'what', 'when', 'where']):
            return "Information Request"
        elif any(word in question_lower for word in ['change', 'update', 'modify']):
            return "Account/Booking Modification"
        else:
            return "General Support"
    
    def _calculate_topic_confidence(self, question: str, identified_topic: str) -> float:
        """Calculate confidence in topic identification."""
        question_lower = question.lower()
        topic_lower = identified_topic.lower()
        
        # Check if topic keywords appear in question
        if topic_lower in self.topic_categories:
            keywords = self.topic_categories[topic_lower]
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            confidence = min(matches / len(keywords) * 2, 1.0)
        else:
            # Basic confidence based on topic word presence
            confidence = 0.7 if any(word in question_lower for word in topic_lower.split()) else 0.5
        
        return round(confidence, 3)

class SentimentTopicAnalyzer:
    """Combined sentiment and topic analysis engine."""
    
    def __init__(self, mistral_evaluator: MistralEvaluator = None):
        self.sentiment_analyzer = TraditionalSentimentAnalyzer()
        
        if mistral_evaluator:
            self.topic_analyzer = MistralTopicAnalyzer(mistral_evaluator)
        else:
            self.topic_analyzer = None
            logger.warning("Mistral evaluator not provided - topic analysis will be limited")
    
    def analyze_conversation(self, question: str, answer: str, 
                           metadata: Dict = None) -> Dict[str, Any]:
        """Perform comprehensive sentiment and topic analysis."""
        results = {
            'question': question,
            'answer': answer,
            'metadata': metadata or {},
            'analysis_timestamp': pd.Timestamp.now().isoformat() if 'pd' in globals() else None
        }
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(answer)
        results['sentiment'] = {
            'polarity': sentiment_result.polarity,
            'subjectivity': sentiment_result.subjectivity,
            'sentiment_label': sentiment_result.sentiment_label,
            'confidence': sentiment_result.confidence,
            'emotional_tone': sentiment_result.emotional_tone,
            'coaching_feedback': sentiment_result.coaching_feedback
        }
        
        # Topic analysis
        if self.topic_analyzer:
            topic_result = self.topic_analyzer.analyze_topic(question, answer)
            results['topic'] = {
                'main_topic': topic_result.main_topic,
                'concern_category': topic_result.concern_category,
                'resolution_approach': topic_result.resolution_approach,
                'key_points': topic_result.key_points,
                'topic_confidence': topic_result.topic_confidence
            }
        else:
            # Basic topic analysis fallback
            results['topic'] = {
                'main_topic': self._basic_topic_extraction(question),
                'concern_category': 'General',
                'resolution_approach': 'Standard response',
                'key_points': [],
                'topic_confidence': 0.5
            }
        
        # Generate combined coaching feedback
        results['combined_coaching'] = self._generate_combined_coaching(
            results['sentiment'], results['topic']
        )
        
        return results
    
    def _basic_topic_extraction(self, question: str) -> str:
        """Basic topic extraction without Mistral."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['reservation', 'booking', 'trip']):
            return "Reservation"
        elif any(word in question_lower for word in ['payment', 'bill', 'refund']):
            return "Billing"
        elif any(word in question_lower for word in ['problem', 'issue', 'error']):
            return "Technical Support"
        else:
            return "General Inquiry"
    
    def _generate_combined_coaching(self, sentiment: Dict, topic: Dict) -> str:
        """Generate combined coaching feedback."""
        feedback_parts = []
        
        # Topic-specific feedback
        topic_name = topic['main_topic']
        if topic_name == "Billing":
            feedback_parts.append("For billing inquiries, ensure you provide clear, accurate information and offer specific next steps.")
        elif topic_name == "Reservation":
            feedback_parts.append("For reservation issues, focus on quick resolution and alternative options when possible.")
        elif topic_name == "Technical Support":
            feedback_parts.append("For technical issues, provide step-by-step guidance and confirm customer understanding.")
        
        # Sentiment-topic combination feedback
        sentiment_label = sentiment['sentiment_label']
        if sentiment_label == "Negative" and topic['concern_category'] == "Problem Resolution":
            feedback_parts.append("When addressing problems, balance acknowledgment with positive, solution-focused language.")
        elif sentiment_label == "Positive" and topic['concern_category'] == "Information Request":
            feedback_parts.append("Excellent positive tone for information requests - this builds customer confidence.")
        
        return " ".join(feedback_parts) if feedback_parts else "Continue maintaining professional communication standards."

def main():
    """Example usage of SentimentTopicAnalyzer."""
    from mistral_model import MistralEvaluator, MistralConfig
    
    # Initialize with Mistral model
    mistral_config = MistralConfig(model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    mistral_evaluator = MistralEvaluator(mistral_config)
    
    analyzer = SentimentTopicAnalyzer(mistral_evaluator)
    
    # Test conversation
    question = "I need help with a reservation I made last week. This is unacceptable service!"
    answer = "I apologize for the trouble. May I have your name and reservation number to look up your booking?"
    
    print("Analyzing conversation...")
    
    # Load Mistral model if available
    if mistral_evaluator.load_model():
        print("Using Mistral model for enhanced analysis")
    else:
        print("Using traditional analysis methods")
    
    result = analyzer.analyze_conversation(question, answer)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    print(f"\nSentiment Analysis:")
    sentiment = result['sentiment']
    print(f"  Sentiment: {sentiment['sentiment_label']} (polarity: {sentiment['polarity']})")
    print(f"  Emotional Tone: {sentiment['emotional_tone']}")
    print(f"  Coaching: {sentiment['coaching_feedback']}")
    
    print(f"\nTopic Analysis:")
    topic = result['topic']
    print(f"  Main Topic: {topic['main_topic']}")
    print(f"  Concern Category: {topic['concern_category']}")
    print(f"  Resolution Approach: {topic['resolution_approach']}")
    print(f"  Key Points: {', '.join(topic['key_points'])}")
    
    print(f"\nCombined Coaching:")
    print(f"  {result['combined_coaching']}")

if __name__ == "__main__":
    main()

