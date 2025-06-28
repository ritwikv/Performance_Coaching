#!/usr/bin/env python3
"""
Call Quality Analysis Engine
Analyzes call transcripts for quality metrics including sentence structure,
repetition, hold requests, and call transfers with AHT correlation
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import Counter
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Data class for quality analysis metrics."""
    clarity_score: float = 0.0
    conciseness_score: float = 0.0
    repetition_score: float = 0.0
    professionalism_score: float = 0.0
    efficiency_score: float = 0.0
    overall_score: float = 0.0
    aht_impact: str = "Unknown"
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class LinguisticAnalyzer:
    """Analyzes linguistic patterns in customer service responses."""
    
    def __init__(self):
        # Common crutch words and phrases
        self.crutch_words = {
            'filler_words': ['um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 'kind of'],
            'redundant_phrases': ['at this point in time', 'in order to', 'due to the fact that'],
            'weak_language': ['maybe', 'perhaps', 'i think', 'i guess', 'probably']
        }
        
        # Hold and wait patterns
        self.hold_patterns = [
            r'please hold',
            r'hold on',
            r'one moment',
            r'just a moment',
            r'bear with me',
            r'please wait',
            r'let me check',
            r'give me a second',
            r'hold the line'
        ]
        
        # Transfer patterns
        self.transfer_patterns = [
            r'transfer you to',
            r'connect you with',
            r'speak to a supervisor',
            r'escalate this',
            r'pass you to',
            r'forward you to',
            r'specialist.*help',
            r'manager.*assist'
        ]
        
        # Professional greeting patterns
        self.greeting_patterns = [
            r'thank you for calling',
            r'good morning',
            r'good afternoon',
            r'how may i assist',
            r'how can i help'
        ]
    
    def analyze_sentence_length(self, text: str) -> Dict[str, Any]:
        """Analyze sentence length and complexity."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {
                'avg_sentence_length': 0,
                'max_sentence_length': 0,
                'long_sentences': [],
                'clarity_score': 0,
                'recommendations': ['No sentences found to analyze']
            }
        
        word_counts = [len(sentence.split()) for sentence in sentences]
        avg_length = statistics.mean(word_counts)
        max_length = max(word_counts)
        
        # Identify long sentences (>25 words typically considered long)
        long_sentences = [
            {'sentence': sentences[i], 'word_count': word_counts[i]}
            for i in range(len(sentences))
            if word_counts[i] > 25
        ]
        
        # Calculate clarity score (inverse relationship with sentence length)
        clarity_score = max(0, min(10, 10 - (avg_length - 15) * 0.3))
        
        recommendations = []
        if avg_length > 20:
            recommendations.append("Consider breaking down long sentences into shorter, clearer statements")
        if long_sentences:
            recommendations.append(f"Found {len(long_sentences)} sentences over 25 words - aim for 15-20 words per sentence")
        if max_length > 40:
            recommendations.append("Some sentences are extremely long - prioritize conciseness")
        
        return {
            'avg_sentence_length': round(avg_length, 2),
            'max_sentence_length': max_length,
            'sentence_count': len(sentences),
            'long_sentences': long_sentences,
            'clarity_score': round(clarity_score, 2),
            'recommendations': recommendations
        }
    
    def analyze_repetition(self, text: str) -> Dict[str, Any]:
        """Analyze repetition and crutch words."""
        text_lower = text.lower()
        
        # Find crutch words
        found_crutch_words = {}
        total_crutch_count = 0
        
        for category, words in self.crutch_words.items():
            found_words = {}
            for word in words:
                count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
                if count > 0:
                    found_words[word] = count
                    total_crutch_count += count
            if found_words:
                found_crutch_words[category] = found_words
        
        # Find repeated phrases (3+ words repeated)
        words = text_lower.split()
        repeated_phrases = []
        
        for i in range(len(words) - 2):
            for length in range(3, min(8, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+length])
                # Count occurrences of this phrase
                phrase_count = len(re.findall(re.escape(phrase), text_lower))
                if phrase_count > 1:
                    repeated_phrases.append({
                        'phrase': phrase,
                        'count': phrase_count
                    })
        
        # Remove duplicates and sort by count
        unique_phrases = {}
        for item in repeated_phrases:
            phrase = item['phrase']
            if phrase not in unique_phrases or unique_phrases[phrase] < item['count']:
                unique_phrases[phrase] = item['count']
        
        repeated_phrases = [
            {'phrase': phrase, 'count': count}
            for phrase, count in sorted(unique_phrases.items(), key=lambda x: x[1], reverse=True)
        ][:5]  # Top 5 repeated phrases
        
        # Calculate repetition score
        word_count = len(words)
        repetition_penalty = (total_crutch_count + sum(p['count'] for p in repeated_phrases)) / max(word_count, 1)
        repetition_score = max(0, min(10, 10 - repetition_penalty * 20))
        
        recommendations = []
        if total_crutch_count > 0:
            recommendations.append(f"Reduce use of filler words - found {total_crutch_count} instances")
        if repeated_phrases:
            recommendations.append("Avoid unnecessary repetition of phrases")
        if repetition_score < 7:
            recommendations.append("Focus on varied language and eliminate redundant expressions")
        
        return {
            'crutch_words_found': found_crutch_words,
            'total_crutch_count': total_crutch_count,
            'repeated_phrases': repeated_phrases,
            'repetition_score': round(repetition_score, 2),
            'recommendations': recommendations
        }
    
    def analyze_hold_requests(self, text: str) -> Dict[str, Any]:
        """Analyze hold and wait requests."""
        text_lower = text.lower()
        
        hold_instances = []
        for pattern in self.hold_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                hold_instances.append({
                    'phrase': match.group(),
                    'position': match.start(),
                    'context': text[max(0, match.start()-20):match.end()+20]
                })
        
        # Analyze professionalism of hold requests
        professional_holds = 0
        for instance in hold_instances:
            phrase = instance['phrase']
            if any(word in phrase for word in ['please', 'moment', 'thank you']):
                professional_holds += 1
        
        professionalism_score = (professional_holds / max(len(hold_instances), 1)) * 10 if hold_instances else 10
        
        recommendations = []
        if len(hold_instances) > 3:
            recommendations.append("Excessive hold requests may frustrate customers - try to minimize")
        if len(hold_instances) > 0 and professional_holds < len(hold_instances):
            recommendations.append("Use more polite language when asking customers to wait")
        if len(hold_instances) == 0:
            recommendations.append("Good - no unnecessary hold requests found")
        
        return {
            'hold_requests_count': len(hold_instances),
            'hold_instances': hold_instances,
            'professional_holds': professional_holds,
            'professionalism_score': round(professionalism_score, 2),
            'recommendations': recommendations
        }
    
    def analyze_transfers(self, text: str) -> Dict[str, Any]:
        """Analyze call transfers and reasons."""
        text_lower = text.lower()
        
        transfer_instances = []
        for pattern in self.transfer_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                transfer_instances.append({
                    'phrase': match.group(),
                    'position': match.start(),
                    'context': text[max(0, match.start()-30):match.end()+30]
                })
        
        # Try to identify transfer reasons
        transfer_reasons = []
        reason_patterns = {
            'supervisor_request': r'supervisor|manager',
            'technical_issue': r'technical|system|computer',
            'billing_issue': r'billing|payment|refund',
            'specialist_needed': r'specialist|expert|department',
            'escalation': r'escalate|higher level'
        }
        
        for reason, pattern in reason_patterns.items():
            if re.search(pattern, text_lower):
                transfer_reasons.append(reason)
        
        # Calculate transfer efficiency score
        has_transfer = len(transfer_instances) > 0
        transfer_score = 8 if not has_transfer else 6  # Transfers generally reduce efficiency
        
        if has_transfer and transfer_reasons:
            transfer_score += 1  # Bonus for clear reason
        
        recommendations = []
        if has_transfer:
            if not transfer_reasons:
                recommendations.append("When transferring, clearly explain the reason to the customer")
            recommendations.append("Try to resolve issues without transfer when possible")
        else:
            recommendations.append("Good - resolved without requiring transfer")
        
        return {
            'transfer_detected': has_transfer,
            'transfer_count': len(transfer_instances),
            'transfer_instances': transfer_instances,
            'transfer_reasons': transfer_reasons,
            'transfer_score': transfer_score,
            'recommendations': recommendations
        }

class AHTCorrelationAnalyzer:
    """Analyzes correlation between quality metrics and Average Handling Time."""
    
    def __init__(self):
        # AHT impact weights for different quality issues
        self.aht_impact_weights = {
            'long_sentences': 0.3,      # Longer explanations = more time
            'repetition': 0.4,          # Repetition wastes time
            'hold_requests': 0.5,       # Holds directly add time
            'transfers': 0.8,           # Transfers significantly increase AHT
            'unclear_communication': 0.6  # Confusion leads to longer calls
        }
    
    def calculate_aht_impact(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate estimated AHT impact based on quality metrics."""
        impact_factors = []
        total_impact_score = 0
        
        # Sentence length impact
        sentence_analysis = quality_analysis.get('sentence_analysis', {})
        if sentence_analysis.get('avg_sentence_length', 0) > 20:
            impact = self.aht_impact_weights['long_sentences']
            impact_factors.append({
                'factor': 'Long sentences',
                'impact': impact,
                'description': 'Verbose explanations increase call duration'
            })
            total_impact_score += impact
        
        # Repetition impact
        repetition_analysis = quality_analysis.get('repetition_analysis', {})
        if repetition_analysis.get('total_crutch_count', 0) > 3:
            impact = self.aht_impact_weights['repetition']
            impact_factors.append({
                'factor': 'Excessive repetition',
                'impact': impact,
                'description': 'Filler words and repetition waste time'
            })
            total_impact_score += impact
        
        # Hold requests impact
        hold_analysis = quality_analysis.get('hold_analysis', {})
        hold_count = hold_analysis.get('hold_requests_count', 0)
        if hold_count > 0:
            impact = self.aht_impact_weights['hold_requests'] * min(hold_count / 3, 1)
            impact_factors.append({
                'factor': 'Hold requests',
                'impact': impact,
                'description': f'{hold_count} hold requests directly add to call time'
            })
            total_impact_score += impact
        
        # Transfer impact
        transfer_analysis = quality_analysis.get('transfer_analysis', {})
        if transfer_analysis.get('transfer_detected', False):
            impact = self.aht_impact_weights['transfers']
            impact_factors.append({
                'factor': 'Call transfer',
                'impact': impact,
                'description': 'Call transfers significantly increase handling time'
            })
            total_impact_score += impact
        
        # Clarity impact
        if sentence_analysis.get('clarity_score', 10) < 6:
            impact = self.aht_impact_weights['unclear_communication']
            impact_factors.append({
                'factor': 'Unclear communication',
                'impact': impact,
                'description': 'Poor clarity leads to confusion and longer calls'
            })
            total_impact_score += impact
        
        # Determine AHT impact level
        if total_impact_score < 0.3:
            aht_impact_level = "Low"
            aht_description = "Minimal impact on call duration"
        elif total_impact_score < 0.7:
            aht_impact_level = "Medium"
            aht_description = "Moderate impact on call duration"
        else:
            aht_impact_level = "High"
            aht_description = "Significant impact on call duration"
        
        return {
            'aht_impact_score': round(total_impact_score, 2),
            'aht_impact_level': aht_impact_level,
            'aht_description': aht_description,
            'impact_factors': impact_factors,
            'estimated_time_increase': f"{int(total_impact_score * 100)}%"
        }

class CallQualityAnalyzer:
    """Main class for comprehensive call quality analysis."""
    
    def __init__(self):
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.aht_analyzer = AHTCorrelationAnalyzer()
    
    def analyze_single_response(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis on a single CSR response."""
        analysis = {
            'text': text,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Perform individual analyses
        analysis['sentence_analysis'] = self.linguistic_analyzer.analyze_sentence_length(text)
        analysis['repetition_analysis'] = self.linguistic_analyzer.analyze_repetition(text)
        analysis['hold_analysis'] = self.linguistic_analyzer.analyze_hold_requests(text)
        analysis['transfer_analysis'] = self.linguistic_analyzer.analyze_transfers(text)
        
        # Calculate AHT impact
        analysis['aht_impact'] = self.aht_analyzer.calculate_aht_impact(analysis)
        
        # Generate overall quality metrics
        analysis['quality_metrics'] = self._calculate_quality_metrics(analysis)
        
        # Generate coaching recommendations
        analysis['coaching_recommendations'] = self._generate_coaching_recommendations(analysis)
        
        return analysis
    
    def analyze_conversation_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze entire DataFrame of conversations."""
        results = []
        
        for idx, row in df.iterrows():
            logger.info(f"Analyzing conversation {idx + 1}/{len(df)}")
            
            # Analyze the CSR answer
            analysis = self.analyze_single_response(row['answer'])
            
            # Add metadata
            result = {
                'call_ID': row.get('call_ID', ''),
                'CSR_ID': row.get('CSR_ID', ''),
                'call_date': row.get('call_date', ''),
                'call_time': row.get('call_time', ''),
                'interaction_sequence': row.get('interaction_sequence', 0),
                'question': row.get('question', ''),
                'answer': row.get('answer', ''),
                **analysis
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _calculate_quality_metrics(self, analysis: Dict[str, Any]) -> QualityMetrics:
        """Calculate overall quality metrics."""
        sentence_score = analysis['sentence_analysis'].get('clarity_score', 0)
        repetition_score = analysis['repetition_analysis'].get('repetition_score', 0)
        hold_score = analysis['hold_analysis'].get('professionalism_score', 0)
        transfer_score = analysis['transfer_analysis'].get('transfer_score', 0)
        
        # Calculate weighted overall score
        overall_score = (
            sentence_score * 0.3 +
            repetition_score * 0.3 +
            hold_score * 0.2 +
            transfer_score * 0.2
        )
        
        aht_impact = analysis['aht_impact'].get('aht_impact_level', 'Unknown')
        
        return QualityMetrics(
            clarity_score=sentence_score,
            conciseness_score=repetition_score,
            professionalism_score=hold_score,
            efficiency_score=transfer_score,
            overall_score=round(overall_score, 2),
            aht_impact=aht_impact
        )
    
    def _generate_coaching_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized coaching recommendations."""
        recommendations = []
        
        # Collect all recommendations from individual analyses
        for analysis_type in ['sentence_analysis', 'repetition_analysis', 'hold_analysis', 'transfer_analysis']:
            if analysis_type in analysis:
                recs = analysis[analysis_type].get('recommendations', [])
                recommendations.extend(recs)
        
        # Add AHT-specific recommendations
        aht_impact = analysis.get('aht_impact', {})
        if aht_impact.get('aht_impact_level') == 'High':
            recommendations.append("Focus on efficiency - current approach significantly increases call duration")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations

def main():
    """Example usage of CallQualityAnalyzer."""
    from data_processor import CallTranscriptProcessor
    
    # Initialize analyzer
    analyzer = CallQualityAnalyzer()
    
    # Process sample transcript
    processor = CallTranscriptProcessor()
    transcript_data = processor.load_json_transcript("Call Transcript Sample 1.json")
    
    if transcript_data:
        records = processor.process_single_transcript(transcript_data)
        df = pd.DataFrame(records)
        
        print("Analyzing call quality...")
        
        # Analyze first response as example
        if not df.empty:
            sample_response = df.iloc[0]['answer']
            analysis = analyzer.analyze_single_response(sample_response)
            
            print(f"\nSample Response: {sample_response[:100]}...")
            print(f"\nQuality Analysis:")
            print(f"  Clarity Score: {analysis['quality_metrics'].clarity_score}/10")
            print(f"  Repetition Score: {analysis['quality_metrics'].conciseness_score}/10")
            print(f"  Overall Score: {analysis['quality_metrics'].overall_score}/10")
            print(f"  AHT Impact: {analysis['quality_metrics'].aht_impact}")
            
            print(f"\nCoaching Recommendations:")
            for i, rec in enumerate(analysis['coaching_recommendations'][:5], 1):
                print(f"  {i}. {rec}")
            
            print(f"\nAHT Impact Analysis:")
            aht_impact = analysis['aht_impact']
            print(f"  Impact Level: {aht_impact['aht_impact_level']}")
            print(f"  Description: {aht_impact['aht_description']}")
            print(f"  Estimated Time Increase: {aht_impact['estimated_time_increase']}")

if __name__ == "__main__":
    main()

