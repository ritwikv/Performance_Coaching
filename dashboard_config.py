"""
Configuration file for the Streamlit Dashboard
Centralized settings and customization options
"""

import streamlit as st
from typing import Dict, List, Any

class DashboardConfig:
    """Configuration class for dashboard settings"""
    
    # App Configuration
    APP_TITLE = "Call Center Performance Dashboard"
    APP_ICON = "📞"
    LAYOUT = "wide"
    
    # File Paths
    DEFAULT_TRANSCRIPT_FILE = "Call Transcript Sample 1.json"
    EVALUATION_RESULTS_JSON = "transcript_evaluation_results.json"
    EVALUATION_RESULTS_CSV = "transcript_evaluation_results.csv"
    MISTRAL_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # UI Colors and Styling
    COLORS = {
        'primary': '#1f77b4',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }
    
    # Score Thresholds
    SCORE_THRESHOLDS = {
        'excellent': 8.5,
        'good': 7.0,
        'needs_improvement': 5.0,
        'poor': 0.0
    }
    
    # Score Labels and Colors
    SCORE_CONFIG = {
        'excellent': {'label': 'Excellent', 'color': '#28a745', 'emoji': '🌟'},
        'good': {'label': 'Good', 'color': '#17a2b8', 'emoji': '👍'},
        'needs_improvement': {'label': 'Needs Improvement', 'color': '#ffc107', 'emoji': '⚠️'},
        'poor': {'label': 'Needs Attention', 'color': '#dc3545', 'emoji': '🚨'}
    }
    
    # Evaluation Metrics Configuration
    METRICS_CONFIG = {
        'english_score': {
            'name': 'English Proficiency',
            'icon': '📝',
            'description': 'Grammar, spelling, and professional language usage'
        },
        'clarity_score': {
            'name': 'Communication Clarity',
            'icon': '🎯',
            'description': 'Sentence structure and message clarity'
        },
        'speech_score': {
            'name': 'Speech Quality',
            'icon': '🗣️',
            'description': 'Absence of crutch words and repetitive language'
        },
        'context_precision': {
            'name': 'Context Precision',
            'icon': '🎯',
            'description': 'Relevance of provided information'
        },
        'context_recall': {
            'name': 'Context Recall',
            'icon': '🧠',
            'description': 'Completeness of information coverage'
        },
        'faithfulness': {
            'name': 'Faithfulness',
            'icon': '✅',
            'description': 'Accuracy of information provided'
        },
        'answer_relevancy': {
            'name': 'Answer Relevancy',
            'icon': '🎯',
            'description': 'How well the answer addresses the question'
        }
    }
    
    # Sentiment Configuration
    SENTIMENT_CONFIG = {
        'positive': {'emoji': '😊', 'color': '#28a745', 'label': 'Positive'},
        'negative': {'emoji': '😔', 'color': '#dc3545', 'label': 'Negative'},
        'neutral': {'emoji': '😐', 'color': '#6c757d', 'label': 'Neutral'}
    }
    
    # Dashboard Sections
    DASHBOARD_SECTIONS = {
        'overview': {
            'title': '📊 Performance Overview',
            'description': 'High-level performance metrics and trends'
        },
        'individual': {
            'title': '👤 Individual Feedback',
            'description': 'Detailed feedback for specific CSR interactions'
        },
        'analytics': {
            'title': '📈 Analytics & Insights',
            'description': 'Advanced analytics and performance insights'
        },
        'coaching': {
            'title': '💬 Coaching Reports',
            'description': 'Personalized coaching recommendations'
        }
    }
    
    # Chart Configuration
    CHART_CONFIG = {
        'height': 400,
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'template': 'plotly_white'
    }
    
    @classmethod
    def get_score_category(cls, score: float) -> str:
        """Get score category based on thresholds"""
        if score >= cls.SCORE_THRESHOLDS['excellent']:
            return 'excellent'
        elif score >= cls.SCORE_THRESHOLDS['good']:
            return 'good'
        elif score >= cls.SCORE_THRESHOLDS['needs_improvement']:
            return 'needs_improvement'
        else:
            return 'poor'
    
    @classmethod
    def get_score_info(cls, score: float) -> Dict[str, str]:
        """Get complete score information"""
        category = cls.get_score_category(score)
        return cls.SCORE_CONFIG[category]
    
    @classmethod
    def get_custom_css(cls) -> str:
        """Get custom CSS for the dashboard"""
        return f"""
        <style>
            .main-header {{
                font-size: 2.5rem;
                color: {cls.COLORS['primary']};
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
            }}
            
            .metric-card {{
                background-color: {cls.COLORS['light']};
                padding: 1.5rem;
                border-radius: 0.75rem;
                border-left: 4px solid {cls.COLORS['primary']};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 0.5rem 0;
            }}
            
            .feedback-card {{
                background-color: white;
                padding: 2rem;
                border-radius: 0.75rem;
                border: 1px solid #e0e0e0;
                margin: 1.5rem 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .score-excellent {{ 
                color: {cls.COLORS['success']}; 
                font-weight: bold; 
                font-size: 1.2rem;
            }}
            
            .score-good {{ 
                color: {cls.COLORS['info']}; 
                font-weight: bold; 
                font-size: 1.2rem;
            }}
            
            .score-needs-improvement {{ 
                color: {cls.COLORS['warning']}; 
                font-weight: bold; 
                font-size: 1.2rem;
            }}
            
            .score-poor {{ 
                color: {cls.COLORS['danger']}; 
                font-weight: bold; 
                font-size: 1.2rem;
            }}
            
            .sidebar-section {{
                background-color: {cls.COLORS['light']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }}
            
            .coaching-highlight {{
                background-color: #e8f4fd;
                border-left: 4px solid {cls.COLORS['info']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }}
            
            .performance-badge {{
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 1rem;
                font-size: 0.875rem;
                font-weight: bold;
                margin: 0.25rem;
            }}
            
            .badge-excellent {{
                background-color: {cls.COLORS['success']};
                color: white;
            }}
            
            .badge-good {{
                background-color: {cls.COLORS['info']};
                color: white;
            }}
            
            .badge-warning {{
                background-color: {cls.COLORS['warning']};
                color: {cls.COLORS['dark']};
            }}
            
            .badge-danger {{
                background-color: {cls.COLORS['danger']};
                color: white;
            }}
            
            .footer {{
                text-align: center;
                color: #666;
                padding: 2rem 0;
                border-top: 1px solid #e0e0e0;
                margin-top: 3rem;
            }}
        </style>
        """

# Global configuration instance
config = DashboardConfig()

