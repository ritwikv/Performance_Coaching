#!/usr/bin/env python3
"""
Configuration file for Call Center Transcript Evaluator
=======================================================

This file contains all configurable parameters for the evaluator.
Modify these settings to customize the behavior according to your needs.
"""

import os
from pathlib import Path

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Path to the Mistral 7B GGUF model file
# Update this to point to your downloaded model
MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Model parameters for llama-cpp-python
MODEL_CONFIG = {
    "n_ctx": 4096,          # Context window size (tokens)
    "n_threads": 4,         # Number of CPU threads to use
    "verbose": False,       # Enable/disable verbose model output
    "temperature": 0.7,     # Sampling temperature (0.0 to 1.0)
    "top_p": 0.9,          # Top-p sampling parameter
    "max_tokens": 512,      # Default max tokens for responses
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Sentence length analysis
SENTENCE_CONFIG = {
    "long_sentence_threshold": 20,  # Words count threshold for long sentences
    "ideal_sentence_length": 15,    # Ideal sentence length in words
}

# Word repetition and crutch words detection
WORD_ANALYSIS_CONFIG = {
    "repetition_threshold": 3,      # Minimum count to consider word as repeated
    "min_word_length": 3,          # Minimum word length to check for repetition
    "crutch_words": [
        # Filler words and phrases
        'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
        'sort of', 'kind of', 'I mean', 'right?', 'okay?', 'so', 'well',
        'anyway', 'obviously', 'clearly', 'honestly', 'frankly', 'essentially',
        
        # Additional professional crutch words
        'moving forward', 'at the end of the day', 'to be honest',
        'if you will', 'as it were', 'per se', 'as such', 'in terms of',
        'with regard to', 'in relation to', 'as far as', 'as per',
    ]
}

# Sentiment analysis configuration
SENTIMENT_CONFIG = {
    "positive_threshold": 0.1,      # Polarity threshold for positive sentiment
    "negative_threshold": -0.1,     # Polarity threshold for negative sentiment
    "subjectivity_threshold": 0.5,  # Subjectivity threshold
}

# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

# Prompts for different evaluation aspects
EVALUATION_PROMPTS = {
    "english_correctness": """
    Please evaluate the following text for English language correctness and provide coaching feedback:
    
    Text: "{text}"
    
    Please provide:
    1. Grammar errors (if any) with specific corrections
    2. Spelling mistakes (if any) with correct spellings
    3. Suggestions for improvement in clarity and professionalism
    4. Overall correctness score (1-10) with brief justification
    5. Coaching advice in a friendly, constructive manner
    
    Format your response as a structured evaluation with clear sections.
    """,
    
    "sentence_improvement": """
    The following sentences are too long and need to be made more crisp and concise:
    
    {long_sentences}
    
    Please provide:
    1. Shorter, clearer alternatives for each sentence
    2. General tips for writing crisp, professional sentences
    3. Specific coaching advice for the customer service agent
    4. Examples of how to break complex ideas into simpler statements
    
    Focus on maintaining professionalism while improving clarity.
    """,
    
    "repetition_coaching": """
    Analyze this text for word repetition and filler words:
    
    Text: "{text}"
    
    Repeated words found: {repeated_words}
    Crutch/filler words found: {crutch_words}
    
    Please provide coaching feedback on:
    1. How to reduce word repetition with specific alternatives
    2. Professional alternatives to crutch words and filler phrases
    3. Tips for more confident and articulate speech
    4. Specific suggestions for improvement in customer service context
    5. Practice exercises or techniques to avoid these patterns
    """,
    
    "sentiment_coaching": """
    Analyze the sentiment of this customer service response:
    
    Text: "{text}"
    
    Sentiment Analysis:
    - Polarity: {polarity:.2f} (range: -1 to 1)
    - Subjectivity: {subjectivity:.2f} (range: 0 to 1)
    - Classification: {sentiment_label}
    
    Please provide coaching feedback in a sentence format about the agent's emotional tone and demeanor.
    Examples:
    - "You maintained a positive and helpful tone throughout your response"
    - "Your tone seemed slightly frustrated; try to maintain calm professionalism"
    - "Great job staying neutral and professional in a difficult situation"
    
    Focus on actionable advice for emotional intelligence in customer service.
    """,
    
    "topic_summary": """
    Analyze the following customer service interaction and summarize the main topic or theme:
    
    Customer Question: "{question}"
    Agent Response: "{answer}"
    
    Please provide:
    1. The main topic/theme (e.g., "Flight Cancellation", "Refund Request", "Booking Issue")
    2. A brief summary of what the interaction was about (1-2 sentences)
    3. The type of customer service issue this represents
    4. The complexity level of the issue (Simple/Moderate/Complex)
    5. Key skills or knowledge areas demonstrated or needed
    
    Keep the response concise and focused on categorization for training purposes.
    """,
    
    "knowledge_extraction": """
    Based on the following call center conversation responses, create comprehensive knowledge documents that capture:
    
    Call center responses:
    {combined_text}
    
    Please create 3-5 distinct knowledge documents, each focusing on:
    1. Key policies and procedures mentioned
    2. Common customer issues and their standard solutions
    3. Best practices for customer service communication
    4. Escalation procedures and supervisor involvement
    5. Refund and compensation guidelines
    
    Format each document clearly with:
    - A descriptive title
    - Main content with bullet points or numbered lists
    - Key takeaways or action items
    
    Make these documents useful for training new customer service representatives.
    """
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Report generation settings
REPORT_CONFIG = {
    "max_text_preview": 200,        # Maximum characters to show in previews
    "include_full_transcripts": True,  # Include full Q&A text in reports
    "generate_summary_stats": True,    # Generate summary statistics
    "save_dataframe_csv": True,       # Save extracted DataFrame as CSV
}

# File paths and naming
FILE_CONFIG = {
    "default_output_dir": "evaluation_results",
    "report_filename_template": "{call_id}_{csr_id}_evaluation_report.txt",
    "csv_filename_template": "{call_id}_{csr_id}_qa_data.csv",
    "knowledge_docs_filename": "knowledge_documents.json",
}

# =============================================================================
# RAGAS CONFIGURATION (for future use)
# =============================================================================

# RAGAS evaluation settings (commented for later implementation)
RAGAS_CONFIG = {
    "metrics": [
        "context_precision",
        "context_recall", 
        "context_entity_recall",
        "answer_relevancy",
        "faithfulness"
    ],
    "batch_size": 10,               # Number of evaluations per batch
    "timeout_seconds": 300,         # Timeout for each evaluation
    "retry_attempts": 3,            # Number of retry attempts for failed evaluations
}

# RAGAS coaching templates
RAGAS_COACHING_TEMPLATES = {
    "context_precision": """
    Your Context Precision score is {score:.2f} out of 1.0.
    
    This measures how well your answer stays focused on the relevant information.
    {score_interpretation}
    
    To improve:
    - Focus on directly answering the customer's specific question
    - Avoid including irrelevant information
    - Structure your response to address the main concern first
    """,
    
    "context_recall": """
    Your Context Recall score is {score:.2f} out of 1.0.
    
    This measures how well you incorporated all relevant available information.
    {score_interpretation}
    
    To improve:
    - Make sure to address all aspects of the customer's question
    - Reference relevant policies or procedures when applicable
    - Don't leave out important details that could help the customer
    """,
    
    "faithfulness": """
    Your Faithfulness score is {score:.2f} out of 1.0.
    
    This measures how accurate your response is based on available information.
    {score_interpretation}
    
    To improve:
    - Only state facts that you can verify
    - Avoid making assumptions or guesses
    - If unsure, acknowledge limitations and offer to find out more
    """,
    
    "answer_relevancy": """
    Your Answer Relevancy score is {score:.2f} out of 1.0.
    
    This measures how well your answer addresses the customer's actual question.
    {score_interpretation}
    
    To improve:
    - Listen carefully to what the customer is really asking
    - Address their main concern directly
    - Ask clarifying questions if the request is unclear
    """
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",                # Logging level (DEBUG, INFO, WARNING, ERROR)
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_to_file": True,           # Save logs to file
    "log_filename": "evaluator.log",
    "max_log_size_mb": 10,         # Maximum log file size in MB
    "backup_count": 5,             # Number of backup log files to keep
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check if model file exists
    if not Path(MODEL_PATH).exists():
        errors.append(f"Model file not found: {MODEL_PATH}")
    
    # Validate thresholds
    if SENTENCE_CONFIG["long_sentence_threshold"] <= 0:
        errors.append("Long sentence threshold must be positive")
    
    if not (0 <= MODEL_CONFIG["temperature"] <= 1):
        errors.append("Temperature must be between 0 and 1")
    
    # Validate output directory
    output_dir = Path(FILE_CONFIG["default_output_dir"])
    try:
        output_dir.mkdir(exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory: {e}")
    
    return errors

def get_model_config():
    """Get model configuration dictionary."""
    return MODEL_CONFIG.copy()

def get_evaluation_config():
    """Get evaluation configuration dictionary."""
    return {
        "sentence": SENTENCE_CONFIG.copy(),
        "word_analysis": WORD_ANALYSIS_CONFIG.copy(),
        "sentiment": SENTIMENT_CONFIG.copy(),
    }

def get_prompts():
    """Get evaluation prompts dictionary."""
    return EVALUATION_PROMPTS.copy()

# =============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# =============================================================================

# Override settings from environment variables if available
if os.getenv("MISTRAL_MODEL_PATH"):
    MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH")

if os.getenv("EVALUATOR_THREADS"):
    MODEL_CONFIG["n_threads"] = int(os.getenv("EVALUATOR_THREADS"))

if os.getenv("EVALUATOR_OUTPUT_DIR"):
    FILE_CONFIG["default_output_dir"] = os.getenv("EVALUATOR_OUTPUT_DIR")

# Validate configuration on import
if __name__ == "__main__":
    validation_errors = validate_config()
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
        print(f"Model path: {MODEL_PATH}")
        print(f"Output directory: {FILE_CONFIG['default_output_dir']}")
        print(f"CPU threads: {MODEL_CONFIG['n_threads']}")

