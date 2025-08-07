#!/usr/bin/env python3
"""
Configuration Management for Call Center Transcript Evaluation System
Centralized configuration for all components
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

@dataclass
class ModelConfig:
    """Configuration for Mistral model."""
    model_path: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = None  # Use all available CPU threads
    n_gpu_layers: int = 0  # CPU only
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    verbose: bool = False

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "./chroma_db"
    collection_name: str = "call_center_knowledge"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7

@dataclass
class QualityConfig:
    """Configuration for quality analysis."""
    max_sentence_length: int = 25  # Words
    crutch_word_threshold: int = 3
    hold_request_threshold: int = 3
    clarity_weight: float = 0.3
    repetition_weight: float = 0.3
    professionalism_weight: float = 0.2
    efficiency_weight: float = 0.2

@dataclass
class DeepEvalConfig:
    """Configuration for DeepEval metrics."""
    relevancy_threshold: float = 0.7
    correctness_threshold: float = 0.7
    enable_custom_metrics: bool = True
    batch_size: int = 1  # CPU optimization

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    use_textblob: bool = True
    use_mistral_sentiment: bool = True
    confidence_threshold: float = 0.6
    emotional_tone_categories: List[str] = field(default_factory=lambda: [
        'empathetic', 'apologetic', 'confident', 'helpful', 'patient', 'professional'
    ])

@dataclass
class StreamlitConfig:
    """Configuration for Streamlit dashboard."""
    page_title: str = "Call Center Performance Coaching Dashboard"
    page_icon: str = "📊"
    layout: str = "wide"
    theme: str = "light"
    max_upload_size: int = 200  # MB
    auto_refresh_interval: int = 30  # seconds

@dataclass
class OutputConfig:
    """Configuration for output and reporting."""
    default_format: str = "json"  # json, csv, excel
    save_intermediate_results: bool = True
    output_directory: str = "./results"
    include_raw_responses: bool = False
    max_summary_words: int = 200
    enable_detailed_logging: bool = True

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    max_concurrent_evaluations: int = 1
    enable_caching: bool = True
    cache_directory: str = "./cache"
    memory_limit_gb: float = 8.0
    timeout_seconds: int = 300
    enable_progress_tracking: bool = True

@dataclass
class SystemConfig:
    """Main system configuration combining all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    deepeval: DeepEvalConfig = field(default_factory=DeepEvalConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Feature toggles
    enable_rag: bool = True
    enable_deepeval: bool = True
    enable_quality_analysis: bool = True
    enable_sentiment_analysis: bool = True
    
    # Environment settings
    debug_mode: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create necessary directories
        os.makedirs(self.output.output_directory, exist_ok=True)
        os.makedirs(self.rag.vector_db_path, exist_ok=True)
        if self.performance.enable_caching:
            os.makedirs(self.performance.cache_directory, exist_ok=True)

class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    DEFAULT_CONFIG_FILE = "system_config.json"
    
    @classmethod
    def load_config(cls, config_file: str = None) -> SystemConfig:
        """Load configuration from file or create default."""
        config_file = config_file or cls.DEFAULT_CONFIG_FILE
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                return cls._dict_to_config(config_dict)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_file}: {e}")
                print("Using default configuration")
        
        return SystemConfig()
    
    @classmethod
    def save_config(cls, config: SystemConfig, config_file: str = None) -> bool:
        """Save configuration to file."""
        config_file = config_file or cls.DEFAULT_CONFIG_FILE
        
        try:
            config_dict = cls._config_to_dict(config)
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config to {config_file}: {e}")
            return False
    
    @classmethod
    def _config_to_dict(cls, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig to dictionary."""
        return {
            'model': {
                'model_path': config.model.model_path,
                'n_ctx': config.model.n_ctx,
                'n_threads': config.model.n_threads,
                'temperature': config.model.temperature,
                'max_tokens': config.model.max_tokens,
                'top_p': config.model.top_p,
                'repeat_penalty': config.model.repeat_penalty
            },
            'rag': {
                'embedding_model': config.rag.embedding_model,
                'vector_db_path': config.rag.vector_db_path,
                'collection_name': config.rag.collection_name,
                'chunk_size': config.rag.chunk_size,
                'chunk_overlap': config.rag.chunk_overlap,
                'top_k_retrieval': config.rag.top_k_retrieval,
                'similarity_threshold': config.rag.similarity_threshold
            },
            'quality': {
                'max_sentence_length': config.quality.max_sentence_length,
                'crutch_word_threshold': config.quality.crutch_word_threshold,
                'hold_request_threshold': config.quality.hold_request_threshold
            },
            'deepeval': {
                'relevancy_threshold': config.deepeval.relevancy_threshold,
                'correctness_threshold': config.deepeval.correctness_threshold,
                'batch_size': config.deepeval.batch_size
            },
            'sentiment': {
                'use_textblob': config.sentiment.use_textblob,
                'use_mistral_sentiment': config.sentiment.use_mistral_sentiment,
                'confidence_threshold': config.sentiment.confidence_threshold
            },
            'output': {
                'default_format': config.output.default_format,
                'save_intermediate_results': config.output.save_intermediate_results,
                'output_directory': config.output.output_directory,
                'max_summary_words': config.output.max_summary_words
            },
            'performance': {
                'max_concurrent_evaluations': config.performance.max_concurrent_evaluations,
                'enable_caching': config.performance.enable_caching,
                'cache_directory': config.performance.cache_directory,
                'memory_limit_gb': config.performance.memory_limit_gb,
                'timeout_seconds': config.performance.timeout_seconds
            },
            'features': {
                'enable_rag': config.enable_rag,
                'enable_deepeval': config.enable_deepeval,
                'enable_quality_analysis': config.enable_quality_analysis,
                'enable_sentiment_analysis': config.enable_sentiment_analysis
            },
            'environment': {
                'debug_mode': config.debug_mode,
                'log_level': config.log_level
            }
        }
    
    @classmethod
    def _dict_to_config(cls, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig."""
        config = SystemConfig()
        
        # Model config
        if 'model' in config_dict:
            model_dict = config_dict['model']
            config.model = ModelConfig(
                model_path=model_dict.get('model_path', config.model.model_path),
                n_ctx=model_dict.get('n_ctx', config.model.n_ctx),
                n_threads=model_dict.get('n_threads', config.model.n_threads),
                temperature=model_dict.get('temperature', config.model.temperature),
                max_tokens=model_dict.get('max_tokens', config.model.max_tokens),
                top_p=model_dict.get('top_p', config.model.top_p),
                repeat_penalty=model_dict.get('repeat_penalty', config.model.repeat_penalty)
            )
        
        # RAG config
        if 'rag' in config_dict:
            rag_dict = config_dict['rag']
            config.rag = RAGConfig(
                embedding_model=rag_dict.get('embedding_model', config.rag.embedding_model),
                vector_db_path=rag_dict.get('vector_db_path', config.rag.vector_db_path),
                collection_name=rag_dict.get('collection_name', config.rag.collection_name),
                chunk_size=rag_dict.get('chunk_size', config.rag.chunk_size),
                chunk_overlap=rag_dict.get('chunk_overlap', config.rag.chunk_overlap),
                top_k_retrieval=rag_dict.get('top_k_retrieval', config.rag.top_k_retrieval),
                similarity_threshold=rag_dict.get('similarity_threshold', config.rag.similarity_threshold)
            )
        
        # Feature toggles
        if 'features' in config_dict:
            features = config_dict['features']
            config.enable_rag = features.get('enable_rag', config.enable_rag)
            config.enable_deepeval = features.get('enable_deepeval', config.enable_deepeval)
            config.enable_quality_analysis = features.get('enable_quality_analysis', config.enable_quality_analysis)
            config.enable_sentiment_analysis = features.get('enable_sentiment_analysis', config.enable_sentiment_analysis)
        
        # Environment settings
        if 'environment' in config_dict:
            env = config_dict['environment']
            config.debug_mode = env.get('debug_mode', config.debug_mode)
            config.log_level = env.get('log_level', config.log_level)
        
        return config
    
    @classmethod
    def validate_config(cls, config: SystemConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check model file exists
        if not os.path.exists(config.model.model_path):
            issues.append(f"Model file not found: {config.model.model_path}")
        
        # Check model parameters
        if config.model.temperature < 0 or config.model.temperature > 2:
            issues.append("Model temperature should be between 0 and 2")
        
        if config.model.max_tokens < 1:
            issues.append("Model max_tokens should be positive")
        
        # Check RAG configuration
        if config.enable_rag:
            if config.rag.chunk_size < 100:
                issues.append("RAG chunk_size should be at least 100")
            
            if config.rag.chunk_overlap >= config.rag.chunk_size:
                issues.append("RAG chunk_overlap should be less than chunk_size")
        
        # Check thresholds
        if config.deepeval.relevancy_threshold < 0 or config.deepeval.relevancy_threshold > 1:
            issues.append("DeepEval relevancy_threshold should be between 0 and 1")
        
        if config.deepeval.correctness_threshold < 0 or config.deepeval.correctness_threshold > 1:
            issues.append("DeepEval correctness_threshold should be between 0 and 1")
        
        # Check output directory
        try:
            os.makedirs(config.output.output_directory, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")
        
        return issues
    
    @classmethod
    def get_environment_config(cls) -> Dict[str, str]:
        """Get configuration from environment variables."""
        return {
            'MISTRAL_MODEL_PATH': os.getenv('MISTRAL_MODEL_PATH', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'),
            'OUTPUT_DIRECTORY': os.getenv('OUTPUT_DIRECTORY', './results'),
            'VECTOR_DB_PATH': os.getenv('VECTOR_DB_PATH', './chroma_db'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
            'DEBUG_MODE': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'ENABLE_RAG': os.getenv('ENABLE_RAG', 'true').lower() == 'true',
            'ENABLE_DEEPEVAL': os.getenv('ENABLE_DEEPEVAL', 'true').lower() == 'true'
        }

def create_default_config_file():
    """Create a default configuration file."""
    config = SystemConfig()
    ConfigManager.save_config(config)
    print(f"Default configuration saved to {ConfigManager.DEFAULT_CONFIG_FILE}")

def main():
    """Example usage of configuration management."""
    # Create default config
    config = SystemConfig()
    
    # Save to file
    ConfigManager.save_config(config, "example_config.json")
    
    # Load from file
    loaded_config = ConfigManager.load_config("example_config.json")
    
    # Validate configuration
    issues = ConfigManager.validate_config(loaded_config)
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Show environment config
    env_config = ConfigManager.get_environment_config()
    print("\nEnvironment configuration:")
    for key, value in env_config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

