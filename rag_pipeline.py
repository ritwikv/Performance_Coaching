#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Pipeline for Call Center Knowledge Base
Creates knowledge documents from transcripts and provides expert-level answers
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Fix Windows symlink issue for HuggingFace Hub
if os.name == 'nt':  # Windows
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Warning: Required packages not installed. Install with:")
    print("pip install sentence-transformers chromadb")
    SentenceTransformer = None
    chromadb = None

from mistral_model import MistralEvaluator, MistralConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    embedding_model: str = "all-MiniLM-L6-v2"  # Lightweight, good for CPU
    vector_db_path: str = "./chroma_db"
    collection_name: str = "call_center_knowledge"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7

class DocumentChunker:
    """Handles document chunking for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_conversation(self, question: str, answer: str, metadata: Dict) -> List[Dict]:
        """Chunk a Q&A pair into knowledge documents."""
        chunks = []
        
        # Create main Q&A chunk
        main_chunk = {
            'content': f"Question: {question}\nAnswer: {answer}",
            'question': question,
            'answer': answer,
            'chunk_type': 'qa_pair',
            'metadata': metadata
        }
        chunks.append(main_chunk)
        
        # If answer is long, create additional chunks focusing on different aspects
        if len(answer.split()) > 100:
            # Split long answers into semantic chunks
            sentences = answer.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk.split()) + len(sentence.split()) <= self.chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunk = {
                            'content': f"Context: {question}\nInformation: {current_chunk.strip()}",
                            'question': question,
                            'answer': current_chunk.strip(),
                            'chunk_type': 'answer_segment',
                            'metadata': metadata
                        }
                        chunks.append(chunk)
                    current_chunk = sentence + ". "
            
            # Add remaining content
            if current_chunk:
                chunk = {
                    'content': f"Context: {question}\nInformation: {current_chunk.strip()}",
                    'question': question,
                    'answer': current_chunk.strip(),
                    'chunk_type': 'answer_segment',
                    'metadata': metadata
                }
                chunks.append(chunk)
        
        return chunks
    
    def chunk_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """Chunk entire DataFrame of conversations."""
        all_chunks = []
        
        for _, row in df.iterrows():
            metadata = {
                'call_id': row.get('call_ID', ''),
                'csr_id': row.get('CSR_ID', ''),
                'call_date': row.get('call_date', ''),
                'call_time': row.get('call_time', ''),
                'interaction_sequence': row.get('interaction_sequence', 0)
            }
            
            chunks = self.chunk_conversation(
                row['question'], 
                row['answer'], 
                metadata
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(df)} conversations")
        return all_chunks

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_model = None
        
    def initialize(self) -> bool:
        """Initialize vector store and embedding model."""
        try:
            # Initialize ChromaDB
            if chromadb is None:
                logger.error("ChromaDB not available")
                return False
            
            self.client = chromadb.PersistentClient(path=self.config.vector_db_path)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.config.collection_name)
                logger.info(f"Loaded existing collection: {self.config.collection_name}")
            except:
                self.collection = self.client.create_collection(name=self.config.collection_name)
                logger.info(f"Created new collection: {self.config.collection_name}")
            
            # Initialize embedding model
            if SentenceTransformer is None:
                logger.error("SentenceTransformer not available")
                return False
            
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False
    
    def add_documents(self, chunks: List[Dict]) -> bool:
        """Add document chunks to vector store."""
        if not self.collection or not self.embedding_model:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk['content'])
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = {
                    'question': chunk['question'],
                    'answer': chunk['answer'],
                    'chunk_type': chunk['chunk_type'],
                    'call_id': str(chunk['metadata'].get('call_id', '')),
                    'csr_id': str(chunk['metadata'].get('csr_id', '')),
                    'call_date': str(chunk['metadata'].get('call_date', '')),
                    'interaction_sequence': str(chunk['metadata'].get('interaction_sequence', 0))
                }
                metadatas.append(metadata)
                ids.append(f"chunk_{i}_{datetime.now().timestamp()}")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents."""
        if not self.collection or not self.embedding_model:
            logger.error("Vector store not initialized")
            return []
        
        try:
            top_k = top_k or self.config.top_k_retrieval
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.config.collection_name,
                'embedding_model': self.config.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

class KnowledgeGenerator:
    """Generates expert knowledge from call transcripts."""
    
    def __init__(self, mistral_evaluator: MistralEvaluator):
        self.mistral_evaluator = mistral_evaluator
    
    def generate_expert_responses(self, questions: List[str], context: str = "travel services") -> List[str]:
        """Generate expert responses for a list of questions."""
        expert_responses = []
        
        for i, question in enumerate(questions):
            logger.info(f"Generating expert response {i+1}/{len(questions)}")
            response = self.mistral_evaluator.generate_expert_answer(question, context)
            expert_responses.append(response)
        
        return expert_responses
    
    def create_knowledge_documents(self, df: pd.DataFrame) -> List[Dict]:
        """Create knowledge documents from processed transcript DataFrame."""
        knowledge_docs = []
        
        # Extract unique questions for expert response generation
        unique_questions = df['question'].unique().tolist()
        logger.info(f"Generating expert responses for {len(unique_questions)} unique questions")
        
        # Generate expert responses
        expert_responses = self.generate_expert_responses(unique_questions)
        
        # Create question-to-expert-response mapping
        expert_map = dict(zip(unique_questions, expert_responses))
        
        # Create knowledge documents
        for _, row in df.iterrows():
            question = row['question']
            original_answer = row['answer']
            expert_answer = expert_map.get(question, "")
            
            knowledge_doc = {
                'question': question,
                'original_csr_answer': original_answer,
                'expert_answer': expert_answer,
                'metadata': {
                    'call_id': row.get('call_ID', ''),
                    'csr_id': row.get('CSR_ID', ''),
                    'call_date': row.get('call_date', ''),
                    'call_time': row.get('call_time', ''),
                    'interaction_sequence': row.get('interaction_sequence', 0)
                },
                'created_at': datetime.now().isoformat()
            }
            knowledge_docs.append(knowledge_doc)
        
        return knowledge_docs

class RAGPipeline:
    """Complete RAG pipeline for call center knowledge base."""
    
    def __init__(self, config: RAGConfig = None, mistral_config: MistralConfig = None):
        self.config = config or RAGConfig()
        self.mistral_evaluator = MistralEvaluator(mistral_config)
        self.chunker = DocumentChunker(self.config.chunk_size, self.config.chunk_overlap)
        self.vector_store = VectorStore(self.config)
        self.knowledge_generator = KnowledgeGenerator(self.mistral_evaluator)
        
    def initialize(self) -> bool:
        """Initialize the RAG pipeline."""
        logger.info("Initializing RAG pipeline...")
        
        # Load Mistral model
        if not self.mistral_evaluator.load_model():
            logger.error("Failed to load Mistral model")
            return False
        
        # Initialize vector store
        if not self.vector_store.initialize():
            logger.error("Failed to initialize vector store")
            return False
        
        logger.info("RAG pipeline initialized successfully")
        return True
    
    def build_knowledge_base(self, df: pd.DataFrame) -> bool:
        """Build knowledge base from processed transcript DataFrame."""
        logger.info("Building knowledge base...")
        
        # Generate knowledge documents
        knowledge_docs = self.knowledge_generator.create_knowledge_documents(df)
        
        # Create chunks for vector storage
        chunks = []
        for doc in knowledge_docs:
            doc_chunks = self.chunker.chunk_conversation(
                doc['question'],
                doc['expert_answer'],
                doc['metadata']
            )
            chunks.extend(doc_chunks)
        
        # Add to vector store
        success = self.vector_store.add_documents(chunks)
        
        if success:
            logger.info(f"Knowledge base built with {len(chunks)} chunks")
            # Save knowledge documents
            self.save_knowledge_documents(knowledge_docs)
        
        return success
    
    def get_expert_answer(self, question: str) -> Dict[str, Any]:
        """Get expert answer for a question using RAG."""
        # Search for relevant context
        relevant_docs = self.vector_store.search(question, top_k=self.config.top_k_retrieval)
        
        if not relevant_docs:
            # Generate new expert answer if no relevant context found
            expert_answer = self.mistral_evaluator.generate_expert_answer(question)
            return {
                'question': question,
                'expert_answer': expert_answer,
                'context_used': [],
                'method': 'direct_generation'
            }
        
        # Use retrieved context to enhance answer generation
        context = "\n".join([doc['content'] for doc in relevant_docs[:3]])
        
        enhanced_prompt = f"""
Based on the following context from previous successful customer interactions, provide an expert response to the customer question:

Context:
{context}

Customer Question: {question}

Provide a comprehensive, professional response that incorporates relevant information from the context while addressing the specific question:
"""
        
        expert_answer = self.mistral_evaluator.generate_response(enhanced_prompt, max_tokens=1024)
        
        return {
            'question': question,
            'expert_answer': expert_answer,
            'context_used': relevant_docs,
            'method': 'rag_enhanced'
        }
    
    def save_knowledge_documents(self, knowledge_docs: List[Dict], filename: str = "knowledge_documents.json"):
        """Save knowledge documents to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(knowledge_docs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(knowledge_docs)} knowledge documents to {filename}")
        except Exception as e:
            logger.error(f"Error saving knowledge documents: {e}")
    
    def load_knowledge_documents(self, filename: str = "knowledge_documents.json") -> List[Dict]:
        """Load knowledge documents from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                knowledge_docs = json.load(f)
            logger.info(f"Loaded {len(knowledge_docs)} knowledge documents from {filename}")
            return knowledge_docs
        except Exception as e:
            logger.error(f"Error loading knowledge documents: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'mistral_model': self.mistral_evaluator.get_model_info(),
            'vector_store': self.vector_store.get_collection_stats(),
            'config': {
                'embedding_model': self.config.embedding_model,
                'chunk_size': self.config.chunk_size,
                'top_k_retrieval': self.config.top_k_retrieval
            }
        }

def main():
    """Example usage of RAG pipeline."""
    from data_processor import CallTranscriptProcessor
    
    # Initialize components
    rag_config = RAGConfig()
    mistral_config = MistralConfig(model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    
    pipeline = RAGPipeline(rag_config, mistral_config)
    
    # Initialize pipeline
    if not pipeline.initialize():
        print("Failed to initialize RAG pipeline")
        return
    
    # Process sample transcript
    processor = CallTranscriptProcessor()
    transcript_data = processor.load_json_transcript("Call Transcript Sample 1.json")
    
    if transcript_data:
        records = processor.process_single_transcript(transcript_data)
        df = pd.DataFrame(records)
        
        print("Building knowledge base...")
        success = pipeline.build_knowledge_base(df)
        
        if success:
            print("Knowledge base built successfully!")
            
            # Test RAG retrieval
            test_question = "I need help with a reservation I made last week"
            result = pipeline.get_expert_answer(test_question)
            
            print(f"\nTest Question: {test_question}")
            print(f"Expert Answer: {result['expert_answer']}")
            print(f"Method: {result['method']}")
            
            # Show pipeline stats
            stats = pipeline.get_pipeline_stats()
            print(f"\nPipeline Statistics:")
            print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
