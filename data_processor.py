#!/usr/bin/env python3
"""
Enhanced Data Processing Pipeline for Call Center Transcripts
Processes JSON call transcripts into structured DataFrames with proper Q&A mapping
"""

import json
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallTranscriptProcessor:
    """
    Processes call center transcripts from JSON format into structured DataFrames
    with proper Question-Answer mapping and metadata extraction.
    """
    
    def __init__(self):
        self.processed_data = []
        self.metadata_columns = ['call_ID', 'CSR_ID', 'call_date', 'call_time']
        
    def load_json_transcript(self, file_path: str) -> Optional[Dict]:
        """Load transcript from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded transcript from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def extract_metadata(self, transcript_data: Dict) -> Dict:
        """Extract metadata from transcript."""
        metadata = {}
        for col in self.metadata_columns:
            metadata[col] = transcript_data.get(col, '')
        return metadata
    
    def parse_conversation(self, call_transcript: List[str]) -> Tuple[List[Dict], str]:
        """
        Parse conversation into Questions, Answers, and Greeting.
        Returns tuple of (conversation_pairs, greeting)
        """
        conversation_pairs = []
        greeting = ""
        
        # Check if CSR starts the conversation (first speaker is CSR)
        csr_starts = False
        if call_transcript and call_transcript[0].strip().startswith('CSR:'):
            csr_starts = True
            greeting = call_transcript[0].replace('CSR:', '').strip()
        
        # Process the conversation
        current_question = ""
        current_answer = ""
        i = 1 if csr_starts else 0  # Start from index 1 if CSR greeting exists
        
        while i < len(call_transcript):
            line = call_transcript[i].strip()
            
            if line.startswith('Customer:'):
                # If we have a pending answer, save the previous Q&A pair
                if current_question and current_answer:
                    conversation_pairs.append({
                        'question': current_question,
                        'answer': current_answer
                    })
                
                # Start new question
                current_question = line.replace('Customer:', '').strip()
                current_answer = ""
                
            elif line.startswith('CSR:') or line.startswith('Supervisor:'):
                # CSR or Supervisor response
                speaker_prefix = 'CSR:' if line.startswith('CSR:') else 'Supervisor:'
                response = line.replace(speaker_prefix, '').strip()
                
                if current_answer:
                    current_answer += " " + response
                else:
                    current_answer = response
            
            i += 1
        
        # Add the last Q&A pair if exists
        if current_question and current_answer:
            conversation_pairs.append({
                'question': current_question,
                'answer': current_answer
            })
        
        return conversation_pairs, greeting
    
    def process_single_transcript(self, transcript_data: Dict) -> List[Dict]:
        """Process a single transcript into structured data."""
        if not isinstance(transcript_data, dict):
            logger.error("Invalid transcript data format")
            return []
        
        # Extract metadata
        metadata = self.extract_metadata(transcript_data)
        
        # Get call transcript
        call_transcript = transcript_data.get('call_transcript', [])
        if not call_transcript:
            logger.warning("No call_transcript found in data")
            return []
        
        # Parse conversation
        conversation_pairs, greeting = self.parse_conversation(call_transcript)
        
        # Create structured records
        records = []
        for i, pair in enumerate(conversation_pairs):
            record = {
                **metadata,  # Include all metadata
                'interaction_sequence': i + 1,
                'greeting': greeting if i == 0 else "",  # Only include greeting in first record
                'question': pair['question'],
                'answer': pair['answer'],
                'question_length': len(pair['question'].split()),
                'answer_length': len(pair['answer'].split()),
                'processed_timestamp': datetime.now().isoformat()
            }
            records.append(record)
        
        return records
    
    def process_multiple_transcripts(self, file_paths: List[str]) -> pd.DataFrame:
        """Process multiple transcript files."""
        all_records = []
        
        for file_path in file_paths:
            transcript_data = self.load_json_transcript(file_path)
            if transcript_data:
                records = self.process_single_transcript(transcript_data)
                all_records.extend(records)
        
        if not all_records:
            logger.warning("No valid records processed")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        logger.info(f"Processed {len(all_records)} conversation pairs from {len(file_paths)} files")
        return df
    
    def analyze_conversation_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze conversation patterns in the processed data."""
        if df.empty:
            return {}
        
        analysis = {
            'total_interactions': len(df),
            'unique_calls': df['call_ID'].nunique(),
            'unique_csrs': df['CSR_ID'].nunique(),
            'avg_question_length': df['question_length'].mean(),
            'avg_answer_length': df['answer_length'].mean(),
            'max_interactions_per_call': df.groupby('call_ID')['interaction_sequence'].max().max(),
            'calls_with_greeting': (df['greeting'] != "").sum(),
            'date_range': {
                'start': df['call_date'].min(),
                'end': df['call_date'].max()
            }
        }
        
        return analysis
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str, format: str = 'csv'):
        """Save processed data to file."""
        try:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(output_path, index=False, engine='openpyxl')
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {e}")

def main():
    """Example usage of the CallTranscriptProcessor."""
    processor = CallTranscriptProcessor()
    
    # Process the sample transcript
    sample_file = "Call Transcript Sample 1.json"
    transcript_data = processor.load_json_transcript(sample_file)
    
    if transcript_data:
        # Process single transcript
        records = processor.process_single_transcript(transcript_data)
        df = pd.DataFrame(records)
        
        print("Processed DataFrame:")
        print(df.head())
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Analyze patterns
        analysis = processor.analyze_conversation_patterns(df)
        print(f"\nConversation Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Save processed data
        processor.save_processed_data(df, "processed_call_transcript.csv")
        processor.save_processed_data(df, "processed_call_transcript.xlsx", format='excel')
        
        print(f"\nSample records:")
        for i, record in enumerate(records[:3]):
            print(f"\nRecord {i+1}:")
            print(f"  Question: {record['question'][:100]}...")
            print(f"  Answer: {record['answer'][:100]}...")
            print(f"  Greeting: {record['greeting']}")

if __name__ == "__main__":
    main()
