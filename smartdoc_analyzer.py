"""
Deep Researcher Agent - A sophisticated AI-powered research system
that uses local embeddings and multi-step reasoning for information synthesis.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Core ML libraries
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from smartdoc_embeddings import EnhancedEmbeddingEngine
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Document processing
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import requests

# Storage and retrieval
# Note: ChromaDB removed to avoid dependency issues

# Utilities
from tqdm import tqdm
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and text extraction from various formats."""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt', '.html', '.md'}
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various document formats."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._extract_pdf_text(file_path)
            elif extension == '.docx':
                return self._extract_docx_text(file_path)
            elif extension in ['.txt', '.md']:
                return self._extract_text_file(file_path)
            elif extension == '.html':
                return self._extract_html_text(file_path)
            else:
                logger.warning(f"Unsupported file format: {extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files with improved extraction."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        # Add page number for better context
                        text += f"[Page {page_num + 1}] {page_text}\n\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text from {file_path}: {e}")
            # Fallback to basic extraction
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"Fallback PDF extraction also failed: {e2}")
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()
    
    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into overlapping chunks for better search accuracy."""
        if len(text) <= chunk_size:
            return [text]
        
        # Clean up the text first
        import re
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 150 characters
                sentence_endings = ['.', '!', '?', '\n\n']
                best_break = end
            
                for i in range(min(150, chunk_size)):
                    if text[end - i] in sentence_endings:
                        best_break = end - i + 1
                        break
            
                # If we found a good break point, use it
                if best_break > start + chunk_size // 2:  # At least half the chunk size
                    end = best_break
                else:
                    # Look for word boundaries as fallback
                    for i in range(min(50, chunk_size)):
                        if text[end - i] == ' ':
                            end = end - i
                            break
            
            chunk = text[start:end].strip()
            
            # Ensure chunk starts with a complete word/sentence
            if chunk and start > 0:
                # Find the first complete word
                words = chunk.split()
                if len(words) > 1:
                    # Skip the first word if it might be incomplete
                    first_word = words[0]
                    if len(first_word) < 3 or not first_word[0].isupper():
                        # Try to find a better starting point
                        space_pos = chunk.find(' ')
                        if space_pos > 0 and space_pos < len(chunk) // 2:
                            chunk = chunk[space_pos + 1:].strip()
            
            if chunk and len(chunk) > 50:  # Only add substantial chunks
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks


class EmbeddingEngine:
    """Handles local embedding generation using Hugging Face models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_online: bool = False, api_token: str = None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Use enhanced embedding engine
        self.enhanced_engine = EnhancedEmbeddingEngine(
            local_model_name=model_name,
            online_model_name="sentence-transformers/all-mpnet-base-v2",
            use_online=use_online,
            api_token=api_token
        )
        
        # For backward compatibility
        self.model = self.enhanced_engine.local_model
        self.embedding_dim = self.enhanced_engine.get_embedding_dimension()
        
        logger.info(f"Loaded model: {model_name}, embedding dimension: {self.embedding_dim}")
        logger.info(f"Online model enabled: {use_online}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.enhanced_engine.generate_embeddings(texts)
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.enhanced_engine.generate_single_embedding(text)
    
    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return self.enhanced_engine.get_model_info()


class VectorStore:
    """Handles vector storage and similarity search."""
    
    def __init__(self, embedding_dim: int, storage_path: str = "./vector_store"):
        self.embedding_dim = embedding_dim
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.documents = []
        self.metadata = []
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """Add documents and their embeddings to the vector store."""
        if metadata is None:
            metadata = [{}] * len(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def clear_all(self):
        """Clear all documents and embeddings from the vector store."""
        try:
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Clear documents and metadata
            self.documents = []
            self.metadata = []
            
            # Save the empty index
            self._save_index()
            
            logger.info("Vector store cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents with enhanced accuracy."""
        if self.index.ntotal == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search with more results initially for better filtering
        search_k = min(k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and idx >= 0:
                # Ensure score is reasonable (between 0 and 1 for cosine similarity)
                normalized_score = max(0.0, min(float(score), 1.0))

                # Enhanced scoring with multiple factors
                # 1. Base similarity score
                base_score = normalized_score
                
                # 2. Length bonus for more comprehensive chunks
                doc_length = len(self.documents[idx])
                length_bonus = min(doc_length / 2000, 0.1)  # Up to 0.1 bonus for longer docs
                
                # 3. Recency bonus (if timestamp available)
                recency_bonus = 0.0
                if 'timestamp' in self.metadata[idx]:
                    try:
                        from datetime import datetime, timedelta
                        doc_time = datetime.fromisoformat(self.metadata[idx]['timestamp'])
                        days_old = (datetime.now() - doc_time).days
                        recency_bonus = max(0, 0.05 - (days_old * 0.001))  # Up to 0.05 bonus for recent docs
                    except:
                        pass
                
                # Calculate final score
                final_score = min(base_score + length_bonus + recency_bonus, 1.0)
                
                # Only include results with meaningful scores
                if final_score > 0.1:  # Minimum threshold
                    results.append((
                        self.documents[idx],
                        final_score,
                        self.metadata[idx]
                    ))

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _save_index(self):
        """Save the FAISS index and metadata."""
        faiss.write_index(self.index, str(self.storage_path / "faiss_index.bin"))
        
        with open(self.storage_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(self.storage_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        index_path = self.storage_path / "faiss_index.bin"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            
            with open(self.storage_path / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
            
            with open(self.storage_path / "metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded existing index with {len(self.documents)} documents")


class QueryProcessor:
    """Handles query preprocessing and optimization for better search results."""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve search accuracy."""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Expand common abbreviations
        abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'db': 'database',
            'sql': 'structured query language',
            'pdf': 'portable document format',
            'doc': 'document',
            'docx': 'document',
            'txt': 'text file'
        }
        
        for abbr, full in abbreviations.items():
            query = query.replace(abbr, full)
        
        return query
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        words = query.split()
        keywords = []
        
        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 2 and word not in self.stop_words:
                keywords.append(word)
        
        return keywords
    
    def generate_search_variations(self, query: str) -> List[str]:
        """Generate variations of the query for better search coverage."""
        variations = [query]
        
        # Add keyword-only version
        keywords = self.extract_keywords(query)
        if keywords:
            variations.append(' '.join(keywords))
        
        # Add individual important keywords
        for keyword in keywords[:3]:  # Top 3 keywords
            if len(keyword) > 3:
                variations.append(keyword)
        
        return list(set(variations))  # Remove duplicates


class ReasoningEngine:
    """Handles multi-step reasoning and query decomposition."""
    
    def __init__(self):
        self.reasoning_steps = []
        self.query_processor = QueryProcessor()
    
    def decompose_query(self, query: str) -> List[str]:
        """Break down a complex query into smaller, manageable sub-queries."""
        # Simple decomposition based on common patterns
        sub_queries = []
        
        # Check for multiple questions
        if '?' in query:
            questions = [q.strip() for q in query.split('?') if q.strip()]
            sub_queries.extend(questions)
        else:
            # Check for conjunctions and complex structures
            conjunctions = [' and ', ' or ', ' but ', ' however ', ' furthermore ', ' additionally ']
            current_query = query
            
            for conj in conjunctions:
                if conj in current_query.lower():
                    parts = current_query.split(conj)
                    sub_queries.extend([part.strip() for part in parts if part.strip()])
                    break
            
            if not sub_queries:
                sub_queries = [query]
        
        return sub_queries
    
    def analyze_findings(self, findings: List[Dict]) -> Dict[str, Any]:
        """Analyze and synthesize findings from multiple sources."""
        if not findings:
            return {"summary": "No relevant information found.", "confidence": 0.0}
        
        # Extract key information
        sources = [f.get('source', 'Unknown') for f in findings]
        texts = [f.get('text', '') for f in findings]
        scores = [f.get('score', 0.0) for f in findings]
        
        # Calculate confidence based on scores
        if scores:
            # Filter out invalid scores
            valid_scores = [s for s in scores if 0 <= s <= 1]
            if valid_scores:
                avg_score = np.mean(valid_scores)
                # Boost confidence based on number of sources and average score
                source_bonus = min(len(valid_scores) * 0.1, 0.3)  # Up to 0.3 bonus for more sources
                confidence = min(avg_score + source_bonus, 1.0)
                
                # Ensure minimum confidence for any results
                if len(valid_scores) > 0:
                    confidence = max(confidence, 0.2)  # Minimum 20% confidence if we have results
            else:
                confidence = 0.0
        else:
            confidence = 0.0
        
        # Create summary
        summary = self._create_summary(texts, sources)
        
        return {
            "summary": summary,
            "confidence": confidence,
            "sources": sources,
            "num_sources": len(findings),
            "avg_relevance_score": avg_score
        }
    
    def _create_summary(self, texts: List[str], sources: List[str]) -> str:
        """Create a coherent summary from multiple text sources."""
        if not texts:
            return "No information available."
        
        # Process texts to create coherent sentences
        processed_texts = []
        for text in texts[:3]:  # Use top 3 most relevant sources
            # Clean up the text
            cleaned_text = text.strip()
            if cleaned_text:
                # Ensure text starts with a complete word/sentence
                words = cleaned_text.split()
                if len(words) > 1:
                    # Check if first word might be incomplete
                    first_word = words[0]
                    if len(first_word) < 3 or not first_word[0].isupper():
                        # Try to find a better starting point
                        space_pos = cleaned_text.find(' ')
                        if space_pos > 0 and space_pos < len(cleaned_text) // 2:
                            cleaned_text = cleaned_text[space_pos + 1:].strip()
                
                # Ensure text ends with proper punctuation
                if not cleaned_text.endswith(('.', '!', '?', ':')):
                    # Try to find a good break point
                    last_period = cleaned_text.rfind('.')
                    last_exclamation = cleaned_text.rfind('!')
                    last_question = cleaned_text.rfind('?')
                    last_colon = cleaned_text.rfind(':')
                    
                    # Find the last sentence ending
                    last_sentence_end = max(last_period, last_exclamation, last_question, last_colon)
                    
                    if last_sentence_end > 0:
                        cleaned_text = cleaned_text[:last_sentence_end + 1]
                    else:
                        # If no sentence ending found, add a period
                        cleaned_text = cleaned_text + "."
                
                processed_texts.append(cleaned_text)
        
        # Join texts with proper spacing
        combined_text = " ".join(processed_texts)
        
        # Clean up any double spaces or weird concatenations
        import re
        combined_text = re.sub(r'\s+', ' ', combined_text)  # Replace multiple spaces with single space
        combined_text = re.sub(r'([a-z])([A-Z])', r'\1. \2', combined_text)  # Add period between sentences
        combined_text = re.sub(r'\.\s*\.', '.', combined_text)  # Remove double periods
        combined_text = re.sub(r'\s+', ' ', combined_text)  # Clean up spaces again
        
        # Truncate if too long, but try to end at a sentence boundary
        if len(combined_text) > 2000:
            truncated = combined_text[:2000]
            # Find the last complete sentence
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            
            last_sentence_end = max(last_period, last_exclamation, last_question)
            if last_sentence_end > 1000:  # Only if we have a reasonable amount of text
                combined_text = truncated[:last_sentence_end + 1] + "..."
            else:
                combined_text = truncated + "..."
        
        return f"Based on analysis of {len(sources)} sources: {combined_text}"


class DeepResearcher:
    """Main Deep Researcher Agent class."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.document_processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine(model_name)
        self.vector_store = VectorStore(self.embedding_engine.embedding_dim)
        self.reasoning_engine = ReasoningEngine()
        
        self.research_history = []
        logger.info("Deep Researcher Agent initialized successfully")
    
    def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the research database with improved chunking."""
        results = {
            "processed": 0,
            "failed": 0,
            "errors": [],
            "chunks_created": 0
        }
        
        for doc_path in tqdm(document_paths, desc="Processing documents"):
            try:
                # Extract text
                text = self.document_processor.extract_text(doc_path)
                if not text.strip():
                    results["failed"] += 1
                    results["errors"].append(f"No text extracted from {doc_path}")
                    continue
                
                # Chunk the document for better search accuracy
                chunks = self.document_processor.chunk_document(text, chunk_size=800, overlap=150)
                
                if not chunks:
                    results["failed"] += 1
                    results["errors"].append(f"No chunks created from {doc_path}")
                    continue
                
                # Generate embeddings for all chunks
                embeddings = self.embedding_engine.generate_embeddings(chunks)
                
                # Create metadata for each chunk
                chunk_metadata = []
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": doc_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "text_length": len(chunk),
                        "chunk_size": len(chunk)
                    }
                    chunk_metadata.append(metadata)
                
                # Add all chunks to vector store
                self.vector_store.add_documents(chunks, embeddings, chunk_metadata)
                results["processed"] += 1
                results["chunks_created"] += len(chunks)
                
                logger.info(f"Processed {doc_path}: {len(chunks)} chunks created")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error processing {doc_path}: {str(e)}")
                logger.error(f"Error processing {doc_path}: {e}")
        
        # Save the updated index
        self.vector_store._save_index()
        
        logger.info(f"Document processing complete: {results['processed']} processed, {results['failed']} failed, {results['chunks_created']} chunks created")
        return results
    
    def research(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform research on a given query with enhanced accuracy."""
        logger.info(f"Starting research for query: {query}")
        
        # Preprocess the query
        processed_query = self.reasoning_engine.query_processor.preprocess_query(query)
        logger.info(f"Processed query: {processed_query}")
        
        # Generate search variations
        search_variations = self.reasoning_engine.query_processor.generate_search_variations(processed_query)
        logger.info(f"Generated {len(search_variations)} search variations")
        
        # Search with multiple query variations
        all_findings = []
        seen_texts = set()  # To avoid duplicates
        
        for variation in search_variations:
            if not variation.strip():
                continue
                
            # Generate embedding for this variation
            query_embedding = self.embedding_engine.generate_single_embedding(variation)
            
            # Search for relevant documents
            search_results = self.vector_store.search(query_embedding, k=max_results)
            
            # Add unique findings
            for text, score, metadata in search_results:
                # Create a hash of the text to avoid duplicates
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_findings.append({
                        "text": text,
                        "score": score,
                        "source": metadata.get("source", "Unknown"),
                        "chunk_info": f"Chunk {metadata.get('chunk_index', '?')}/{metadata.get('total_chunks', '?')}",
                        "metadata": metadata,
                        "query_variation": variation
                    })
        
        # Sort by score and take top results
        all_findings.sort(key=lambda x: x["score"], reverse=True)
        findings = all_findings[:max_results]
        
        # Analyze findings using reasoning engine
        analysis = self.reasoning_engine.analyze_findings(findings)
        
        # Create research result
        result = {
            "query": query,
            "processed_query": processed_query,
            "search_variations": search_variations,
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "analysis": analysis,
            "num_documents_searched": len(self.vector_store.documents),
            "total_chunks_searched": len(self.vector_store.documents)
        }
        
        # Store in research history
        self.research_history.append(result)
        
        logger.info(f"Research completed. Found {len(findings)} relevant documents from {len(all_findings)} total matches.")
        return result
    
    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get the history of research queries."""
        return self.research_history
    
    def export_research(self, research_result: Dict[str, Any], format: str = "markdown") -> str:
        """Export research results to various formats."""
        if format.lower() == "markdown":
            return self._export_markdown(research_result)
        elif format.lower() == "json":
            return json.dumps(research_result, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_markdown(self, research_result: Dict[str, Any]) -> str:
        """Export research results as Markdown."""
        md = f"# Research Report\n\n"
        md += f"**Query:** {research_result['query']}\n\n"
        md += f"**Timestamp:** {research_result['timestamp']}\n\n"
        md += f"**Documents Searched:** {research_result['num_documents_searched']}\n\n"
        
        analysis = research_result['analysis']
        md += f"## Summary\n\n{analysis['summary']}\n\n"
        md += f"**Confidence Score:** {analysis['confidence']:.2f}\n\n"
        md += f"**Sources Used:** {analysis['num_sources']}\n\n"
        
        md += "## Detailed Findings\n\n"
        for i, finding in enumerate(research_result['findings'], 1):
            md += f"### Finding {i}\n\n"
            md += f"**Source:** {finding['source']}\n\n"
            md += f"**Relevance Score:** {finding['score']:.3f}\n\n"
            md += f"**Content:**\n{finding['text'][:500]}...\n\n"
        
        return md


# Example usage and testing
if __name__ == "__main__":
    # Initialize the researcher
    researcher = DeepResearcher()
    
    # Example: Add some sample documents (you would replace these with actual documents)
    sample_docs = [
        "Artificial intelligence is transforming various industries.",
        "Machine learning algorithms can process large amounts of data.",
        "Deep learning models require significant computational resources.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information."
    ]
    
    # Add sample documents
    for i, doc in enumerate(sample_docs):
        embedding = researcher.embedding_engine.generate_single_embedding(doc)
        metadata = {"source": f"sample_doc_{i}.txt", "timestamp": datetime.now().isoformat()}
        researcher.vector_store.add_documents([doc], embedding.reshape(1, -1), [metadata])
    
    # Perform research
    result = researcher.research("What is artificial intelligence?")
    print("\nResearch Result:")
    print(json.dumps(result, indent=2))
    
    # Export as markdown
    markdown_report = researcher.export_research(result, "markdown")
    print("\nMarkdown Report:")
    print(markdown_report)
