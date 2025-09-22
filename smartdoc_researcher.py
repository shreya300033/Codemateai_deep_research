"""
Advanced Deep Researcher Agent with enhanced capabilities including
summarization, interactive query refinement, and advanced export features.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import re

# Core ML libraries
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Document processing
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import requests

# Storage and retrieval
# Note: ChromaDB removed to avoid dependency issues

# Export functionality
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import markdown
from markdown.extensions import codehilite, fenced_code

# Utilities
from tqdm import tqdm
import pickle
import hashlib

# Import base classes
from smartdoc_analyzer import DocumentProcessor, EmbeddingEngine, VectorStore, ReasoningEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationEngine:
    """Advanced summarization capabilities using Hugging Face models."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading summarization model: {model_name}")
        
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                min_length=50
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, using extractive summarization: {e}")
            self.summarizer = None
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Summarize text using abstractive or extractive methods."""
        if not text.strip():
            return ""
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        if self.summarizer and len(text) > 100:
            try:
                # Use abstractive summarization
                result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Abstractive summarization failed: {e}")
        
        # Fallback to extractive summarization
        return self._extractive_summarize(text, max_length)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for summarization."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """Simple extractive summarization by selecting key sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # Simple scoring based on word frequency and position
        word_freq = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if len(word) > 3:  # Only count meaningful words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if len(word) > 3)
            # Boost score for sentences at the beginning
            position_bonus = 1.0 - (i / len(sentences)) * 0.3
            sentence_scores.append((score * position_bonus, sentence))
        
        # Select top sentences (limit to max_length)
        sentence_scores.sort(reverse=True)
        selected_sentences = []
        current_length = 0
        
        for score, sentence in sentence_scores:
            if current_length + len(sentence) <= max_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            if len(selected_sentences) >= 3:  # Limit to 3 sentences max
                break
        
        if not selected_sentences:
            # Fallback: return first few sentences
            selected_sentences = sentences[:2]
        
        return '. '.join(selected_sentences) + '.'


class InteractiveQueryRefiner:
    """Handles interactive query refinement and follow-up questions."""
    
    def __init__(self):
        self.conversation_history = []
        self.current_context = {}
    
    def refine_query(self, original_query: str, research_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent follow-up questions based on research results."""
        follow_up_questions = []
        
        # Analyze the research results to identify gaps or interesting areas
        findings = research_results.get('findings', [])
        analysis = research_results.get('analysis', {})
        
        if not findings:
            follow_up_questions.append("Could you provide more specific information about this topic?")
            return follow_up_questions
        
        # Extract key concepts from findings for context-aware questions
        key_concepts = self._extract_key_concepts(findings)
        confidence = analysis.get('confidence', 0.0)
        
        # Generate intelligent questions based on content analysis
        if confidence < 0.5:
            follow_up_questions.extend([
                "What are the main challenges or limitations in this area?",
                "Are there any recent developments or updates?",
                "What are the different perspectives on this topic?",
                "What specific examples can you provide?"
            ])
        elif confidence < 0.8:
            follow_up_questions.extend([
                "What are the practical applications and real-world use cases?",
                "How does this compare to alternative approaches or methods?",
                "What are the future implications and trends?",
                "What are the key technical requirements or prerequisites?"
            ])
        else:
            # High confidence - generate detailed, specific questions
            follow_up_questions.extend([
                "What are the specific technical details and implementation approaches?",
                "Can you provide concrete examples and case studies?",
                "How can this be practically implemented in real-world scenarios?",
                "What are the best practices and common pitfalls to avoid?"
            ])
        
        # Add context-specific questions based on findings content
        if key_concepts:
            if 'AI' in key_concepts or 'artificial intelligence' in key_concepts:
                follow_up_questions.extend([
                    "What are the different types and approaches to AI?",
                    "How is AI being applied across different industries?",
                    "What are the ethical considerations and challenges in AI?"
                ])
            elif 'machine learning' in key_concepts or 'ML' in key_concepts:
                follow_up_questions.extend([
                    "What are the main machine learning algorithms and techniques?",
                    "How do different ML approaches compare in terms of performance?",
                    "What are the data requirements and preprocessing steps?"
                ])
            elif 'deep learning' in key_concepts or 'neural networks' in key_concepts:
                follow_up_questions.extend([
                    "What are the different types of neural network architectures?",
                    "How do you choose the right network structure for a problem?",
                    "What are the training challenges and optimization techniques?"
                ])
        
        # Add comparison questions if multiple sources
        sources = analysis.get('sources', [])
        if len(sources) > 1:
            follow_up_questions.append("How do the different sources compare or complement each other?")
        
        # Add implementation questions
        follow_up_questions.extend([
            "What are the step-by-step implementation guidelines?",
            "What tools, frameworks, or technologies are typically used?"
        ])
        
        return follow_up_questions[:6]  # Limit to 6 questions
    
    def _extract_key_concepts(self, findings: List[Dict]) -> List[str]:
        """Extract key concepts from findings for context-aware question generation."""
        concepts = []
        
        for finding in findings[:3]:  # Use top 3 findings
            text = finding.get('text', '').lower()
            
            # Extract key AI/tech terms
            ai_terms = ['artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 
                        'neural networks', 'algorithms', 'algorithms', 'data science', 'nlp', 
                        'computer vision', 'robotics', 'automation', 'chatgpt', 'llm', 'llms']
            
            for term in ai_terms:
                if term in text:
                    concepts.append(term)
        
        return concepts
    
    def process_follow_up(self, follow_up_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a follow-up question with context from previous research."""
        self.conversation_history.append({
            "query": follow_up_query,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        # Enhance the query with context
        enhanced_query = self._enhance_query_with_context(follow_up_query, context)
        
        return {
            "original_query": follow_up_query,
            "enhanced_query": enhanced_query,
            "context_used": bool(context)
        }
    
    def _enhance_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance query with relevant context from previous research."""
        if not context:
            return query
        
        # Extract key terms from previous findings
        findings = context.get('findings', [])
        key_terms = set()
        
        for finding in findings[:3]:  # Use top 3 findings
            text = finding.get('text', '')
            # Extract potential key terms (simple approach)
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            key_terms.update(words[:5])  # Limit to 5 terms per finding
        
        if key_terms:
            context_terms = ', '.join(list(key_terms)[:3])
            return f"{query} (related to: {context_terms})"
        
        return query


class AdvancedExportEngine:
    """Advanced export functionality for PDF and Markdown formats with modern styling."""
    
    def __init__(self):
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'accent': '#f093fb',
            'success': '#48bb78',
            'warning': '#ed8936',
            'error': '#f56565',
            'info': '#4299e1',
            'dark': '#2d3748',
            'light': '#f7fafc',
            'gray': '#718096'
        }
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup modern custom styles for PDF export."""
        # Main title style
        self.styles.add(ParagraphStyle(
            name='ModernTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=1,  # Center alignment
            textColor=colors.HexColor(self.colors['primary']),
            fontName='Helvetica-Bold'
        ))
        
        # Section headings
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=16,
            textColor=colors.HexColor(self.colors['secondary']),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor(self.colors['primary']),
            borderPadding=8,
            backColor=colors.HexColor('#f7fafc')
        ))
        
        # Subsection headings
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor(self.colors['dark']),
            fontName='Helvetica-Bold'
        ))
        
        # Body text with better spacing
        self.styles.add(ParagraphStyle(
            name='ModernBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            spaceBefore=6,
            textColor=colors.HexColor(self.colors['dark']),
            fontName='Helvetica',
            leading=14
        ))
        
        # Highlighted text
        self.styles.add(ParagraphStyle(
            name='Highlighted',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            spaceBefore=6,
            textColor=colors.HexColor(self.colors['primary']),
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#e6fffa'),
            borderWidth=1,
            borderColor=colors.HexColor(self.colors['success']),
            borderPadding=4
        ))
        
        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            textColor=colors.HexColor(self.colors['gray']),
            fontName='Helvetica-Oblique'
        ))
        
        # Quote style
        self.styles.add(ParagraphStyle(
            name='Quote',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            spaceBefore=8,
            leftIndent=20,
            rightIndent=20,
            textColor=colors.HexColor(self.colors['dark']),
            fontName='Helvetica-Oblique',
            backColor=colors.HexColor('#f7fafc'),
            borderWidth=1,
            borderColor=colors.HexColor(self.colors['gray']),
            borderPadding=8
        ))
    
    def export_to_pdf(self, research_result: Dict[str, Any], output_path: str) -> str:
        """Export research results to modern PDF format with enhanced styling."""
        doc = SimpleDocTemplate(
            output_path, 
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        story = []
        
        # Header with logo and title
        story.append(self._create_header())
        story.append(Spacer(1, 20))
        
        # Main title
        title = Paragraph("📚 SmartDoc Explorer Research Report", self.styles['ModernTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Research metadata card
        story.append(self._create_metadata_card(research_result))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("📋 Executive Summary", self.styles['SectionHeading']))
        analysis = research_result.get('analysis', {})
        
        if isinstance(analysis, dict):
            summary_text = analysis.get('summary', 'No summary available.')
            confidence = analysis.get('confidence', 0.0)
        elif isinstance(analysis, str):
            summary_text = analysis
            confidence = 0.0
        else:
            summary_text = str(analysis) if analysis else 'No summary available.'
            confidence = 0.0
        
        # Add confidence indicator
        if confidence > 0:
            confidence_text = f"<b>Confidence Level:</b> {confidence:.1%}"
            story.append(Paragraph(confidence_text, self.styles['Highlighted']))
            story.append(Spacer(1, 8))
        
        story.append(Paragraph(summary_text, self.styles['ModernBody']))
        story.append(Spacer(1, 20))
        
        # Key Statistics
        story.append(Paragraph("📊 Key Statistics", self.styles['SectionHeading']))
        story.append(self._create_statistics_table(research_result))
        story.append(Spacer(1, 20))
        
        # Detailed Findings
        story.append(Paragraph("🔍 Detailed Findings", self.styles['SectionHeading']))
        findings = research_result.get('findings', [])
        
        if findings:
            for i, finding in enumerate(findings[:10], 1):  # Limit to top 10 findings
                story.append(self._create_finding_card(finding, i))
                story.append(Spacer(1, 12))
        else:
            story.append(Paragraph("No specific findings available.", self.styles['ModernBody']))
        
        # Follow-up Questions
        follow_up_questions = research_result.get('follow_up_questions', [])
        if follow_up_questions:
            story.append(Paragraph("💡 Suggested Follow-up Questions", self.styles['SectionHeading']))
            for i, question in enumerate(follow_up_questions[:5], 1):
                story.append(Paragraph(f"{i}. {question}", self.styles['Quote']))
                story.append(Spacer(1, 6))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(self._create_footer())
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _create_header(self):
        """Create a modern header for the PDF."""
        header_data = [
            ['📚 SmartDoc Explorer', 'AI-Powered Research Platform'],
            ['Generated Report', f'{datetime.now().strftime("%B %d, %Y at %I:%M %p")}']
        ]
        
        header_table = Table(header_data, colWidths=[3*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(self.colors['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 16),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.whitesmoke)
        ]))
        
        return header_table
    
    def _create_metadata_card(self, research_result):
        """Create a metadata information card."""
        query = research_result.get('query', 'Unknown query')
        timestamp = research_result.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        num_docs = research_result.get('num_documents_searched', 0)
        findings_count = len(research_result.get('findings', []))
        
        metadata_data = [
            ['Research Query', query],
            ['Generated', timestamp],
            ['Documents Searched', str(num_docs)],
            ['Relevant Findings', str(findings_count)]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor(self.colors['light'])),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor(self.colors['dark'])),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor(self.colors['dark'])),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(self.colors['gray']))
        ]))
        
        return metadata_table
    
    def _create_statistics_table(self, research_result):
        """Create a statistics table."""
        findings = research_result.get('findings', [])
        if not findings:
            return Paragraph("No statistics available.", self.styles['ModernBody'])
        
        # Calculate statistics
        scores = [f.get('score', 0.0) for f in findings if isinstance(f, dict)]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Findings', str(len(findings))],
            ['Average Relevance', f'{avg_score:.3f}'],
            ['Highest Relevance', f'{max_score:.3f}'],
            ['Lowest Relevance', f'{min_score:.3f}']
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['secondary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(self.colors['gray']))
        ]))
        
        return stats_table
    
    def _create_finding_card(self, finding, index):
        """Create a styled finding card."""
        if not isinstance(finding, dict):
            return Paragraph(f"Finding {index}: {str(finding)}", self.styles['ModernBody'])
        
        source = finding.get('source', 'Unknown Source')
        score = finding.get('score', 0.0)
        text = finding.get('text', 'No content available.')
        
        # Truncate text for PDF
        if len(text) > 500:
            text = text[:500] + "..."
        
        # Create finding card
        finding_title = f"Finding {index}"
        source_info = f"<b>Source:</b> {source} | <b>Relevance:</b> {score:.3f}"
        
        # Use a table for better formatting
        finding_data = [
            [finding_title, f"Relevance: {score:.3f}"],
            [source_info, ""],
            [text, ""]
        ]
        
        finding_table = Table(finding_data, colWidths=[5.5*inch, 0.5*inch])
        finding_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor(self.colors['primary'])),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.whitesmoke),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor(self.colors['success'])),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor(self.colors['light'])),
            ('BACKGROUND', (0, 2), (-1, 2), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
            ('FONTNAME', (0, 2), (-1, 2), 'Helvetica'),
            ('FONTSIZE', (0, 0), (0, 0), 12),
            ('FONTSIZE', (1, 0), (1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, 1), 9),
            ('FONTSIZE', (0, 2), (-1, 2), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(self.colors['gray']))
        ]))
        
        return finding_table
    
    def _create_footer(self):
        """Create a footer for the PDF."""
        footer_text = f"Generated by SmartDoc Explorer | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | AI-Powered Research Platform"
        footer = Paragraph(footer_text, self.styles['Metadata'])
        return footer
    
    def export_to_markdown(self, research_result: Dict[str, Any], output_path: str) -> str:
        """Export research results to Markdown format."""
        md_content = self._generate_markdown_content(research_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return output_path
    
    def _generate_markdown_content(self, research_result: Dict[str, Any]) -> str:
        """Generate modern markdown content for research results."""
        md_content = []
        
        # Header with branding
        md_content.append("# 📚 SmartDoc Explorer Research Report")
        md_content.append("")
        md_content.append("*AI-Powered Research Platform*")
        md_content.append("")
        md_content.append("---")
        md_content.append("")
        
        # Research metadata
        query = research_result.get('query', 'Unknown query')
        timestamp = research_result.get('timestamp', 'Unknown timestamp')
        num_docs = research_result.get('num_documents_searched', 0)
        findings_count = len(research_result.get('findings', []))
        
        md_content.append("## 📋 Research Information")
        md_content.append("")
        md_content.append(f"| Field | Value |")
        md_content.append(f"|-------|-------|")
        md_content.append(f"| **Research Query** | {query} |")
        md_content.append(f"| **Generated** | {timestamp} |")
        md_content.append(f"| **Documents Searched** | {num_docs} |")
        md_content.append(f"| **Relevant Findings** | {findings_count} |")
        md_content.append("")
        
        # Executive Summary
        analysis = research_result.get('analysis', {})
        md_content.append("## 📋 Executive Summary")
        md_content.append("")
        
        if isinstance(analysis, dict):
            summary_text = analysis.get('summary', 'No summary available.')
            confidence = analysis.get('confidence', 0.0)
        elif isinstance(analysis, str):
            summary_text = analysis
            confidence = 0.0
        else:
            summary_text = str(analysis) if analysis else 'No summary available.'
            confidence = 0.0
        
        # Add confidence indicator
        if confidence > 0:
            md_content.append(f"**🎯 Confidence Level:** {confidence:.1%}")
            md_content.append("")
        
        md_content.append(summary_text)
        md_content.append("")
        
        # Key Statistics
        findings = research_result.get('findings', [])
        if findings:
            md_content.append("## 📊 Key Statistics")
            md_content.append("")
            
            scores = [f.get('score', 0.0) for f in findings if isinstance(f, dict)]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0
            
            md_content.append(f"| Metric | Value |")
            md_content.append(f"|--------|-------|")
            md_content.append(f"| **Total Findings** | {len(findings)} |")
            md_content.append(f"| **Average Relevance** | {avg_score:.3f} |")
            md_content.append(f"| **Highest Relevance** | {max_score:.3f} |")
            md_content.append(f"| **Lowest Relevance** | {min_score:.3f} |")
            md_content.append("")
        
        # Detailed findings
        md_content.append("## 🔍 Detailed Findings")
        md_content.append("")
        
        if findings:
            for i, finding in enumerate(findings, 1):
                md_content.append(f"### 📄 Finding {i}")
                md_content.append("")
                
                if isinstance(finding, dict):
                    source = finding.get('source', 'Unknown Source')
                    score = finding.get('score', 0.0)
                    text = finding.get('text', '')
                    
                    # Create a nice card-like format
                    md_content.append(f"**📁 Source:** `{source}`")
                    md_content.append(f"**⭐ Relevance Score:** `{score:.3f}`")
                    md_content.append("")
                    md_content.append("> " + text.replace('\n', '\n> '))
                else:
                    md_content.append(f"> {str(finding)}")
                
                md_content.append("")
                md_content.append("---")
                md_content.append("")
        else:
            md_content.append("*No specific findings available.*")
            md_content.append("")
        
        # Follow-up Questions
        follow_up_questions = research_result.get('follow_up_questions', [])
        if follow_up_questions:
            md_content.append("## 💡 Suggested Follow-up Questions")
            md_content.append("")
            for i, question in enumerate(follow_up_questions, 1):
                md_content.append(f"{i}. {question}")
            md_content.append("")
        
        # Footer
        md_content.append("---")
        md_content.append("")
        md_content.append(f"*Generated by SmartDoc Explorer | {timestamp} | AI-Powered Research Platform*")
        
        return "\n".join(md_content)


class AdvancedDeepResearcher:
    """Enhanced Deep Researcher Agent with advanced capabilities."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_online: bool = False, api_token: str = None):
        # Initialize base components
        self.document_processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine(model_name, use_online, api_token)
        self.vector_store = VectorStore(self.embedding_engine.embedding_dim)
        self.reasoning_engine = ReasoningEngine()
        
        # Initialize advanced components
        self.summarization_engine = SummarizationEngine()
        self.query_refiner = InteractiveQueryRefiner()
        self.export_engine = AdvancedExportEngine()
        
        self.research_history = []
        self.conversation_context = {}
        
        logger.info("Advanced Deep Researcher Agent initialized successfully")
    
    def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the research database with enhanced processing and chunking."""
        results = {
            "processed": 0,
            "failed": 0,
            "errors": [],
            "summaries": [],
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
                
                # Generate summary for the entire document
                if not hasattr(self, 'summarization_engine') or self.summarization_engine is None:
                    self.summarization_engine = SummarizationEngine()
                
                try:
                    summary = self.summarization_engine.summarize_text(text, max_length=200)
                except Exception as e:
                    logger.warning(f"Summarization failed, using text excerpt: {e}")
                    summary = text[:200] + "..." if len(text) > 200 else text
                results["summaries"].append({
                    "source": doc_path,
                    "summary": summary
                })
                
                # Chunk the document for better search accuracy
                chunks = self.document_processor.chunk_document(text, chunk_size=800, overlap=150)
                
                if not chunks:
                    results["failed"] += 1
                    results["errors"].append(f"No chunks created from {doc_path}")
                    continue
                
                # Generate embeddings for all chunks
                embeddings = self.embedding_engine.generate_embeddings(chunks)
                
                # Create enhanced metadata for each chunk
                chunk_metadata = []
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": doc_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "text_length": len(chunk),
                        "chunk_size": len(chunk),
                        "document_summary": summary,
                        "word_count": len(chunk.split())
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
    
    def clear_documents(self):
        """Clear all documents from the vector store and reset the database."""
        try:
            # Clear the vector store
            self.vector_store.clear_all()
            
            # Reset research history
            self.research_history = []
            
            # Reinitialize summarization engine (don't set to None)
            self.summarization_engine = SummarizationEngine()
            
            logger.info("All documents cleared from the research database")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return False
    
    def research(self, query: str, max_results: int = 10, enable_refinement: bool = True) -> Dict[str, Any]:
        """Perform enhanced research with summarization and refinement capabilities."""
        logger.info(f"Starting advanced research for query: {query}")
        
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
                        "query_variation": variation,
                        "summary": metadata.get("document_summary", "")
                    })
        
        # Sort by score and take top results
        all_findings.sort(key=lambda x: x["score"], reverse=True)
        findings = all_findings[:max_results]
        
        # Analyze findings using reasoning engine
        analysis = self.reasoning_engine.analyze_findings(findings)
        
        # Generate comprehensive summary
        if findings:
            if not hasattr(self, 'summarization_engine') or self.summarization_engine is None:
                self.summarization_engine = SummarizationEngine()
            
            all_texts = [f["text"] for f in findings[:5]]  # Use top 5 findings
            try:
                comprehensive_summary = self.summarization_engine.summarize_text(
                    " ".join(all_texts), 
                    max_length=300
                )
                analysis["comprehensive_summary"] = comprehensive_summary
            except Exception as e:
                logger.warning(f"Comprehensive summarization failed: {e}")
                analysis["comprehensive_summary"] = " ".join(all_texts)[:300] + "..."
        
        # Generate follow-up questions if enabled
        follow_up_questions = []
        if enable_refinement:
            follow_up_questions = self.query_refiner.refine_query(query, {
                "findings": findings,
                "analysis": analysis
            })
        
        # Create enhanced research result
        result = {
            "query": query,
            "processed_query": processed_query,
            "search_variations": search_variations,
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "analysis": analysis,
            "follow_up_questions": follow_up_questions,
            "num_documents_searched": len(self.vector_store.documents),
            "total_chunks_searched": len(self.vector_store.documents),
            "reasoning_steps": self.reasoning_engine.reasoning_steps
        }
        
        # Store in research history
        self.research_history.append(result)
        
        # Update conversation context
        self.conversation_context = {
            "last_query": query,
            "last_findings": findings,
            "last_analysis": analysis
        }
        
        logger.info(f"Advanced research completed. Found {len(findings)} relevant documents.")
        return result
    
    def ask_follow_up(self, follow_up_query: str) -> Dict[str, Any]:
        """Process a follow-up question with context from previous research."""
        if not self.conversation_context:
            return self.research(follow_up_query)
        
        # Process the follow-up with context
        refinement_result = self.query_refiner.process_follow_up(
            follow_up_query, 
            self.conversation_context
        )
        
        # Perform research with enhanced query
        enhanced_query = refinement_result["enhanced_query"]
        result = self.research(enhanced_query, enable_refinement=False)
        
        # Add follow-up context
        result["is_follow_up"] = True
        result["original_follow_up"] = follow_up_query
        result["enhanced_query"] = enhanced_query
        
        return result
    
    def export_research(self, research_result: Dict[str, Any], format: str = "markdown", output_path: str = None) -> str:
        """Export research results with enhanced formatting."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"research_report_{timestamp}.{format}"
        
        if format.lower() == "pdf":
            return self.export_engine.export_to_pdf(research_result, output_path)
        elif format.lower() == "markdown":
            return self.export_engine.export_to_markdown(research_result, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of all research conducted."""
        if not self.research_history:
            return {"message": "No research conducted yet"}
        
        total_queries = len(self.research_history)
        total_documents = len(self.vector_store.documents)
        
        # Calculate average confidence
        confidences = [r.get("analysis", {}).get("confidence", 0) for r in self.research_history]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            "total_queries": total_queries,
            "total_documents": total_documents,
            "average_confidence": avg_confidence,
            "recent_queries": [r["query"] for r in self.research_history[-5:]]
        }


# Example usage
if __name__ == "__main__":
    # Initialize the advanced researcher
    researcher = AdvancedDeepResearcher()
    
    # Example: Add some sample documents
    sample_docs = [
        "Artificial intelligence is transforming various industries including healthcare, finance, and transportation.",
        "Machine learning algorithms can process large amounts of data to identify patterns and make predictions.",
        "Deep learning models require significant computational resources but can achieve state-of-the-art performance.",
        "Natural language processing enables computers to understand and generate human language effectively.",
        "Computer vision allows machines to interpret and analyze visual information from images and videos."
    ]
    
    # Add sample documents
    for i, doc in enumerate(sample_docs):
        embedding = researcher.embedding_engine.generate_single_embedding(doc)
        metadata = {
            "source": f"sample_doc_{i}.txt", 
            "timestamp": datetime.now().isoformat(),
            "summary": researcher.summarization_engine.summarize_text(doc)
        }
        researcher.vector_store.add_documents([doc], embedding.reshape(1, -1), [metadata])
    
    # Perform research
    result = researcher.research("What are the applications of artificial intelligence?")
    print("\nAdvanced Research Result:")
    print(json.dumps(result, indent=2))
    
    # Export as markdown
    markdown_path = researcher.export_research(result, "markdown")
    print(f"\nMarkdown report exported to: {markdown_path}")
    
    # Export as PDF
    pdf_path = researcher.export_research(result, "pdf")
    print(f"PDF report exported to: {pdf_path}")
