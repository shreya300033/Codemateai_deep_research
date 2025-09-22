# SmartDoc Explorer - Usage Guide

## Overview

SmartDoc Explorer is a sophisticated AI-powered research platform that uses local embeddings and multi-step reasoning to search, analyze, and synthesize information from large-scale data sources. It operates entirely locally without relying on external web search APIs.

## Key Features

- **Local Embedding Generation**: Uses Hugging Face models for local embedding generation
- **Multi-step Reasoning**: Breaks down complex queries into smaller, manageable tasks
- **Efficient Storage & Retrieval**: Vector-based document indexing and similarity search
- **Document Processing**: Supports PDF, DOCX, TXT, HTML, and Markdown files
- **Interactive Query Refinement**: Follow-up questions and deeper exploration
- **Export Capabilities**: Generate research reports in PDF and Markdown formats
- **AI-powered Explanations**: Shows reasoning steps and decision-making process

## Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for large document collections)
- 2GB+ free disk space for vector storage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: GPU Support

For faster processing with GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run smartdoc_explorer.py
```

Then open your browser to `http://localhost:8501`

### 2. Command Line Interface

```bash
# Interactive mode
python smartdoc_cli.py --interactive

# Single query mode
python smartdoc_cli.py --query "What is artificial intelligence?" --documents doc1.pdf doc2.txt

# Export results
python smartdoc_cli.py --query "Machine learning applications" --format markdown --output report.md
```

### 3. Python API

```python
from smartdoc_researcher import AdvancedDeepResearcher

# Initialize researcher
researcher = AdvancedDeepResearcher()

# Add documents
researcher.add_documents(["document1.pdf", "document2.txt"])

# Conduct research
result = researcher.research("Your research query")

# Export results
researcher.export_research(result, "pdf", "report.pdf")
```

## Detailed Usage

### Document Management

#### Supported File Formats

- **PDF**: `.pdf` files
- **Microsoft Word**: `.docx` files
- **Plain Text**: `.txt` files
- **HTML**: `.html` files
- **Markdown**: `.md` files

#### Adding Documents

```python
# Single document
researcher.add_documents(["document.pdf"])

# Multiple documents
researcher.add_documents(["doc1.pdf", "doc2.docx", "doc3.txt"])

# From directory
import glob
documents = glob.glob("documents/*.pdf")
researcher.add_documents(documents)
```

#### Document Processing Results

The system provides detailed feedback on document processing:

```python
results = researcher.add_documents(["doc1.pdf", "doc2.txt"])
print(f"Processed: {results['processed']}")
print(f"Failed: {results['failed']}")
print(f"Errors: {results['errors']}")
```

### Research Queries

#### Basic Research

```python
# Simple query
result = researcher.research("What is machine learning?")

# Advanced query with options
result = researcher.research(
    query="What are the applications of AI in healthcare?",
    max_results=15,
    enable_refinement=True
)
```

#### Query Refinement

The system automatically generates follow-up questions:

```python
result = researcher.research("What is artificial intelligence?")
follow_up_questions = result['follow_up_questions']

# Ask follow-up questions
follow_up_result = researcher.ask_follow_up("What are the challenges in AI development?")
```

#### Research Results Structure

```python
result = researcher.research("Your query")

# Access different parts of the result
query = result['query']
timestamp = result['timestamp']
findings = result['findings']  # List of relevant documents
analysis = result['analysis']  # Summary and confidence scores
follow_up_questions = result['follow_up_questions']

# Analysis details
confidence = analysis['confidence']
summary = analysis['summary']
num_sources = analysis['num_sources']
```

### Export Functionality

#### Markdown Export

```python
# Export to markdown
markdown_path = researcher.export_research(result, "markdown", "report.md")

# Download the file
with open(markdown_path, 'r') as f:
    markdown_content = f.read()
```

#### PDF Export

```python
# Export to PDF
pdf_path = researcher.export_research(result, "pdf", "report.pdf")

# The PDF includes:
# - Executive summary
# - Confidence scores
# - Detailed findings
# - Source information
```

#### Custom Export

```python
# Get raw data for custom processing
result = researcher.research("Your query")
json_data = json.dumps(result, indent=2)
```

### Advanced Features

#### Multi-step Reasoning

The system automatically breaks down complex queries:

```python
# Complex query gets decomposed
result = researcher.research("What is AI and how does it work in healthcare and finance?")
# The system will:
# 1. Identify key concepts (AI, healthcare, finance)
# 2. Search for each concept
# 3. Synthesize findings
# 4. Generate comprehensive summary
```

#### Confidence Scoring

```python
result = researcher.research("Your query")
confidence = result['analysis']['confidence']

if confidence >= 0.7:
    print("High confidence in results")
elif confidence >= 0.4:
    print("Medium confidence in results")
else:
    print("Low confidence - consider refining query")
```

#### Research History

```python
# Access research history
history = researcher.get_research_summary()
print(f"Total queries: {history['total_queries']}")
print(f"Documents indexed: {history['total_documents']}")
print(f"Average confidence: {history['average_confidence']}")

# Get specific research results
research_history = researcher.research_history
for research in research_history:
    print(f"Query: {research['query']}")
    print(f"Confidence: {research['analysis']['confidence']}")
```

## Configuration

### Model Selection

```python
# Use different embedding models
researcher = AdvancedDeepResearcher(model_name="sentence-transformers/all-mpnet-base-v2")

# Available models:
# - sentence-transformers/all-MiniLM-L6-v2 (default, fast)
# - sentence-transformers/all-mpnet-base-v2 (larger, more accurate)
# - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
```

### Storage Configuration

```python
# Custom storage path
from smartdoc_analyzer import VectorStore
vector_store = VectorStore(embedding_dim=384, storage_path="./custom_storage")
```

### Performance Tuning

```python
# Batch processing for large document collections
researcher = AdvancedDeepResearcher()
# Process documents in batches
for batch in document_batches:
    researcher.add_documents(batch)
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Problem**: Out of memory errors with large documents
**Solution**: 
- Process documents in smaller batches
- Use smaller embedding models
- Increase system RAM

#### 2. Slow Processing

**Problem**: Slow embedding generation
**Solution**:
- Use GPU acceleration if available
- Use smaller embedding models
- Process documents in parallel

#### 3. Low Quality Results

**Problem**: Poor search results or low confidence scores
**Solution**:
- Add more relevant documents
- Use more specific queries
- Check document quality and format

#### 4. Export Issues

**Problem**: PDF export fails
**Solution**:
- Install reportlab: `pip install reportlab`
- Check file permissions
- Ensure sufficient disk space

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
researcher = AdvancedDeepResearcher()
```

## Examples

### Example 1: Academic Research

```python
# Research academic papers
researcher = AdvancedDeepResearcher()

# Add research papers
papers = glob.glob("papers/*.pdf")
researcher.add_documents(papers)

# Research specific topic
result = researcher.research("What are the latest developments in quantum computing?")

# Export comprehensive report
researcher.export_research(result, "pdf", "quantum_computing_research.pdf")
```

### Example 2: Business Intelligence

```python
# Analyze business documents
researcher = AdvancedDeepResearcher()

# Add business documents
business_docs = ["market_analysis.pdf", "competitor_report.docx", "financial_data.txt"]
researcher.add_documents(business_docs)

# Research market trends
result = researcher.research("What are the key market trends in our industry?")

# Get follow-up insights
follow_up = researcher.ask_follow_up("What are the competitive advantages mentioned?")
```

### Example 3: Technical Documentation

```python
# Search technical documentation
researcher = AdvancedDeepResearcher()

# Add technical docs
tech_docs = glob.glob("docs/*.md")
researcher.add_documents(tech_docs)

# Find specific information
result = researcher.research("How to implement authentication in the API?")

# Get implementation details
details = researcher.ask_follow_up("What are the security considerations?")
```

## Best Practices

### 1. Document Preparation

- Ensure documents are in supported formats
- Remove unnecessary formatting or metadata
- Use descriptive filenames
- Organize documents by topic or category

### 2. Query Formulation

- Be specific and clear in your queries
- Use relevant keywords and terminology
- Break complex questions into simpler parts
- Use follow-up questions to dig deeper

### 3. Performance Optimization

- Process documents in batches
- Use appropriate embedding models for your use case
- Monitor memory usage with large document collections
- Regularly clean up old vector stores

### 4. Result Interpretation

- Check confidence scores before trusting results
- Review source documents for verification
- Use follow-up questions for clarification
- Export results for further analysis

## Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Review the test cases in `test_researcher.py`
3. Run the example usage script: `python example_usage.py`
4. Check the logs for detailed error information

## License

This project is licensed under the MIT License. See the LICENSE file for details.

