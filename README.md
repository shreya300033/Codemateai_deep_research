# 📚 Deep Research AI

A sophisticated AI-powered research platform that can search, analyze, and synthesize information from large-scale data sources using local embeddings and multi-step reasoning.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.25+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🌐 **Modern Web Interface**: Clean, responsive Streamlit-based UI with real-time analytics
- 💻 **Enhanced CLI**: Rich terminal interface with progress bars and beautiful formatting
- 🧠 **Local AI Processing**: Uses Hugging Face models for local embedding generation
- 🔄 **Multi-step Reasoning**: Breaks down complex queries into manageable tasks
- 📊 **Vector-based Search**: Efficient document indexing and similarity search
- 📄 **Multi-format Support**: PDF, DOCX, TXT, HTML, and Markdown documents
- 💬 **Interactive Queries**: Follow-up questions and conversational research
- 📈 **Export Capabilities**: Generate research reports in PDF and Markdown formats
- 📊 **Analytics Dashboard**: Visual insights into research patterns and performance
- 🚀 **Easy Launch**: Simple launcher script for quick access

## 🚀 Quick Start

### Option 1: Easy Launcher (Recommended)
```bash
python smartdoc_launcher.py
```

### Option 2: Direct Launch
```bash
# Web Interface
streamlit run smartdoc_explorer.py

# CLI Interface
python smartdoc_cli.py --interactive

# Demonstration
python smartdoc_demo.py
```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd hack1
```

2. **Install Python 3.8 or higher**

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch the application:**
```bash
python smartdoc_launcher.py
```

## 🎯 Usage

### Web Interface
- 🌐 Modern, intuitive web interface
- 📤 Drag-and-drop document upload
- 🔍 Real-time research with visual feedback
- 📊 Interactive analytics dashboard
- 📄 One-click report export

### Command Line Interface
- 💻 Rich terminal interface with progress bars
- 🔄 Interactive mode for conversational research
- 📝 Batch processing capabilities
- 🎨 Beautiful formatting with colors and tables

### Python API
```python
from smartdoc_researcher import AdvancedDeepResearcher

# Initialize researcher
researcher = AdvancedDeepResearcher()

# Add documents
researcher.add_documents(["doc1.pdf", "doc2.txt"])

# Conduct research
result = researcher.research("Your research query")

# Export results
researcher.export_research(result, "pdf", "report.pdf")
```

## 🏗️ Architecture

- **📄 Document Processor**: Handles various file formats and extracts text
- **🧠 Embedding Engine**: Generates local embeddings using Hugging Face models
- **🗄️ Vector Store**: Efficient storage and retrieval using FAISS
- **🔄 Reasoning Engine**: Multi-step query decomposition and analysis
- **📊 Response Generator**: Synthesizes findings into coherent reports
- **🌐 Web Interface**: Modern Streamlit-based UI with real-time analytics
- **💻 CLI Interface**: Rich terminal interface with enhanced UX

## 📊 Supported File Types

- 📄 PDF documents
- 📝 Microsoft Word (.docx)
- 📄 Plain text (.txt)
- 🌐 HTML files
- 📝 Markdown files

## 🎨 UI Features

### Web Interface
- 🎨 Modern gradient design with clean typography
- 📊 Real-time metrics and analytics
- 🔄 Interactive progress indicators
- 📱 Responsive design for all devices
- 🎯 Intuitive navigation and user flow

### CLI Interface
- 🎨 Rich terminal formatting with colors and emojis
- 📊 Progress bars and loading indicators
- 📋 Beautiful tables and structured output
- 🔄 Interactive prompts and confirmations
- 📈 Real-time status updates

## 🔧 Configuration

The application can be configured through `config.py`:

- **Model Selection**: Choose different embedding models
- **Performance Settings**: Adjust batch sizes and workers
- **Storage Options**: Configure vector store and export paths
- **UI Themes**: Customize colors and styling

## 📚 Documentation

- 📖 **README.md**: This overview
- 📋 **USAGE_GUIDE.md**: Detailed usage instructions
- 🔍 **QUERY_GUIDE.md**: Query examples and tips
- 🎬 **smartdoc_demo.py**: Interactive demonstration script

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Streamlit for the web framework
- FAISS for vector search
- Rich for beautiful CLI output
