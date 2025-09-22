"""
Demonstration script for SmartDoc Explorer
This script showcases the key features and capabilities of the system.
"""

import os
import sys
from pathlib import Path
import tempfile
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smartdoc_researcher import AdvancedDeepResearcher

def create_sample_documents():
    """Create sample documents for demonstration."""
    sample_docs = {
        "ai_basics.txt": """
Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. 
The field has evolved significantly since its inception in the 1950s, with major breakthroughs in machine learning, 
deep learning, and neural networks. AI systems can now perform tasks that were previously thought to require human 
intelligence, such as image recognition, natural language processing, and strategic game playing.

The development of AI has been driven by advances in computational power, the availability of large datasets, 
and improvements in algorithms. Modern AI systems use techniques such as supervised learning, unsupervised learning, 
and reinforcement learning to acquire knowledge and improve their performance over time.
        """,
        
        "machine_learning.txt": """
Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical 
models that enable computer systems to improve their performance on a specific task through experience. Unlike 
traditional programming, where explicit instructions are provided, machine learning systems learn patterns from data.

There are three main types of machine learning:
1. Supervised Learning: Learning with labeled training data
2. Unsupervised Learning: Finding patterns in data without labels
3. Reinforcement Learning: Learning through interaction with an environment

Machine learning has applications in various fields including healthcare, finance, transportation, and entertainment.
        """,
        
        "deep_learning.txt": """
Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep networks) 
to model and understand complex patterns in data. Deep learning has revolutionized many areas of AI, particularly in 
computer vision, natural language processing, and speech recognition.

Key concepts in deep learning include:
- Neural Networks: Computing systems inspired by biological neural networks
- Backpropagation: Algorithm for training neural networks
- Convolutional Neural Networks (CNNs): Specialized for image processing
- Recurrent Neural Networks (RNNs): Designed for sequential data
- Transformers: Architecture that has revolutionized natural language processing

Deep learning requires significant computational resources and large amounts of data to train effectively.
        """,
        
        "applications.txt": """
AI Applications in Various Industries:

Healthcare:
- Medical image analysis and diagnosis
- Drug discovery and development
- Personalized treatment plans
- Robotic surgery assistance

Finance:
- Algorithmic trading
- Fraud detection
- Credit scoring
- Risk assessment

Transportation:
- Autonomous vehicles
- Traffic optimization
- Route planning
- Predictive maintenance

Entertainment:
- Recommendation systems
- Content generation
- Game AI
- Virtual assistants

These applications demonstrate the broad impact of AI across different sectors of the economy.
        """,
        
        "challenges.txt": """
Challenges and Considerations in AI Development:

Technical Challenges:
- Data quality and availability
- Computational requirements
- Model interpretability
- Bias and fairness in AI systems

Ethical Considerations:
- Privacy concerns
- Job displacement
- Algorithmic bias
- Autonomous decision-making

Regulatory and Legal Issues:
- Liability and accountability
- Intellectual property rights
- Safety standards
- International cooperation

Future Directions:
- Artificial General Intelligence (AGI)
- Human-AI collaboration
- Sustainable AI development
- Ethical AI frameworks
        """
    }
    
    # Create temporary directory for sample documents
    temp_dir = Path(tempfile.mkdtemp())
    
    for filename, content in sample_docs.items():
        file_path = temp_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    return temp_dir, list(sample_docs.keys())

def demonstrate_basic_research():
    """Demonstrate basic research functionality."""
    print("=" * 80)
    print("🔍 DEEP RESEARCHER AGENT - DEMONSTRATION")
    print("=" * 80)
    print("A sophisticated AI-powered research tool for intelligent document analysis")
    print("=" * 80)
    
    # Initialize researcher
    print("\n📋 Step 1: Initializing Deep Researcher Agent...")
    print("   Loading AI models and setting up vector database...")
    researcher = AdvancedDeepResearcher()
    print("   ✅ Researcher initialized successfully")
    
    # Create and add sample documents
    print("\n📄 Step 2: Creating and adding sample documents...")
    print("   Generating sample AI and machine learning content...")
    temp_dir, doc_names = create_sample_documents()
    
    # Get file paths
    doc_paths = [str(temp_dir / name) for name in doc_names]
    
    # Add documents to researcher
    print("   Processing documents and generating embeddings...")
    results = researcher.add_documents(doc_paths)
    print(f"   ✅ Processed {results['processed']} documents successfully")
    
    if results['failed'] > 0:
        print(f"   ⚠️  {results['failed']} documents failed to process")
    
    print(f"   📊 Total documents in database: {len(researcher.vector_store.documents)}")
    
    return researcher, temp_dir

def demonstrate_research_queries(researcher):
    """Demonstrate various research queries."""
    print("\n🔍 Step 3: Demonstrating Research Queries")
    print("   Testing various AI and machine learning questions...")
    print("-" * 60)
    
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of AI in healthcare?",
        "What challenges does AI development face?",
        "How does deep learning differ from traditional machine learning?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("   " + "─" * 50)
        
        # Conduct research
        print("   🔍 Conducting research...")
        result = researcher.research(query, max_results=3)
        
        # Display results
        analysis = result['analysis']
        print(f"   📊 Confidence Score: {analysis['confidence']:.2f}")
        print(f"   📚 Sources Found: {analysis['num_sources']}")
        print(f"   📄 Summary: {analysis['summary']}")
        
        # Show follow-up questions
        if result.get('follow_up_questions'):
            print("   💡 Suggested follow-up questions:")
            for j, follow_up in enumerate(result['follow_up_questions'][:2], 1):
                print(f"      {j}. {follow_up}")
        
        print("   ✅ Query completed successfully")

def demonstrate_follow_up_questions(researcher):
    """Demonstrate follow-up question functionality."""
    print("\n🔄 Step 4: Demonstrating Follow-up Questions")
    print("   Testing conversational research capabilities...")
    print("-" * 60)
    
    # Initial research
    print("\n📝 Initial Query: What is artificial intelligence?")
    print("   🔍 Conducting initial research...")
    initial_result = researcher.research("What is artificial intelligence?")
    print(f"   📊 Initial Confidence: {initial_result['analysis']['confidence']:.2f}")
    
    # Follow-up questions
    follow_up_queries = [
        "What are the main types of AI?",
        "How is AI being used in different industries?",
        "What are the technical challenges in AI development?"
    ]
    
    for i, follow_up in enumerate(follow_up_queries, 1):
        print(f"\n📝 Follow-up {i}: {follow_up}")
        print("   🔍 Processing follow-up query...")
        follow_up_result = researcher.ask_follow_up(follow_up)
        print(f"   📊 Confidence: {follow_up_result['analysis']['confidence']:.2f}")
        print(f"   📄 Summary: {follow_up_result['analysis']['summary'][:200]}...")
        print("   ✅ Follow-up completed")

def demonstrate_export_functionality(researcher):
    """Demonstrate export functionality."""
    print("\n📄 Step 5: Demonstrating Export Functionality")
    print("   Testing report generation capabilities...")
    print("-" * 60)
    
    # Conduct research for export
    print("\n📝 Conducting research for export...")
    result = researcher.research("What are the key applications and challenges of artificial intelligence?")
    print(f"   📊 Research completed with confidence: {result['analysis']['confidence']:.2f}")
    
    # Export to Markdown
    print("\n📝 Exporting to Markdown...")
    markdown_path = researcher.export_research(result, "markdown", "demo_research.md")
    print(f"   ✅ Markdown report exported to: {markdown_path}")
    
    # Export to PDF
    print("\n📄 Exporting to PDF...")
    pdf_path = researcher.export_research(result, "pdf", "demo_research.pdf")
    print(f"   ✅ PDF report exported to: {pdf_path}")
    
    # Show file sizes
    markdown_size = Path(markdown_path).stat().st_size
    pdf_size = Path(pdf_path).stat().st_size
    print(f"\n📊 File sizes:")
    print(f"   📝 Markdown: {markdown_size:,} bytes")
    print(f"   📄 PDF: {pdf_size:,} bytes")

def demonstrate_advanced_features(researcher):
    """Demonstrate advanced features."""
    print("\n🚀 Step 6: Demonstrating Advanced Features")
    print("   Testing multi-step reasoning and analytics...")
    print("-" * 60)
    
    # Multi-step reasoning
    print("\n🧠 Multi-step Reasoning Example:")
    complex_query = "How does AI impact healthcare and what are the challenges?"
    print(f"   📝 Complex Query: {complex_query}")
    print("   🔍 Processing complex multi-step query...")
    result = researcher.research(complex_query)
    print(f"   📊 Confidence: {result['analysis']['confidence']:.2f}")
    print(f"   📚 Sources Used: {result['analysis']['num_sources']}")
    print("   ✅ Complex reasoning completed")
    
    # Research history
    print("\n📈 Research History Summary:")
    print("   📊 Analyzing research session data...")
    summary = researcher.get_research_summary()
    print(f"   📝 Total Queries: {summary['total_queries']}")
    print(f"   📄 Documents Indexed: {summary['total_documents']}")
    print(f"   📊 Average Confidence: {summary['average_confidence']:.2f}")
    
    # Recent queries
    if summary.get('recent_queries'):
        print("\n   📚 Recent Queries:")
        for query in summary['recent_queries'][-3:]:
            print(f"      • {query}")

def demonstrate_performance_metrics(researcher):
    """Demonstrate performance metrics and system status."""
    print("\n📊 Step 7: Performance Metrics and System Status")
    print("   Analyzing system performance and capabilities...")
    print("-" * 60)
    
    # System summary
    summary = researcher.get_research_summary()
    
    print(f"\n🎯 System Performance:")
    print(f"   📝 Total Research Queries: {summary['total_queries']}")
    print(f"   📄 Documents in Database: {summary['total_documents']}")
    print(f"   📊 Average Confidence Score: {summary['average_confidence']:.2f}")
    
    # Model information
    print(f"\n🤖 Model Information:")
    print(f"   🧠 Embedding Model: {researcher.embedding_engine.model_name}")
    print(f"   📐 Embedding Dimension: {researcher.embedding_engine.embedding_dim}")
    print(f"   💻 Device: {researcher.embedding_engine.device}")
    
    # Vector store information
    print(f"\n🗄️  Vector Store Information:")
    print(f"   📄 Total Documents: {len(researcher.vector_store.documents)}")
    print(f"   🔢 Index Size: {researcher.vector_store.index.ntotal} vectors")

def cleanup(temp_dir):
    """Clean up temporary files."""
    print("\n🧹 Step 8: Cleanup")
    print("   Cleaning up temporary files...")
    print("-" * 60)
    
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print("   ✅ Temporary files cleaned up successfully")
    except Exception as e:
        print(f"   ⚠️  Error cleaning up: {e}")

def main():
    """Main demonstration function."""
    try:
        # Initialize and setup
        researcher, temp_dir = demonstrate_basic_research()
        
        # Demonstrate core functionality
        demonstrate_research_queries(researcher)
        demonstrate_follow_up_questions(researcher)
        demonstrate_export_functionality(researcher)
        demonstrate_advanced_features(researcher)
        demonstrate_performance_metrics(researcher)
        
        # Cleanup
        cleanup(temp_dir)
        
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n🚀 The Deep Researcher Agent is ready for use!")
        print("\n📋 Next steps:")
        print("   1. 🌐 Run 'streamlit run app.py' for the web interface")
        print("   2. 💻 Run 'python cli.py --interactive' for command line interface")
        print("   3. 📄 Check the generated reports: demo_research.md and demo_research.pdf")
        print("   4. 📚 Read USAGE_GUIDE.md for detailed documentation")
        print("\n✨ Happy researching!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check the error message and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
