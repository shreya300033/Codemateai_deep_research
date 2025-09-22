"""
Deep Research AI - Advanced AI Research Platform
A cutting-edge Streamlit application for intelligent document analysis and research
"""

import streamlit as st
import os
import sys
import tempfile
import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smartdoc_researcher import AdvancedDeepResearcher
from config import create_directories, get_env_config

# Page configuration
st.set_page_config(
    page_title="Deep Research AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern vibrant styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Main Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.5)); }
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2rem 0 1.5rem 0;
        position: relative;
        padding-left: 1rem;
    }
    
    .sub-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    
    /* Status Messages */
    .success-message {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        border: none;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3);
        font-weight: 500;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        border: none;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(245, 101, 101, 0.3);
        font-weight: 500;
    }
    
    .info-message {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        border: none;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(66, 153, 225, 0.3);
        font-weight: 500;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        border: none;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3);
        font-weight: 500;
    }
    
    /* Research Results */
    .research-result {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .research-result:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    /* Confidence Bar */
    .confidence-bar {
        background: linear-gradient(90deg, #f56565 0%, #ed8936 25%, #ecc94b 50%, #48bb78 75%, #38a169 100%);
        height: 12px;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        border-right: 3px solid #667eea;
    }
    
    .sidebar .sidebar-content .block-container {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .sidebar .stMarkdown h3 {
        color: #f7fafc;
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar .stMarkdown h2 {
        color: #f7fafc;
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    
    /* File Uploader */
    .sidebar .stFileUploader > div {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .sidebar .stFileUploader > div:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        transform: scale(1.02);
    }
    
    /* Follow-up Questions */
    .follow-up-question {
        background: linear-gradient(135deg, #e6fffa 0%, #f0fff4 100%);
        border: 2px solid #38a169;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    .follow-up-question:hover {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(56, 161, 105, 0.3);
    }
    
    /* Query Input */
    .query-input {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        font-size: 1.1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        color: #2d3748 !important;
    }
    
    .query-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: #ffffff !important;
        color: #2d3748 !important;
    }
    
    /* Force text visibility in all inputs */
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        color: #2d3748 !important;
        background-color: #f7fafc !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
        color: #2d3748 !important;
        background-color: #ffffff !important;
        border-color: #667eea !important;
    }
    
    /* Ensure placeholder text is visible */
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #718096 !important;
        opacity: 1 !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        color: #2d3748 !important;
    }
    
    .stSelectbox > div > div > div {
        color: #2d3748 !important;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        color: #2d3748 !important;
        font-size: 14px !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: #ffffff !important;
        color: #2d3748 !important;
    }
    
    /* General Input Override */
    input[type="text"], input[type="number"], input[type="email"], input[type="password"] {
        color: #2d3748 !important;
        background: #f7fafc !important;
    }
    
    input[type="text"]:focus, input[type="number"]:focus, input[type="email"]:focus, input[type="password"]:focus {
        color: #2d3748 !important;
        background: #ffffff !important;
    }
    
    textarea {
        color: #2d3748 !important;
        background: #f7fafc !important;
    }
    
    textarea:focus {
        color: #2d3748 !important;
        background: #ffffff !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Text Area */
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        color: #2d3748 !important;
        font-size: 14px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: #ffffff !important;
        color: #2d3748 !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #718096 !important;
        opacity: 1;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        color: #2d3748 !important;
        font-size: 14px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: #ffffff !important;
        color: #2d3748 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #718096 !important;
        opacity: 1;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #f7fafc;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'researcher' not in st.session_state:
    st.session_state.researcher = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def generate_topic_keywords(query: str, result: Dict[str, Any]) -> str:
    """Generate topic-specific keywords for filename"""
    # Extract key terms from query
    query_words = re.findall(r'\b[A-Za-z]{3,}\b', query.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
    
    # Filter out stop words and get meaningful terms
    meaningful_words = [word for word in query_words if word not in stop_words and len(word) > 3]
    
    # Take first 3-4 meaningful words
    topic_words = meaningful_words[:4]
    
    # If we have topic words, use them
    if topic_words:
        topic_keywords = '_'.join(topic_words)
    else:
        # Fallback to generic terms
        topic_keywords = 'research'
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"{topic_keywords}_report_{timestamp}"

def create_export_filename(query: str, result: Dict[str, Any], format: str) -> str:
    """Create a descriptive filename for export"""
    base_name = generate_topic_keywords(query, result)
    return f"{base_name}.{format}"

def initialize_researcher():
    """Initialize the Deep Researcher Agent"""
    try:
        with st.spinner("Initializing Deep Researcher Agent..."):
            researcher = AdvancedDeepResearcher()
            st.session_state.researcher = researcher
            st.session_state.documents_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize researcher: {str(e)}")
        return False

def display_metrics():
    """Display system metrics"""
    if st.session_state.researcher:
        try:
            summary = st.session_state.researcher.get_research_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Queries",
                    value=summary.get('total_queries', 0),
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Documents Indexed",
                    value=summary.get('total_documents', 0),
                    delta=None
                )
            
            with col3:
                avg_conf = summary.get('average_confidence', 0)
                st.metric(
                    label="Avg Confidence",
                    value=f"{avg_conf:.2f}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="Recent Queries",
                    value=len(summary.get('recent_queries', [])),
                    delta=None
                )
        except Exception as e:
            st.warning(f"Could not load metrics: {str(e)}")

def display_research_result(result):
    """Display research result in a clean format"""
    with st.container():
        st.markdown('<div class="research-result">', unsafe_allow_html=True)
        
        # Query
        st.markdown(f"**Query:** {result['query']}")
        
        # Analysis (main content)
        st.markdown("**Analysis:**")
        analysis_content = result.get('analysis', 'No analysis available')
        
        if isinstance(analysis_content, str):
            # Clean up the analysis text
            cleaned_analysis = analysis_content.strip()
            # Replace any weird concatenations
            import re
            cleaned_analysis = re.sub(r'([a-z])([A-Z])', r'\1. \2', cleaned_analysis)
            cleaned_analysis = re.sub(r'\s+', ' ', cleaned_analysis)
            st.write(cleaned_analysis)
        elif isinstance(analysis_content, dict):
            if 'summary' in analysis_content:
                summary = analysis_content['summary']
                # Clean up the summary text
                import re
                cleaned_summary = re.sub(r'([a-z])([A-Z])', r'\1. \2', summary)
                cleaned_summary = re.sub(r'\s+', ' ', cleaned_summary)
                st.write(cleaned_summary)
            else:
                st.write(str(analysis_content))
        else:
            st.write(str(analysis_content))
        
        # Findings count
        findings_count = len(result.get('findings', []))
        st.markdown(f"**Relevant Documents Found:** {findings_count}")
        
        # Show findings if available
        if result.get('findings'):
            st.markdown("**Key Findings:**")
            for i, finding in enumerate(result['findings'][:3], 1):
                if isinstance(finding, dict):
                    text = finding.get('text', str(finding))
                    source = finding.get('source', 'Unknown source')
                    
                    # Clean up the text for better display
                    cleaned_text = text.strip()
                    
                    # Clean up concatenation issues
                    import re
                    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1. \2', cleaned_text)
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                    cleaned_text = re.sub(r'\.\s*\.', '.', cleaned_text)
                    
                    if len(cleaned_text) > 200:
                        # Try to find a good break point
                        last_period = cleaned_text[:200].rfind('.')
                        last_exclamation = cleaned_text[:200].rfind('!')
                        last_question = cleaned_text[:200].rfind('?')
                        
                        last_sentence_end = max(last_period, last_exclamation, last_question)
                        if last_sentence_end > 50:  # Only if we have a reasonable amount of text
                            cleaned_text = cleaned_text[:last_sentence_end + 1] + "..."
                        else:
                            cleaned_text = cleaned_text[:200] + "..."
                    
                    st.markdown(f"{i}. {cleaned_text}")
                    st.caption(f"Source: {source}")
                else:
                    st.markdown(f"{i}. {str(finding)[:200]}...")
        
        # Follow-up questions
        if result.get('follow_up_questions'):
            st.markdown("**💡 Suggested Follow-up Questions:**")
            st.markdown("Click on any question below to explore deeper:")
            
            # Create clickable buttons for follow-up questions
            for i, question in enumerate(result['follow_up_questions'][:4], 1):
                if st.button(f"❓ {question}", key=f"followup_{i}", use_container_width=True):
                    # Set the question as current query
                    st.session_state.current_query = question
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header with new design
    st.markdown("""
    <div class="fade-in-up">
        <h1 class="main-header">🧠 Deep Research AI</h1>
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.3rem; color: #4a5568; font-weight: 500; margin: 0;">
                🚀 Advanced AI-powered research platform for intelligent document analysis and knowledge discovery
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add feature highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-weight: 600;">AI-Powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
            <div style="font-weight: 600;">Lightning Fast</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔒</div>
            <div style="font-weight: 600;">Secure & Local</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 600;">Analytics</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar with new design
    with st.sidebar:
        # Header with logo and title
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
            <h2 style="margin: 0; color: white; font-size: 1.8rem; font-weight: 700;">Deep Research AI</h2>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 1rem; font-weight: 500;">Advanced AI Research Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status with new design
        st.markdown("### 🚀 System Status")
        if st.session_state.researcher:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); padding: 1rem; border-radius: 15px; 
                        color: white; text-align: center; margin: 1rem 0; box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3);">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✅</div>
                <div style="font-weight: 600; font-size: 1.1rem;">Researcher Active</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Document count
            doc_count = len(st.session_state.researcher.vector_store.documents)
            if doc_count > 0:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); padding: 1rem; border-radius: 15px; 
                            color: white; text-align: center; margin: 1rem 0; box-shadow: 0 8px 25px rgba(66, 153, 225, 0.3);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">{doc_count} Documents Loaded</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); padding: 1rem; border-radius: 15px; 
                            color: white; text-align: center; margin: 1rem 0; box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📄</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">No Documents Loaded</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%); padding: 1rem; border-radius: 15px; 
                        color: white; text-align: center; margin: 1rem 0; box-shadow: 0 8px 25px rgba(245, 101, 101, 0.3);">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">❌</div>
                <div style="font-weight: 600; font-size: 1.1rem;">Researcher Not Initialized</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Initialize Researcher", type="primary", use_container_width=True):
                if initialize_researcher():
                    st.success("Researcher initialized successfully!")
                    st.rerun()
                else:
                    st.error("Failed to initialize researcher")
        
        st.markdown("---")
        
        # Document Management Section
        st.markdown("### 📄 Document Management")
        
        # File uploader with better styling
        uploaded_files = st.file_uploader(
            "Choose files to analyze",
            type=['pdf', 'docx', 'txt', 'html', 'md'],
            accept_multiple_files=True,
            help="Upload documents for research and analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_files and st.session_state.researcher:
            # Show uploaded files
            st.markdown(f"**📁 {len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                st.caption(f"• {file.name}")
            
            # Upload options
            st.markdown("**⚙️ Upload Options:**")
            clear_existing = st.checkbox(
                "Clear existing documents", 
                value=True, 
                help="Uncheck to add to existing documents"
            )
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Process Documents", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Processing documents..."):
                            temp_files = []
                            
                            # Save uploaded files to temporary locations
                            for uploaded_file in uploaded_files:
                                try:
                                    # Get file extension
                                    file_extension = uploaded_file.name.split('.')[-1]
                                    
                                    # Create temporary file
                                    temp_file = tempfile.NamedTemporaryFile(
                                        delete=False, 
                                        suffix=f".{file_extension}",
                                        prefix="upload_"
                                    )
                                    
                                    # Write file content
                                    temp_file.write(uploaded_file.getvalue())
                                    temp_file.close()
                                    temp_files.append(temp_file.name)
                                    
                                    st.info(f"📄 Saved: {uploaded_file.name} → {temp_file.name}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Failed to save {uploaded_file.name}: {str(e)}")
                                    continue
                            
                            if not temp_files:
                                st.error("❌ No files were successfully saved for processing")
                                return
                            
                            # Clear existing documents if requested
                            if clear_existing:
                                st.info("🧹 Clearing existing documents...")
                                clear_result = st.session_state.researcher.clear_documents()
                                if clear_result:
                                    st.success("✅ Existing documents cleared")
                                else:
                                    st.warning("⚠️ Warning: Could not clear existing documents")
                            else:
                                st.info("📄 Adding to existing documents...")
                            
                            # Process documents
                            st.info("🔄 Processing documents...")
                            results = st.session_state.researcher.add_documents(temp_files)
                            
                            # Clean up temporary files
                            for temp_file in temp_files:
                                try:
                                    os.unlink(temp_file)
                                except Exception as e:
                                    st.warning(f"⚠️ Could not delete temporary file {temp_file}: {e}")
                            
                            # Show results
                            if results['processed'] > 0:
                                st.success(f"✅ Successfully processed {results['processed']} documents!")
                                st.session_state.documents_loaded = True
                                
                                # Show document count
                                doc_count = len(st.session_state.researcher.vector_store.documents)
                                st.info(f"📊 Total documents in database: {doc_count}")
                            else:
                                st.error("❌ No documents were processed successfully")
                            
                            if results['failed'] > 0:
                                st.warning(f"⚠️ Failed to process {results['failed']} documents")
                                if results.get('errors'):
                                    for error in results['errors']:
                                        st.error(f"   • {error}")
                    
                    except Exception as e:
                        st.error(f"❌ Upload failed: {str(e)}")
                        st.error("Please check the file format and try again")
            
            with col2:
                if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                    clear_result = st.session_state.researcher.clear_documents()
                    st.session_state.documents_loaded = False
                    if clear_result:
                        st.success("All documents cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear documents")
        
        # Quick Actions Section
        if st.session_state.researcher:
            st.markdown("---")
            st.markdown("### ⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Sample", type="secondary", use_container_width=True, help="Load sample document for testing"):
                    try:
                        # Create a sample document
                        sample_content = """
                        This is a sample document about artificial intelligence.
                        AI is a branch of computer science that aims to create intelligent machines.
                        Machine learning is a subset of AI that focuses on algorithms.
                        Deep learning uses neural networks with multiple layers.
                        """
                        
                        # Clear existing documents
                        st.session_state.researcher.clear_documents()
                        
                        # Save sample document
                        sample_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="sample_")
                        sample_file.write(sample_content.encode('utf-8'))
                        sample_file.close()
                        
                        # Process sample document
                        results = st.session_state.researcher.add_documents([sample_file.name])
                        
                        # Clean up
                        os.unlink(sample_file.name)
                        
                        if results['processed'] > 0:
                            st.success("✅ Sample document processed successfully!")
                            st.session_state.documents_loaded = True
                        else:
                            st.error("❌ Failed to process sample document")
                            
                    except Exception as e:
                        st.error(f"❌ Test failed: {str(e)}")
            
            with col2:
                if st.button("📊 Status", type="secondary", use_container_width=True, help="Check system status"):
                    doc_count = len(st.session_state.researcher.vector_store.documents)
                    st.info(f"📊 Documents in database: {doc_count}")
                    
                    if doc_count > 0:
                        st.success("✅ System is working correctly")
                    else:
                        st.warning("⚠️ No documents loaded")
        
        # Export Section
        if 'current_result' in st.session_state:
            st.markdown("---")
            st.markdown("### 📄 Export Results")
            
            # Export buttons with better styling
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 PDF", help="Download as PDF", use_container_width=True):
                    try:
                        filename = create_export_filename(
                            st.session_state.current_result['query'], 
                            st.session_state.current_result, 
                            "pdf"
                        )
                        path = st.session_state.researcher.export_research(
                            st.session_state.current_result, "pdf", filename
                        )
                        
                        with open(path, 'rb') as file:
                            file_content = file.read()
                        
                        st.download_button(
                            label="📥 Download PDF",
                            data=file_content,
                            file_name=filename,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"PDF export failed: {str(e)}")
            
            with col2:
                if st.button("📝 MD", help="Download as Markdown", use_container_width=True):
                    try:
                        filename = create_export_filename(
                            st.session_state.current_result['query'], 
                            st.session_state.current_result, 
                            "md"
                        )
                        path = st.session_state.researcher.export_research(
                            st.session_state.current_result, "markdown", filename
                        )
                        
                        with open(path, 'r', encoding='utf-8') as file:
                            file_content = file.read()
                        
                        st.download_button(
                            label="📥 Download MD",
                            data=file_content,
                            file_name=filename,
                            mime="text/markdown",
                            type="primary",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Markdown export failed: {str(e)}")
        
        # Research History Section
        if st.session_state.research_history:
            st.markdown("---")
            st.markdown("### 📚 Recent Research")
            for i, query in enumerate(st.session_state.research_history[-5:], 1):
                if st.button(f"{i}. {query[:40]}...", key=f"history_{i}", use_container_width=True):
                    st.session_state.current_query = query
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #7f8c8d; font-size: 0.8rem;">
            <p style="margin: 0;">🧠 Deep Research AI</p>
            <p style="margin: 0.25rem 0 0 0;">v3.0 - Advanced AI Research</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area with new design
    if not st.session_state.researcher:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); padding: 3rem; border-radius: 25px; 
                    color: white; text-align: center; margin: 2rem 0; box-shadow: 0 15px 50px rgba(66, 153, 225, 0.3);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎉</div>
            <h2 style="margin: 0 0 1rem 0; font-size: 2.5rem; font-weight: 700;">Welcome to Deep Research AI!</h2>
            <p style="font-size: 1.3rem; margin: 0 0 2rem 0; opacity: 0.9;">This advanced AI research platform can help you:</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📖</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">Analyze Documents</div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Search through documents with AI</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">Intelligent Queries</div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Conduct smart research queries</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">Generate Insights</div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Get summaries and analytics</div>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">Export Reports</div>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Download research reports</div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚀</div>
                <div style="font-weight: 600; font-size: 1.2rem;">Get Started Now!</div>
                <div style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">Click "Initialize Researcher" in the sidebar to begin</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Key Features")
            st.markdown("""
            - **Local Processing**: All analysis happens on your machine
            - **Multi-format Support**: PDF, DOCX, TXT, HTML, Markdown
            - **Smart Search**: Vector-based similarity search
            - **AI Summarization**: Intelligent content summarization
            - **Export Options**: PDF and Markdown reports
            """)
        
        with col2:
            st.markdown("### 🚀 Quick Actions")
            st.markdown("""
            1. Initialize the researcher
            2. Upload your documents
            3. Ask research questions
            4. Export your findings
            """)
        
        return
    
    # Display metrics
    display_metrics()
    
    # Research interface with new design
    st.markdown('<h2 class="sub-header">🔍 Research Interface</h2>', unsafe_allow_html=True)
    
    # Add research tips
    with st.expander("💡 Research Tips", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e6fffa 0%, #f0fff4 100%); padding: 1.5rem; border-radius: 15px; 
                    border: 2px solid #38a169; margin: 1rem 0;">
            <h4 style="color: #2d3748; margin-top: 0;">🎯 How to get the best results:</h4>
            <ul style="color: #4a5568; margin: 0;">
                <li><strong>Be specific:</strong> Ask detailed questions about your documents</li>
                <li><strong>Use keywords:</strong> Include relevant terms from your documents</li>
                <li><strong>Ask follow-ups:</strong> Use the suggested questions to dive deeper</li>
                <li><strong>Try variations:</strong> Rephrase your questions for different perspectives</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Query input with new styling
    st.markdown("### 💬 Ask Your Question")
    query = st.text_area(
        "Enter your research query:",
        value=st.session_state.get('current_query', ''),
        height=120,
        help="Ask any question about your uploaded documents",
        placeholder="Example: What are the main findings in the research papers about machine learning?"
    )
    
    # Research controls with new design
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown("### ⚙️ Settings")
        max_results = st.slider("Max Results", 1, 20, 10, help="Maximum number of results to return")
    
    with col2:
        st.markdown("### 🚀 Actions")
        if st.button("🔍 Research", type="primary", use_container_width=True):
            if query.strip():
                # Check if documents are loaded
                doc_count = len(st.session_state.researcher.vector_store.documents)
                if doc_count == 0:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); padding: 1.5rem; border-radius: 15px; 
                                color: white; text-align: center; margin: 1rem 0; box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div>
                        <div style="font-weight: 600; font-size: 1.1rem;">No documents loaded!</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">Please upload and process documents first.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    return
                
                with st.spinner("🔍 Conducting research..."):
                    try:
                        result = st.session_state.researcher.research(query, max_results=max_results)
                        st.session_state.research_history.append(query)
                        st.session_state.current_result = result
                        st.success("✅ Research completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Research failed: {str(e)}")
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); padding: 1rem; border-radius: 15px; 
                            color: white; text-align: center; margin: 1rem 0; box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3);">
                    <div style="font-weight: 600;">Please enter a research query</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        if 'current_result' in st.session_state:
            st.markdown("### 📊 Export Results")
            export_format = st.selectbox("Export Format", ["Markdown", "PDF"], key="export_format", 
                                        help="Choose the format for your research report")
            
            if export_format == "Markdown":
                filename = create_export_filename(
                    st.session_state.current_result['query'], 
                    st.session_state.current_result, 
                    "md"
                )
                try:
                    path = st.session_state.researcher.export_research(
                        st.session_state.current_result, "markdown", filename
                    )
                    
                    with open(path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                    
                    st.download_button(
                        label="📄 Download Markdown",
                        data=file_content,
                        file_name=filename,
                        mime="text/markdown",
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Markdown export failed: {str(e)}")
            
            else:  # PDF
                filename = create_export_filename(
                    st.session_state.current_result['query'], 
                    st.session_state.current_result, 
                    "pdf"
                )
                try:
                    path = st.session_state.researcher.export_research(
                        st.session_state.current_result, "pdf", filename
                    )
                    
                    with open(path, 'rb') as file:
                        file_content = file.read()
                    
                    st.download_button(
                        label="📄 Download PDF",
                        data=file_content,
                        file_name=filename,
                        mime="application/pdf",
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"PDF export failed: {str(e)}")
        else:
            st.info("Conduct research first to enable export")
    
    # Display results
    if 'current_result' in st.session_state:
        st.markdown('<h2 class="sub-header">📋 Research Results</h2>', unsafe_allow_html=True)
        display_research_result(st.session_state.current_result)
    
    # Research history visualization
    if st.session_state.research_history:
        st.markdown('<h2 class="sub-header">📈 Research Analytics</h2>', unsafe_allow_html=True)
        
        # Create a simple chart of research activity
        history_df = pd.DataFrame({
            'Query': st.session_state.research_history,
            'Timestamp': [datetime.now()] * len(st.session_state.research_history)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Research Activity**")
            st.bar_chart(history_df['Query'].value_counts())
        
        with col2:
            st.markdown("**Query Length Distribution**")
            query_lengths = [len(q) for q in st.session_state.research_history]
            st.bar_chart(pd.DataFrame({'Length': query_lengths}))
    
    # Footer with new design
    st.markdown("""
    <div class="footer">
        <div style="font-size: 1.5rem; margin-bottom: 1rem;">🧠</div>
        <h3 style="margin: 0 0 0.5rem 0; color: #f7fafc;">Deep Research AI</h3>
        <p style="margin: 0 0 1rem 0; color: #a0aec0; font-size: 1.1rem;">Powered by Advanced AI | Built with Streamlit</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; color: #667eea;">🚀</div>
                <div style="font-size: 0.9rem; color: #a0aec0;">Fast</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; color: #48bb78;">🔒</div>
                <div style="font-size: 0.9rem; color: #a0aec0;">Secure</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; color: #ed8936;">🧠</div>
                <div style="font-size: 0.9rem; color: #a0aec0;">Smart</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; color: #4299e1;">📊</div>
                <div style="font-size: 0.9rem; color: #a0aec0;">Analytics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Create necessary directories
    create_directories()
    
    # Run the app
    main()
