#!/usr/bin/env python3
"""
SmartDoc Explorer Launcher
A simple launcher script for the SmartDoc Explorer
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 80)
    print("📚 SmartDoc Explorer Launcher")
    print("=" * 80)
    print("A sophisticated AI-powered research tool")
    print("=" * 80)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import rich
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install dependencies with: pip install -r requirements.txt")
        return False

def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("🌐 Launching web interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "smartdoc_explorer.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch web interface: {e}")
    except KeyboardInterrupt:
        print("\n👋 Web interface closed")

def launch_cli():
    """Launch the CLI interface"""
    print("💻 Launching CLI interface...")
    try:
        subprocess.run([sys.executable, "smartdoc_cli.py", "--interactive"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch CLI: {e}")
    except KeyboardInterrupt:
        print("\n👋 CLI closed")

def run_demo():
    """Run the demonstration script"""
    print("🎬 Running demonstration...")
    try:
        subprocess.run([sys.executable, "smartdoc_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run demo: {e}")
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    print("\n🚀 Choose an option:")
    print("1. 🌐 Launch Web Interface (Streamlit)")
    print("2. 💻 Launch CLI Interface")
    print("3. 🎬 Run Demonstration")
    print("4. 📚 Show Help")
    print("5. 🚪 Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                launch_web_interface()
                break
            elif choice == "2":
                launch_cli()
                break
            elif choice == "3":
                run_demo()
                break
            elif choice == "4":
                show_help()
                break
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0

def show_help():
    """Show help information"""
    print("\n📚 SmartDoc Explorer Help")
    print("=" * 50)
    print("\n🔍 What is SmartDoc Explorer?")
    print("   A sophisticated AI-powered research tool that can analyze documents,")
    print("   answer questions, and generate research reports using local AI models.")
    
    print("\n🚀 Getting Started:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run this launcher: python launch.py")
    print("   3. Choose your preferred interface")
    
    print("\n🌐 Web Interface:")
    print("   - Modern, user-friendly web interface")
    print("   - Upload documents and ask questions")
    print("   - Visual analytics and export options")
    print("   - Run with: streamlit run smartdoc_explorer.py")
    
    print("\n💻 CLI Interface:")
    print("   - Command-line interface for power users")
    print("   - Interactive mode and batch processing")
    print("   - Rich terminal output with progress bars")
    print("   - Run with: python smartdoc_cli.py --interactive")
    
    print("\n🎬 Demonstration:")
    print("   - Shows all features with sample data")
    print("   - Perfect for first-time users")
    print("   - Run with: python smartdoc_demo.py")
    
    print("\n📄 Supported File Types:")
    print("   - PDF documents")
    print("   - Microsoft Word (.docx)")
    print("   - Plain text (.txt)")
    print("   - HTML files")
    print("   - Markdown files")
    
    print("\n🔧 Troubleshooting:")
    print("   - Make sure all dependencies are installed")
    print("   - Check that you have sufficient disk space")
    print("   - For GPU support, install CUDA-compatible versions")
    
    print("\n📚 Documentation:")
    print("   - README.md: Project overview")
    print("   - USAGE_GUIDE.md: Detailed usage instructions")
    print("   - QUERY_GUIDE.md: Query examples and tips")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

