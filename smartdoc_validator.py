#!/usr/bin/env python3
"""
SmartDoc Explorer - System Validation Script
Quick validation of all system components and requirements
"""

import sys
import os
import time
from pathlib import Path

def print_header():
    """Print validation header"""
    print("=" * 80)
    print("📚 SMARTDOC EXPLORER - SYSTEM VALIDATION")
    print("=" * 80)
    print("Validating system components and requirements...")
    print("=" * 80)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'sentence_transformers', 'numpy', 'pandas',
        'scikit_learn', 'faiss', 'streamlit', 'reportlab', 'markdown',
        'tqdm', 'rich', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("  ✅ All dependencies installed")
    return True

def check_file_structure():
    """Check if all required files exist"""
    print("\n📁 Checking File Structure...")
    
    required_files = [
        'smartdoc_explorer.py', 'smartdoc_cli.py', 'smartdoc_demo.py', 'smartdoc_launcher.py',
        'smartdoc_researcher.py', 'smartdoc_analyzer.py',
        'smartdoc_analytics.py', 'smartdoc_reasoning.py',
        'test_requirements.py', 'config.py',
        'requirements.txt', 'README.md'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("  ✅ All required files present")
    return True

def test_basic_functionality():
    """Test basic system functionality"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        # Test imports
        from smartdoc_researcher import AdvancedDeepResearcher
        from smartdoc_analytics import ResearchAnalytics
        from smartdoc_reasoning import AdvancedReasoningEngine
        print("  ✅ Module imports successful")
        
        # Test researcher initialization
        researcher = AdvancedDeepResearcher()
        print("  ✅ Researcher initialization successful")
        
        # Test reasoning engine
        reasoning_engine = AdvancedReasoningEngine()
        steps = reasoning_engine.decompose_query("What is AI?")
        print(f"  ✅ Reasoning engine working ({len(steps)} steps generated)")
        
        # Test analytics
        analytics = ResearchAnalytics()
        print("  ✅ Analytics system working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_document_processing():
    """Test document processing capabilities"""
    print("\n📄 Testing Document Processing...")
    
    try:
        from smartdoc_researcher import AdvancedDeepResearcher
        
        # Create test document
        test_content = "This is a test document about artificial intelligence and machine learning."
        test_file = "test_document.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test document processing
        researcher = AdvancedDeepResearcher()
        results = researcher.add_documents([test_file])
        
        if results['processed'] > 0:
            print("  ✅ Document processing successful")
            
            # Test query
            result = researcher.research("What is AI?")
            if result['analysis']['confidence'] > 0:
                print("  ✅ Query processing successful")
            else:
                print("  ⚠️  Query processing returned low confidence")
            
            # Clean up
            os.remove(test_file)
            return True
        else:
            print("  ❌ Document processing failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Document processing test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality"""
    print("\n📤 Testing Export Functionality...")
    
    try:
        from smartdoc_researcher import AdvancedDeepResearcher
        
        # Create test document
        test_content = "This is a test document about artificial intelligence."
        test_file = "test_export.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test research and export
        researcher = AdvancedDeepResearcher()
        researcher.add_documents([test_file])
        result = researcher.research("What is AI?")
        
        # Test Markdown export
        md_path = researcher.export_research(result, "markdown", "test_export.md")
        if Path(md_path).exists():
            print("  ✅ Markdown export successful")
        else:
            print("  ❌ Markdown export failed")
            return False
        
        # Test PDF export
        pdf_path = researcher.export_research(result, "pdf", "test_export.pdf")
        if Path(pdf_path).exists():
            print("  ✅ PDF export successful")
        else:
            print("  ❌ PDF export failed")
            return False
        
        # Clean up
        os.remove(test_file)
        os.remove(md_path)
        os.remove(pdf_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Export functionality test failed: {e}")
        return False

def test_web_interface():
    """Test web interface components"""
    print("\n🌐 Testing Web Interface...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit available")
        
        # Test if smartdoc_explorer.py can be imported
        import smartdoc_explorer
        print("  ✅ Web interface module importable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Web interface test failed: {e}")
        return False

def test_cli_interface():
    """Test CLI interface"""
    print("\n💻 Testing CLI Interface...")
    
    try:
        import smartdoc_cli
        print("  ✅ CLI module importable")
        
        # Test if CLI can be run with help
        import subprocess
        result = subprocess.run([sys.executable, "smartdoc_cli.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ CLI help command successful")
        else:
            print("  ⚠️  CLI help command failed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI interface test failed: {e}")
        return False

def run_performance_test():
    """Run basic performance test"""
    print("\n⚡ Testing Performance...")
    
    try:
        from smartdoc_researcher import AdvancedDeepResearcher
        
        # Create test document
        test_content = "This is a test document about artificial intelligence and machine learning."
        test_file = "perf_test.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test processing speed
        start_time = time.time()
        researcher = AdvancedDeepResearcher()
        results = researcher.add_documents([test_file])
        processing_time = time.time() - start_time
        
        if processing_time < 30:  # Should process in under 30 seconds
            print(f"  ✅ Document processing speed: {processing_time:.2f}s")
        else:
            print(f"  ⚠️  Document processing slow: {processing_time:.2f}s")
        
        # Test query speed
        start_time = time.time()
        result = researcher.research("What is AI?")
        query_time = time.time() - start_time
        
        if query_time < 5:  # Should query in under 5 seconds
            print(f"  ✅ Query response speed: {query_time:.2f}s")
        else:
            print(f"  ⚠️  Query response slow: {query_time:.2f}s")
        
        # Clean up
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def main():
    """Main validation function"""
    print_header()
    
    tests = [
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Basic Functionality", test_basic_functionality),
        ("Document Processing", test_document_processing),
        ("Export Functionality", test_export_functionality),
        ("Web Interface", test_web_interface),
        ("CLI Interface", test_cli_interface),
        ("Performance", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The Deep Researcher Agent is ready to use!")
        print("\n🚀 Quick Start:")
        print("   python smartdoc_launcher.py          # Easy launcher")
        print("   streamlit run smartdoc_explorer.py      # Web interface")
        print("   python smartdoc_cli.py --interactive  # CLI interface")
        print("   python smartdoc_demo.py            # Demonstration")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Please check the errors above and fix them before using the system.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

