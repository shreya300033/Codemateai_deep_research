"""
Command Line Interface for SmartDoc Explorer
A clean, modern CLI with enhanced user experience
"""

import argparse
import json
import sys
import re
from pathlib import Path
from datetime import datetime
import os
from typing import List, Dict, Any

# Rich library for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Install with 'pip install rich' for better CLI experience.")

from smartdoc_researcher import AdvancedDeepResearcher

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

class CLIDisplay:
    """Enhanced CLI display with rich formatting"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
    
    def print_header(self):
        """Print application header"""
        if self.console:
            header = """
📚 SmartDoc Explorer CLI
A sophisticated AI-powered research tool
            """
            self.console.print(Panel(header, style="bold blue", box=box.ROUNDED))
        else:
            print("=" * 60)
            print("📚 SmartDoc Explorer CLI")
            print("A sophisticated AI-powered research tool")
            print("=" * 60)
    
    def print_success(self, message: str):
        """Print success message"""
        if self.console:
            self.console.print(f"✅ {message}", style="green")
        else:
            print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        if self.console:
            self.console.print(f"❌ {message}", style="red")
        else:
            print(f"❌ {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        if self.console:
            self.console.print(f"⚠️  {message}", style="yellow")
        else:
            print(f"⚠️  {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        if self.console:
            self.console.print(f"ℹ️  {message}", style="blue")
        else:
            print(f"ℹ️  {message}")
    
    def print_loading(self, message: str):
        """Print loading message with spinner"""
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(message, total=None)
                return progress, task
        else:
            print(f"🔄 {message}")
            return None, None
    
    def print_research_result(self, result: Dict[str, Any]):
        """Print research result in a formatted way"""
        if self.console:
            # Create a table for the result
            table = Table(title="Research Results", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Query", result['query'])
            table.add_row("Confidence", f"{result['analysis']['confidence']:.2f}")
            table.add_row("Sources Found", str(result['analysis']['num_sources']))
            table.add_row("Summary", result['analysis']['summary'][:100] + "..." if len(result['analysis']['summary']) > 100 else result['analysis']['summary'])
            
            self.console.print(table)
            
            # Follow-up questions
            if result.get('follow_up_questions'):
                self.console.print("\n[bold]Suggested Follow-up Questions:[/bold]")
                for i, question in enumerate(result['follow_up_questions'][:3], 1):
                    self.console.print(f"{i}. {question}")
        else:
            print(f"\nQuery: {result['query']}")
            print(f"Confidence: {result['analysis']['confidence']:.2f}")
            print(f"Sources Found: {result['analysis']['num_sources']}")
            print(f"\nSummary: {result['analysis']['summary']}")
            
            if result.get('follow_up_questions'):
                print("\nSuggested follow-up questions:")
                for i, question in enumerate(result['follow_up_questions'], 1):
                    print(f"{i}. {question}")

def initialize_researcher(display: CLIDisplay) -> AdvancedDeepResearcher:
    """Initialize the researcher with progress indication"""
    progress, task = display.print_loading("Initializing Deep Researcher Agent...")
    
    try:
        researcher = AdvancedDeepResearcher()
        if progress:
            progress.update(task, completed=True)
        display.print_success("Researcher initialized successfully!")
        return researcher
    except Exception as e:
        if progress:
            progress.update(task, completed=True)
        display.print_error(f"Failed to initialize researcher: {str(e)}")
        return None

def process_documents(researcher: AdvancedDeepResearcher, doc_paths: List[str], display: CLIDisplay, clear_existing: bool = True):
    """Process documents with progress indication"""
    progress, task = display.print_loading(f"Processing {len(doc_paths)} documents...")
    
    try:
        # Clear existing documents if requested
        if clear_existing:
            display.print_info("Clearing existing documents...")
            researcher.clear_documents()
        
        results = researcher.add_documents(doc_paths)
        if progress:
            progress.update(task, completed=True)
        
        if results['processed'] > 0:
            display.print_success(f"Successfully processed {results['processed']} documents!")
            display.print_info(f"Total documents in database: {len(researcher.vector_store.documents)}")
        if results['failed'] > 0:
            display.print_warning(f"Failed to process {results['failed']} documents")
    except Exception as e:
        if progress:
            progress.update(task, completed=True)
        display.print_error(f"Error processing documents: {str(e)}")

def interactive_mode(researcher: AdvancedDeepResearcher, display: CLIDisplay, max_results: int):
    """Enhanced interactive mode"""
    display.print_info("Interactive mode started. Type 'help' for commands, 'quit' to exit.")
    
    while True:
        try:
            if RICH_AVAILABLE:
                query = Prompt.ask("\n[bold cyan]Enter your research query[/bold cyan]")
            else:
                query = input("\nEnter your research query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'help':
                display.print_info("Available commands:")
                print("  - Enter any research query to search")
                print("  - 'quit', 'exit', 'q' to exit")
                print("  - 'help' to show this message")
                print("  - 'clear' to clear screen")
                print("  - 'clear_docs' to clear all documents")
                print("  - 'status' to show document status")
                continue
            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif query.lower() == 'clear_docs':
                if researcher.clear_documents():
                    display.print_success("All documents cleared!")
                else:
                    display.print_error("Failed to clear documents")
                continue
            elif query.lower() == 'status':
                doc_count = len(researcher.vector_store.documents)
                display.print_info(f"Documents in database: {doc_count}")
                continue
            elif not query.strip():
                display.print_warning("Please enter a valid query")
                continue
            
            progress, task = display.print_loading("Conducting research...")
            
            try:
                result = researcher.research(query, max_results=max_results)
                if progress:
                    progress.update(task, completed=True)
                
                display.print_research_result(result)
                
                # Ask if user wants to export
                if RICH_AVAILABLE and Confirm.ask("\nWould you like to export this result?"):
                    export_result(researcher, result, display)
                    
            except Exception as e:
                if progress:
                    progress.update(task, completed=True)
                display.print_error(f"Research failed: {str(e)}")
                
        except KeyboardInterrupt:
            display.print_info("\nExiting interactive mode...")
            break
        except Exception as e:
            display.print_error(f"Unexpected error: {str(e)}")

def export_result(researcher: AdvancedDeepResearcher, result: Dict[str, Any], display: CLIDisplay):
    """Export research result"""
    if RICH_AVAILABLE:
        format_choice = Prompt.ask("Choose export format", choices=["json", "markdown", "pdf"], default="markdown")
    else:
        format_choice = input("Choose export format (json/markdown/pdf): ").strip().lower()
    
    filename = create_export_filename(result['query'], result, format_choice)
    
    try:
        display.print_info(f"Exporting to {format_choice.upper()} format...")
        path = researcher.export_research(result, format_choice, filename)
        
        # Get file size
        file_size = os.path.getsize(path)
        file_size_mb = file_size / (1024 * 1024)
        
        display.print_success(f"✅ Export completed successfully!")
        display.print_info(f"📄 File: {path}")
        display.print_info(f"📊 Size: {file_size_mb:.2f} MB")
        
        # Ask if user wants to open the file
        if RICH_AVAILABLE and Confirm.ask("Would you like to open the exported file?"):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(path)
                elif os.name == 'posix':  # macOS and Linux
                    os.system(f'open "{path}"' if sys.platform == 'darwin' else f'xdg-open "{path}"')
                display.print_success("File opened successfully!")
            except Exception as e:
                display.print_warning(f"Could not open file automatically: {e}")
                display.print_info(f"Please open manually: {path}")
        
    except Exception as e:
        display.print_error(f"Export failed: {str(e)}")

def main():
    """Main CLI function with enhanced interface"""
    parser = argparse.ArgumentParser(
        description="SmartDoc Explorer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --interactive
  python cli.py --query "What is AI?" --documents doc1.pdf doc2.txt
  python cli.py --query "Machine learning" --output results.json --format json
        """
    )
    
    parser.add_argument("--query", "-q", type=str, help="Research query")
    parser.add_argument("--documents", "-d", nargs="+", help="Document paths to process")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "markdown", "pdf"], default="json", help="Output format")
    parser.add_argument("--max-results", "-m", type=int, default=10, help="Maximum number of results")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--version", "-v", action="version", version="SmartDoc Explorer CLI v1.0.0")
    
    args = parser.parse_args()
    
    # Initialize display
    display = CLIDisplay()
    display.print_header()
    
    # Initialize researcher
    researcher = initialize_researcher(display)
    if not researcher:
        return 1
    
    # Process documents if provided
    if args.documents:
        process_documents(researcher, args.documents, display)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(researcher, display, args.max_results)
    
    # Single query mode
    elif args.query:
        progress, task = display.print_loading("Conducting research...")
        
        try:
            result = researcher.research(args.query, max_results=args.max_results)
            if progress:
                progress.update(task, completed=True)
            
            # Output results
            if args.output:
                if args.format == "json":
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2)
                else:
                    researcher.export_research(result, args.format, args.output)
                display.print_success(f"Results saved to {args.output}")
            else:
                display.print_research_result(result)
                
        except Exception as e:
            if progress:
                progress.update(task, completed=True)
            display.print_error(f"Research failed: {str(e)}")
            return 1
    
    else:
        display.print_info("No query provided. Use --help for usage information.")
        display.print_info("Try: python cli.py --interactive")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
