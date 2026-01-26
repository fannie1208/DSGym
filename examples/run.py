#!/usr/bin/env python3
"""
DSGym Single Task Runner

A simple script to run a single custom task with DSGym.
Perfect for quick testing and experimentation.

Usage:
    # Basic usage
    python run.py --question "Analyze sales trends" --data /path/to/sales.csv --model gpt-4

    # With context and ground truth
    python run.py \
        --question "What is the correlation between temperature and sales?" \
        --data /path/to/weather_sales.csv \
        --context "This is retail data from 2023" \
        --ground-truth "Strong positive correlation (r=0.85)" \
        --model gpt-4

    # Multiple data files
    python run.py \
        --question "Compare sales across regions" \
        --data /path/to/sales.csv /path/to/regions.csv \
        --model gpt-4
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")

# Set multiprocessing start method
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsgym.datasets import create_custom_task
from dsgym.agents import ReActDSAgent
from dsgym.eval import Evaluator


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a single custom task with DSGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run.py --question "Analyze sales trends" --data sales.csv --model gpt-4
  
  # With context
  python run.py \\
    --question "What factors affect customer satisfaction?" \\
    --data customer_data.csv \\
    --context "E-commerce customer survey data from 2023" \\
    --model gpt-4
  
  # Multiple files
  python run.py \\
    --question "Compare performance across regions" \\
    --data sales.csv regions.csv demographics.csv \\
    --model gpt-4
        """
    )
    
    # Required arguments
    parser.add_argument("--question", "-q", type=str, required=True,
                       help="The data science question to analyze")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'claude-3')")
    
    # Data arguments
    parser.add_argument("--data", "-d", nargs="+", 
                       help="Data file path(s) to analyze")
    parser.add_argument("--context", "-c", type=str,
                       help="Additional context about the data")
    parser.add_argument("--ground-truth", "-gt", type=str,
                       help="Expected answer (for evaluation)")
    
    # Model configuration
    parser.add_argument("--backend", type=str, default="litellm",
                       choices=["litellm", "vllm", "sglang"],
                       help="Backend to use (default: litellm)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-turns", type=int, default=15,
                       help="Maximum turns for agent (default: 15)")
    
    # Environment configuration
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code execution manager URL")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory (default: ./outputs)")
    parser.add_argument("--save-conversation", action="store_true",
                       help="Save full conversation to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    
    return parser


def print_header(args):
    """Print task header."""
    print("🚀 DSGym Single Task Runner")
    print("=" * 50)
    print(f"Question: {args.question}")
    if args.data:
        print(f"Data files: {', '.join(args.data)}")
    if args.context:
        print(f"Context: {args.context}")
    print(f"Model: {args.model} ({args.backend})")
    print("=" * 50)


def print_result(result, args):
    """Print task result."""
    print("\n📊 Result:")
    print("-" * 30)
    
    if hasattr(result, 'prediction') and result.prediction:
        print("Answer:")
        print(result.prediction)
    else:
        print("No answer generated")
    
    if args.ground_truth and hasattr(result, 'metrics') and result.metrics:
        print(f"\nEvaluation:")
        for metric_name, metric_data in result.metrics.items():
            if isinstance(metric_data, dict) and 'score' in metric_data:
                print(f"  {metric_name}: {metric_data['score']}")
    
    if hasattr(result, 'success'):
        print(f"\nStatus: {'✅ Success' if result.success else '❌ Failed'}")
    
    if args.verbose and hasattr(result, 'execution_time'):
        print(f"Execution time: {result.execution_time:.2f}s")


def save_conversation(result, args):
    """Save conversation to file if requested."""
    if not args.save_conversation:
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename
    import time
    timestamp = int(time.time())
    filename = f"conversation_{timestamp}.txt"
    filepath = os.path.join(args.output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Question: {args.question}\n")
        f.write(f"Model: {args.model}\n")
        f.write("=" * 50 + "\n\n")
        
        # Try to extract conversation from result
        conversation = getattr(result, 'raw_response', None)
        if conversation and isinstance(conversation, list):
            for msg in conversation:
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    f.write(f"{role.upper()}:\n{content}\n\n")
        elif hasattr(result, 'prediction'):
            f.write(f"RESULT:\n{result.prediction}\n")
    
    print(f"💾 Conversation saved to: {filepath}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print_header(args)
    
    try:
        # Create custom task
        print("📝 Creating task...")
        task = create_custom_task(
            query=args.question,
            data_files=args.data,
            context=args.context,
            ground_truth=args.ground_truth
        )
        
        # Initialize agent
        print("🤖 Initializing agent...")
        agent_config = {
            "manager_url": args.manager_url,
            "max_turns": args.max_turns,
            "temperature": args.temperature,
            "output_dir": args.output_dir,
        }
        
        agent = ReActDSAgent(
            backend=args.backend,
            model=args.model,
            **agent_config
        )
        
        # Create evaluator
        evaluator = Evaluator()
        
        # Run evaluation
        print("🏃 Running task...")
        result = evaluator.evaluate(
            agent=agent,
            tasks=[task],
            save_results=False,
            show_progress=args.verbose
        )
        
        # Extract single result
        if result and result.get("results"):
            task_result = result["results"][0]
            
            # Print result
            print_result(task_result, args)
            
            # Save conversation if requested
            save_conversation(task_result, args)
            
            return 0
        else:
            print("❌ No results generated")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Task interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)