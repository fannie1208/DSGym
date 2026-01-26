#!/usr/bin/env python3
"""
Unified DSGym Evaluation Script

This script provides a unified interface for evaluating agents on any DSGym dataset.
It consolidates the functionality of all individual evaluate_*.py scripts.

Usage:
    python evaluate.py --dataset daeval --model gpt-4 --limit 10
    python evaluate.py --dataset discoverybench --split test --model gpt-4 
    python evaluate.py --dataset qrdata --dataset-type synthetic --model gpt-4
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")

# Set multiprocessing start method to avoid resource tracking issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsgym.datasets import DatasetRegistry, load_tasks_from_dataset, create_custom_task
from dsgym.agents import ReActDSAgent
from dsgym.eval import Evaluator
from dsgym.eval.utils import EvaluationConfig


# Dataset configuration - now using ReActDSAgent as default for all datasets
DATASET_CONFIG = {
    "daeval": {
        "agent_type": "react",
        "extra_params": [],
        "result_processor": "daeval"
    },
    "discoverybench": {
        "agent_type": "react", 
        "extra_params": ["--split"],
        "result_processor": "default"
    },
    "qrdata": {
        "agent_type": "react",
        "extra_params": ["--dataset-type", "--synthetic-path"],
        "result_processor": "default"
    },
    "dabstep": {
        "agent_type": "react",
        "extra_params": [],
        "result_processor": "dabstep"
    },
    "dsbio": {
        "agent_type": "react", 
        "extra_params": [],
        "result_processor": "default"
    }
}


def create_parser():
    """Create the argument parser with all possible parameters."""
    parser = argparse.ArgumentParser(description="Evaluate agent on DSGym datasets")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIG.keys()),
                       help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')")
    
    # Common arguments
    parser.add_argument("--backend", type=str, default="litellm", 
                       choices=["litellm", "vllm", "sglang"],
                       help="Backend to use for model inference")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Number of samples to evaluate")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Index of first sample to evaluate (0-based)")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code sandbox manager URL")
    parser.add_argument("--max-turns", type=int, default=15,
                       help="Maximum turns per sample")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses environment variable if not provided)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (auto-set based on backend if not provided)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                       help="Maximum model sequence length for vLLM backend (default: 32768)")
    
    # Dataset-specific arguments
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Dataset split to use (for discoverybench)")
    parser.add_argument("--dataset-type", type=str, default="original",
                       choices=["original", "synthetic"],
                       help="Type of dataset to use (for qrdata)")
    parser.add_argument("--synthetic-path", type=str, default=None,
                       help="Path to synthetic dataset (for qrdata)")
    
    return parser


def create_agent(args, dataset_config):
    """Create agent instance based on configuration."""
    # Agent configuration
    agent_config = {
        "manager_url": args.manager_url,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
    }
    
    # Add vLLM-specific parameters
    if args.backend in ["vllm", "sglang"]:
        agent_config["max_model_len"] = args.max_model_len
    
    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key
    
    # Choose agent type
    agent_type = dataset_config["agent_type"]
    if agent_type == "react":
        agent_class = ReActDSAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class(
        backend=args.backend,
        model=args.model,
        **agent_config
    )


def load_dataset(args):
    """Load dataset with appropriate configuration."""
    dataset = DatasetRegistry.load(args.dataset)
    
    # Build load configuration
    load_config = {
        "limit": args.limit,
        "start_index": args.start_index
    }
    
    # Add dataset-specific parameters
    if args.dataset == "discoverybench":
        load_config["split"] = args.split
    elif args.dataset == "qrdata":
        load_config["dataset_type"] = args.dataset_type
        if args.dataset_type == "synthetic" and args.synthetic_path:
            load_config["synthetic_dataset_path"] = args.synthetic_path
    
    tasks = dataset.load(**load_config)
    return dataset, tasks


def process_results(args, results, dataset_config):
    """Process and display results based on dataset type."""
    print("✅ Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    # Print summary statistics
    print("\n📈 Results Summary:")
    print("-" * 30)

    aggregated_metrics = results.get("metrics", {})
    evaluation_results = results.get("results", [])

    if aggregated_metrics:
        for metric_name, metric_value in aggregated_metrics.items():
            if metric_name not in ["total_evaluation_time", "metrics_used"]:
                if isinstance(metric_value, dict) and 'mean_score' in metric_value:
                    print(f"{metric_name}: {metric_value['mean_score']:.3f}")
                else:
                    print(f"{metric_name}: {metric_value}")

    # Dataset-specific result processing
    result_processor = dataset_config["result_processor"]
    
    if result_processor == "daeval":
        _process_daeval_results(evaluation_results)
    elif result_processor == "dabstep":
        _process_dabstep_results(evaluation_results)
    else:
        _process_default_results(evaluation_results)


def _process_daeval_results(evaluation_results):
    """Process DAEval-specific results."""
    if evaluation_results and len(evaluation_results) > 0:
        print(f"\n📝 Sample Results (showing first 3):")
        print("-" * 40)
        for i, sample_result in enumerate(evaluation_results[:3]):
            print(f"Sample {i+1}:")
            if hasattr(sample_result, 'prediction'):
                pred_preview = str(sample_result.prediction)[:100]
                if len(str(sample_result.prediction)) > 100:
                    pred_preview += "..."
                print(f"  Prediction: {pred_preview}")

            # Check if prediction follows DAEval format
            prediction_str = str(sample_result.prediction) if hasattr(sample_result, 'prediction') else ""
            has_format = "@" in prediction_str and "[" in prediction_str and "]" in prediction_str
            print(f"  Format compliance: {'✅' if has_format else '❌'}")

            if hasattr(sample_result, 'metrics'):
                for metric_name, metric_score in sample_result.metrics.items():
                    print(f"  {metric_name}: {metric_score}")
            print()


def _process_dabstep_results(evaluation_results):
    """Process DABStep-specific results."""
    # Count successful completions
    successful = sum(1 for r in evaluation_results if r.success)
    print(f"Successful completions: {successful}/{len(evaluation_results)}")

    # Note about metrics
    print("\n⚠️  Note: Metric scores are None (no ground truth available)")
    
    # Print sample predictions (first 3)
    if evaluation_results and len(evaluation_results) > 0:
        print(f"\n📝 Sample Predictions (showing first 3):")
        print("-" * 40)
        for i, sample_result in enumerate(evaluation_results[:3]):
            print(f"Sample {i+1}:")
            if hasattr(sample_result, 'prediction'):
                pred_preview = str(sample_result.prediction)[:200]
                if len(str(sample_result.prediction)) > 200:
                    pred_preview += "..."
                print(f"  Prediction: {pred_preview}")
            print()


def _process_default_results(evaluation_results):
    """Process default results display."""
    if evaluation_results and len(evaluation_results) > 0:
        print(f"\n📝 Sample Results (showing first 3):")
        print("-" * 40)
        for i, sample_result in enumerate(evaluation_results[:3]):
            print(f"Sample {i+1}:")
            if hasattr(sample_result, 'prediction'):
                pred_preview = str(sample_result.prediction)[:100]
                if len(str(sample_result.prediction)) > 100:
                    pred_preview += "..."
                print(f"  Prediction: {pred_preview}")

            if hasattr(sample_result, 'metrics'):
                for metric_name, metric_score in sample_result.metrics.items():
                    print(f"  {metric_name}: {metric_score}")
            print()


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    
    print(f"🚀 Starting {args.dataset.upper()} Evaluation with DSGym")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Samples: {args.limit if args.limit is not None else 'all'}")
    print(f"Start index: {args.start_index}")
    
    # Set default max_workers based on backend
    if args.max_workers is None:
        if args.backend == "litellm":
            args.max_workers = 24
        else:  # vllm or sglang
            args.max_workers = 1
    
    print(f"Max workers: {args.max_workers}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize agent
    print("🤖 Initializing agent...")
    try:
        agent = create_agent(args, dataset_config)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    # Load dataset
    print(f"📊 Loading {args.dataset} dataset...")
    try:
        dataset, tasks = load_dataset(args)
        print(f"✅ Loaded {len(tasks)} tasks from {args.dataset}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Create evaluator with dataset-specific metrics
    print("⚖️ Setting up evaluator...")
    evaluator = Evaluator(
        dataset=dataset,  # This will automatically use dataset metrics and configs
        parallel_workers=args.max_workers
    )
    
    # Create evaluation config
    config = EvaluationConfig(
        model_name=args.model,
        backend_type=args.backend,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        run_name=f"{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
    )
    
    # Run evaluation
    print("🏃 Starting evaluation...")
    try:
        results = evaluator.evaluate(
            agent=agent,
            tasks=tasks,
            config=config,
            save_results=True
        )
        
        # Process and display results
        process_results(args, results, dataset_config)
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)