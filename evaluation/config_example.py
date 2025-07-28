# -*- coding: utf-8 -*-
"""
Configuration Example
Example configuration and usage for the video caption evaluation pipeline.
"""

import os
from main import VideoEvaluationPipeline
import argparse


def create_example_config():
    """Create an example configuration for the evaluation pipeline."""
    
    # Example configuration - modify these paths according to your setup
    config = {
        # Data paths
        'test_data_path': '/nlp_group/yulinhao/vlm/eval_video_caption_0127/data_lhyu_filtered/all_test/key_point_labeled_mcts_data_080.json',
        'test_data_threshold': '080',
        'categoried_kp_path': '/nlp_group/yulinhao/vlm/caption_test_data/test_data_mcts_0127/util/category_kp/video_with_verified_categoried_deplicated_kp',
        
        # Model prediction paths
        'model_predictions_path': '/path/to/model/predictions.jsonl',
        'candidate_captions_path': '/path/to/candidate/captions.jsonl',
        'golden_kp_path': '/path/to/golden/keypoints.jsonl',
        
        # Output configuration
        'output_dir': './evaluation_results',
        'model_name': 'InternVL2_5-8B',
        
        # API configuration
        'openai_api_key': os.getenv('OPENAI_API_KEY'),  # Set as environment variable
        
        # Evaluation categories
        'categories': [
            "Appearance Description", 
            "Action Description", 
            "Environment Description", 
            "Object Description", 
            "Camera Movement and Composition"
        ]
    }
    
    return config


def run_evaluation_example():
    """Run an example evaluation with sample configuration."""
    
    # Get example configuration
    config = create_example_config()
    
    # Create arguments object
    args = argparse.Namespace(**config)
    
    # Validate required paths exist
    required_paths = [
        'test_data_path', 
        'categoried_kp_path',
        'model_predictions_path', 
        'candidate_captions_path', 
        'golden_kp_path'
    ]
    
    for path_key in required_paths:
        path = getattr(args, path_key)
        if not os.path.exists(path):
            print(f"Warning: Required file not found: {path}")
            print(f"Please update the path for {path_key} in the configuration.")
    
    # Run evaluation pipeline
    try:
        pipeline = VideoEvaluationPipeline(args)
        results = pipeline.run_full_pipeline()
        
        print("\nExample evaluation completed!")
        return results
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("Please check your configuration and file paths.")
        return None


def batch_evaluation_example():
    """Example of evaluating multiple models in batch."""
    
    # List of models to evaluate
    model_configs = [
        {
            'model_name': 'InternVL2_5-8B',
            'model_predictions_path': '/path/to/InternVL2_5-8B_predictions.jsonl',
            'candidate_captions_path': '/path/to/InternVL2_5-8B_captions.jsonl'
        },
        {
            'model_name': 'InternVL2_5-26B', 
            'model_predictions_path': '/path/to/InternVL2_5-26B_predictions.jsonl',
            'candidate_captions_path': '/path/to/InternVL2_5-26B_captions.jsonl'
        },
        # Add more models as needed
    ]
    
    # Base configuration
    base_config = create_example_config()
    all_results = {}
    
    for model_config in model_configs:
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_config['model_name']}")
        print(f"{'='*80}")
        
        # Update configuration for current model
        config = base_config.copy()
        config.update(model_config)
        config['output_dir'] = f"./evaluation_results/{model_config['model_name']}"
        
        # Create arguments and run evaluation
        args = argparse.Namespace(**config)
        
        try:
            pipeline = VideoEvaluationPipeline(args)
            results = pipeline.run_full_pipeline()
            all_results[model_config['model_name']] = results
            
        except Exception as e:
            print(f"Error evaluating {model_config['model_name']}: {e}")
            continue
    
    # Display comparison results
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    
    if all_results:
        # Print header
        categories = base_config['categories']
        header = "Model" + " " * 20
        for cat in categories:
            header += f"{cat[:15]:>18}"
        header += f"{'Overall':>18}"
        print(header)
        print("-" * len(header))
        
        # Print results for each model
        for model_name, results in all_results.items():
            if not results:
                continue
                
            line = f"{model_name:25}"
            
            # Category F1 scores
            for category in categories:
                f1_score = results['f1']['category_score'].get(category, 0.0)
                line += f"{f1_score*100:15.1f}   "
            
            # Overall F1 score
            overall_f1 = results['f1']['overall']
            line += f"{overall_f1*100:15.1f}"
            
            print(line)
    
    return all_results


if __name__ == "__main__":
    # Choose which example to run:
    
    print("Choose evaluation mode:")
    print("1. Single model evaluation")
    print("2. Batch model evaluation")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_evaluation_example()
    elif choice == "2":
        batch_evaluation_example()
    else:
        print("Invalid choice. Running single model evaluation by default.")
        run_evaluation_example()