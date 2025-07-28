# -*- coding: utf-8 -*-
"""
Main Entry Point
Complete pipeline for evaluating video caption models using precision, recall, and F1 scores.
"""

import argparse
import json
import os
from typing import List, Dict, Any

from precision_prompt_builder import PrecisionPromptBuilder
from recall_prompt_builder import RecallPromptBuilder
from gpt_api_caller import GPT4oAPICaller
from metric_calculator import MetricCalculator


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video Caption Evaluation Pipeline")
    
    # Data paths
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data containing key points')
    parser.add_argument('--test_data_threshold', type=str, default='080',
                       help='Threshold for filtering test data')
    parser.add_argument('--categoried_kp_path', type=str, required=True,
                       help='Path to categorized key points file')
    
    # Model prediction paths
    parser.add_argument('--model_predictions_path', type=str, required=True,
                       help='Path to model prediction results')
    parser.add_argument('--candidate_captions_path', type=str, required=True,
                       help='Path to candidate captions file')
    parser.add_argument('--golden_kp_path', type=str, required=True,
                       help='Path to golden key points file')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output files')
    parser.add_argument('--precision_prompt_path', type=str,
                       help='Path to save precision prompts')
    parser.add_argument('--recall_prompt_path', type=str,
                       help='Path to save recall prompts')
    
    # API configuration
    parser.add_argument('--openai_api_key', type=str,
                       help='OpenAI API key for GPT-4o calls')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the model being evaluated')
    
    # Evaluation categories
    parser.add_argument('--categories', type=str, nargs='+',
                       default=["Appearance Description", "Action Description", 
                               "Environment Description", "Object Description", 
                               "Camera Movement and Composition"],
                       help='List of evaluation categories')
    
    return parser.parse_args()


class VideoEvaluationPipeline:
    """Complete pipeline for video caption evaluation."""
    
    def __init__(self, args):
        """
        Initialize the evaluation pipeline.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.categories = args.categories
        self.output_dir = args.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default paths if not provided
        if not args.precision_prompt_path:
            args.precision_prompt_path = os.path.join(
                self.output_dir, f"{args.model_name}_precision_prompts.jsonl"
            )
        if not args.recall_prompt_path:
            args.recall_prompt_path = os.path.join(
                self.output_dir, f"{args.model_name}_recall_prompts.jsonl"
            )
        
        # Initialize components
        self.precision_builder = PrecisionPromptBuilder(
            args.test_data_path, args.test_data_threshold
        )
        self.recall_builder = RecallPromptBuilder()
        
        if args.openai_api_key:
            self.api_caller = GPT4oAPICaller(args.openai_api_key)
        else:
            self.api_caller = None
            
        self.metric_calculator = MetricCalculator(
            self.categories, args.test_data_threshold
        )
        
        # Load categorized key points
        self.metric_calculator.load_categorized_kp(args.categoried_kp_path)
    
    def step1_create_precision_prompts(self) -> str:
        """
        Step 1: Create precision evaluation prompts.
        
        Returns:
            str: Path to the created precision prompts file
        """
        print("Step 1: Creating precision evaluation prompts...")
        
        self.precision_builder.create_prompts(
            self.args.model_predictions_path,
            self.args.precision_prompt_path
        )
        
        print(f"Precision prompts saved to: {self.args.precision_prompt_path}")
        return self.args.precision_prompt_path
    
    def step2_create_recall_prompts(self) -> str:
        """
        Step 2: Create recall evaluation prompts.
        
        Returns:
            str: Path to the created recall prompts file
        """
        print("Step 2: Creating recall evaluation prompts...")
        
        self.recall_builder.create_prompts(
            self.args.candidate_captions_path,
            self.args.golden_kp_path,
            self.args.recall_prompt_path
        )
        
        print(f"Recall prompts saved to: {self.args.recall_prompt_path}")
        return self.args.recall_prompt_path
    
    def step3_call_gpt4o(self, precision_prompt_path: str, recall_prompt_path: str) -> tuple:
        """
        Step 3: Call GPT-4o API for evaluation.
        
        Args:
            precision_prompt_path (str): Path to precision prompts
            recall_prompt_path (str): Path to recall prompts
            
        Returns:
            tuple: Paths to precision and recall results
        """
        print("Step 3: Calling GPT-4o API for evaluation...")
        
        if self.api_caller is None:
            print("Warning: No API key provided. Skipping GPT-4o calls.")
            print("Please implement API calls manually or provide an API key.")
            
            # Return expected output paths for manual implementation
            precision_results_path = os.path.join(
                self.output_dir, f"{self.args.model_name}_precision_results.jsonl"
            )
            recall_results_path = os.path.join(
                self.output_dir, f"{self.args.model_name}_recall_results.jsonl"
            )
            
            print(f"Expected precision results path: {precision_results_path}")
            print(f"Expected recall results path: {recall_results_path}")
            
            # TODO: Implement actual API calls here
            return precision_results_path, recall_results_path
        
        # Call API for precision evaluation
        precision_results_path = os.path.join(
            self.output_dir, f"{self.args.model_name}_precision_results.jsonl"
        )
        print("Processing precision prompts...")
        self.api_caller.process_prompt_file(precision_prompt_path, precision_results_path)
        
        # Call API for recall evaluation  
        recall_results_path = os.path.join(
            self.output_dir, f"{self.args.model_name}_recall_results.jsonl"
        )
        print("Processing recall prompts...")
        self.api_caller.process_prompt_file(recall_prompt_path, recall_results_path)
        
        return precision_results_path, recall_results_path
    
    def step4_calculate_metrics(self, precision_results_path: str, 
                              recall_results_path: str) -> Dict[str, Any]:
        """
        Step 4: Calculate precision, recall, and F1 scores.
        
        Args:
            precision_results_path (str): Path to precision evaluation results
            recall_results_path (str): Path to recall evaluation results
            
        Returns:
            Dict[str, Any]: Dictionary containing all calculated metrics
        """
        print("Step 4: Calculating metrics...")
        
        # Check if result files exist
        if not os.path.exists(precision_results_path):
            print(f"Warning: Precision results file not found: {precision_results_path}")
            print("Please run GPT-4o evaluation first or provide the results file.")
            return {}
        
        if not os.path.exists(recall_results_path):
            print(f"Warning: Recall results file not found: {recall_results_path}")
            print("Please run GPT-4o evaluation first or provide the results file.")
            return {}
        
        # Calculate precision scores
        print("Calculating precision scores...")
        precision_scores = self.metric_calculator.calculate_precision(
            precision_results_path, self.args.model_name
        )
        
        # Calculate recall scores
        print("Calculating recall scores...")
        recall_scores = self.metric_calculator.calculate_recall(
            recall_results_path, self.args.model_name
        )
        
        # Calculate F1 scores
        print("Calculating F1 scores...")
        f1_scores = self.metric_calculator.calculate_f1_score(
            precision_scores, recall_scores
        )
        
        # Combine all results
        results = {
            'model_name': self.args.model_name,
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores
        }
        
        return results
    
    def step5_save_and_display_results(self, results: Dict[str, Any]) -> None:
        """
        Step 5: Save and display the evaluation results.
        
        Args:
            results (Dict[str, Any]): Dictionary containing all calculated metrics
        """
        if not results:
            print("No results to display.")
            return
            
        print("Step 5: Saving and displaying results...")
        
        # Save results to JSON file
        results_json_path = os.path.join(
            self.output_dir, f"{self.args.model_name}_evaluation_results.json"
        )
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {results_json_path}")
        
        # Display results
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS FOR {self.args.model_name}")
        print("="*80)
        
        print("\nCategory-wise Results (Precision/Recall/F1):")
        print("-" * 60)
        
        for category in self.categories:
            precision = results['precision']['category_score'].get(category, 0.0)
            recall = results['recall']['category_score'].get(category, 0.0)
            f1 = results['f1']['category_score'].get(category, 0.0)
            
            print(f"{category:40}: {precision*100:5.1f} / {recall*100:5.1f} / {f1*100:5.1f}")
        
        print("-" * 60)
        overall_precision = results['precision']['overall']
        overall_recall = results['recall']['overall']
        overall_f1 = results['f1']['overall']
        
        print(f"{'Overall':40}: {overall_precision*100:5.1f} / {overall_recall*100:5.1f} / {overall_f1*100:5.1f}")
        
        # Display LaTeX format
        print("\nLaTeX Table Format:")
        print("-" * 40)
        latex_output = self.metric_calculator.format_results_for_display(
            results['precision'], results['recall'], results['f1'], self.args.model_name
        )
        print(latex_output)
        
        # Save LaTeX format
        latex_path = os.path.join(
            self.output_dir, f"{self.args.model_name}_latex_results.txt"
        )
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)
        
        print(f"LaTeX format saved to: {latex_path}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Dict[str, Any]: Final evaluation results
        """
        print(f"Starting evaluation pipeline for model: {self.args.model_name}")
        print("="*80)
        
        try:
            # Step 1: Create precision prompts
            precision_prompt_path = self.step1_create_precision_prompts()
            
            # Step 2: Create recall prompts
            recall_prompt_path = self.step2_create_recall_prompts()
            
            # Step 3: Call GPT-4o API
            precision_results_path, recall_results_path = self.step3_call_gpt4o(
                precision_prompt_path, recall_prompt_path
            )
            
            # Step 4: Calculate metrics
            results = self.step4_calculate_metrics(
                precision_results_path, recall_results_path
            )
            
            # Step 5: Save and display results
            self.step5_save_and_display_results(results)
            
            print("\nPipeline completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise


def main():
    """Main entry point."""
    args = get_args()
    
    # Create and run pipeline
    pipeline = VideoEvaluationPipeline(args)
    results = pipeline.run_full_pipeline()
    
    return results


if __name__ == "__main__":
    main()