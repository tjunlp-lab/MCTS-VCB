# -*- coding: utf-8 -*-
"""
Metric Calculator
Processes GPT-4o responses and calculates precision, recall, and F1 scores.
"""

import json
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class LazyDecoder(json.JSONDecoder):
    """Custom JSON decoder that handles malformed JSON strings."""
    
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)


class MetricCalculator:
    """Calculator for precision, recall, and F1 scores."""
    
    def __init__(self, categories: List[str], threshold: str = "080"):
        """
        Initialize the MetricCalculator.
        
        Args:
            categories (List[str]): List of evaluation categories
            threshold (str): Threshold value for filtering
        """
        self.categories = categories
        self.threshold = threshold
        self.index2kp_list_with_threshold = {}
    
    def load_categorized_kp(self, categoried_kp_path: str) -> None:
        """
        Load categorized key points from file.
        
        Args:
            categoried_kp_path (str): Path to the categorized key points file
        """
        with open(categoried_kp_path, "r") as in_file:
            for line in in_file:
                data = json.loads(line)
                index = int(data["index"])
                self.index2kp_list_with_threshold[index] = [
                    item for item in data["passed_kp_list"]
                ]
    
    def calculate_recall(self, judge_res_path: str, model_name: str) -> Dict[str, Any]:
        """
        Calculate recall scores from judge results.
        
        Args:
            judge_res_path (str): Path to the judge results file
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict[str, Any]: Recall scores by category and overall
        """
        category_statistic = {
            category: {"entailment": 0, "neutral": 0, "contradiction": 0} 
            for category in self.categories
        }
        recall_score_per_model = {}
        all_entail_count = 0
        all_label_count = 0
        
        with open(judge_res_path, "r") as infile:
            fail_parse_num = 0
            for line in infile:
                data = json.loads(line)
                video_index = int(data["index"].split("_")[0])
                batch_index = int(data["index"].split("_")[1])  # starts from 0
                batch_size = 30
                sample_answer = data["answer"]
                
                # Clean the response
                if "```json" in sample_answer:
                    sample_answer = sample_answer.split('```json')[1]
                if "```" in sample_answer:
                    sample_answer = sample_answer.split('```')[0]
                
                try:
                    judgements = json.loads(sample_answer, cls=LazyDecoder)
                    if len(judgements) != data["kp_num"]:
                        continue
                    
                    for key_index, key in enumerate(judgements):
                        # Extract judgment
                        judgment_text = judgements[key]["judgement"].lower()
                        if "entailment" in judgment_text:
                            judge_res = "entailment"
                        elif "contradiction" in judgment_text:
                            judge_res = "contradiction"
                        elif "neutral" in judgment_text:
                            judge_res = "neutral"
                        else:
                            continue
                        
                        # Get corresponding key point
                        kp_item = self.index2kp_list_with_threshold[video_index][
                            batch_index * batch_size + key_index
                        ]
                        
                        if kp_item["text"] != data["kp_list"][key_index]:
                            raise ValueError("Key point mismatch")
                        
                        # Filter by threshold
                        if (kp_item["threshold"] is None or 
                            int(kp_item["threshold"]) > int(self.threshold)):
                            continue
                        
                        # Update counts
                        if judge_res == "entailment":
                            all_entail_count += 1
                        all_label_count += 1
                        
                        # Update category statistics
                        kp_category = kp_item["category"]
                        if kp_category is not None:
                            category_statistic[kp_category][judge_res] += 1
                
                except Exception as e:
                    fail_parse_num += 1
        
        recall_score_per_model["fail_num"] = fail_parse_num
        recall_score_per_model["category_score"] = {}
        
        # Calculate category scores
        for category, stats in category_statistic.items():
            total = stats["entailment"] + stats["neutral"] + stats["contradiction"]
            if total > 0:
                recall_score_per_model["category_score"][category] = round(
                    stats["entailment"] / total, 5
                )
            else:
                recall_score_per_model["category_score"][category] = 0.0
        
        # Calculate overall score
        if all_label_count > 0:
            recall_score_per_model["overall"] = round(all_entail_count / all_label_count, 5)
        else:
            recall_score_per_model["overall"] = 0.0
        
        return recall_score_per_model
    
    def calculate_precision(self, judge_res_path: str, model_name: str) -> Dict[str, Any]:
        """
        Calculate precision scores from judge results.
        
        Args:
            judge_res_path (str): Path to the judge results file
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict[str, Any]: Precision scores by category and overall
        """
        category_statistic = {
            category: {"entailment": 0, "neutral": 0, "contradiction": 0} 
            for category in self.categories
        }
        precision_score_per_model = {}
        all_entail_count = 0
        all_label_count = 0
        
        with open(judge_res_path, "r") as infile:
            fail_parse_num = 0
            for line in infile:
                data = json.loads(line)
                video_index = int(data["index"])
                kp_list = data["kp_list"]
                sample_answer = data["answer"]
                
                # Clean the response
                if "```json" in sample_answer:
                    sample_answer = sample_answer.split('```json')[1]
                if "```" in sample_answer:
                    sample_answer = sample_answer.split('```')[0]
                
                try:
                    judgements = json.loads(sample_answer, cls=LazyDecoder)
                    if len(judgements) != data["kp_num"]:
                        continue
                    
                    for key_index, key in enumerate(judgements):
                        # Extract judgment
                        judgment_text = judgements[key]["judgement"].lower()
                        if "entailment" in judgment_text:
                            judge_res = "entailment"
                        elif "contradiction" in judgment_text:
                            judge_res = "contradiction"
                        elif "neutral" in judgment_text:
                            judge_res = "neutral"
                        else:
                            continue
                        
                        # Update counts
                        if judge_res == "entailment":
                            all_entail_count += 1
                        all_label_count += 1
                        
                        # Update category statistics
                        kp_category = kp_list[key_index]["category"]
                        if kp_category is not None:
                            category_statistic[kp_category][judge_res] += 1
                
                except Exception as e:
                    fail_parse_num += 1
        
        precision_score_per_model["fail_num"] = fail_parse_num
        precision_score_per_model["category_score"] = {}
        
        # Calculate category scores
        for category, stats in category_statistic.items():
            total = stats["entailment"] + stats["neutral"] + stats["contradiction"]
            if total > 0:
                precision_score_per_model["category_score"][category] = round(
                    stats["entailment"] / total, 5
                )
            else:
                precision_score_per_model["category_score"][category] = 0.0
        
        # Calculate overall score
        if all_label_count > 0:
            precision_score_per_model["overall"] = round(all_entail_count / all_label_count, 5)
        else:
            precision_score_per_model["overall"] = 0.0
        
        return precision_score_per_model
    
    def calculate_f1_score(self, precision_scores: Dict[str, Any], 
                          recall_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate F1 scores from precision and recall scores.
        
        Args:
            precision_scores (Dict[str, Any]): Precision scores by category
            recall_scores (Dict[str, Any]): Recall scores by category
            
        Returns:
            Dict[str, Any]: F1 scores by category and overall
        """
        f1_scores = {"category_score": {}, "overall": 0.0}
        
        # Calculate category F1 scores
        for category in self.categories:
            if (category in precision_scores["category_score"] and 
                category in recall_scores["category_score"]):
                
                precision = precision_scores["category_score"][category]
                recall = recall_scores["category_score"][category]
                
                if precision + recall > 0:
                    f1 = round(2 * (precision * recall) / (precision + recall), 5)
                else:
                    f1 = 0.0
                
                f1_scores["category_score"][category] = f1
        
        # Calculate overall F1 score
        overall_precision = precision_scores["overall"]
        overall_recall = recall_scores["overall"]
        
        if overall_precision + overall_recall > 0:
            overall_f1 = round(
                2 * (overall_precision * overall_recall) / (overall_precision + overall_recall), 5
            )
        else:
            overall_f1 = 0.0
        
        f1_scores["overall"] = overall_f1
        
        return f1_scores
    
    def format_results_for_display(self, precision_scores: Dict[str, Any], 
                                 recall_scores: Dict[str, Any], 
                                 f1_scores: Dict[str, Any], 
                                 model_name: str) -> str:
        """
        Format results for display in LaTeX table format.
        
        Args:
            precision_scores (Dict[str, Any]): Precision scores
            recall_scores (Dict[str, Any]): Recall scores  
            f1_scores (Dict[str, Any]): F1 scores
            model_name (str): Name of the model
            
        Returns:
            str: Formatted results string
        """
        result_parts = [model_name.replace("_", "\\_")]
        
        # Add category scores
        for category in self.categories:
            precision = precision_scores["category_score"].get(category, 0.0)
            recall = recall_scores["category_score"].get(category, 0.0)  
            f1 = f1_scores["category_score"].get(category, 0.0)
            
            result_parts.append(
                f"{round(precision*100,1)}/{round(recall*100,1)}/{round(f1*100,1)}"
            )
        
        # Add overall scores
        overall_precision = precision_scores["overall"]
        overall_recall = recall_scores["overall"]
        overall_f1 = f1_scores["overall"]
        
        result_parts.append(
            f"{round(overall_precision*100,1)}/{round(overall_recall*100,1)}/{round(overall_f1*100,1)}"
        )
        
        return " & ".join(result_parts) + " & \\\\\\"