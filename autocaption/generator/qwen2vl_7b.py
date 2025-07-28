"""
Qwen2-VL-7B model wrapper for video captioning tasks.
Provides single and batch inference capabilities with proper error handling.
"""

import os
import sys
import json
import copy
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Required dependencies not found: {e}")
    DEPENDENCIES_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    logging.warning("qwen_vl_utils not available, some features may not work")
    QWEN_UTILS_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from util import ACTION_DICT, ACTION_PROMPT
except ImportError:
    logging.error("Failed to import util module")
    ACTION_DICT = {}
    ACTION_PROMPT = {}

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the Qwen2-VL-7B model."""
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_tokens: int = 256
    tensor_parallel_size: int = 1
    max_images_per_prompt: int = 16
    trust_remote_code: bool = True
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Load configuration from environment variables."""
        return cls(
            model_path=os.getenv("QWEN_MODEL_PATH", cls.model_path),
            temperature=float(os.getenv("QWEN_TEMPERATURE", cls.temperature)),
            top_p=float(os.getenv("QWEN_TOP_P", cls.top_p)),
            top_k=int(os.getenv("QWEN_TOP_K", cls.top_k)),
            repetition_penalty=float(os.getenv("QWEN_REP_PENALTY", cls.repetition_penalty)),
            max_tokens=int(os.getenv("QWEN_MAX_TOKENS", cls.max_tokens)),
            tensor_parallel_size=int(os.getenv("QWEN_TENSOR_PARALLEL", cls.tensor_parallel_size)),
        )


class Qwen2VL7B:
    """
    Wrapper for Qwen2-VL-7B model providing video captioning capabilities.
    Supports both single and batch inference with proper error handling.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model, processor, and sampling parameters.
        
        Args:
            config (Optional[ModelConfig]): Model configuration. Uses default if None.
            
        Raises:
            RuntimeError: If required dependencies are not available.
            Exception: If model initialization fails.
        """
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Required dependencies (vllm, transformers) are not available")
        
        self.config = config or ModelConfig.from_env()
        logger.info(f"Initializing Qwen2VL7B with model: {self.config.model_path}")
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_tokens=self.config.max_tokens,
            stop_token_ids=[],
        )
        
        # Initialize processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            logger.info("Processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            raise
        
        # Initialize model
        try:
            self.model = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                limit_mm_per_prompt={"image": self.config.max_images_per_prompt},
                trust_remote_code=self.config.trust_remote_code,
                enforce_eager=self.config.enforce_eager,
                disable_custom_all_reduce=self.config.disable_custom_all_reduce,
            )
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _validate_node(self, node) -> bool:
        """
        Validate that the node has required metadata.
        
        Args:
            node: MCTS node object
            
        Returns:
            bool: True if node is valid, False otherwise
        """
        if not hasattr(node, 'meta_data'):
            logger.error("Node missing meta_data attribute")
            return False
        
        meta_data = node.meta_data
        if not isinstance(meta_data, dict):
            logger.error("Node meta_data is not a dictionary")
            return False
        
        if 'video_path' not in meta_data:
            logger.error("Node meta_data missing video_path")
            return False
        
        video_path = meta_data['video_path']
        if not video_path or not isinstance(video_path, str):
            logger.error("Invalid video_path in node meta_data")
            return False
        
        return True
    
    def _prepare_messages(self, video_path: str, prompt: str, num_frames: int = 64) -> List[Dict[str, Any]]:
        """
        Prepare messages for the model input.
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Text prompt
            num_frames (int): Number of frames to extract from video
            
        Returns:
            List[Dict[str, Any]]: Formatted messages for the model
        """
        content = [
            {"type": "video", "video": video_path, "nframes": num_frames},
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": content}]
    
    def _process_model_input(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process messages into model input format.
        
        Args:
            messages (List[Dict[str, Any]]): Input messages
            
        Returns:
            Dict[str, Any]: Processed input for the model
            
        Raises:
            RuntimeError: If qwen_vl_utils is not available
        """
        if not QWEN_UTILS_AVAILABLE:
            raise RuntimeError("qwen_vl_utils is required for processing vision info")
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
    
    def get_completion(self, node, action_prompt: str) -> str:
        """
        Generate a single completion for a given node and action prompt.

        Args:
            node: The current MCTS node containing metadata like video path.
            action_prompt (str): Textual prompt describing the action.

        Returns:
            str: Generated response from the model.
            
        Raises:
            ValueError: If node validation fails.
            Exception: If model inference fails.
        """
        if not self._validate_node(node):
            raise ValueError("Invalid node provided")
        
        video_path = node.meta_data["video_path"]
        logger.debug(f"Generating completion for video: {video_path}")
        
        try:
            # Prepare input
            messages = self._prepare_messages(video_path, action_prompt)
            llm_input = self._process_model_input(messages)
            
            # Generate response
            outputs = self.model.generate([llm_input], sampling_params=self.sampling_params)
            
            if not outputs or not outputs[0].outputs:
                logger.error("Model returned empty output")
                return ""
            
            result = outputs[0].outputs[0].text
            logger.debug(f"Generated completion of length {len(result)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return f"Error: Failed to generate completion - {str(e)}"

    def get_completion_batch(self, node, action_prompt_list: List[str]) -> List[str]:
        """
        Generate multiple completions for a list of action prompts.

        Args:
            node: The current MCTS node containing metadata like video path.
            action_prompt_list (List[str]): A list of action prompts to query the model with.

        Returns:
            List[str]: List of generated texts corresponding to each input prompt.
            
        Raises:
            ValueError: If node validation fails or empty prompt list provided.
            Exception: If batch inference fails.
        """
        if not self._validate_node(node):
            raise ValueError("Invalid node provided")
        
        if not action_prompt_list:
            logger.warning("Empty action prompt list provided")
            return []
        
        video_path = node.meta_data["video_path"]
        logger.info(f"Generating {len(action_prompt_list)} batch completions for video: {video_path}")
        
        try:
            # Prepare all inputs
            all_llm_inputs = []
            for prompt in action_prompt_list:
                messages = self._prepare_messages(video_path, prompt)
                llm_input = self._process_model_input(messages)
                all_llm_inputs.append(llm_input)
            
            # Generate batch responses
            outputs = self.model.generate(all_llm_inputs, sampling_params=self.sampling_params)
            
            if len(outputs) != len(action_prompt_list):
                logger.error(f"Output count mismatch: expected {len(action_prompt_list)}, got {len(outputs)}")
                return ["Error: Output count mismatch"] * len(action_prompt_list)
            
            # Extract results
            results = []
            for i, output in enumerate(outputs):
                if output.outputs:
                    results.append(output.outputs[0].text)
                    logger.debug(f"Batch completion {i+1}: length {len(output.outputs[0].text)}")
                else:
                    logger.warning(f"Empty output for prompt {i+1}")
                    results.append("")
            
            logger.info(f"Successfully generated {len(results)} batch completions")
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch completions: {e}")
            error_msg = f"Error: Failed to generate batch completions - {str(e)}"
            return [error_msg] * len(action_prompt_list)

    def generate_action_prompt(self, node, next_action_type_list: List[str]) -> List[str]:
        """
        Generate a list of action prompts based on the next possible action types.

        Args:
            node: The current MCTS node, possibly containing prior taken actions.
            next_action_type_list (List[str]): A list of action type identifiers (keys from ACTION_DICT).

        Returns:
            List[str]: List of prompts suitable for passing into the model.
            
        Raises:
            ValueError: If invalid action types provided.
        """
        if not next_action_type_list:
            logger.warning("Empty action type list provided")
            return []
        
        if not ACTION_DICT or not ACTION_PROMPT:
            logger.error("ACTION_DICT or ACTION_PROMPT not properly initialized")
            return ["Error: Action dictionaries not initialized"] * len(next_action_type_list)
        
        logger.debug(f"Generating prompts for actions: {next_action_type_list}")
        
        action_prompt_list = []
        
        try:
            for action_type in next_action_type_list:
                if action_type not in ACTION_PROMPT:
                    logger.error(f"Unknown action type: {action_type}")
                    action_prompt_list.append(f"Error: Unknown action type {action_type}")
                    continue
                
                if action_type != ACTION_DICT.get("ACTION2"):
                    # Use default prompt for non-ACTION2 types
                    action_prompt_list.append(ACTION_PROMPT[action_type])
                else:
                    # Special handling for ACTION2 to avoid repeating previously mentioned details
                    prompt = self._generate_action2_prompt(node)
                    action_prompt_list.append(prompt)
            
            logger.info(f"Generated {len(action_prompt_list)} action prompts")
            return action_prompt_list
            
        except Exception as e:
            logger.error(f"Error generating action prompts: {e}")
            return [f"Error: Failed to generate prompt - {str(e)}"] * len(next_action_type_list)
    
    def _generate_action2_prompt(self, node) -> str:
        """
        Generate ACTION2 prompt with context awareness to avoid repetition.
        
        Args:
            node: Current MCTS node
            
        Returns:
            str: ACTION2 prompt with repetition avoidance
        """
        base_prompt = ACTION_PROMPT.get(ACTION_DICT.get("ACTION2", ""), "")
        
        if not hasattr(node, 'taken_action_list') or not node.taken_action_list:
            return base_prompt
        
        try:
            # Extract previously mentioned details from ACTION2 actions
            prev_details = []
            for action in node.taken_action_list:
                if isinstance(action, (list, tuple)) and len(action) >= 2:
                    action_type, action_data = action[0], action[1]
                    if action_type == ACTION_DICT.get("ACTION2"):
                        if isinstance(action_data, (list, tuple)) and len(action_data) > 0:
                            # Extract detail from tuple format (detail, category, aspects, instruction)
                            detail = action_data[0] if isinstance(action_data[0], str) else str(action_data[0])
                            prev_details.append(detail)
                        elif isinstance(action_data, str):
                            # Simple string format
                            prev_details.append(action_data)
            
            if prev_details:
                # Remove duplicates and shuffle for variety
                unique_details = list(set(prev_details))
                random.shuffle(unique_details)
                
                # Create enumerated list
                detail_str = " ".join([f"{i+1}. {item}" for i, item in enumerate(unique_details)])
                
                enhanced_prompt = (
                    f"{base_prompt} We have mentioned {len(unique_details)} details before, "
                    f"please do not mention them again. They are: {detail_str}."
                )
                
                logger.debug(f"Generated ACTION2 prompt with {len(unique_details)} excluded details")
                return enhanced_prompt
            else:
                return base_prompt
                
        except Exception as e:
            logger.warning(f"Error generating ACTION2 prompt with context: {e}")
            return base_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration and statistics
        """
        return {
            "model_path": self.config.model_path,
            "sampling_params": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "max_tokens": self.config.max_tokens,
            },
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_images_per_prompt": self.config.max_images_per_prompt,
            "dependencies_available": DEPENDENCIES_AVAILABLE,
            "qwen_utils_available": QWEN_UTILS_AVAILABLE,
        }
    
    def update_sampling_params(self, **kwargs) -> None:
        """
        Update sampling parameters dynamically.
        
        Args:
            **kwargs: Sampling parameter updates
        """
        valid_params = {
            'temperature', 'top_p', 'top_k', 'repetition_penalty', 
            'max_tokens', 'stop_token_ids'
        }
        
        updates = {k: v for k, v in kwargs.items() if k in valid_params}
        
        if updates:
            # Create new sampling params with updates
            current_params = {
                'temperature': self.sampling_params.temperature,
                'top_p': self.sampling_params.top_p,
                'top_k': self.sampling_params.top_k,
                'repetition_penalty': self.sampling_params.repetition_penalty,
                'max_tokens': self.sampling_params.max_tokens,
                'stop_token_ids': self.sampling_params.stop_token_ids,
            }
            current_params.update(updates)
            
            self.sampling_params = SamplingParams(**current_params)
            logger.info(f"Updated sampling parameters: {updates}")
        else:
            logger.warning(f"No valid parameters to update from: {list(kwargs.keys())}")
