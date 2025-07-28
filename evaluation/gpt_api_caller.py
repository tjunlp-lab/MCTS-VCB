# -*- coding: utf-8 -*-
"""
GPT-4o API Caller
Handles calling GPT-4o API for evaluation prompts.
Warning: This file is created by Claude AI, which may not work correctly. Users should edit themselves.
"""

import json
import time
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI


class GPT4oAPICaller:
    """Class for calling GPT-4o API to evaluate prompts."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3, delay: float = 1.0):
        """
        Initialize the GPT-4o API caller.
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model name to use (default: gpt-4o)
            max_retries (int): Maximum number of retries for failed requests
            delay (float): Delay between requests in seconds
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.delay = delay
    
    def call_api(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        """
        Call GPT-4o API with a single prompt.
        
        Args:
            prompt (str): The prompt to send to the API
            temperature (float): Temperature parameter for generation
            
        Returns:
            Optional[str]: The API response or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"Max retries reached for prompt: {prompt[:100]}...")
                    return None
    
    def process_prompt_file(self, input_path: str, output_path: str) -> None:
        """
        Process a file of prompts and save the results.
        
        Args:
            input_path (str): Path to the input prompt file
            output_path (str): Path to save the API responses
        """
        processed_count = 0
        failed_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    data = json.loads(line.strip())
                    prompt = data['question']
                    
                    print(f"Processing prompt {processed_count + 1}...")
                    
                    # Call API
                    response = self.call_api(prompt)
                    
                    if response is not None:
                        # Create response data
                        response_data = {
                            'index': data['index'],
                            'kp_num': data['kp_num'],
                            'kp_list': data.get('kp_list', []),
                            'answer': response
                        }
                        
                        outfile.write(json.dumps(response_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    # Add delay between requests
                    time.sleep(self.delay)
        
        print(f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")


# TODO: Implement the actual API call
def call_gpt4o_api(prompts: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """
    Call GPT-4o API for a list of prompts.
    
    Args:
        prompts (List[Dict[str, Any]]): List of prompt data
        api_key (str): OpenAI API key
        
    Returns:
        List[Dict[str, Any]]: List of API responses
        
    Note:
        This is a placeholder function. Implement based on your specific needs:
        - Set up proper API credentials
        - Handle rate limiting
        - Implement error handling and retries
        - Add logging for monitoring
    """
    # TODO: Implement actual GPT-4o API calling logic
    caller = GPT4oAPICaller(api_key)
    results = []
    
    for prompt_data in prompts:
        response = caller.call_api(prompt_data['question'])
        if response:
            result = {
                'index': prompt_data['index'],
                'kp_num': prompt_data['kp_num'],
                'kp_list': prompt_data.get('kp_list', []),
                'answer': response
            }
            results.append(result)
    
    return results