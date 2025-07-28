import random
import json
import copy
import os
import re
import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logging.warning(f"Optional dependency not found: {e}")
    logging.warning("Some features may not be available")

# Configuration class for better config management
@dataclass
class APIConfig:
    """Configuration for API endpoints and credentials."""
    openai_api_key: str = "EMPTY"
    openai_api_base: str = "http://localhost:8000/v1"
    model_name: str = "Qwen2.5-7B-Instruct"
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
            model_name=os.getenv("MODEL_NAME", "Qwen2.5-7B-Instruct")
        )

# Initialize global configuration
API_CONFIG = APIConfig.from_env()

# Load the sentence embedding model once (with error handling)
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    logging.warning(f"Failed to load sentence transformer model: {e}")
    model = None

def get_openai_client() -> Optional[OpenAI]:
    """Create OpenAI client with proper error handling."""
    try:
        return OpenAI(
            api_key=API_CONFIG.openai_api_key,
            base_url=API_CONFIG.openai_api_base,
        )
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None

# Define action dictionary
ACTION_DICT = {
    "ACTION1": "ACTION1",
    "ACTION2": "ACTION2", 
    "ACTION3": "ACTION3",
    "ACTION4": "ACTION4",
    "ACTION5": "ACTION5",
    "ACTION6": "ACTION6",
    # "ACTION7": "ACTION7",
}

# Define prompts corresponding to each action type
# ⚠️ DO NOT modify the following prompt content
ACTION_PROMPT = {
    ACTION_DICT["ACTION1"]: "What should you do first? I think you should first describe the video in detail, focusing on providing a comprehensive overview of the entire video content, including the main subjects, actions, and settings. So please describe the video overall first.",
    ACTION_DICT["ACTION2"]: "Next, I think you should carefully observe the details of the page, I think there are these details to consider, People or Animals, Plants, Food, Buildings, Vehicles, Other Objects and so on. Please ouput only one detail you want to focus on.",
    ACTION_DICT["ACTION3"]: "Please describe the video from a new temporal perspective. Focus on changes that occur before and after a specific camera transition or time point in the video.",
    ACTION_DICT["ACTION4"]: "Please describe the video from a new spatial perspective. Focus on different areas of the frame, such as the left side, right side, foreground, or background.",
    ACTION_DICT["ACTION5"]: "Please provide a detailed description of the background in the video. Focus on the setting, environment, and any contextual elements that contribute to the overall scene.",
    ACTION_DICT["ACTION6"]: "Please describe the changes in camera shots and movements throughout the video. Focus on different types of shots, camera movements, angles, transitions, and any special effects used.",
}


def weighted_random_choice(action_set: set, num_choices: int) -> List[str]:
    """
    Randomly select a subset of actions from action_set, giving higher weight to ACTION2.

    Args:
        action_set (set): Set of available actions.
        num_choices (int): Number of actions to sample.

    Returns:
        List[str]: A list of randomly selected actions with ACTION2 heavily weighted.
        
    Raises:
        ValueError: If num_choices exceeds the size of action_set.
    """
    if num_choices > len(action_set):
        raise ValueError(f"Cannot choose {num_choices} items from set of size {len(action_set)}")
    
    action_list = list(action_set)
    weights = [2 if action == ACTION_DICT["ACTION2"] else 1 for action in action_list]

    chosen = []
    remaining_items = copy.deepcopy(action_list)
    remaining_weights = copy.deepcopy(weights)

    for _ in range(num_choices):
        if not remaining_items:
            break
            
        chosen_item = random.choices(remaining_items, remaining_weights, k=1)[0]
        chosen.append(chosen_item)

        # Remove chosen item from future draws
        index = remaining_items.index(chosen_item)
        del remaining_items[index]
        del remaining_weights[index]

    random.shuffle(chosen)
    return chosen


def shuffle_lists(list1: List, list2: List) -> Tuple[List, List]:
    """
    Shuffle two lists in unison while keeping the first element pair fixed.

    Args:
        list1 (List): First list to shuffle.
        list2 (List): Second list to shuffle.

    Returns:
        Tuple[List, List]: Shuffled versions of the two input lists.
        
    Raises:
        ValueError: If the lengths of the two lists are not equal.
    """
    if len(list1) != len(list2):
        raise ValueError("The lengths of both lists must be equal.")
    
    if len(list1) <= 1:
        return list1.copy(), list2.copy()
    
    combined = list(zip(list1, list2))
    first_pair = combined[0]
    remaining_pairs = combined[1:]
    random.shuffle(remaining_pairs)

    shuffled_list1 = [first_pair[0]] + [pair[0] for pair in remaining_pairs]
    shuffled_list2 = [first_pair[1]] + [pair[1] for pair in remaining_pairs]

    return shuffled_list1, shuffled_list2


def get_next_action_list(executed_actions: List[str]) -> Optional[List[str]]:
    """
    Determine the next set of actions that can be executed based on prerequisites
    and execution history.

    Args:
        executed_actions (List[str]): List of actions already taken.

    Returns:
        Optional[List[str]]: A list containing one randomly chosen action 
                             that is eligible for execution. Returns None 
                             if no further action is valid.
    """
    # Define all available actions
    all_actions = [ACTION_DICT[action] for action in ACTION_DICT.keys()]

    # Define prerequisite actions for each action
    prerequisites = {
        ACTION_DICT["ACTION1"]: [],
        ACTION_DICT["ACTION2"]: [ACTION_DICT["ACTION3"], ACTION_DICT["ACTION4"], ACTION_DICT["ACTION5"]],
        ACTION_DICT["ACTION3"]: [ACTION_DICT["ACTION1"]],
        ACTION_DICT["ACTION4"]: [ACTION_DICT["ACTION1"]],
        ACTION_DICT["ACTION5"]: [ACTION_DICT["ACTION1"]],
        ACTION_DICT["ACTION6"]: [ACTION_DICT["ACTION2"]],
    }

    # Define how many times each action is allowed to be executed
    multi_execution_actions = {
        ACTION_DICT["ACTION1"]: 2,
        ACTION_DICT["ACTION2"]: 1000,
        ACTION_DICT["ACTION3"]: 2,
        ACTION_DICT["ACTION4"]: 2,
        ACTION_DICT["ACTION5"]: 2,
        ACTION_DICT["ACTION6"]: 2,
    }

    # Count how many times each action has been executed
    execution_counts = {action: executed_actions.count(action) for action in all_actions}

    # Determine which actions are currently valid based on prerequisites and execution limits
    possible_actions = []
    for action in all_actions:
        # Skip actions that have reached their execution limit
        if execution_counts[action] >= multi_execution_actions[action]:
            continue
        # Include the action only if all its prerequisites are satisfied
        if all(prerequisite in executed_actions for prerequisite in prerequisites[action]):
            possible_actions.append(action)

    # Select one action randomly from the list of valid actions
    if possible_actions:
        num_choices = min(1, len(possible_actions))  # Limit to a single action
        return weighted_random_choice(possible_actions, num_choices)
    else:
        return None

def print_node(node) -> None:
    """
    Print the node information in JSON format.

    Args:
        node: Node object with various metadata and children.
    """
    try:
        node_dict = {
            'node_id': getattr(node, 'node_id', 'unknown'),
            'meta_data': getattr(node, 'meta_data', {}),
            'taken_action_list': getattr(node, 'taken_action_list', []),
            'taken_action_action_raw_res_list': getattr(node, 'taken_action_action_raw_res_list', []),
            'parent_node': getattr(node.parent_node, 'node_id', None) if hasattr(node, 'parent_node') and node.parent_node else None,
            'mc_value': getattr(node, 'mc_value', 0),
            'children': [getattr(child, 'node_id', 'unknown') for child in getattr(node, 'children', [])],
            'visit_times': getattr(node, 'visit_times', 0),
            'q_value': getattr(node, 'q_value', 0)
        }
        print(json.dumps(node_dict, indent=4, ensure_ascii=False))
    except Exception as e:
        logging.error(f"Error printing node: {e}")


def write_node(outpath: Union[str, Path], node) -> bool:
    """
    Write node data to a file in JSON lines format.

    Args:
        outpath (Union[str, Path]): Output file path.
        node: Node object to serialize.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure directory exists
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        
        node_dict = {
            'node_id': getattr(node, 'node_id', 'unknown'),
            'meta_data': getattr(node, 'meta_data', {}),
            'taken_action_list': getattr(node, 'taken_action_list', []),
            'taken_action_action_raw_res_list': getattr(node, 'taken_action_action_raw_res_list', []),
            'parent_node': getattr(node.parent_node, 'node_id', None) if hasattr(node, 'parent_node') and node.parent_node else None,
            'mc_value': getattr(node, 'mc_value', 0),
            'children': [getattr(child, 'node_id', 'unknown') for child in getattr(node, 'children', [])],
            'visit_times': getattr(node, 'visit_times', 0),
            'q_value': getattr(node, 'q_value', 0)
        }
        
        with open(outpath, "a", encoding='utf-8') as ofile:
            ofile.write(json.dumps(node_dict, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logging.error(f"Error writing node to {outpath}: {e}")
        return False


def get_detail_prompt(action2_res: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Generate a detail-focused video prompt by parsing the model's generated description.

    Args:
        action2_res (str): The model's response to ACTION2 prompt.

    Returns:
        Optional[Tuple[str, str, str, str]]: Extracted detail, category, describe aspects, and reformulated prompt.
        Returns None if extraction fails.
    """
    client = get_openai_client()
    if not client:
        logging.error("OpenAI client not available")
        return None
        
    prompt = """
For Vision Large Language Models, generating video content description is a very important task, which is called "video caption" task. At the same time, it is hoped that the model can pay more attention to the details of the video when describing the video.
To do that, I'll start by giving the model the video to determine the details that need attention. 
Then I will give you the "MODEL ANSWER" (what the model think itself should focus on) and need you to extract the following information from this "MODEL ANSWER":
1. What video detail does the model need to focus on?
2. What category does this detail fall into?
3. What describe aspects can model focus on for this detail?

Here are some categories of details and describe aspects to focus on under the category, if the details extracted from the answer can not be classified into the following categories, you can think of your own:
1. People or Animals:  Describe their expressions, facial features, postures, clothing, age,  and quantity. Please include any notable actions or interactions they are involved in.
2. Plants: Describe the quantity, types, size, color, and any notable features such as flowers or fruits.
3. Food: Describe the quantity, types, colors, and presentation (e.g., plated, packaged).
4. Buildings:  Describe the quantity, types, architectural style, appearance, shapes, and any distinctive features (e.g., windows,  doors, decorations).
5. Vehicles: Describe the types, appearance, quantity, color, and any notable features (e.g., brand, model, condition).
6. Other Objects: Describe the types, colors, appearance, size, and any distinctive features or uses.

Please answer in this format:
Detail: [the detail what the model think itself should focus on in MODEL ANSWER.]
Category: [What category does this detail fall into. Can either pick from the above categories or thick by yourself.]
Relevant Describe Aspects: [The possible describe aspects when focusing on the detail.]

Here is the MODEL ANSWER: {model_answer}
Output:
"""
    
    try:
        chat_outputs = client.chat.completions.create(
            model=API_CONFIG.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.format(model_answer=action2_res)},
            ],
            timeout=30  # Add timeout
        )

        text = chat_outputs.choices[0].message.content
        logging.info(f"Detail prompt response: {text}")

        # Use regex to extract structured response
        pattern1 = r'Detail\s*:\s*(.*?)(?=\n|Category:)'
        pattern2 = r'Category\s*:\s*(.*?)(?=\n|Relevant)'
        pattern3 = r'Relevant\s+Describe\s+Aspects\s*:\s*(.*?)(?=\n|$)'

        match1 = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)
        match2 = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
        match3 = re.search(pattern3, text, re.DOTALL | re.IGNORECASE)

        if match1 and match2 and match3:
            detail = match1.group(1).strip()
            category = match2.group(1).strip()
            relevant_describe_aspects = match3.group(1).strip()
            
            instruction = f"I want you to focus on the {detail.lower()} in the video and describe it in the following detail: {relevant_describe_aspects}"
            return detail, category, relevant_describe_aspects, instruction
        else:
            logging.warning("Failed to parse detail prompt response")
            return None
    except Exception as e:
        logging.error(f"Error in get_detail_prompt: {e}")
        return None


def cal_node_similarity(node) -> float:
    """
    Calculate the average cosine similarity between the current node's latest action output
    and the outputs of all ancestor nodes that executed the same action.

    Args:
        node: Current node in the search tree.

    Returns:
        float: The average similarity score. Returns 0.0 if no matching ancestor found or model unavailable.
    """
    if not model:
        logging.warning("Sentence transformer model not available, returning 0.0")
        return 0.0
        
    # Check if the node has any action taken
    if not getattr(node, 'taken_action_list', []):
        return 0.0

    try:
        # Get the last action and its corresponding response
        taken_actions = getattr(node, 'taken_action_list', [])
        taken_responses = getattr(node, 'taken_action_action_raw_res_list', [])
        
        if not taken_actions or not taken_responses:
            return 0.0
            
        current_action = taken_actions[-1]
        current_response = taken_responses[-1]

        # Collect ancestor responses with the same action
        matching_responses = []
        ancestor = getattr(node, 'parent_node', None)

        while ancestor:
            ancestor_actions = getattr(ancestor, 'taken_action_list', [])
            ancestor_responses = getattr(ancestor, 'taken_action_action_raw_res_list', [])
            
            for i, action in enumerate(ancestor_actions):
                if action == current_action and i < len(ancestor_responses):
                    matching_responses.append(ancestor_responses[i])
            ancestor = getattr(ancestor, 'parent_node', None)

        # If no matching action responses in the path, return 0
        if not matching_responses:
            return 0.0

        # Embed the current response and all matching ancestor responses
        sentences = [current_response] + matching_responses
        embeddings = model.encode(sentences)

        current_embedding = embeddings[0]
        other_embeddings = embeddings[1:]

        # Compute cosine similarities and return the average
        similarities = cosine_similarity([current_embedding], other_embeddings)[0]
        return float(np.mean(similarities))
    except Exception as e:
        logging.error(f"Error calculating node similarity: {e}")
        return 0.0


def gpt4o_extract_keypoints(caption: str) -> List[str]:
    """
    Extract key points from the caption using GPT-4o.
    
    Note: This is a placeholder implementation. In production, implement with actual GPT-4o API.

    Args:
        caption (str): The caption text to be structured.

    Returns:
        List[str]: A list of one-sentence key points describing the caption content.
    """
    # Placeholder implementation - split by sentences as basic keypoints
    import re
    sentences = re.split(r'[.!?]+', caption)
    keypoints = [sent.strip() for sent in sentences if sent.strip()]
    return keypoints[:5]  # Return up to 5 keypoints


def gpt4o_generate_yesno_questions(keypoint: str) -> List[str]:
    """
    Convert a key point into a list of yes/no questions for factual verification.
    
    Note: This is a placeholder implementation. In production, implement with actual GPT-4o API.

    Args:
        keypoint (str): A single key point extracted from the caption.

    Returns:
        List[str]: A list of yes/no questions based on the key point.
    """
    # Basic question generation based on keypoint
    questions = [
        f"Is this statement about the video accurate: {keypoint}?",
        f"Can you confirm that the following is shown in the video: {keypoint}?",
    ]
    return questions


def qwen_answer_question(video_path: str, question: str) -> str:
    """
    Use Qwen2.5-VL-7B to answer a visual question based on the video (frames).

    Args:
        video_path (str): Path or URL to the input video frame(s).
        question (str): A yes/no question about the visual content.

    Returns:
        str: Model's answer, typically 'yes' or 'no'.
    """
    client = get_openai_client()
    if not client:
        logging.error("OpenAI client not available")
        return "no"
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": video_path},
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages,
            timeout=30
        )
        return response.choices[0].message.content.lower().strip()
    except Exception as e:
        logging.error(f"Error in qwen_answer_question: {e}")
        return "no"


def gpt4o_answer_question(video_path: str, question: str) -> str:
    """
    Use GPT-4o to answer a visual yes/no question (placeholder implementation).

    Args:
        video_path (str): Path or URL to video frame(s).
        question (str): Yes/no question to be answered.

    Returns:
        str: 'yes' or 'no' depending on model's visual understanding.
    """
    # Placeholder implementation - in production, use actual GPT-4o API
    logging.warning("gpt4o_answer_question is using placeholder implementation")
    return "yes"  # Default to yes for testing


def verify_keypoint_with_images(keypoint: str, video_path: str) -> bool:
    """
    Verify whether a key point is supported by the video using both models.

    Args:
        keypoint (str): The key point to be verified.
        video_path (str): Video image path or frame path(s).

    Returns:
        bool: True if both models agree on the key point being valid, False otherwise.
    """
    try:
        yesno_questions = gpt4o_generate_yesno_questions(keypoint)
        
        for q in yesno_questions:
            answer_qwen = qwen_answer_question(video_path, q)
            answer_gpt4o = gpt4o_answer_question(video_path, q)

            if not (answer_qwen.startswith("yes") and answer_gpt4o.startswith("yes")):
                return False
        return True
    except Exception as e:
        logging.error(f"Error in verify_keypoint_with_images: {e}")
        return False


def evaluate_caption_node(node) -> float:
    """
    Evaluate the factual accuracy of the caption in a given node using multi-frame visual QA.

    This function will:
    - Extract the caption from the node.
    - Use GPT-4o to extract structured key points.
    - For each key point, ask multiple yes/no questions to Qwen and GPT-4o.
    - Aggregate how many key points are verified by both models.
    - Store evaluation results and score back into the node.

    Args:
        node: The search tree node containing the caption and metadata.

    Returns:
        float: A score between 0.0 and 1.0 indicating the proportion of key points that are verified.
    """
    try:
        taken_responses = getattr(node, 'taken_action_action_raw_res_list', [])
        if not taken_responses:
            return 0.0
            
        caption = taken_responses[-1]
        meta_data = getattr(node, 'meta_data', {})
        
        # Get image list from metadata, with fallback
        image_list = meta_data.get("image_list", meta_data.get("video_path", ""))

        keypoints = gpt4o_extract_keypoints(caption)
        if not keypoints:
            return 0.0
            
        verified_results = []
        verified_count = 0
        
        for kp in keypoints:
            passed = verify_keypoint_with_images(kp, image_list)
            verified_results.append({
                "text": kp,
                "verified": passed
            })
            if passed:
                verified_count += 1

        score = round(verified_count / len(keypoints), 4) if keypoints else 0.0

        # Store evaluation results inside the node
        setattr(node, 'keypoints', verified_results)
        setattr(node, 'mc_value', score)  # Store for MCTS value propagation
        
        return score
    except Exception as e:
        logging.error(f"Error in evaluate_caption_node: {e}")
        return 0.0