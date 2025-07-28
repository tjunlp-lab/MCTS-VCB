"""
Monte Carlo Tree Search (MCTS) implementation for video captioning.
Provides Node class for tree structure and Caption_MCTS class for search algorithm.
"""

import random
import math
import copy
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from util import (
    get_next_action_list, 
    get_detail_prompt, 
    cal_node_similarity,
    evaluate_caption_node
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global node ID tracker
node_id_suffix = 0


class Node:
    """
    Represents a node in the MCTS search tree for video captioning.
    """
    
    def __init__(
        self, 
        meta_data: Dict[str, Any], 
        taken_action_list: List[Union[str, Tuple]], 
        taken_action_action_raw_res_list: List[str], 
        parent_node: Optional['Node'] = None, 
        is_root_node: bool = False
    ):
        """
        Initialize a new node in the MCTS tree.

        Args:
            meta_data (Dict[str, Any]): Information about the video and problem instance.
            taken_action_list (List[Union[str, Tuple]]): List of actions taken from the root to this node.
            taken_action_action_raw_res_list (List[str]): Raw responses generated for each taken action.
            parent_node (Optional[Node]): The parent node in the tree. Defaults to None.
            is_root_node (bool): Whether this node is the root of the tree.
        """
        global node_id_suffix
        
        if is_root_node:
            node_id_suffix = 0
        
        # Generate unique node ID
        video_name = meta_data.get('video_name', 'unknown')
        index = meta_data.get('index', 0)
        self.node_id = f"{video_name}_{index}_{node_id_suffix}"
        node_id_suffix += 1

        # Core node data
        self.meta_data = meta_data
        self.taken_action_list = taken_action_list.copy() if taken_action_list else []
        self.taken_action_action_raw_res_list = taken_action_action_raw_res_list.copy() if taken_action_action_raw_res_list else []
        
        # Tree structure
        self.parent_node = parent_node
        self.children: List['Node'] = []
        
        # MCTS statistics
        self.visit_times = 0
        self.q_value = 0.0
        self.mc_value: Optional[float] = None
        
        # Additional metrics
        self.node_similarity = 0.0
        
        # Evaluation results (for caption quality assessment)
        self.keypoints: Optional[List[Dict[str, Any]]] = None
        
        logger.debug(f"Created node {self.node_id} with parent {parent_node.node_id if parent_node else 'None'}")

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been expanded from this node."""
        # This would need to be implemented based on the specific action space
        # For now, assume a node is fully expanded if it has children
        return len(self.children) > 0

    def add_child(self, child: 'Node') -> None:
        """Add a child node to this node."""
        if child not in self.children:
            self.children.append(child)
            child.parent_node = self

    def get_path_to_root(self) -> List['Node']:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent_node
        return path[::-1]  # Reverse to get root-to-current order

    def get_depth(self) -> int:
        """Get the depth of this node (distance from root)."""
        depth = 0
        current = self.parent_node
        while current is not None:
            depth += 1
            current = current.parent_node
        return depth

    def __str__(self) -> str:
        return f"Node({self.node_id}, actions={len(self.taken_action_list)}, children={len(self.children)})"

    def __repr__(self) -> str:
        return self.__str__()


class Caption_MCTS:
    """
    Monte Carlo Tree Search implementation for video captioning tasks.
    Uses a multimodal language model to generate and evaluate captions.
    """
    
    def __init__(self, exploration_constant: float = 0.125, random_seed: Optional[int] = None):
        """
        Initialize the Caption MCTS system.
        
        Args:
            exploration_constant (float): UCB exploration constant (c_puct).
            random_seed (Optional[int]): Random seed for reproducible results.
        """
        self.c_puct = exploration_constant
        
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"Set random seed to {random_seed}")
        
        # Initialize the generator
        try:
            from generator.qwen2vl_7b import Qwen2VL7B
            self.generator = Qwen2VL7B()
            logger.info("Successfully initialized Qwen2VL7B generator")
        except ImportError as e:
            logger.error(f"Failed to import generator: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise

    def cal_node_mc_value(self, node: Node) -> float:
        """
        Calculate the Monte Carlo value for a node.

        Args:
            node (Node): Node to evaluate.

        Returns:
            float: MC value between 0 and 1.
        """
        
        return evaluate_caption_node(node)

    def cal_Q(self, node: Node, alpha: float = 0.5, beta: float = 0.9) -> float:
        """
        Compute the exploitation value (Q) of a node.

        Args:
            node (Node): Node to evaluate.
            alpha (float): Weighting factor for mc_value.
            beta (float): Weighting factor for similarity penalty.

        Returns:
            float: Q value of the node.
        """
        if node.mc_value is None:
            node.mc_value = self.cal_node_mc_value(node)
        
        # Calculate similarity penalty to encourage diversity
        node.node_similarity = cal_node_similarity(node)
        
        # Use exponential weighting to emphasize high-quality, diverse responses
        v1 = alpha ** (1 - node.mc_value)  # Higher mc_value gives higher v1
        v2 = beta ** node.node_similarity   # Higher similarity gives lower v2
        
        q_value = v1 * v2
        
        logger.debug(f"Node {node.node_id}: mc_value={node.mc_value:.3f}, "
                    f"similarity={node.node_similarity:.3f}, Q={q_value:.3f}")
        
        return q_value

    def cal_U(self, node: Node, nodes: List[Node], c_puct: Optional[float] = None) -> float:
        """
        Compute the exploration bonus (U) for a node using UCB1.

        Args:
            node (Node): Node to evaluate.
            nodes (List[Node]): List of all nodes (used for parent visit count).
            c_puct (Optional[float]): Exploration constant. Uses instance default if None.

        Returns:
            float: U value (exploration bonus).
        """
        if c_puct is None:
            c_puct = self.c_puct
        
        parent_node = node.parent_node
        if parent_node is None:
            return 0.0
        
        # UCB1 exploration term
        numerator = math.sqrt(parent_node.visit_times)
        denominator = 1 + node.visit_times
        
        u_value = c_puct * (numerator / denominator)
        
        logger.debug(f"Node {node.node_id}: parent_visits={parent_node.visit_times}, "
                    f"node_visits={node.visit_times}, U={u_value:.3f}")
        
        return u_value

    def select(self, nodes: List[Node]) -> Tuple[Optional[Node], float]:
        """
        Select the best node to expand using Upper Confidence Bound (UCB).

        Args:
            nodes (List[Node]): All nodes in the current tree.

        Returns:
            Tuple[Optional[Node], float]: The selected node and its UCB value.
        """
        if not nodes:
            logger.warning("No nodes available for selection")
            return None, 0.0
        
        # Only consider leaf nodes (nodes without children)
        leaf_nodes = [node for node in nodes if node.is_leaf()]
        
        if not leaf_nodes:
            logger.warning("No leaf nodes available for selection")
            return None, 0.0
        
        chosen_node = None
        best_ucb = -float('inf')
        
        for node in leaf_nodes:
            q_value = self.cal_Q(node)
            u_value = self.cal_U(node, nodes)
            ucb = q_value + u_value
            
            logger.debug(f"Node {node.node_id}: Q={q_value:.3f}, U={u_value:.3f}, UCB={ucb:.3f}")
            
            if ucb > best_ucb:
                chosen_node = node
                best_ucb = ucb
                
            # Update node's Q value for future reference
            node.q_value = q_value
        
        if chosen_node:
            logger.info(f"Selected node {chosen_node.node_id} with UCB={best_ucb:.3f}")
        else:
            logger.warning("No node selected during selection phase")
        
        return chosen_node, best_ucb

    def expand(self, node: Node) -> List[Tuple[str, str]]:
        """
        Generate possible next actions from a node.

        Args:
            node (Node): The node to expand.

        Returns:
            List[Tuple[str, str]]: List of (action_type, action_prompt) pairs.
        """
        try:
            # Get action types based on current node state
            action_types = [action[0] if isinstance(action, tuple) else action 
                          for action in node.taken_action_list]
            next_action_type_list = get_next_action_list(action_types)
            
            logger.debug(f"Expanding node {node.node_id} with actions: {next_action_type_list}")
            
            if not next_action_type_list:
                logger.warning(f"No next actions available for node {node.node_id}")
                return []
            
            # Generate prompts for the actions
            action_prompt_list = self.generator.generate_action_prompt(node, next_action_type_list)
            
            if len(action_prompt_list) != len(next_action_type_list):
                logger.error(f"Mismatch between action types ({len(next_action_type_list)}) "
                           f"and prompts ({len(action_prompt_list)})")
                return []
            
            actions = list(zip(next_action_type_list, action_prompt_list))
            logger.info(f"Generated {len(actions)} actions for node {node.node_id}")
            
            return actions
            
        except Exception as e:
            logger.error(f"Error expanding node {node.node_id}: {e}")
            return []

    def simulate(self, node: Node, actions: List[Tuple[str, str]], all_nodes: List[Node]) -> None:
        """
        Simulate the result of taking action(s) from the given node.

        Args:
            node (Node): The node to simulate from.
            actions (List[Tuple[str, str]]): List of (action_type, action_prompt) pairs.
            all_nodes (List[Node]): List of all nodes in the tree.
        """
        if not actions:
            logger.warning(f"No actions to simulate for node {node.node_id}")
            return
        
        try:
            action_count = len(actions)
            logger.info(f"Simulating {action_count} actions from node {node.node_id}")
            
            # Single action processing (ACTION2 with detail generation)
            self._simulate_single_action(node, actions[0], all_nodes)
            
            # Backpropagate the results
            self.backpropagate(node)
            
        except Exception as e:
            logger.error(f"Error simulating actions for node {node.node_id}: {e}")

    def _simulate_single_action(self, node: Node, action: Tuple[str, str], all_nodes: List[Node]) -> None:
        """
        Simulate a single action (typically ACTION2 with detail follow-up).
        
        Args:
            node (Node): Parent node
            action (Tuple[str, str]): Single action to simulate
            all_nodes (List[Node]): All nodes list to update
        """
        try:
            action_type, action_prompt = action
            
            # Get initial response
            res = self.generator.get_completion(node, action_prompt)
            
            # Generate detail prompt based on the response
            detail_prompt_result = get_detail_prompt(res)
            if detail_prompt_result is None:
                logger.warning(f"Failed to generate detail prompt for node {node.node_id}")
                return
            
            detail, category, relevant_aspects, instruction = detail_prompt_result
            logger.info(f"Generated detail prompt for {detail} in category {category}")
            
            # Get detailed response
            detail_res = self.generator.get_completion(node, instruction)
            
            # Create new node with both responses
            taken_action_list = (
                node.taken_action_list + 
                [(action_type, action_prompt)] + 
                [(action_type, (detail, category, relevant_aspects, instruction))]
            )
            taken_res_list = node.taken_action_action_raw_res_list + [res, detail_res]

            new_node = Node(
                meta_data=node.meta_data,
                taken_action_list=taken_action_list,
                taken_action_action_raw_res_list=taken_res_list,
                parent_node=node
            )
            
            node.add_child(new_node)
            all_nodes.append(new_node)
            
            # Calculate initial values
            new_node.mc_value = self.cal_node_mc_value(new_node)
            new_node.q_value = self.cal_Q(new_node)
            
            logger.debug(f"Created detailed node {new_node.node_id} with MC value {new_node.mc_value:.3f}")
            
        except Exception as e:
            logger.error(f"Error in single action simulation: {e}")

    def backpropagate(self, leaf_node: Node) -> None:
        """
        Backpropagate the evaluation results up the tree.

        Args:
            leaf_node (Node): The leaf node to start backpropagation from.
        """
        try:
            current_node = leaf_node
            
            while current_node is not None:
                current_node.visit_times += 1
                
                if current_node.children:
                    # Update Q value based on children's Q values
                    child_q_values = [child.q_value for child in current_node.children if child.q_value > 0]
                    if child_q_values:
                        current_node.q_value = sum(child_q_values) / len(child_q_values)
                else:
                    # Leaf node - use its own calculated Q value
                    if current_node.q_value == 0:
                        current_node.q_value = self.cal_Q(current_node)
                
                logger.debug(f"Backpropagated to node {current_node.node_id}: "
                           f"visits={current_node.visit_times}, Q={current_node.q_value:.3f}")
                
                current_node = current_node.parent_node
                
        except Exception as e:
            logger.error(f"Error in backpropagation from node {leaf_node.node_id}: {e}")

    def process_annotation(self, meta_data: Dict[str, Any], all_nodes: List[Node], max_rollout_times: int) -> None:
        """
        Perform MCTS rollout to annotate a single video.

        Args:
            meta_data (Dict[str, Any]): Metadata of the video.
            all_nodes (List[Node]): The initial list of nodes.
            max_rollout_times (int): Maximum number of rollout iterations.
        """
        video_name = meta_data.get('video_name', 'unknown')
        logger.info(f"Starting MCTS annotation for {video_name} with max_rollout_times={max_rollout_times}")
        
        successful_rollouts = 0
        failed_rollouts = 0
        
        for rollout_i in range(max_rollout_times):
            try:
                logger.debug(f"Rollout {rollout_i + 1}/{max_rollout_times} for {video_name}")
                
                # Selection phase
                node, uct = self.select(all_nodes)
                if node is None:
                    logger.info(f"No more nodes to expand for {video_name}. Stopping at rollout {rollout_i + 1}")
                    break
                
                # Expansion phase
                actions = self.expand(node)
                if not actions:
                    logger.warning(f"No actions generated for node {node.node_id} at rollout {rollout_i + 1}")
                    failed_rollouts += 1
                    continue
                
                # Simulation phase
                initial_node_count = len(all_nodes)
                self.simulate(node, actions, all_nodes)
                new_nodes_created = len(all_nodes) - initial_node_count
                
                if new_nodes_created > 0:
                    successful_rollouts += 1
                    logger.debug(f"Rollout {rollout_i + 1} successful: created {new_nodes_created} new nodes")
                else:
                    failed_rollouts += 1
                    logger.warning(f"Rollout {rollout_i + 1} created no new nodes")
                
            except KeyboardInterrupt:
                logger.info(f"MCTS annotation interrupted by user at rollout {rollout_i + 1}")
                break
            except Exception as e:
                failed_rollouts += 1
                logger.error(f"Error in rollout {rollout_i + 1} for {video_name}: {e}")
                continue
        
        total_nodes = len(all_nodes)
        logger.info(f"MCTS annotation completed for {video_name}. "
                   f"Total nodes: {total_nodes}, "
                   f"Successful rollouts: {successful_rollouts}, "
                   f"Failed rollouts: {failed_rollouts}")

        """
        Get statistics about the current search tree.
        
        Args:
            all_nodes (List[Node]): All nodes in the tree
            
        Returns:
            Dict[str, Any]: Tree statistics
        """
        if not all_nodes:
            return {}
        
        depths = [node.get_depth() for node in all_nodes]
        q_values = [node.q_value for node in all_nodes if node.q_value > 0]
        visit_counts = [node.visit_times for node in all_nodes]
        
        return {
            'total_nodes': len(all_nodes),
            'max_depth': max(depths) if depths else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'avg_q_value': sum(q_values) / len(q_values) if q_values else 0,
            'total_visits': sum(visit_counts),
            'leaf_nodes': len([n for n in all_nodes if n.is_leaf()]),
        }