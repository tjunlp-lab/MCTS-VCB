#!/usr/bin/env python3
"""
Main entry point for Caption MCTS annotation system.
Supports distributed processing using MPI for scalable video captioning.
"""

import json
import sys
import os
import logging
import argparse
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    logging.warning("MPI not available, running in single-process mode")

from mcts import Caption_MCTS, Node
from util import write_node

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Process %(process)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('caption_mcts.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def setup_mpi() -> tuple:
    """
    Initialize MPI environment and return communication parameters.
    
    Returns:
        tuple: (comm_world, rank, local_rank, size) or (None, 0, 0, 1) if MPI unavailable
    """
    if not MPI_AVAILABLE:
        return None, 0, 0, 1
    
    try:
        comm_world = MPI.COMM_WORLD
        rank = comm_world.Get_rank()
        size = comm_world.Get_size()
        local_rank = rank % 8  # Assume 8 GPUs per node
        
        logger.info(f"MPI initialized - Rank: {rank}, Size: {size}, Local Rank: {local_rank}")
        return comm_world, rank, local_rank, size
    except Exception as e:
        logger.error(f"Failed to initialize MPI: {e}")
        return None, 0, 0, 1


def validate_input_file(input_path: str) -> bool:
    """
    Validate that the input file exists and is readable.
    
    Args:
        input_path (str): Path to input JSONL file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not Path(input_path).exists():
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    if not Path(input_path).is_file():
        logger.error(f"Input path is not a file: {input_path}")
        return False
    
    # Try to read first line to validate format
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                json.loads(first_line)  # Validate JSON format
        logger.info(f"Input file validation passed: {input_path}")
        return True
    except Exception as e:
        logger.error(f"Invalid input file format: {e}")
        return False


def get_handled_filename(output_path: str) -> List[str]:
    """
    Retrieve the list of video filenames that have already been processed.

    Args:
        output_path (str): Path to the output JSONL file.

    Returns:
        List[str]: List of video names already present in output.
    """
    if not Path(output_path).exists():
        logger.info(f"Output file does not exist yet: {output_path}")
        return []

    handled_filename_set = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as in_file:
            for line_num, line in enumerate(in_file, 1):
                try:
                    data = json.loads(line.strip())
                    if 'meta_data' in data and 'video_name' in data['meta_data']:
                        handled_filename_set.add(data["meta_data"]["video_name"])
                    else:
                        logger.warning(f"Missing video_name in line {line_num} of {output_path}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in line {line_num} of {output_path}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading output file {output_path}: {e}")
        return []

    handled_list = list(handled_filename_set)
    logger.info(f"Found {len(handled_list)} already processed files")
    return handled_list


def read_problems(args: argparse.Namespace, process_id: int) -> List[Dict[str, Any]]:
    """
    Read and filter problems assigned to the current process.

    Args:
        args (argparse.Namespace): Command-line arguments containing input/output paths and process count.
        process_id (int): ID of the current process (used for distributed splitting).

    Returns:
        List[Dict[str, Any]]: List of unprocessed problem data assigned to this process.
    """
    input_path = args.input_path
    output_path = args.output_path
    process_num = args.process_num

    if not validate_input_file(input_path):
        logger.error("Input file validation failed")
        return []

    handled_filename_list = get_handled_filename(output_path)
    handled_filename_set = set(handled_filename_list)

    all_data_to_process = []
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'video_name' not in data:
                        logger.warning(f"Missing video_name in line {line_num}")
                        continue
                    
                    if data["video_name"] not in handled_filename_set:
                        # Add index if missing
                        if 'index' not in data:
                            data['index'] = line_num - 1
                        all_data_to_process.append(data)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return []

    # Distribute data across processes
    data_to_process_in_process = all_data_to_process[process_id::process_num]
    
    logger.info(f"Process {process_id}: Total problems: {len(all_data_to_process)}, "
                f"Assigned: {len(data_to_process_in_process)}, "
                f"Already handled: {len(handled_filename_list)}")
    
    return data_to_process_in_process


def setup_output_directory(output_path: str, process_id: int) -> str:
    """
    Create output directory structure and return the temporary output path.
    
    Args:
        output_path (str): Base output path
        process_id (int): Process ID
        
    Returns:
        str: Path to temporary output file for this process
    """
    # Create temporary directory for individual processes
    temp_dir = Path(output_path).parent / f"{Path(output_path).stem}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_output_path = temp_dir / str(process_id)
    
    # Ensure parent directories exist
    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory setup complete: {temp_output_path}")
    return str(temp_output_path)


def process_single_problem(caption_mcts: Caption_MCTS, problem: Dict[str, Any], 
                          max_rollout_times: int) -> List[Node]:
    """
    Process a single problem instance using MCTS.
    
    Args:
        caption_mcts (Caption_MCTS): MCTS instance
        problem (Dict[str, Any]): Problem data
        max_rollout_times (int): Maximum rollout iterations
        
    Returns:
        List[Node]: List of all nodes created during processing
    """
    all_nodes = []
    
    try:
        # Create root node
        root_node = Node(
            meta_data=problem,
            taken_action_list=[],
            taken_action_action_raw_res_list=[],
            is_root_node=True
        )
        all_nodes.append(root_node)

        # Initialize MCTS values
        root_node.mc_value = 1.0
        root_node.q_value = 1.0

        logger.debug(f"Created root node {root_node.node_id} for video {problem.get('video_name', 'unknown')}")

        # Expand and simulate initial state
        try:
            action = caption_mcts.expand(root_node)
            caption_mcts.simulate(root_node, action, all_nodes)
        except Exception as e:
            logger.error(f"Error in initial expand/simulate for {problem.get('video_name', 'unknown')}: {e}")
            return all_nodes

        # Run MCTS annotation process
        try:
            caption_mcts.process_annotation(
                meta_data=problem,
                all_nodes=all_nodes,
                max_rollout_times=max_rollout_times
            )
        except Exception as e:
            logger.error(f"Error in process_annotation for {problem.get('video_name', 'unknown')}: {e}")

        logger.info(f"Successfully processed {problem.get('video_name', 'unknown')} "
                   f"with {len(all_nodes)} nodes")
        
    except Exception as e:
        logger.error(f"Error processing problem {problem.get('video_name', 'unknown')}: {e}")
    
    return all_nodes


def main(args: argparse.Namespace, process_id: int) -> None:
    """
    Main function to run Caption MCTS annotation for a given process.

    Args:
        args (argparse.Namespace): Command-line arguments.
        process_id (int): ID of the current process.
    """
    logger.info(f"Starting main process {process_id}")
    
    # Initialize MCTS system
    try:
        caption_mcts = Caption_MCTS()
        logger.info("Caption MCTS initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Caption MCTS: {e}")
        return

    # Read problems for this process
    problems = read_problems(args, process_id)
    if not problems:
        logger.warning(f"No problems to process for process {process_id}")
        return

    # Setup output directory
    temp_output_path = setup_output_directory(args.output_path, process_id)

    # Process each problem
    total_processed = 0
    total_errors = 0
    
    for problem_index, problem in enumerate(tqdm(problems, desc=f"Process {process_id}")):
        try:
            start_time = time.time()
            
            all_nodes = process_single_problem(
                caption_mcts, 
                problem, 
                max_rollout_times=args.max_rollout_times
            )
            
            # Write results
            nodes_written = 0
            for node in all_nodes:
                if write_node(temp_output_path, node):
                    nodes_written += 1
                else:
                    logger.warning(f"Failed to write node {getattr(node, 'node_id', 'unknown')}")
            
            processing_time = time.time() - start_time
            total_processed += 1
            
            logger.info(f"Process {process_id}: Completed problem {problem_index + 1}/{len(problems)} "
                       f"({problem.get('video_name', 'unknown')}) in {processing_time:.2f}s. "
                       f"Nodes written: {nodes_written}/{len(all_nodes)}")
            
        except KeyboardInterrupt:
            logger.info(f"Process {process_id} interrupted by user")
            break
        except Exception as e:
            total_errors += 1
            logger.error(f"Process {process_id}: Error processing problem {problem_index + 1}: {e}")
            continue

    logger.info(f"Process {process_id} completed. "
                f"Processed: {total_processed}/{len(problems)}, "
                f"Errors: {total_errors}")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run Caption MCTS annotation with MPI parallelism.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to the input file (JSONL)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to the output file (JSONL)')
    parser.add_argument('--process_num', type=int, required=True,
                       help='Total number of processes to run')
    parser.add_argument('--gpu_nums_one_process', type=int, required=True,
                       help='Number of GPUs used per process')
    parser.add_argument('--max_rollout_times', type=int, default=25,
                       help='Maximum number of MCTS rollout iterations')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate arguments
    if args.process_num <= 0:
        logger.error("process_num must be positive")
        sys.exit(1)
    
    if args.gpu_nums_one_process <= 0:
        logger.error("gpu_nums_one_process must be positive")
        sys.exit(1)

    if args.max_rollout_times <= 0:
        logger.error("max_rollout_times must be positive")
        sys.exit(1)

    # Initialize MPI
    comm_world, rank, local_rank, size = setup_mpi()
    
    # Determine which processes should run
    if MPI_AVAILABLE and size > 1:
        # MPI mode: each rank handles multiple process IDs based on GPU allocation
        for process_id in range(args.process_num):
            if rank == process_id * args.gpu_nums_one_process:
                # Set CUDA devices for this process
                cuda_devices = ",".join(
                    map(str, range(local_rank, local_rank + args.gpu_nums_one_process))
                )
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
                logger.info(f"Process {process_id} using CUDA devices: {cuda_devices}")
                
                try:
                    main(args, process_id=process_id)
                except Exception as e:
                    logger.error(f"Process {process_id} failed: {e}")
                    sys.exit(1)
    else:
        # Single process mode
        logger.info("Running in single-process mode")
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Use first GPU
        try:
            main(args, process_id=0)
        except Exception as e:
            logger.error(f"Single process failed: {e}")
            sys.exit(1)

    logger.info("All processes completed successfully")