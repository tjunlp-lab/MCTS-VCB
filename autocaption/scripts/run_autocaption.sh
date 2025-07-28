#!/bin/bash

# AutoCaption Run Script
# Usage: ./scripts/run_autocaption.sh [config_file] [input_file] [output_file]

set -e  # Exit on any error

# Default values
CONFIG_FILE="${1:-config/config.yaml}"
INPUT_FILE="${2:-data/input.jsonl}"
OUTPUT_FILE="${3:-results/output.jsonl}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PROCESS_NUM="${PROCESS_NUM:-1}"
GPU_NUMS="${GPU_NUMS:-1}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-25}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if required files exist
check_requirements() {
    log_step "Checking requirements..."
    
    if [ ! -f "$INPUT_FILE" ]; then
        log_error "Input file not found: $INPUT_FILE"
        exit 1
    fi
    
    if [ ! -f "main.py" ]; then
        log_error "main.py not found. Are you in the correct directory?"
        exit 1
    fi
    
    # Check Python dependencies
    python -c "import torch, transformers, vllm" 2>/dev/null || {
        log_error "Required Python packages not installed. Run: pip install -r requirements.txt"
        exit 1
    }
    
    # Check GPU availability
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_warn "CUDA not available. Some features may not work properly."
    else
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        log_info "Found $GPU_COUNT GPU(s)"
    fi
    
    log_info "Requirements check passed"
}

# Setup directories
setup_directories() {
    log_step "Setting up directories..."
    
    mkdir -p logs temp results data
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    
    log_info "Directories created"
}

# Run the main caption generation
run_autocaption() {
    log_step "Starting AutoCaption process..."
    
    # Set environment variables
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export TOKENIZERS_PARALLELISM=false
    
    # Build command
    CMD="python main.py"
    CMD="$CMD --input_path $INPUT_FILE"
    CMD="$CMD --output_path $OUTPUT_FILE"
    CMD="$CMD --process_num $PROCESS_NUM"
    CMD="$CMD --gpu_nums_one_process $GPU_NUMS"
    CMD="$CMD --max_rollout_times $MAX_ROLLOUTS"
    CMD="$CMD --log_level $LOG_LEVEL"
    
    # Check if MPI is available and requested
    if command -v mpirun >/dev/null && [ "$PROCESS_NUM" -gt 1 ]; then
        log_info "Running with MPI: $PROCESS_NUM processes, $GPU_NUMS GPUs per process"
        CMD="mpirun -np $((PROCESS_NUM * GPU_NUMS)) $CMD"
    else
        log_info "Running single process mode"
    fi
    
    log_info "Executing: $CMD"
    
    # Run the command with timestamp
    START_TIME=$(date)
    $CMD
    END_TIME=$(date)
    
    log_info "Process started at: $START_TIME"
    log_info "Process ended at: $END_TIME"
}

# Print configuration
print_config() {
    log_step "Configuration:"
    echo "  Input file: $INPUT_FILE"
    echo "  Output file: $OUTPUT_FILE"
    echo "  Config file: $CONFIG_FILE"
    echo "  Process count: $PROCESS_NUM"
    echo "  GPUs per process: $GPU_NUMS"
    echo "  Max rollouts: $MAX_ROLLOUTS"
    echo "  Log level: $LOG_LEVEL"
    echo "  Start model server: ${START_MODEL_SERVER:-false}"
}

# Main execution
main() {
    log_info "AutoCaption Run Script Starting..."
    
    print_config
    check_requirements
    setup_directories
    run_autocaption
    
    # Check if output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_SIZE=$(wc -l < "$OUTPUT_FILE")
        log_info "Process completed successfully. Output file contains $OUTPUT_SIZE lines."
    else
        log_warn "Output file not found. Process may have failed."
        exit 1
    fi
}

# Show help
show_help() {
    echo "AutoCaption Run Script"
    echo ""
    echo "Usage: $0 [config_file] [input_file] [output_file]"
    echo ""
    echo "Environment variables:"
    echo "  LOG_LEVEL         Logging level (DEBUG|INFO|WARNING|ERROR)"
    echo "  PROCESS_NUM       Number of processes to run"
    echo "  GPU_NUMS          Number of GPUs per process"
    echo "  MAX_ROLLOUTS      Maximum MCTS rollouts"
    echo "  START_MODEL_SERVER  Set to 'true' to start vLLM server"
    echo "  MODEL_SERVER_PORT   Port for model server (default: 8000)"
    echo "  MODEL_NAME          Model name for server"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 config/custom.yaml data/videos.jsonl results/captions.jsonl"
    echo "  START_MODEL_SERVER=true PROCESS_NUM=2 $0"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac