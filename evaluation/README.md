# Video Caption Evaluation Pipeline

A comprehensive evaluation pipeline for video caption generation models using precision, recall, and F1 scores across different semantic categories.

## Overview

This pipeline evaluates video caption models by comparing generated captions with human-annotated key points. It calculates:

- **Precision**: How accurate are the model's generated caption breakdown points?
- **Recall**: How well does the model capture the human-annotated key points?
- **F1 Score**: Harmonic mean of precision and recall

The evaluation is performed across multiple semantic categories:
- Appearance Description
- Action Description  
- Environment Description
- Object Description
- Camera Movement and Composition

## File Structure

```
├── main.py                     # Main entry point
├── precision_prompt_builder.py # Creates prompts for precision evaluation
├── recall_prompt_builder.py    # Creates prompts for recall evaluation
├── gpt_api_caller.py           # Handles GPT-4o API calls
├── metric_calculator.py        # Calculates precision, recall, F1 scores
├── config_example.py           # Configuration examples and batch processing
└── README.md                   # This file
```

## Installation

1. Install required dependencies:
```bash
pip install openai pandas numpy
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Single Model Evaluation

```bash
python main.py \
    --test_data_path /path/to/test_data.json \
    --test_data_threshold 080 \
    --categoried_kp_path /path/to/categorized_keypoints.jsonl \
    --model_predictions_path /path/to/model_predictions.jsonl \
    --candidate_captions_path /path/to/candidate_captions.jsonl \
    --golden_kp_path /path/to/golden_keypoints.jsonl \
    --output_dir ./results \
    --model_name InternVL2_5-8B \
    --openai_api_key $OPENAI_API_KEY
```

### Using the Configuration Example

```bash
# Edit config_example.py with your paths
python config_example.py
```

## Pipeline Steps

### Step 1: Create Precision Prompts
- Compares LMM caption breakdown points with human-generated key points
- Creates evaluation prompts using the provided template
- Outputs prompts in JSONL format

### Step 2: Create Recall Prompts  
- Compares human key points with video captions
- Batches key points for efficient processing
- Outputs prompts in JSONL format

### Step 3: GPT-4o API Evaluation
- Sends prompts to GPT-4o for relationship classification
- Handles rate limiting and retries
- Returns judgments: entailment, contradiction, or neutral

### Step 4: Calculate Metrics
- Processes GPT-4o responses to extract judgments
- Calculates precision, recall, and F1 scores by category
- Handles parsing errors and edge cases

### Step 5: Save and Display Results
- Saves results in JSON format
- Displays formatted results table
- Generates LaTeX table format for papers

## Input File Formats

### Test Data (JSON Lines)
```json
{"index": "video_id", "category": "Action Description", "key_point": "A person is walking", "threshold": "080"}
```

### Model Predictions (JSON Lines)
```json
{"index": "video_id", "fined_atom_desc": [{"text": "A person walks down the street", "category": "Action Description"}]}
```

### Candidate Captions (JSON Lines)
```json
{"index": "video_id", "pred_caption": "A person is walking down a busy street in the city."}
```

### Golden Key Points (JSON Lines)
```json
{"index": "video_id", "passed_kp_list": [{"text": "A person is walking", "category": "Action Description", "threshold": "080"}]}
```

## Output Format

### Results JSON
```json
{
  "model_name": "InternVL2_5-8B",
  "precision": {
    "category_score": {
      "Appearance Description": 0.65128,
      "Action Description": 0.64975
    },
    "overall": 0.69552
  },
  "recall": {
    "category_score": {
      "Appearance Description": 0.36500,
      "Action Description": 0.42000  
    },
    "overall": 0.40100
  },
  "f1": {
    "category_score": {
      "Appearance Description": 0.46800,
      "Action Description": 0.51000
    },
    "overall": 0.50800
  }
}
```

### LaTeX Table Format
```latex
InternVL2\_5-8B & 65.1/36.5/46.8 & 65.0/42.0/51.0 & 76.0/48.7/59.4 & 72.1/30.3/42.7 & 54.7/31.7/40.1 & 69.6/40.1/50.8 & \\\\
```

## Batch Processing

For evaluating multiple models:

```python
from config_example import batch_evaluation_example
results = batch_evaluation_example()
```

## API Rate Limiting

The pipeline includes built-in rate limiting and retry mechanisms:
- Exponential backoff for failed requests
- Configurable delay between requests
- Maximum retry attempts

## Error Handling

- Validates input file existence
- Handles malformed JSON responses from GPT-4o
- Skips failed evaluations and continues processing
- Provides detailed error messages and warnings

## Customization

### Adding New Categories
```python
categories = [
    "Appearance Description",
    "Action Description", 
    "Environment Description",
    "Object Description",
    "Camera Movement and Composition",
    "Your New Category"  # Add here
]
```

### Modifying Evaluation Prompts
The prompts are defined in the respective builder classes and can be modified while keeping the examples intact.

### Changing Threshold Values
```python
--test_data_threshold 070  # Use 0.70 threshold instead of 0.80
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Set the OPENAI_API_KEY environment variable
2. **File Not Found**: Check all input file paths
3. **Parsing Errors**: Check JSON format of input files
4. **API Rate Limits**: Increase delay between requests

### Debug Mode
Enable verbose logging by modifying the print statements in the pipeline steps.

## Contributing

To extend the pipeline:
1. Add new evaluation metrics in `metric_calculator.py`
2. Modify prompt templates in the builder classes
3. Add new output formats in the display functions

## License

This project is part of the video caption evaluation research framework.