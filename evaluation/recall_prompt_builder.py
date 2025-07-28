# -*- coding: utf-8 -*-
"""
Recall Prompt Builder
Creates prompts for evaluating recall by comparing key points with video captions.
"""

import json
from typing import Dict, List, Any


class RecallPromptBuilder:
    """Builder class for creating recall evaluation prompts."""
    
    # Keep the original system prompt unchanged
    SYSTEM_PROMPT = """
Please objectively classify the relationship between each provided key point and the given video caption.
Analyze each key point individually to determine its relationship with the video caption.

For each key point, classify the relationship into one of the following categories:

1. "entailment" means that the key point is accurately reflected within the video caption.
2. "contradiction" means that some detail in the key point contradicts the information mentioned in the video caption.
3. "neutral" means that the relationship is neither "entailment" nor "contradiction".
For each key point, provide a brief analysis explaining the reasoning behind your judgment.

Please present the result in a JSON dict format: {{'1': {{'judgement': judgement_1, 'analysis': analysis_1}}, ...,  'n': {{'judgement': judgement_n, 'analysis': analysis_n}}}}.

For clarity, consider these examples:

## Example 1:
### Video Caption
The video shows an aerial view of a city with a large, white church in the center. The church has two tall towers and a large dome. Surrounding the church are many buildings, some of which are tall and some are shorter. The buildings are mostly white, with some having green roofs. There are also some trees in the area. The sky is clear and blue. The video shows the church and the surrounding buildings from different angles.

### Key Points
1. An aerial shot captures a cityscape view.
2. The clear sky above indicates favorable weather at the time of filming.
3. The urban landscape displays a range of buildings with various sizes and elevations.
4. The video consistently features smooth transitions from broad views to detailed close-ups.
5. The camera moves smoothly and with intent.
6. The detailed designs and structural components of the cathedral are now clearer.
7. The church displays complex architectural details.
8. The parks and green areas offer a stark contrast to the urban density.
9. The intricate stonework and pointed arches at the entrance of the cathedral are highlighted.
10. The city's grid of streets and alleys is evident.
11. The rooftops display diverse colors and designs.
12. The cathedral's exterior showcases elaborate features like high spires and pointed arch windows.
13. The camera consistently remains stable and focused on the church throughout the video.
14. The scene vividly portrays a lively urban area.
15. The urban landscape is heavily populated.
16. The trees contribute greenery to the cityscape.
17. The illumination is intense.
18. This fluid and intentional shift allows viewers to understand the building's context within the city as well as its detailed architecture.
19. The portrayal of the city and its architectural elements is vivid and true-to-life.
20. The presence of trees introduces a splash of green to the predominantly concrete scenery.
21. Natural lighting highlights the cathedral and the surrounding are2. 

### Result:
```json
{{
    "1": {{
        "judgement": "entailment",
        "analysis": "The video caption describes an aerial view of a city, which aligns with the key point about capturing a cityscape view."
    }},
    "2": {{
        "judgement": "entailment",
        "analysis": "The video caption mentions a clear and blue sky, which indicates favorable weather, aligning with the key point."
    }},
    "3": {{
        "judgement": "entailment",
        "analysis": "The caption describes buildings of various heights, which matches the key point about a range of building sizes and elevations."
    }},
    "4": {{
        "judgement": "neutral",
        "analysis": "The video caption does not mention the type of transitions used in the video, so this point cannot be confirmed or contradicted."
    }},
    "5": {{
        "judgement": "neutral",
        "analysis": "The video caption does not provide information about the camera movement, so this point cannot be confirmed or contradicted."
    }},
    "6": {{
        "judgement": "neutral",
        "analysis": "The caption does not specify the clarity of the church's architectural details, so this point cannot be confirmed or contradicted."
    }},
    "7": {{
        "judgement": "entailment",
        "analysis": "The caption mentions the church having two tall towers and a large dome, which suggests complex architectural details."
    }},
    "8": {{
        "judgement": "contradiction",
        "analysis": "The caption does not mention parks or green areas, only some trees, which contradicts the key point about parks offering contrast."
    }},
    "9": {{
        "judgement": "neutral",
        "analysis": "The caption does not describe specific architectural features like stonework or pointed arches, so this point cannot be confirmed or contradicted."
    }},
    "10": {{
        "judgement": "neutral",
        "analysis": "The caption does not mention the city's grid of streets and alleys, so this point cannot be confirmed or contradicted."
    }},
    "11": {{
        "judgement": "neutral",
        "analysis": "The caption mentions buildings mostly being white with some green roofs, but does not elaborate on diverse rooftop designs."
    }},
    "12": {{
        "judgement": "neutral",
        "analysis": "The caption does not specify features like high spires or pointed arch windows, so this point cannot be confirmed or contradicted."
    }},
    "13": {{
        "judgement": "contradiction",
        "analysis": "The caption describes views from different angles, not a consistent focus on the church, contradicting the key point."
    }},
    "14": {{
        "judgement": "neutral",
        "analysis": "The caption does not describe the urban area as lively, so this point cannot be confirmed or contradicted."
    }},
    "15": {{
        "judgement": "neutral",
        "analysis": "The caption does not mention population density, so this point cannot be confirmed or contradicted."
    }},
    "16": {{
        "judgement": "entailment",
        "analysis": "The caption mentions trees in the area, which aligns with the key point about trees contributing greenery."
    }},
    "17": {{
        "judgement": "neutral",
        "analysis": "The caption does not mention the intensity of illumination, so this point cannot be confirmed or contradicted."
    }},
    "18": {{
        "judgement": "neutral",
        "analysis": "The caption does not describe the shift in perspective or its impact on understanding the building's context."
    }},
    "19": {{
        "judgement": "entailment",
        "analysis": "The caption describes the city and its buildings, which aligns with the key point about a vivid portrayal of architectural elements."
    }},
    "20": {{
        "judgement": "entailment",
        "analysis": "The caption mentions trees adding greenery, which aligns with the key point about trees introducing green to the scenery."
    }},
    "21": {{
        "judgement": "neutral",
        "analysis": "The caption does not mention natural lighting specifically highlighting the cathedral, so this point cannot be confirmed or contradicted."
    }}
}}
```

## Example 1:
### Video Caption
The video depicts a train station with a train on the tracks. The train is stationary at the beginning of the video, and a person is seen walking on the platform. The train then starts to move, and the person continues to walk alongside it. As the train gains speed, the person stops walking and stands still. The train continues to move, and the person remains stationary on the platform. The train eventually leaves the station, and the platform is left empty. The video captures the movement of the train and the person's actions as they interact with the train. The station appears to be well-lit, and the train is visible in the background. Overall, the video provides a glimpse into the daily life of a train station and the interactions between people and trains.

### Key Points
1. The footage shows a subway train traveling along its route.
2. The video begins with a top-down view of a lengthy train.
3. The train is positioned centrally within the shot.
4. The platform is covered with tiled flooring.
5. The station's floor is covered with tiles in a white and grey pattern.
6. An elevated walkway or bridge spans above the platform.
7. The train is situated within a subway station.
8. The platform becomes visible.
9. The footage begins with a fixed shot of a locomotive stationed at a platform.
10. The metro station features an arched ceiling.
11. The illumination at the station is adequate.
12. An individual in blue attire is seen walking on the subway platform.
13. The station's structure is noticeable in the background.
14. The subway station features two parallel railway tracks.
15. The walls of the train station are covered with tiles.
16. The concluding shot presents an aerial view of the station.

### Result:
```json
{{
    "1": {{
        "judgement": "contradiction",
        "analysis": "The video caption describes a train station with a train on the tracks, but it does not specify that it is a subway train. Therefore, this key point contradicts the information provided."
    }},
    "2": {{
        "judgement": "contradiction",
        "analysis": "The video caption does not mention a top-down view of the train. It describes the train being stationary and then moving, but not the specific camera angle at the beginning."
    }},
    "3": {{
        "judgement": "neutral",
        "analysis": "The video caption does not specify the position of the train within the shot, so this point cannot be confirmed or contradicted."
    }},
    "4": {{
        "judgement": "neutral",
        "analysis": "The video caption does not mention the type of flooring on the platform, so this point cannot be confirmed or contradicted."
    }},
    "5": {{
        "judgement": "neutral",
        "analysis": "The video caption does not describe the pattern or color of the tiles on the station's floor, so this point cannot be confirmed or contradicted."
    }},
    "6": {{
        "judgement": "neutral",
        "analysis": "The video caption does not mention an elevated walkway or bridge above the platform, so this point cannot be confirmed or contradicted."
    }},
    "7": {{
        "judgement": "neutral",
        "analysis": "The video caption describes a train station but does not specify that it is a subway station, so this point cannot be confirmed or contradicted."
    }},
    "8": {{
        "judgement": "entailment",
        "analysis": "The video caption describes the platform and the person's actions on it, which aligns with the key point about the platform becoming visible."
    }},
    "9": {{
        "judgement": "entailment",
        "analysis": "The video caption describes the train being stationary at the beginning, which aligns with the key point about a fixed shot of a locomotive stationed at a platform."
    }},
    "10": {{
        "judgement": "neutral",
        "analysis": "The video caption does not mention the ceiling of the station, so this point cannot be confirmed or contradicted."
    }},
    "11": {{
        "judgement": "entailment",
        "analysis": "The video caption mentions that the station appears to be well-lit, which aligns with the key point about adequate illumination."
    }},
    "12": {{
        "judgement": "neutral",
        "analysis": "The video caption mentions a person walking on the platform but does not specify the color of their attire, so this point cannot be confirmed or contradicted."
    }},
    "13": {{
        "judgement": "entailment",
        "analysis": "The video caption mentions the station and the train being visible in the background, which aligns with the key point about the station's structure being noticeable."
    }},
    "14": {{
        "judgement": "neutral",
        "analysis": "The video caption does not mention the number of railway tracks, so this point cannot be confirmed or contradicted."
    }},
    "15": {{
        "judgement": "neutral",
        "analysis": "The video caption does not describe the walls of the train station, so this point cannot be confirmed or contradicted."
    }},
    "16": {{
        "judgement": "contradiction",
        "analysis": "The video caption does not mention an aerial view of the station in the concluding shot, so this point contradicts the provided information."
    }}
}}
```
With these examples in mind, please help me classify the relationship between each provided key point and the given video caption.

### Video Caption
{caption}

### Key Points
{key_points}

### Result:
"""

    def __init__(self, batch_size: int = 30):
        """
        Initialize the RecallPromptBuilder.
        
        Args:
            batch_size (int): Number of key points to process in each batch
        """
        self.batch_size = batch_size
    
    def create_prompts(self, candidate_path: str, golden_path: str, output_path: str) -> None:
        """
        Create recall evaluation prompts by comparing key points with video captions.
        
        Args:
            candidate_path (str): Path to the candidate captions file
            golden_path (str): Path to the golden key points file
            output_path (str): Path to save the generated prompts
        """
        # Load candidate captions
        index2caption = {}
        with open(candidate_path, "r") as infile:
            for line in infile:
                data = json.loads(line)
                index2caption[data["index"]] = data["pred_caption"]
        
        # Process key points and create prompts
        with open(output_path, "w") as outfile:
            with open(golden_path, "r") as infile:
                for line in infile:
                    data = json.loads(line)
                    if data["index"] not in index2caption:
                        # Skip if no caption available (e.g., model refusal)
                        continue
                    
                    # Extract key point texts
                    filtered_pass_kp_list = [item["text"] for item in data["passed_kp_list"]]
                    
                    # Split into batches
                    filtered_pass_kp_list_splits = [
                        filtered_pass_kp_list[i:i + self.batch_size] 
                        for i in range(0, len(filtered_pass_kp_list), self.batch_size)
                    ]
                    
                    # Create prompts for each batch
                    for batch_index, filtered_pass_kp_list_split in enumerate(filtered_pass_kp_list_splits):
                        key_point = ""
                        for i, point in enumerate(filtered_pass_kp_list_split):
                            key_point += '{}. {}\n'.format(i+1, point)
                        
                        # Create prompt data
                        prompt_data = {
                            "index": f"{data['index']}_{batch_index}",
                            "kp_num": len(filtered_pass_kp_list_split), 
                            "kp_list": filtered_pass_kp_list_split, 
                            "question": self.SYSTEM_PROMPT.format(
                                key_points=key_point,
                                caption=index2caption[data["index"]]
                            )
                        }
                        
                        outfile.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")