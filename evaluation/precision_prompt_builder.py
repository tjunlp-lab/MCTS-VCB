# -*- coding: utf-8 -*-
"""
Precision Prompt Builder
Creates prompts for evaluating precision by comparing LMM caption breakdown points 
with human-generated key points.
"""

import json
import pandas as pd
from typing import Dict, List, Tuple, Any


class PrecisionPromptBuilder:
    """Builder class for creating precision evaluation prompts."""
    
    # Keep the original prompt template unchanged
    MLLM_EVAL_PROMPT = '''
Please objectively classify the relationship between each video caption breakdown provided by the Large Multimodal Model (LMM) and provided human-generated key points.
Analyze each breakdown point individually to determine its relationship with the human-generated key points.

For each breakdown point, classify the relationship into one of the following categories:
1. "entailment" means that the breakdown point is accurately reflected within one or more of the human-generated key points.
2. "contradiction" means that breakdown point some detail in the breakdown point contradicts with the infomation mentioned human-generated key points.
3. "neutral" means that the relationship is neither "entailment" nor "contradiction".

For each breakdown point, provide a brief analysis explaining the reasoning behind your judgment.

Please present the result in a JSON dict format: {{'breakdown_point_1': {{'judgement': judgement_1, 'analysis': analysis_1}}, ...,  'breakdown_point_n': {{'judgement': judgement_n, 'analysis': analysis_n}}}}.

For clarity, consider these examples:

#### Example 1:
### Human-Generated Video Key Points:
1.The background is where the ocean meets the blue sky.
2.A cityscape can be seen in the distance.
3.A person is practicing yoga.
4.The person is a woman.
5.She is wearing a blue top.
6.Her hair is blue.
7.The woman closes her eyes, raises her right hand, and slightly leans to her left. She opens her eyes, raises both hands horizontally, turns to her right, and faces the camera sideways, lifting both hands above her head.
8.The camera slightly moves following the action, first up to the right, then down to the left.

### LMM Caption Breakdown Point to Evaluate:
1. An individual is performing yoga near a body of water.
2. A cityscape is visible in the background.
3. The person begins by stretching their arms out horizontally at shoulder height.
4. This posture emphasizes the extension and strengthening of the arms.
5. The individual then transitions into a side stretch.
6. One arm is raised high above their head while leaning to the opposite side.
7. This movement creates a curve in the body.
8. This movement highlights flexibility.
9. This movement targets the muscles of the side body.
10. The person stretches their arms upwards, reaching towards the sky.
11. This enhances the stretch in the spine.
12. This engages the core muscles.
13. The individual then returns to the horizontal arm extension pose.
14. The individual maintains a strong and balanced posture.
15. The sequence repeats.
16. The individual alternates between the side stretch and the arm extension poses.
17. The individual maintains a fluid and controlled flow throughout the routine.
18. The backdrop of the serene water and clear sky adds a calming effect.
19. The calming effect complements the graceful movements of the yoga practice.

## Result:
```json
{{
  "1":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions an individual performing yoga near a body of water, which is consistent with the human-generated key points indicating the ocean in the background."
  }},
  "2":{{
    "judgement": "entailment",
    "analysis": "The breakdown point states that a cityscape is visible in the background, which matches the human-generated key point about the cityscape in the distance."
  }},
  "3":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions the person begins by stretching their arms out horizontally at shoulder height, which contradicts the human-generated key point that describes the person closing her eyes, raising her right hand, and slightly leaning to her left."
  }},
  "4":{{
    "judgement": "neutral",
    "analysis": "The breakdown point emphasizes the extension and strengthening of the arms, which is not specifically mentioned in the human-generated key points."
  }},
  "5":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the individual transitioning into a side stretch, which aligns with the human-generated key point describing the person leaning to her left."
  }},
  "6":{{
    "judgement": "entailment",
    "analysis": "The breakdown point describes one arm being raised high above the head while leaning to the opposite side, which is consistent with the human-generated key point about the woman raising her right hand and leaning to her left."
  }},
  "7":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions creating a curve in the body, which is not specifically detailed in the human-generated key points."
  }},
  "8":{{
    "judgement": "neutral",
    "analysis": "The breakdown point highlights flexibility, which is not specifically mentioned in the human-generated key points."
  }},
  "9":{{
    "judgement": "neutral",
    "analysis": "The breakdown point targets the muscles of the side body, which is not specifically mentioned in the human-generated key points."
  }},
  "10":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the person stretching their arms upwards, reaching towards the sky, which aligns with the human-generated key point about the woman lifting both hands above her head."
  }},
  "11":{{
    "judgement": "neutral",
    "analysis": "The breakdown point enhances the stretch in the spine, which is not specifically mentioned in the human-generated key points."
  }},
  "12":{{
    "judgement": "neutral",
    "analysis": "The breakdown point engages the core muscles, which is not specifically mentioned in the human-generated key points."
  }},
  "13":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions the individual returning to the horizontal arm extension pose, which contradicts the human-generated key points that do not describe this specific action."
  }},
  "14":{{
    "judgement": "neutral",
    "analysis": "The breakdown point maintains a strong and balanced posture, which is not specifically mentioned in the human-generated key points."
  }},
  "15":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions the sequence repeating, which contradicts the human-generated key points that describe a specific sequence of actions without repetition."
  }},
  "16":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions alternating between the side stretch and the arm extension poses, which contradicts the human-generated key points that describe a specific sequence of actions without alternation."
  }},
  "17":{{
    "judgement": "neutral",
    "analysis": "The breakdown point maintains a fluid and controlled flow throughout the routine, which is not specifically mentioned in the human-generated key points."
  }},
  "18":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the backdrop of serene water and clear sky, which aligns with the human-generated key point about the ocean meeting the blue sky."
  }},
  "19":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the calming effect complementing the graceful movements, which is not specifically mentioned in the human-generated key points."
  }}
}}
```

#### Example 2:
### Human-Generated Video Key Points:
1.The person on the right side of the video has injured his right foot.
2.The person on the left side of the video is bandaging his foot.
3.The bandage is white.
4.The ankle, heel, and sole are all wrapped in the bandage.
5.The person on the left side of the video is wearing a top with black sleeves.
6.The person on the left side of the video is wearing black pants.
7.The person on the left side of the video is wearing glasses.
8.Indoor scene.
9.The lighting is bright.
10.White walls.
11.Green plastic lawn.
12.The camera remains static, shooting from a high angle.

### LMM Caption Breakdown Point to Evaluate:
1. A person is applying white athletic tape to another individual's lower leg.
2. The process involves wrapping the tape around the ankle.
3. The process involves wrapping the tape around the calf.
4. The tape is secured in place with deliberate movements.
5. The application is meticulous.
6. The application indicates care for proper support or injury prevention.
7. The background features an artificial green turf surface.
8. The background features part of a wall.
9. The setting suggests an indoor environment possibly designed for sports or physical activities.
10. There are no significant changes in the environment throughout the sequence.
11. There are no significant changes in the actions performed throughout the sequence.
12. The focus is solely on the taping procedure.

### Result:
```json
{{
  "1":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions a person applying white athletic tape to another individual's lower leg, which is consistent with the human-generated key points about bandaging the foot with a white bandage."
  }},
  "2":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions wrapping the tape around the ankle, which matches the human-generated key point that the ankle is wrapped in the bandage."
  }},
  "3":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions wrapping the tape around the calf, which contradicts the human-generated key points that only the ankle, heel, and sole are wrapped."
  }},
  "4":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the tape is secured in place with deliberate movements, which is not directly addressed in the human-generated key points."
  }},
  "5":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the application is meticulous, which is not directly addressed in the human-generated key points."
  }},
  "6":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the application indicates care for proper support or injury prevention, which is not directly addressed in the human-generated key points."
  }},
  "7":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the background features an artificial green turf surface, which matches the human-generated key point about the green plastic lawn."
  }},
  "8":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the background features part of a wall, which is consistent with the human-generated key point about white walls."
  }},
  "9":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the setting suggests an indoor environment, which matches the human-generated key point about the indoor scene."
  }},
  "10":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions there are no significant changes in the environment throughout the sequence, which is not directly addressed in the human-generated key points."
  }},
  "11":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions there are no significant changes in the actions performed throughout the sequence, which is not directly addressed in the human-generated key points."
  }},
  "12":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the focus is solely on the taping procedure, which is not directly addressed in the human-generated key points."
  }}
}}
```

#### Example 3:
### Human-Generated Video Key Points:
1.A motorcyclist is riding a motorcycle.
2.The racer is wearing a black and white racing suit.
3.Wearing a helmet with red accents.
4.Riding a black and white motorcycle.
5.The racing environment is in a mountainous open area.
6.Riding past stirs up dust.
7.Red mesh barriers separate the spectators and the racer.
8.On both sides of the track, some people wearing blue vests are taking photos with phones or cameras.
9.The racer first rides the motorcycle over a small hill, then leaps into the air and dives downward, continues forward, and then makes a right turn.
10.The camera continuously follows the movement of the racer, with a close-up action in the latter half of the scene.

### LMM Caption Breakdown Point to Evaluate:
1. The video features a motocross rider navigating through an off-road, rugged terrain.
2. The subject is seen riding a dirt bike with impressive skill.
3. The rider maneuvers through a rocky and uneven track.
4. The rider jumps over mounds of dirt.
5. The rider navigates sharp turns.
6. The rider often lifts both the front and rear wheels off the ground.
7. The track is set in a rugged, mountainous area.
8. The track has loose gravel and dirt.
9. The track is surrounded by steep cliff-like walls.
10. Spectators are lined along certain sections of the track.
11. Some spectators are standing.
12. Some spectators are taking photos.
13. Spectators are separated by orange safety nets.
14. The rider is dressed in full motocross gear.
15. The rider is wearing a helmet.


### Result:
```json
{{
  "1":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions a motocross rider navigating through an off-road, rugged terrain, which is consistent with the human-generated key point about the racing environment being in a mountainous open area."
  }},
  "2":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the subject riding a dirt bike with impressive skill, which aligns with the human-generated key point about a motorcyclist riding a motorcycle."
  }},
  "3":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the rider maneuvering through a rocky and uneven track, which is not directly addressed in the human-generated key points."
  }},
  "4":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the rider jumping over mounds of dirt, which is consistent with the human-generated key point about the racer leaping into the air."
  }},
  "5":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the rider navigating sharp turns, which matches the human-generated key point about the racer making a right turn."
  }},
  "6":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the rider often lifting both the front and rear wheels off the ground, which is not directly addressed in the human-generated key points."
  }},
  "7":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the track being set in a rugged, mountainous area, which aligns with the human-generated key point about the racing environment being in a mountainous open area."
  }},
  "8":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions the track having loose gravel and dirt, which is not directly addressed in the human-generated key points."
  }},
  "9":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions the track being surrounded by steep cliff-like walls, which contradicts the human-generated key points that do not mention any steep cliff-like walls."
  }},
  "10":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions spectators lined along certain sections of the track, which is consistent with the human-generated key point about spectators separated by red mesh barriers."
  }},
  "11":{{
    "judgement": "neutral",
    "analysis": "The breakdown point mentions some spectators standing, which is not directly addressed in the human-generated key points."
  }},
  "12":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions some spectators taking photos, which matches the human-generated key point about people wearing blue vests taking photos with phones or cameras."
  }},
  "13":{{
    "judgement": "contradiction",
    "analysis": "The breakdown point mentions spectators separated by orange safety nets, which contradicts the human-generated key point about red mesh barriers."
  }},
  "14":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the rider dressed in full motocross gear, which aligns with the human-generated key point about the racer wearing a black and white racing suit."
  }},
  "15":{{
    "judgement": "entailment",
    "analysis": "The breakdown point mentions the rider wearing a helmet, which is consistent with the human-generated key point about the racer wearing a helmet with red accents."
  }}
}}
```

With these examples in mind, please help me evaluate whether each breakdown point is accurately reflected in the provided human-generated key points.

### Human-Generated Video Key Points:
{key_point}

### LMM Caption Breakdown Point to Evaluate:
{caption}

### Result:
'''

    def __init__(self, test_data_path: str, test_data_threshold: str):
        """
        Initialize the PrecisionPromptBuilder.
        
        Args:
            test_data_path (str): Path to the test data file containing key points
            test_data_threshold (str): Threshold value for filtering test data
        """
        self.test_data_path = test_data_path
        self.test_data_threshold = test_data_threshold
        self.index2sample = {}
        self._load_test_data()
    
    def _load_test_data(self) -> None:
        """Load and process test data from the given path."""
        data_df = pd.read_json(self.test_data_path, orient='records', lines=True)
        
        for index, row in data_df.iterrows():
            index = str(row.iloc[0])
            if index not in self.index2sample:
                self.index2sample[index] = {}
            
            key_point = row.iloc[2].lstrip('0123456789').lstrip('.').strip()
            if 'key_points' not in self.index2sample[index]:
                self.index2sample[index]['key_points'] = []
            self.index2sample[index]['key_points'].append((key_point, row.iloc[3]))
    
    def create_prompts(self, model_predictions_path: str, output_path: str) -> None:
        """
        Create precision evaluation prompts by comparing model predictions with key points.
        
        Args:
            model_predictions_path (str): Path to model prediction results
            output_path (str): Path to save the generated prompts
        """
        # Load model predictions
        index2answer = {}
        with open(model_predictions_path, 'r') as fin:
            for line in fin:
                sample = json.loads(line)
                index2answer[str(sample['index'])] = sample['fined_atom_desc']
        
        # Generate prompts
        count = 0
        with open(output_path, 'w') as fout:
            for index in self.index2sample:
                if index not in index2answer:
                    # Skip if model has no prediction (e.g., refusal cases)
                    continue
                
                # Format key points
                key_point = ''
                key_points = self.index2sample[index]['key_points']
                for i, point in enumerate(key_points):
                    key_point += '{}.{}\n'.format(i+1, point[0])
                key_point = key_point.rstrip('\n')
                
                # Format extracted breakdown points
                extracted_key_point = ""
                for i, point in enumerate(index2answer[index]):
                    extracted_key_point += '{}.{}\n'.format(i+1, point["text"])
                extracted_key_point = extracted_key_point.rstrip('\n')
                
                # Create prompt
                prompt = self.MLLM_EVAL_PROMPT.format(key_point=key_point, caption=extracted_key_point)
                
                # Write to output
                fout.write(json.dumps({
                    'idx': str(count), 
                    'question': prompt, 
                    'index': index, 
                    "kp_num": len(index2answer[index]), 
                    "kp_list": index2answer[index]
                }, ensure_ascii=False) + '\n')
                count += 1