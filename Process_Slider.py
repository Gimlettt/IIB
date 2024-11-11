import pandas as pd
import numpy as np

# Define constants for slider properties and target values
SLIDER_IDS = ["MRTKSlider_4", "MRTKSlider_5", "MRTKSlider_6"]
SLIDER_TARGETVALUES = {
    1: {"MRTKSlider_4": 0.15, "MRTKSlider_5": 0.75, "MRTKSlider_6": 0.36},
    2: {"MRTKSlider_4": 0.85, "MRTKSlider_5": 0.25, "MRTKSlider_6": 0.91},
    3: {"MRTKSlider_4": 0.45, "MRTKSlider_5": 1, "MRTKSlider_6": 0.56},
    4: {"MRTKSlider_4": 0.3, "MRTKSlider_5": 0, "MRTKSlider_6": 0.09},
    5: {"MRTKSlider_4": 0.95, "MRTKSlider_5": 0.25, "MRTKSlider_6": 0.73}
}
THRESHOLD = 0.2

# Lambda function to calculate the absolute distance from the score to the target
slider_distance_from_score = lambda score, target_value: abs(1 - score * np.maximum(target_value, 1 - target_value))

# Function to process slider events, calculate distances, and mark frames
def process_slider_events(slider_events, threshold=THRESHOLD):
    # Exclude training data
    filtered_events = slider_events[~slider_events.str.contains("Training")]

    # List to hold marked frames
    marked_frames = []

    # Loop through each slider event to calculate distances
    for idx, event in filtered_events.items():
        # Check if the event contains "Sliders #", otherwise skip
        if "Sliders #" not in event:
            continue

        # Extract task number and slider ID
        try:
            task_number = int(event.split("Sliders #")[1][0])  # Extract task number
            slider_id = next((sid for sid in SLIDER_IDS if sid in event), None)
        except (IndexError, ValueError):
            print(f"Could not extract task number from event: {event}")
            continue

        # Proceed if slider_id is valid and contains a subscores section
        if slider_id and f"(SUBSCORE {slider_id}):" in event:
            try:
                # Extract the score value from the event text
                score_text = event.split(f"(SUBSCORE {slider_id}): ")[1]
                score = float(score_text) if score_text else np.nan
            except (IndexError, ValueError):
                print(f"Could not extract score from event: {event}")
                continue

            # Get the target value for this slider and task
            target_value = SLIDER_TARGETVALUES.get(task_number, {}).get(slider_id, np.nan)

            # Calculate the distance if score and target are available
            if not np.isnan(score) and not np.isnan(target_value):
                distance = slider_distance_from_score(score, target_value)

                # Check if the distance exceeds the threshold
                if distance > threshold:
                    marked_frames.append((idx, task_number, slider_id, score, target_value, distance))
    
    # Convert to DataFrame 
    marked_frames_df = pd.DataFrame(marked_frames, columns=["Frame Index", "Task Number", "Slider ID", "Score", "Target Value", "Distance"])

    # Save the results to a CSV file
    marked_frames_df.to_csv("marked_frames.csv", index=False)
    print("Marked frames saved to 'marked_frames.csv'")
    
    return marked_frames_df
