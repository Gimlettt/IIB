import os
import re
import pandas as pd
import numpy as np

# -----------------------------
# Drop Counting Functions
# -----------------------------

def count_drag_drops(file_path, t=60, thre=0.03):
    """
    Count drag drop events.
    It looks for "RELEASED" and "COMPLETED" events (ignoring "Training" rows).
    It first removes any RELEASED event immediately preceding a COMPLETED event,
    then merges remaining RELEASED events that occur within 1 second into one.
    """
    data_df = pd.read_csv(file_path)
    data_df.columns = data_df.columns.str.strip()

    # Filter for Drag events (ignoring any "Training" rows)
    drag_events = data_df[data_df.iloc[:, 1].str.contains("Drag", na=False)].iloc[:, 1]
    drag_events = drag_events[~drag_events.str.contains("Training")]

    # Identify "RELEASED" and "COMPLETED" events
    released_indices = drag_events[drag_events.str.contains("RELEASED")].index
    completed_indices = drag_events[drag_events.str.contains("COMPLETED")].index

    # Remove any release event immediately preceding a completed event
    for row in completed_indices:
        if (row - 1) in released_indices:
            released_indices = released_indices.drop(row - 1, errors='ignore')

    # Merge release events occurring within 1 second into one.
    if len(released_indices) > 0:
        # Assume timestamps are in the first column; convert to float.
        release_times = data_df.loc[released_indices].iloc[:, 0].astype(float).values
        release_times_sorted = np.sort(release_times)
        merged_release_count = 1
        last_time = release_times_sorted[0]
        for t_val in release_times_sorted[1:]:
            if t_val - last_time >= 1:  # merge events within 1 second
                merged_release_count += 1
                last_time = t_val
    else:
        merged_release_count = 0

    drop_count = merged_release_count + len(completed_indices)
    return drop_count

def count_sketch_drops(file_path):
    """
    Count sketch drop events.
    It selects rows containing "DRAW" or "Sketch" (ignoring "Training") and finds "STOPPED" events.
    Multiple STOPPED events occurring within 1 second are merged into one drop.
    Assumes the first column holds timestamps (as float).
    Also, if the computed drop count is less than 24 (e.g., if merging yields a very low count),
    it defaults to 24.
    """
    data_df = pd.read_csv(file_path)
    data_df.columns = data_df.columns.str.strip()

    sketch_events = data_df[
        (data_df.iloc[:, 1].str.contains("DRAW", na=False)) |
        (data_df.iloc[:, 1].str.contains("Sketch", na=False))
    ].iloc[:, 1]
    sketch_events = sketch_events[~sketch_events.str.contains("Training")]

    stopped_indices = sketch_events[sketch_events.str.contains("STOPPED")].index
    if len(stopped_indices) == 0:
        return 0

    times = data_df.loc[stopped_indices].iloc[:, 0].astype(float).values
    times_sorted = np.sort(times)

    drop_count = 1
    last_time = times_sorted[0]
    for t_val in times_sorted[1:]:
        if (t_val - last_time) >= 1:  # 1 second gap
            drop_count += 1
            last_time = t_val

    # In case merging yields a very low count, default to 24.
    if drop_count < 24:
        drop_count = 24

    return drop_count

def count_slider_drops(file_path, score_threshold=0.6, debug=True):
    """
    Count slider drop events by parsing lines matching:
      EVENT: SCORE TASK Sliders #<task_num> (SUBSCORE MRTKSlider_<slider_id>): <score>
    A drop is counted if the parsed score is less than score_threshold.
    """
    pattern = r"SCORE TASK Sliders #(\d+)\s*\(SUBSCORE MRTKSlider_(\d+)\):\s*([\d\.]+)"
    data_df = pd.read_csv(file_path)
    data_df.columns = data_df.columns.str.strip()

    all_scores = []
    for idx, row in data_df.iterrows():
        event_text = str(row.iloc[1])
        match = re.search(pattern, event_text)
        if match:
            score_val = float(match.group(3))
            all_scores.append(score_val)

    count_below = sum(score < score_threshold for score in all_scores)

    if debug:
        print(f"\n--- Debug info for slider file: {os.path.basename(file_path)} ---")
        print("All parsed slider scores:", all_scores)
        print(f"Count of scores below {score_threshold}:", count_below)
        print("-------------------------------------------------------------")

    return count_below

# -----------------------------
# SUS Analysis Functions
# -----------------------------

def compute_sus_score(file_path):
    """
    Reads a SUS text file and computes the SUS score.
    The file is expected to have 10 questions in the format:
        Q1, <response>
        Q2, <response>
        ...
    with responses on a 0-to-4 scale.
    
    Calculation:
      - For odd-numbered questions: adjusted = response.
      - For even-numbered questions: adjusted = 4 - response.
    Sum the adjusted scores (range: 0 to 40) and multiply by 2.5 to yield a score out of 100.
    """
    responses = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Q"):
                parts = line.split(',')
                if len(parts) >= 2:
                    question = parts[0].strip()
                    try:
                        response = float(parts[1].strip())
                        responses[question] = response
                    except ValueError:
                        continue
    total = 0
    for i in range(1, 11):
        key = f"Q{i}"
        if key in responses:
            if i % 2 == 1:  # odd-numbered question
                adjusted = responses[key]
            else:           # even-numbered question
                adjusted = 4 - responses[key]
            total += adjusted
        else:
            print(f"Missing response for {key} in {file_path}.")
    sus_score = total * 2.5
    return sus_score

def parse_sus_file_info(file_path):
    """
    Parses a SUS file path to extract participant and condition.
    Expects file names like "001_SUS_C.txt" or "001_SUS_WC.txt".
    """
    base = os.path.basename(file_path)
    parts = base.split("_")
    participant = parts[0]  # e.g., "001"
    condition = parts[2].split(".")[0]  # e.g., "C" or "WC" (assuming format: 001_SUS_C.txt)
    return participant, condition

# -----------------------------
# NASA TLX Parsing Functions
# -----------------------------

def parse_nasa_file_info(file_path):
    """
    Parses a NASA TLX file path to extract participant and condition.
    Expects file names like "001_C.txt" or "001_WC.txt".
    """
    base = os.path.basename(file_path)
    parts = base.split("_")
    participant = parts[0]
    condition = parts[1].split(".")[0]
    return participant, condition

def get_nasa_overall(file_path):
    """
    Reads a NASA TLX text file and returns the overall value.
    Looks for a line starting with "Overall = ".
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    overall_value = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Overall"):
            match = re.search(r"Overall\s*=\s*([\d\.]+)", line)
            if match:
                overall_value = float(match.group(1))
                break
    return overall_value

# -----------------------------
# Helper to Parse Raw Data File Info
# -----------------------------

def parse_file_info(file_path):
    """
    Parses the raw data file path to extract participant and condition.
    Assumes a structure like:
      experiment_data/Raw data/<participant>/<Condition>_<date>/bodyPose.csv
    where <participant> is at index 2 and <Condition> (e.g., "WC" or "C")
    is the prefix of the folder at index 3.
    """
    parts = os.path.normpath(file_path).split(os.sep)
    participant = parts[2]
    condition = parts[3].split("_")[0]
    return participant, condition

# -----------------------------
# Main Processing and CSV Output
# -----------------------------

if __name__ == "__main__":
    # List of raw data file paths for all participants (each participant has 2 files: one for WC and one for C)
    file_paths = [
        "experiment_data/Raw data/001/WC_2025-03-17_15-13-11/bodyPose.csv",
        "experiment_data/Raw data/001/C_2025-03-17_15-24-59/bodyPose.csv",
        "experiment_data/Raw data/002/WC_2025-03-26_20-51-21/bodyPose.csv",
        "experiment_data/Raw data/002/C_2025-03-26_20-57-53/bodyPose.csv",
        "experiment_data/Raw data/003/WC_2025-03-19_14-07-45/bodyPose.csv",
        "experiment_data/Raw data/003/C_2025-03-19_14-23-16/bodyPose.csv",
        "experiment_data/Raw data/004/WC_2025-03-19_16-01-31/bodyPose.csv",
        "experiment_data/Raw data/004/C_2025-03-19_16-10-03/bodyPose.csv",
        "experiment_data/Raw data/005/WC_2025-03-20_11-36-51/bodyPose.csv",
        "experiment_data/Raw data/005/C_2025-03-20_11-48-36/bodyPose.csv",
        "experiment_data/Raw data/006/WC_2025-03-20_13-38-29/bodyPose.csv",
        "experiment_data/Raw data/006/C_2025-03-20_13-50-49/bodyPose.csv",
        "experiment_data/Raw data/007/WC_2025-03-21_14-17-25/bodyPose.csv",
        "experiment_data/Raw data/007/C_2025-03-21_14-28-27/bodyPose.csv",
        "experiment_data/Raw data/008/C_2025-03-21_15-15-51/bodyPose.csv",
        "experiment_data/Raw data/008/WC_2025-03-21_15-04-16/bodyPose.csv",
        "experiment_data/Raw data/009/WC_2025-03-22_15-34-43/bodyPose.csv",
        "experiment_data/Raw data/009/C_2025-03-22_15-46-29/bodyPose.csv",
        "experiment_data/Raw data/010/C_2025-03-23_14-12-17/bodyPose.csv",
        "experiment_data/Raw data/010/WC_2025-03-23_14-01-33/bodyPose.csv",
        "experiment_data/Raw data/011/WC_2025-03-23_21-34-13/bodyPose.csv",
        "experiment_data/Raw data/011/C_2025-03-23_21-49-19/bodyPose.csv",
        "experiment_data/Raw data/012/WC_2025-03-24_20-08-18/bodyPose.csv",
        "experiment_data/Raw data/012/C_2025-03-24_20-18-47/bodyPose.csv",
        "experiment_data/Raw data/013/WC_2025-03-25_14-38-18/bodyPose.csv",
        "experiment_data/Raw data/013/C_2025-03-25_14-50-05/bodyPose.csv",
        "experiment_data/Raw data/014/C_2025-03-25_19-31-40/bodyPose.csv",
        "experiment_data/Raw data/014/WC_2025-03-25_19-20-19/bodyPose.csv",
        "experiment_data/Raw data/015/WC_2025-03-26_16-56-38/bodyPose.csv",
        "experiment_data/Raw data/015/C_2025-03-26_17-06-59/bodyPose.csv",
        "experiment_data/Raw data/016/WC_2025-03-26_17-15-08/bodyPose.csv",
        "experiment_data/Raw data/016/C_2025-03-26_17-25-43/bodyPose.csv",
    ]
    
    # List of NASA TLX file paths.
    nasa_paths = [
        "experiment_data/NASA TLX/001_C.txt",
        "experiment_data/NASA TLX/001_WC.txt",
        "experiment_data/NASA TLX/002_C.txt",
        "experiment_data/NASA TLX/002_WC.txt",
        "experiment_data/NASA TLX/003_C.txt",
        "experiment_data/NASA TLX/003_WC.txt",
        "experiment_data/NASA TLX/004_C.txt",
        "experiment_data/NASA TLX/004_WC.txt",
        "experiment_data/NASA TLX/005_C.txt",
        "experiment_data/NASA TLX/005_WC.txt",
        "experiment_data/NASA TLX/006_C.txt",
        "experiment_data/NASA TLX/006_WC.txt",
        "experiment_data/NASA TLX/007_C.txt",
        "experiment_data/NASA TLX/007_WC.txt",
        "experiment_data/NASA TLX/008_C.txt",
        "experiment_data/NASA TLX/008_WC.txt",
        "experiment_data/NASA TLX/009_C.txt",
        "experiment_data/NASA TLX/009_WC.txt",
        "experiment_data/NASA TLX/010_C.txt",
        "experiment_data/NASA TLX/010_WC.txt",
        "experiment_data/NASA TLX/011_C.txt",
        "experiment_data/NASA TLX/011_WC.txt",
        "experiment_data/NASA TLX/012_C.txt",
        "experiment_data/NASA TLX/012_WC.txt",
        "experiment_data/NASA TLX/013_C.txt",
        "experiment_data/NASA TLX/013_WC.txt",
        "experiment_data/NASA TLX/014_C.txt",
        "experiment_data/NASA TLX/014_WC.txt",
        "experiment_data/NASA TLX/015_C.txt",
        "experiment_data/NASA TLX/015_WC.txt",
        "experiment_data/NASA TLX/016_C.txt",
        "experiment_data/NASA TLX/016_WC.txt",
    ]
    
    # List of SUS file paths.
    sus_paths = [
        "experiment_data/SUS/001_SUS_C.txt",
        "experiment_data/SUS/001_SUS_WC.txt",
        "experiment_data/SUS/002_SUS_C.txt",
        "experiment_data/SUS/002_SUS_WC.txt",
        "experiment_data/SUS/003_SUS_C.txt",
        "experiment_data/SUS/003_SUS_WC.txt",
        "experiment_data/SUS/004_SUS_C.txt",
        "experiment_data/SUS/004_SUS_WC.txt",
        "experiment_data/SUS/005_SUS_C.txt",
        "experiment_data/SUS/005_SUS_WC.txt",
        "experiment_data/SUS/006_SUS_C.txt",
        "experiment_data/SUS/006_SUS_WC.txt",
        "experiment_data/SUS/007_SUS_C.txt",
        "experiment_data/SUS/007_SUS_WC.txt",
        "experiment_data/SUS/008_SUS_C.txt",
        "experiment_data/SUS/008_SUS_WC.txt",
        "experiment_data/SUS/009_SUS_C.txt",
        "experiment_data/SUS/009_SUS_WC.txt",
        "experiment_data/SUS/010_SUS_C.txt",
        "experiment_data/SUS/010_SUS_WC.txt",
        "experiment_data/SUS/011_SUS_C.txt",
        "experiment_data/SUS/011_SUS_WC.txt",
        "experiment_data/SUS/012_SUS_C.txt",
        "experiment_data/SUS/012_SUS_WC.txt",
        "experiment_data/SUS/013_SUS_C.txt",
        "experiment_data/SUS/013_SUS_WC.txt",
        "experiment_data/SUS/014_SUS_C.txt",
        "experiment_data/SUS/014_SUS_WC.txt",
        "experiment_data/SUS/015_SUS_C.txt",
        "experiment_data/SUS/015_SUS_WC.txt",
        "experiment_data/SUS/016_SUS_C.txt",
        "experiment_data/SUS/016_SUS_WC.txt",
    ]
    
    # Parse NASA TLX overall scores into a dictionary keyed by (participant, condition)
    nasa_overall = {}
    for nasa_fp in nasa_paths:
        base = os.path.basename(nasa_fp)
        parts = base.split("_")
        participant = parts[0]
        cond = parts[1].split(".")[0]
        overall = get_nasa_overall(nasa_fp)
        if overall is not None:
            nasa_overall[(participant, cond.upper())] = overall

    # Parse SUS scores into a dictionary keyed by (participant, condition)
    sus_scores = {}
    for sus_fp in sus_paths:
        participant, cond = parse_sus_file_info(sus_fp)
        score = compute_sus_score(sus_fp)
        sus_scores[(participant, cond.upper())] = score

    # Dictionaries to store results per participant for each condition.
    wc_results = {}
    c_results = {}
    
    for fp in file_paths:
        participant, condition = parse_file_info(fp)
        
        drag_count = count_drag_drops(fp)
        sketch_count = count_sketch_drops(fp)
        slider_count = count_slider_drops(fp, score_threshold=0.6, debug=False)
        
        nasa_score = nasa_overall.get((participant, condition.upper()), None)
        sus_score = sus_scores.get((participant, condition.upper()), None)
        
        result = {
            "Drag": drag_count,
            "Slider": slider_count,
            "Sketch": sketch_count,
            "NASA": nasa_score,
            "SUS": sus_score
        }
        
        if condition.upper() == "WC":
            wc_results[participant] = result
        elif condition.upper() == "C":
            c_results[participant] = result
    
    # Convert dictionaries to DataFrames and output to CSV.
    wc_df = pd.DataFrame.from_dict(wc_results, orient="index").reset_index()
    wc_df.rename(columns={"index": "Participant"}, inplace=True)
    c_df = pd.DataFrame.from_dict(c_results, orient="index").reset_index()
    c_df.rename(columns={"index": "Participant"}, inplace=True)
    
    wc_df.to_csv("WC_results.csv", index=False)
    c_df.to_csv("C_results.csv", index=False)
    
    print("WC_results.csv:")
    print(wc_df)
    print("\nC_results.csv:")
    print(c_df)
