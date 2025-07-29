import pandas as pd
import numpy as np
import os
import re

WINDOW_SIZE = 9  # number of frames per window

def extract_window(row_number, data):
    """
    Mirrors your original extract_rows logic:
      - 25% of WINDOW_SIZE before the drop frame,
      - 75% after,
      - drop non-numeric rows,
      - if too few rows, append subsequent rows until WINDOW_SIZE,
      - trim any extras.
    Returns a DataFrame of shape (WINDOW_SIZE, num_numeric_columns).
    """
    total = len(data)
    # 25% before, 75% after
    before = int(WINDOW_SIZE * 0.25)
    after  = WINDOW_SIZE - before

    start = max(0, row_number - before)
    end   = min(total, row_number + after)

    # pull numeric rows
    df_win = data.iloc[start:end].apply(pd.to_numeric, errors='coerce').dropna()
    
    # if too short, append subsequent rows
    desired = WINDOW_SIZE
    while len(df_win) < desired and end < total:
        need = desired - len(df_win)
        nxt_end = min(total, end + need)
        more = data.iloc[end:nxt_end].apply(pd.to_numeric, errors='coerce').dropna()
        if more.empty:
            break
        df_win = pd.concat([df_win, more], ignore_index=True)
        end = nxt_end

    # finally, trim or (rarely) pad by repeating last row
    if len(df_win) > desired:
        df_win = df_win.iloc[:desired]
    elif len(df_win) < desired:
        last = df_win.iloc[[-1]]
        pad_count = desired - len(df_win)
        df_win = pd.concat([df_win, pd.concat([last]*pad_count, ignore_index=True)],
                           ignore_index=True)

    return df_win.reset_index(drop=True)


def get_calibration_position(events):
    calibration_event = events[events.str.contains("CALIBRATION HEADPOS")].iloc[-1]  # Get the last calibration event
    match = re.search(r'CALIBRATION HEADPOS \((-?\d+\.\d+); (-?\d+\.\d+); (-?\d+\.\d+)\)', calibration_event)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    else:
        print("Calibration event not found or malformed.")
        return None, None, None

data_files_sketching = [
    ('1', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 'sketch_frame/Sketching_label_001.csv'),
    ('2', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 'sketch_frame/Sketching_label_002.csv'),
    ('3', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 'sketch_frame/Sketching_label_003.csv'),
    ('4', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 'sketch_frame/Sketching_label_004.csv'),
    ('5', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 'sketch_frame/Sketching_label_005.csv'),
    ('6', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 'sketch_frame/Sketching_label_006.csv'),
    ('7', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 'sketch_frame/Sketching_label_007.csv'),
    ('8', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 'sketch_frame/Sketching_label_008.csv')
]
data_files_slider = [
    ('1', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 'slider_frame/001_slider.csv'),
    ('2', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 'slider_frame/002_slider.csv'),
    ('3', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 'slider_frame/003_slider.csv'),
    ('4', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 'slider_frame/004_slider.csv'),
    ('5', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 'slider_frame/005_slider.csv'),
    ('6', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 'slider_frame/006_slider.csv'),
    ('7', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 'slider_frame/007_slider.csv'),
    ('8', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 'slider_frame/008_slider.csv')
]
data_files_drag = [
    ('1', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 'drag_frame/Drag_001.csv'),
    ('2', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 'drag_frame/Drag_002.csv'),
    ('3', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 'drag_frame/Drag_003.csv'),
    ('4', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 'drag_frame/Drag_004.csv'),
    ('5', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 'drag_frame/Drag_005.csv'),
    ('6', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 'drag_frame/Drag_006.csv'),
    ('7', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 'drag_frame/Drag_007.csv'),
    ('8', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 'drag_frame/Drag_008.csv')
]
# Define tasks
tasks = ['drag', 'slider', 'sketching']
data_files_mapping = {
    'slider': data_files_slider,
    'drag': data_files_drag,
    'sketching': data_files_sketching 
}

all_windows = []
all_labels  = []
all_meta    = []  # optional: store (task, file_id, row_number)

for task_name, file_list in data_files_mapping.items():
    for file_id, data_path, label_path in file_list:
        # --- load raw CSV and labels ---
        data = pd.read_csv(data_path)
        events = data.iloc[:,1].astype(str)
        _, _, plane_z = get_calibration_position(events)
        
        labels_df = pd.read_csv(label_path)  # must have columns 'row_number' and 'label'
        
        for _, row in labels_df.iterrows():
            rn    = int(row['row_number'])
            lab   = int(row['label'])
            window_df = extract_window(rn, data)
            
            # keep only numeric columns (time + all coords)
            #feat = window_df.to_numpy(dtype=np.float32)  # shape (9, F)
            # keep just the rightHandIndexTip_pos_z column, shape (9, 1)
            # feat = window_df[[' rightHandIndexTip_pos_z']].to_numpy(dtype=np.float32) 
            #feat = window_df[[' rightHandIndexTip_pos_x',' rightHandIndexTip_pos_y',' rightHandIndexTip_pos_z']].to_numpy(dtype=np.float32) # shape (9, 3)
            feat = window_df[
    [
        " rightHandIndexTip_pos_x",
        " rightHandIndexTip_pos_y",
        " rightHandIndexTip_pos_z",
        " rightArmUpper_pos_x",
        " rightArmUpper_pos_y",
        " rightArmUpper_pos_z",
        " rightArmUpperTwist1_pos_x",
        " rightArmUpperTwist1_pos_y",
        " rightArmUpperTwist1_pos_z",
        " rightArmUpperTwist2_pos_x",
        " rightArmUpperTwist2_pos_y",
        " rightArmUpperTwist2_pos_z",
        " rightArmUpperTwist3_pos_x",
        " rightArmUpperTwist3_pos_y",
        " rightArmUpperTwist3_pos_z",
        " rightArmLower_pos_x",
        " rightArmLower_pos_y",
        " rightArmLower_pos_z",
        " rightArmLowerTwist1_pos_x",
        " rightArmLowerTwist1_pos_y",
        " rightArmLowerTwist1_pos_z",
        " rightArmLowerTwist2_pos_x",
        " rightArmLowerTwist2_pos_y",
        " rightArmLowerTwist2_pos_z",
        " rightArmLowerTwist3_pos_x",
        " rightArmLowerTwist3_pos_y",
        " rightArmLowerTwist3_pos_z",
        " rightHandWrist_pos_x",
        " rightHandWrist_pos_y",
        " rightHandWrist_pos_z"
    ]
].to_numpy(dtype=np.float32)  # shape (9, 30)

            all_windows.append(feat)
            all_labels.append(lab)
            all_meta.append((task_name, file_id, rn))

# stack into arrays
X = np.stack(all_windows, axis=0)     # shape (N, 9, F)
y = np.array(all_labels, dtype=np.int64)  # shape (N,)

# (optional) metadata as structured array or Pandas
meta_df = pd.DataFrame(all_meta, columns=['task','file_id','row_number'])

# save everything
os.makedirs('dataset', exist_ok=True)
np.savez('dataset/transformer_dataset_arm.npz', X=X, y=y)
meta_df.to_csv('dataset/transformer_metadata_arm.csv', index=False)

print(f"Saved {X.shape[0]} samples; each sample is {X.shape[1]}×{X.shape[2]}")
print(" → transformer_dataset.npz (X, y)")
print(" → transformer_metadata.csv (task, file_id, row_number)")
