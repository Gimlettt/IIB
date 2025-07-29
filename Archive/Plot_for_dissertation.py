import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil

# ──────────────────────────────
# Configuration
# ──────────────────────────────
WINDOW_SIZE = 9
CHAR_COL = ' rightHandIndexTip_pos_z'

TASK_ORDER = ['drag', 'slider', 'sketching']



characteristic_name = ' rightHandIndexTip_pos_z'
DATA_FILES = {
    'slider': [
        ('1', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 'slider_frame/001_slider.csv'),
        ('2', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 'slider_frame/002_slider.csv'),
        ('3', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 'slider_frame/003_slider.csv'),
        ('4', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 'slider_frame/004_slider.csv'),
        ('5', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 'slider_frame/005_slider.csv'),
        ('6', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 'slider_frame/006_slider.csv'),
        ('7', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 'slider_frame/007_slider.csv'),
        ('8', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 'slider_frame/008_slider.csv')
    ],
    'drag': [
        ('1', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 'drag_frame/Drag_001.csv'),
        ('2', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 'drag_frame/Drag_002.csv'),
        ('3', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 'drag_frame/Drag_003.csv'),
        ('4', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 'drag_frame/Drag_004.csv'),
        ('5', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 'drag_frame/Drag_005.csv'),
        ('6', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 'drag_frame/Drag_006.csv'),
        ('7', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 'drag_frame/Drag_007.csv'),
        ('8', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 'drag_frame/Drag_008.csv')
    ],
    'sketching': [
        ('1', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 'sketch_frame/Sketching_label_001.csv'),
        ('2', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 'sketch_frame/Sketching_label_002.csv'),
        ('3', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 'sketch_frame/Sketching_label_003.csv'),
        ('4', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 'sketch_frame/Sketching_label_004.csv'),
        ('5', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 'sketch_frame/Sketching_label_005.csv'),
        ('6', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 'sketch_frame/Sketching_label_006.csv'),
        ('7', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 'sketch_frame/Sketching_label_007.csv'),
        ('8', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 'sketch_frame/Sketching_label_008.csv')
    ]
}

def extract_rows(data: pd.DataFrame, row_num: int) -> pd.DataFrame:
    """Extract a WINDOW_SIZE slice centered (25% before, 75% after) around row_num."""
    start = max(0, row_num - int(WINDOW_SIZE * 0.25))
    end   = min(len(data), row_num + int(WINDOW_SIZE * 0.75))
    df = data.iloc[start:end].apply(pd.to_numeric, errors='coerce').dropna()
    # If too few, append further rows
    while len(df) < WINDOW_SIZE:
        more_end = min(len(data), end + (WINDOW_SIZE - len(df)))
        df = pd.concat([df, data.iloc[end:more_end].apply(pd.to_numeric, errors='coerce').dropna()])
        if end == more_end:
            break
        end = more_end
    return df.head(WINDOW_SIZE)

def get_calibration_position(events: pd.Series):
    """Find last CALIBRATION HEADPOS and parse (x,y,z)."""
    last = events[events.str.contains("CALIBRATION HEADPOS", na=False)].iloc[-1]
    m = re.search(r'\((-?[\d\.]+); (-?[\d\.]+); (-?[\d\.]+)\)', last)
    if not m:
        raise ValueError("Malformed calibration event")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

def calculate_max_deviation(df: pd.DataFrame, plane_z: float) -> float:
    """Return max |z - plane_z| * 100 (cm)."""
    dev = np.abs(df[CHAR_COL] - plane_z)
    return dev.max() * 100

def collect_deviations(task: str, data: pd.DataFrame, labels: pd.DataFrame, plane_z: float):
    out = {'intended': [], 'unintended': []}
    for _, r in labels.iterrows():
        slice_df = extract_rows(data, int(r['row_number']))
        dev = calculate_max_deviation(slice_df, plane_z)
        key = 'intended' if r['label']==1 else 'unintended'
        out[key].append(dev)
    return out

def plot_combined_deviation(all_deviations, save_path):
    TASK_ORDER = ['drag', 'slider', 'sketching']
    SHORT = {'sketching':'sket','slider':'slid','drag':'drag'}

    plt.figure(figsize=(14, 8))
    labels, data = [], []

    # iterate in the order you specify
    for task in TASK_ORDER:
        vals = all_deviations[task]
        short = SHORT[task]
        labels += [f"{short}_I", f"{short}_U"]
        data   += [vals['intended'], vals['unintended']]

    bp = plt.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color='black', linewidth=2)
    )

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor('#1f77b4' if i % 2 == 0 else '#ff7f0e')

    plt.ylabel('Deviation (cm)', fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=28)
    plt.grid(axis='y')
    plt.ylim(0, 20)
    plt.savefig(save_path)
    plt.close()

# ──────────────────────────────
# Main
# ──────────────────────────────
if __name__ == "__main__":
    all_deviations = {}

    for task, files in DATA_FILES.items():
        devs = {'intended': [], 'unintended': []}
        tmpdir = f"tmp_{task}"
        os.makedirs(tmpdir, exist_ok=True)

        for pid, data_path, label_path in files:
            # 1) load
            df       = pd.read_csv(data_path)
            labels   = pd.read_csv(label_path)
            # 2) find calibration Z
            events   = df[df.iloc[:,1].astype(str).str.contains("EVENT:", na=False)].iloc[:,1]
            _,_,cz  = get_calibration_position(events)
            plane_z  = cz + 0.55
            # 3) collect deviations
            part = collect_deviations(task, df, labels, plane_z)
            devs['intended']  += part['intended']
            devs['unintended']+= part['unintended']

        shutil.rmtree(tmpdir)
        all_deviations[task] = devs

    plot_combined_deviation(all_deviations, f"Window_{WINDOW_SIZE}_deviation.png")
    print("Saved deviation plot to Window_{}_deviation.png".format(WINDOW_SIZE))







