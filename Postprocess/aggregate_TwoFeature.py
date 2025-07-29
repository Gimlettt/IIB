import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re
#####
# This code is for creating the df for the inference, the data frame will contain 2 features,velocity and the second one.

# Note: Time and position units in the data are in seconds and meters, respectively.

# Function 1: Extract rows with specified row number, adjusting the position of the drop frame within the window.
def extract_rows(row_number, output_file="extracted_rows.csv"):
    # Ensure the adjusted row number is within the DataFrame range
    start = max(0, row_number - int(WINDOW_SIZE * 0.25))
    end = min(len(data), row_number + int(WINDOW_SIZE * 0.75))

    # Extract rows within the specified range and drop non-numeric data
    extracted_data = data.iloc[start:end].apply(pd.to_numeric, errors='coerce').dropna()

    # Ensure enough rows are extracted
    desired_rows = WINDOW_SIZE
    while len(extracted_data) < desired_rows:
        missing_rows = desired_rows - len(extracted_data)
        
        # Add subsequent rows to fill the gap
        subsequent_start = end
        subsequent_end = min(len(data), end + missing_rows)
        subsequent_rows = data.iloc[subsequent_start:subsequent_end].apply(pd.to_numeric, errors='coerce').dropna()
        
        # Update extracted_data and end position
        extracted_data = pd.concat([extracted_data, subsequent_rows])
        end = subsequent_end
        
        # Break the loop if no more rows are available
        if subsequent_start >= len(data):
            print(f"Warning: Unable to extract {desired_rows} rows. Only {len(extracted_data)} rows available.")
            break

    # Trim to desired number of rows if we have extra
    extracted_data = extracted_data.head(desired_rows)

    # Save to CSV file
    extracted_data.to_csv(output_file, index=False)

    return extracted_data

# Function 2: Calculate velocity for the drop event (UNCHANGED)
def calculate_velocity_z(extracted_data):
    # Convert columns to numeric and drop rows with NaN values
    extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()

    # Ensure there are at least two data points to compute velocity
    if len(extracted_data) < 2:
        print("Not enough data points to compute velocity.")
        return np.nan

    # Compute the time differences (assuming the first column is time)
    time_diff = extracted_data.iloc[:, 0].diff()
    # Compute the differences in z-position (using the specified column name)
    z_diff = extracted_data[' rightHandIndexTip_pos_z'].diff()

    # Compute velocity for each consecutive pair (in m/s)
    velocities = z_diff / time_diff

    # Drop the first row which is NaN because of the diff() operation
    velocities = velocities.iloc[1:]

    # Calculate the average velocity
    average_velocity = velocities.mean()

    # Convert average velocity to cm/s (1 m/s = 100 cm/s)
    average_velocity_cm_s = average_velocity * 100

    return average_velocity_cm_s


# # Function 3: Calculate acceleration for the drop event
# def calculate_velocity_xy(extracted_data):
#     extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()
#     # extracted_data is just sliced row data
#     dx = extracted_data[' rightHandIndexTip_pos_x'].iloc[-1] - extracted_data[' rightHandIndexTip_pos_x'].iloc[0]
#     dy = extracted_data[' rightHandIndexTip_pos_y'].iloc[-1] - extracted_data[' rightHandIndexTip_pos_y'].iloc[0]

#     initial_time = extracted_data.iloc[0, 0]  # Assuming time is in the first column
#     final_time = extracted_data.iloc[-1, 0]
    
#     # Calculate the overall Z velocity
#     overall_velocity = np.sqrt(dx**2 + dy**2) / (final_time - initial_time)
#     return overall_velocity*100  # Convert to cm/s

def calculate_max_deviation_from_plane(extracted_data, plane_z):
    extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Compute deviation from the plane for each frame
    deviations = np.abs(extracted_data[' rightHandIndexTip_pos_z'] - plane_z)
    
    # Return the maximum deviation in the window
    return np.max(deviations)


def collect_metrics(label_data, data, folder_name, characteristic_one, characteristic_two, plane_z):
    metrics = {
        f'intended_{characteristic_one}': [],
        f'unintended_{characteristic_one}': [],
        f'intended_{characteristic_two}': [],
        f'unintended_{characteristic_two}': [],
        'intended_row_numbers': [],
        'unintended_row_numbers': []
    }
    
    for _, row in label_data.iterrows():
        row_number = row['row_number']
        label = row['label']  # 1 for intended, 0 for unintended
        output_file = os.path.join(folder_name, f"extracted_data_{Task}__{row_number}.csv")
        extracted_data = extract_rows(row_number, output_file=output_file)
        z_velocity = calculate_velocity_z(extracted_data)
        feature_two = calculate_max_deviation_from_plane(extracted_data, plane_z)
        
        if label == 1:
            metrics[f'intended_{characteristic_one}'].append(z_velocity)
            metrics[f'intended_{characteristic_two}'].append(feature_two)
            metrics['intended_row_numbers'].append(row_number)
        else:
            metrics[f'unintended_{characteristic_one}'].append(z_velocity)
            metrics[f'unintended_{characteristic_two}'].append(feature_two)
            metrics['unintended_row_numbers'].append(row_number)
    
    return metrics

def get_calibration_position(events):
    calibration_event = events[events.str.contains("CALIBRATION HEADPOS")].iloc[-1]  # Get the last calibration event
    match = re.search(r'CALIBRATION HEADPOS \((-?\d+\.\d+); (-?\d+\.\d+); (-?\d+\.\d+)\)', calibration_event)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    else:
        print("Calibration event not found or malformed.")
        return None, None, None
# Load the CSV data (replace with your actual file paths)
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


# Set the window size for velocity and acceleration calculation.
WINDOW_SIZE = 9  # Keeping the original window size for velocity calculation


characteristic_one = 'velocity_z'
#characteristic_two = 'velocity_xy'
characteristic_two = 'deviation'


# Initialize the aggregated data list
aggregated_data = []

for Task_name in tasks:
    data_files = data_files_mapping[Task_name]
    all_intended = []    # List of all intended velocities
    all_unintended = []  # List of all unintended velocities

    # Create a parent output directory for extracted data files
    parent_output_dir = f"extracted_datafile_{Task_name}"
    os.makedirs(parent_output_dir, exist_ok=True)  # Create the directory

    # Main: Loop through each pair of data and label files
    for file_id, data_path, label_path in data_files:

        data = pd.read_csv(data_path)
        events = data[data.iloc[:, 1].str.contains("EVENT:", na=False)].iloc[:, 1]  # filter the data with "EVENT:" string and keep these rows, than keep only the second column
        calibrate_x, calibrate_y, calibrate_z = get_calibration_position(events)
        plane_z = calibrate_z + 0.55

        label_data = pd.read_csv(label_path)

        
        # Set task and folder for this pair of files
        Task = f'{Task_name}_{file_id}'
        output_dir = os.path.join(parent_output_dir, f"extracted_{Task}")  # Subfolder for each task
        os.makedirs(output_dir, exist_ok=True)  # Create the directory
        
        # Collect velocities and accelerations for this person
        metrics = collect_metrics(label_data, data, output_dir, characteristic_one, characteristic_two,plane_z)
        
        
        # Aggregate all intended and unintended velocities and accelerations across all files, for overall box plot
        # For each intended velocity and acceleration, append a dict
        for vel, acc, rn in zip(metrics[f'intended_{characteristic_one}'],
                                  metrics[f'intended_{characteristic_two}'],
                                  metrics['intended_row_numbers']):
            aggregated_data.append({
                f'{characteristic_one}': vel,
                f'{characteristic_two}': acc,
                'label': 1,
                'task': Task_name,
                'file_id': file_id,
                'row_number': rn
            })
        
        # Append unintended drops
        for vel, acc, rn in zip(metrics[f'unintended_{characteristic_one}'],
                                  metrics[f'unintended_{characteristic_two}'],
                                  metrics['unintended_row_numbers']):
            aggregated_data.append({
                f'{characteristic_one}': vel,
                f'{characteristic_two}': acc,
                'label': 0,
                'task': Task_name,
                'file_id': file_id,
                'row_number': rn
            })

# After processing all tasks and files, create a DataFrame from aggregated_data
final_df = pd.DataFrame(aggregated_data)

# Display the first few rows to verify
print(final_df.head())

# Save the final DataFrame to a CSV file
final_csv_path = "final_aggregated_data_TwoFeature.csv"
final_df.to_csv(final_csv_path, index=False)
print(f"Final aggregated data saved to {final_csv_path}")

