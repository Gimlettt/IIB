import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
#####
# This code is for creating the df for the inference, the data frame will contain velocity for each drop

# Note: time and position units in the data are in seconds and meters, respectively

# Function 1: Extract rows with specified row number, note we adjust the position of the drop frame within the window in this function
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


# Function 2: Calculate velocity for the drop event
def calculate_velocity(extracted_data, characteristic_name):
    extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()
    # extracted_data is just sliced row data
    initial_value = extracted_data[characteristic_name].iloc[0]
    final_value = extracted_data[characteristic_name].iloc[-1]
    initial_time = extracted_data.iloc[0, 0]  # Assuming time is in the first column
    final_time = extracted_data.iloc[-1, 0]
    
    # Calculate the overall Z velocity
    overall_velocity = (final_value - initial_value) / (final_time - initial_time)
    return overall_velocity


# extract rows from row data using the labelled data, this version is slightly different from the one used in Velocity.py, this one also include task and row info
# each drop corresponds to a row in the labelled data, and one csv file is created for each drop
def collect_velocities(label_data, data, folder_name, characteristic_name):
    velocities = []

    for _, row in label_data.iterrows():
        row_number = row['row_number']
        label = row['label']  # 1 for intended, 0 for unintended

        # Put the extracted data (for one drop) in a folder, specify the file path
        output_file = os.path.join(folder_name, f"extracted_data_{Task}__{row_number}.csv")

        # Extract rows and save to the specified file path
        extracted_data = extract_rows(row_number, output_file=output_file)

        # Calculate a single Z-axis velocity for the extracted data
        z_velocity = calculate_velocity(extracted_data, characteristic_name)
        z_velocity = z_velocity  # Convert to cm/s

        # Append velocity with associated metadata
        velocities.append({
            'z_velocity_cm_s': z_velocity,
            'label': label,
            'row_number': row_number
        })

        print(f"Z-axis velocity for row {row_number} (label {label}): {z_velocity}")

    return velocities





# Load the CSV data (replace 'data.csv' and 'score_data.csv' with your file paths)
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
# Set the window size for velocity calculation.
WINDOW_SIZE = 9
characteristic_name = ' rightHandIndexTip_pos_z'

tasks = ['drag', 'slider', 'sketching']


data_files_mapping = {
    'slider': data_files_slider,
    'drag': data_files_drag,
    'sketching': data_files_sketching 
}

# Initialize the aggregated data list
aggregated_data = []

# Aggregation Step with additional file_id and row_number
# Aggregation Step
for Task_name in tasks:
    data_files = data_files_mapping[Task_name]

    # For each task, save all the files into one folder
    parent_output_dir = f"extracted_datafile_{Task_name}"
    os.makedirs(parent_output_dir, exist_ok=True)  # Create the directory

    # Main: Loop through each pair of data and label files
    for file_id, data_path, label_path in data_files:
        # Load the data and label CSVs
        data = pd.read_csv(data_path)
        label_data = pd.read_csv(label_path)

        # Set task and folder for this pair of files
        Task = f'{Task_name}_{file_id}'
        output_dir = os.path.join(parent_output_dir, f"extracted_{Task}")  # Subfolder for each task
        os.makedirs(output_dir, exist_ok=True)  # Create the directory

        # Collect velocities for this person
        velocities = collect_velocities(label_data, data, output_dir, characteristic_name)

        # Append all velocities to the aggregated data with task and file_id
        for velocity_info in velocities:
            aggregated_data.append({
                'velocity_z': velocity_info['z_velocity_cm_s'],
                'label': velocity_info['label'],
                'task': Task_name,
                'file_id': file_id,
                'row_number': velocity_info['row_number']
            })
# After processing all tasks and files, create a DataFrame from aggregated_data
final_df = pd.DataFrame(aggregated_data)

# Display the first few rows to verify
print(final_df.head())

# Save the final DataFrame to a CSV file
final_csv_path = "final_aggregated_velocity_data.csv"
final_df.to_csv(final_csv_path, index=False)
print(f"Final aggregated data saved to {final_csv_path}")
