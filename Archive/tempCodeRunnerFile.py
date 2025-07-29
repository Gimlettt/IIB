import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import shutil

# Function 1: Extract rows with specified row number
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

# extract rows from row data using the labelled data, and save for box plot
# each drop corresponds to a row in the labelled data, and one csv file is created for each drop
def collect_velocities(label_data, data, folder_name, characteristic_name):
    velocities = {'intended': [], 'unintended': []}
    
    for _, row in label_data.iterrows():
        row_number = row['row_number']
        label = row['label']  # 1 for intended, 0 for unintended
        # put the extracted data(for one drop) in a folder, specify the file path
        output_file = os.path.join(folder_name, f"extracted_data_{Task}__{row_number}.csv")
        #folder_name is the mother folder
        
        # Extract rows and save to the specified file path
        extracted_data = extract_rows(row_number, output_file=output_file)
        # Calculate a single Z-axis velocity for the extracted data
        z_velocity = calculate_velocity(extracted_data, characteristic_name)
        z_velocity = z_velocity * 100  # Convert to cm/s
        
        # Append the single velocity to the respective list
        if label == 1:
            velocities['intended'].append(z_velocity)
        else:
            velocities['unintended'].append(z_velocity)
        print(f"Z-axis velocity for row {row_number} (label {label}): {z_velocity}")
    
    return velocities



# Main processing

WINDOW_SIZE = 9
characteristic_name = ' rightHandIndexTip_pos_z'
data_files_mapping = {
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

all_velocities = {}

# Process each task
for task_name, data_files in data_files_mapping.items():
    velocities = {'intended': [], 'unintended': []}

    for file_id, data_path, label_path in data_files:
        data = pd.read_csv(data_path)
        label_data = pd.read_csv(label_path)
        Task = f'{task_name}_{file_id}'

        # Temporary directory for intermediate files
        temp_dir = os.path.join(f"temp_{task_name}")
        os.makedirs(temp_dir, exist_ok=True)

        # Collect velocities
        task_velocities = collect_velocities(label_data, data, temp_dir, characteristic_name)
        velocities['intended'].extend(task_velocities['intended'])
        velocities['unintended'].extend(task_velocities['unintended'])

        # Clean up temporary CSV files
        shutil.rmtree(temp_dir)

    all_velocities[task_name] = velocities




def plot_combined_tasks(all_velocities, save_path):
    plt.figure(figsize=(14, 8))

    labels = []
    data = []

    # Organize data for plotting
    for task_name, velocities in all_velocities.items():
        if task_name == 'sketching':
            task_name = 'sket'
        elif task_name == 'slider':
            task_name = 'slid'
        elif task_name == 'dra':
            task_name = 'drag'
        labels.append(f"{task_name}_I")
        labels.append(f"{task_name}_U")
        data.append(velocities['intended'])
        data.append(velocities['unintended'])


    # Create a combined box plot
    box = plt.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color='black', linewidth=2)
    )

    # Assign colors based on index (odd indices blue, even indices orange)
    for i, patch in enumerate(box['boxes']):
        if i % 2 == 0:  # Even index
            patch.set_facecolor('#1f77b4')  # Blue
        else:  # Odd index
            
            patch.set_facecolor('#ff7f0e')  # Orange
    # Add improved legend
    intended_color = "#1f77b4"  # Soft blue for intended
    unintended_color = "#ff7f0e"  # Soft orange for unintended


    # Add labels and formatting
    plt.ylabel('Z-Axis Velocity (cm/s)', fontsize=36)
    plt.xticks(fontsize=36)  # Change x-tick font size
    plt.yticks(fontsize=28)  # Change y-tick font size
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.ylim(-250, 20)  # Set y-axis limits
    plt.savefig(save_path)
    plt.close()


# Plot all tasks combined into one figure
plot_combined_tasks(all_velocities, f'Window_{WINDOW_SIZE}.png')
