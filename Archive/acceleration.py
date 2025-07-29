import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil

# Note: time and position units in the data are in seconds and meters, respectively

# Function 1: Extract rows with specified row number and ratio
def extract_rows(row_number, window_size, ratio_before, ratio_after, data, output_file="extracted_rows.csv"):
    """
    Extract a window of rows around the specified row_number based on the given ratio.
    
    Parameters:
    - row_number (int): The index of the central row.
    - window_size (int): Total number of rows to extract.
    - ratio_before (float): Ratio of the window before the row_number.
    - ratio_after (float): Ratio of the window after the row_number.
    - data (pd.DataFrame): The DataFrame containing the data.
    - output_file (str): Path to save the extracted data.
    
    Returns:
    - extracted_data (pd.DataFrame): The extracted subset of data.
    """
    # Calculate number of rows before and after based on the ratio
    rows_before = int(window_size * ratio_before)
    rows_after = window_size - rows_before

    # Ensure the adjusted row number is within the DataFrame range
    start = max(0, row_number - rows_before)
    end = min(len(data), row_number + rows_after)

    # Extract rows within the specified range and drop non-numeric data
    extracted_data = data.iloc[start:end].apply(pd.to_numeric, errors='coerce').dropna()

    # Ensure enough rows are extracted
    desired_rows = window_size
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

# Function 2: Calculate acceleration for the drop event
def calculate_acceleration(extracted_data, characteristic_name):
    """
    Calculate acceleration using the finite difference method.
    
    Parameters:
    - extracted_data (pd.DataFrame): The extracted subset of data.
    - characteristic_name (str): The column name for the position data.
    
    Returns:
    - acceleration (float): The average acceleration in cm/s² over the window.
    """
    extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Extract position and time
    positions = extracted_data[characteristic_name].values  # in meters
    times = extracted_data.iloc[:, 0].values  # assuming first column is time in seconds
    
    # Compute velocities using first-order differences
    velocities = np.diff(positions) / np.diff(times)  # in m/s
    
    # Compute accelerations using first-order differences of velocities
    accelerations = np.diff(velocities) / np.diff(times[:-1])  # in m/s²
    
    # Average acceleration over the window
    avg_acceleration = np.mean(accelerations) if len(accelerations) > 0 else np.nan
    
    # Convert to cm/s²
    avg_acceleration_cm_s2 = avg_acceleration * 100  # m/s² to cm/s²
    
    return np.absolute(avg_acceleration_cm_s2)
# Function 3: Collect accelerations based on labels and ratio
def collect_accelerations(label_data, data, folder_name, characteristic_name, window_size, ratio_before, ratio_after):
    """
    Collect acceleration data for each labeled drop event.
    
    Parameters:
    - label_data (pd.DataFrame): DataFrame containing labels and row numbers.
    - data (pd.DataFrame): The main data DataFrame.
    - folder_name (str): Directory to save extracted data files.
    - characteristic_name (str): The column name for the position data.
    - window_size (int): Total number of rows to extract around each event.
    - ratio_before (float): Ratio of the window before the event.
    - ratio_after (float): Ratio of the window after the event.
    
    Returns:
    - accelerations (dict): Dictionary with lists of 'intended' and 'unintended' accelerations.
    """
    accelerations = {'intended': [], 'unintended': []}
    
    for _, row in label_data.iterrows():
        row_number = row['row_number']
        label = row['label']  # 1 for intended, 0 for unintended
        # Specify the file path with ratio information
        output_file = os.path.join(folder_name, f"extracted_data_{Task}__{row_number}.csv")
        
        # Extract rows and save to the specified file path
        extracted_data = extract_rows(row_number, window_size, ratio_before, ratio_after, data, output_file=output_file)
        
        # Calculate a single acceleration for the extracted data
        acceleration = calculate_acceleration(extracted_data, characteristic_name)
        
        # Append the single acceleration to the respective list
        if label == 1:
            accelerations['intended'].append(acceleration)
        else:
            accelerations['unintended'].append(acceleration)
        print(f"Acceleration for row {row_number} (label {label}): {acceleration:.2f} cm/s²")
    
    return accelerations

# Plotting functions adjusted for acceleration
def plot_per_participant(all_accelerations, participant_count=8, task_name="", window_size=9):
    """
    Plot acceleration data per participant.
    
    Parameters:
    - all_accelerations (list): List of tuples containing participant labels and their accelerations.
    - participant_count (int): Number of participants.
    - task_name (str): Name of the task for the plot title.
    - window_size (int): Window size used for extraction.
    """
    participant_labels = [f"P{i + 1}" for i in range(participant_count)]  

    # Plot the box plots
    plt.figure(figsize=(14, 8))  # Larger figure size
    boxprops = dict(linewidth=2)
    flierprops = dict(marker='o', color='gray', alpha=0.7)  # Gray fliers
    medianprops = dict(color='black', linewidth=2)  # Black median lines

    # Use improved colors
    intended_color = "#1f77b4"  # Soft blue for intended
    unintended_color = "#ff7f0e"  # Soft orange for unintended

    # Define spacing for wider boxes
    box_width = 0.6  # Wider boxes
    spacing = 3.0  # Increased spacing between groups

    # Create boxplots and calculate midpoints
    midpoints = []  # To store the x positions for participant labels
    for i in range(participant_count):
        # Plot intended box
        intended_idx = i * 2
        plt.boxplot(all_accelerations[intended_idx][1], positions=[i * spacing + 1], widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=intended_color, edgecolor='black'),
                    medianprops=medianprops,
                    flierprops=flierprops,
                    whis=3.0)

        # Plot unintended box
        unintended_idx = i * 2 + 1
        plt.boxplot(all_accelerations[unintended_idx][1], positions=[i * spacing + 2], widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=unintended_color, edgecolor='black'),
                    medianprops=medianprops,
                    flierprops=flierprops,
                    whis=3.0)

        # Calculate midpoint for x-axis label
        midpoints.append(i * spacing + 1.5)

    # Set participant labels centered between boxes
    plt.xticks(
        ticks=midpoints,  # Use calculated midpoints
        labels=participant_labels,
        fontsize=12
    )

    # Add labels and title
    plt.xlabel('Participant', fontsize=14)
    plt.ylabel('Z-Axis Acceleration (cm/s²)', fontsize=14)
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.title(f'Z-Axis Acceleration in {task_name} across participants', fontsize=16)

    # Add improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)  # Change legend location to lower right

    plt.show()

def plot_combined_drops(all_intended, all_unintended, task_name=""):
    """
    Plot combined acceleration data for intended and unintended drops.
    
    Parameters:
    - all_intended (list): List of accelerations for intended drops.
    - all_unintended (list): List of accelerations for unintended drops.
    - task_name (str): Name of the task for the plot title.
    """
    plt.figure(figsize=(10, 8))  # Increase figure size for better readability

    # Create the boxplot with patch_artist=True for box colors
    box = plt.boxplot(
        [all_intended, all_unintended],
        patch_artist=True,  # Enables custom coloring of boxes
        labels=['Intended', 'Unintended'],  # Labels for the boxes
        medianprops=dict(color='black', linewidth=2)  # Black median line for better contrast
    )

    # Improved colors
    intended_color = "#1f77b4"  # Soft blue for intended
    unintended_color = "#ff7f0e"  # Soft orange for unintended
    colors = [intended_color, unintended_color]

    # Apply colors to the boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)  # Apply the face color

    # Add labels and title
    plt.xlabel('Drop Type', fontsize=14)
    plt.ylabel('Z-Axis Acceleration (cm/s²)', fontsize=14)
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.title(f'Z-Axis Acceleration in {task_name} for All Participants', fontsize=16)

    # Add an improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)  # Change legend location to lower right

    plt.show()

def plot_mean_z_acceleration(all_accelerations, participant_count=8, task_name=""):
    """
    Plot the mean Z-axis acceleration for each participant.
    
    Parameters:
    - all_accelerations (list): List of tuples containing participant labels and their accelerations.
    - participant_count (int): Number of participants.
    - task_name (str): Name of the task for the plot title.
    """
    # Calculate mean Z acceleration for each participant
    intended_means = []
    unintended_means = []

    for i in range(participant_count):
        # Extract intended and unintended accelerations for each participant
        intended = all_accelerations[i * 2][1]  # Even indices for intended (_1)
        unintended = all_accelerations[i * 2 + 1][1]  # Odd indices for unintended (_0)

        # Debugging: Print the unintended accelerations
        print(f"Participant {i + 1} - Unintended: {unintended}")

        # Compute mean accelerations, handle missing data
        intended_means.append(np.mean(intended) if intended else np.nan)  # Use NaN for missing data
        unintended_means.append(np.mean(unintended) if unintended else np.nan)  # Use NaN for missing data

    # Debugging: Print the calculated means
    print("Unintended Means:", unintended_means)

    # Filter out NaN values for plotting
    filtered_intended_means = [x for x in intended_means if not np.isnan(x)]
    filtered_unintended_means = [x for x in unintended_means if not np.isnan(x)]

    # Create the box plot
    plt.figure(figsize=(10, 8))
    box = plt.boxplot(
        [filtered_intended_means, filtered_unintended_means],
        patch_artist=True,  # Enables custom coloring of boxes
        labels=['Intended', 'Unintended'],  # Labels for the boxes
        medianprops=dict(color='black', linewidth=2)  # Black median line for better contrast
    )

    # Improved colors
    intended_color = "#1f77b4"  # Soft blue for intended
    unintended_color = "#ff7f0e"  # Soft orange for unintended
    colors = [intended_color, unintended_color]

    # Apply colors to the boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)  # Apply the face color

    # Add labels and title
    plt.xlabel('Drop Type', fontsize=14)
    plt.ylabel('Z-Axis Mean Acceleration (cm/s²)', fontsize=14)
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.title(f'Mean Z-Axis Acceleration for Each Participant in {task_name}', fontsize=16)

    # Add an improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)

    plt.show()

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

# Task configuration: Change to 'slider', 'drag', or 'sketching' as needed
#Task_name = 'drag'
Task_name = 'slider'
#Task_name = 'sketching'

data_files_mapping = {
    'slider': data_files_slider,
    'drag': data_files_drag,
    'sketching': data_files_sketching 
}

data_files = data_files_mapping.get(Task_name, [])

# Set the window size for acceleration calculation.
WINDOW_SIZE = 9
characteristic_name = ' rightHandIndexTip_pos_z'

# Initialize the data structure for overall box plot
all_accelerations = []
all_intended = []
all_unintended = []

# Create a parent output directory for extracted data files
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
    
    # Define ratio for row extraction (fixed as 0.25 before and 0.75 after)
    ratio_before = 0.25
    ratio_after = 0.75
    
    # Collect accelerations for this person
    accelerations = collect_accelerations(label_data, data, output_dir, characteristic_name, WINDOW_SIZE, ratio_before, ratio_after)
    
    # Append accelerations to overall list with labeled IDs
    all_accelerations.append((f"{file_id}_1", accelerations['intended']))
    all_accelerations.append((f"{file_id}_0", accelerations['unintended']))
    
    # Aggregate all intended and unintended accelerations across all files, for overall box plot
    all_intended.extend(accelerations['intended'])
    all_unintended.extend(accelerations['unintended'])

# Call function to plot per participant
plot_per_participant(all_accelerations, participant_count=8, task_name=Task_name, window_size=WINDOW_SIZE)

# Call function to plot combined drops
plot_combined_drops(all_intended, all_unintended, task_name=Task_name)

# Call function to plot mean acceleration
plot_mean_z_acceleration(all_accelerations, participant_count=8, task_name=Task_name)
