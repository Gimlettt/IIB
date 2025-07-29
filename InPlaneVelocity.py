import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Note: time and position units in the data are in seconds and meters, respectively

# Function 1: Extract the rows within one window for certain drop frame, and save them in one csv file, note we adjust the position of the drop frame within the window in this function
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
def calculate_velocity(extracted_data, characteristic_name_x,characteristic_name_y):
    extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()
    # extracted_data is just sliced row data
    dx = extracted_data[' rightHandIndexTip_pos_x'].iloc[-1] - extracted_data[' rightHandIndexTip_pos_x'].iloc[0]
    dy = extracted_data[' rightHandIndexTip_pos_y'].iloc[-1] - extracted_data[' rightHandIndexTip_pos_y'].iloc[0]

    initial_time = extracted_data.iloc[0, 0]  # Assuming time is in the first column
    final_time = extracted_data.iloc[-1, 0]
    
    # Calculate the overall Z velocity
    overall_velocity = np.sqrt(dx**2 + dy**2) / (final_time - initial_time)
    return overall_velocity


# extract rows from row data using the labelled data, and save for box plot
# each drop corresponds to a row in the labelled data, and one csv file is created for each drop
def collect_velocities(label_data, data, folder_name):
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
        velocity = calculate_velocity(extracted_data, characteristic_name_x, characteristic_name_y)
        velocity = velocity * 100  # Convert to cm/s
        
        # Append the single velocity to the respective list
        if label == 1:
            velocities['intended'].append(velocity)
        else:
            velocities['unintended'].append(velocity)
        print(f"Z-axis velocity for row {row_number} (label {label}): {velocity}")
    
    return velocities

def plot_per_participant(all_velocities, participant_count=8, task_name="", window_size=12):
    participant_labels = [f"P{i + 1}" for i in range(participant_count)]  

    # Plot the box plots
    plt.figure(figsize=(14, 8))  # Larger figure size
    boxprops = dict(linewidth=2)
    flierprops = dict(marker='o', color='gray', alpha=0.7)  # Gray fliers
    medianprops = dict(color='black', linewidth=2)  # Black median lines

    # Use improved colors
    intended_color = "#1f77b4"  # Soft blue for intended
    unintended_color = "#ff7f0e"  # Soft orange for unintended

    colors = [intended_color if '_1' in label else unintended_color for label, _ in all_velocities]

    # Define spacing for wider boxes
    box_width = 0.6  # Wider boxes
    spacing = 3.0  # Increased spacing between groups

    # Create boxplots and calculate midpoints
    midpoints = []  # To store the x positions for participant labels
    for i in range(participant_count):
        # Plot intended box
        intended_idx = i * 2
        plt.boxplot(all_velocities[intended_idx][1], positions=[i * spacing + 1], widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=intended_color, edgecolor='black'),
                    medianprops=medianprops,
                    flierprops=flierprops,
                    whis=3.0)

        # Plot unintended box
        unintended_idx = i * 2 + 1
        plt.boxplot(all_velocities[unintended_idx][1], positions=[i * spacing + 2], widths=box_width,
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
        fontsize=36
    )

    # Add labels and title
    plt.ylabel('In Plane Velocity (cm/s)', fontsize=36)
    plt.yticks(fontsize=28)  # Increase y-tick font size
    plt.grid(axis='y')  # Add horizontal grid lines
    #plt.title(f'Z-Axis Velocity in Task {task_name} across participants', fontsize=36)

    # Add improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower left', fontsize=24)  # Change legend location to lower right

    plt.show()
def plot_combined_drops(all_intended, all_unintended, task_name=""):
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
    plt.ylabel('Z-Axis Velocity (cm/s)', fontsize=14)
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.title(f'In-Plane Velocity in {task_name} for All Participants', fontsize=16)

    # Add an improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)  # Change legend location to lower right

    plt.show()
def plot_mean_z_velocity(all_velocities, participant_count=8, task_name=""):


    # Calculate mean Z velocity for each participant
    intended_means = []
    unintended_means = []

    for i in range(participant_count):
        # Extract intended and unintended velocities for each participant
        intended = all_velocities[i * 2][1]  # Even indices for intended (_1)
        unintended = all_velocities[i * 2 + 1][1]  # Odd indices for unintended (_0)

        # Debugging: Print the unintended velocities
        print(f"Participant {i + 1} - Unintended: {unintended}")

        # Compute mean velocities, handle missing data
        intended_means.append(np.mean(intended) if intended else np.nan)  # Use NaN for missing data
        unintended_means.append(np.mean(unintended) if unintended else np.nan)  # Use NaN for missing data

    # Debugging: Print the calculated means
    print("Unintended Means:", unintended_means)

    # Filter out NaN values for plotting
    filtered_intended_means = [x for x in intended_means if not np.isnan(x)]
    filtered_unintended_means = [x for x in unintended_means if not np.isnan(x)]

    # Create the box plot
    plt.figure(figsize=(14, 8))
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
    plt.ylabel('Z-Axis Mean Velocity (cm/s)', fontsize=28)
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.title(f'Mean In-Plane Velocity for each participants in {task_name}', fontsize=16)

    # Add an improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower left', fontsize=24)

    plt.show()





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

Task_name = 'drag'
#Task_name = 'slider'
#Task_name = 'sketching'
data_files_mapping = {
    'slider': data_files_slider,
    'drag': data_files_drag,
    'sketching': data_files_sketching 
}

data_files = data_files_mapping.get(Task_name, [])


# Set the window size for velocity calculation.
WINDOW_SIZE = 9
characteristic_name_x = ' rightHandIndexTip_pos_x'
characteristic_name_y = ' rightHandIndexTip_pos_y'


# Initialize the data structure for overall box plot
all_velocities = []
all_intended = []
all_unintended = []

#for each save, save all the file into one folder
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
    velocities = collect_velocities(label_data, data, output_dir)
    
    # Append velocities to overall list with labeled IDs
    all_velocities.append((f"{file_id}_1", velocities['intended']))
    all_velocities.append((f"{file_id}_0", velocities['unintended']))
    
    # Aggregate all intended and unintended velocities across all files, for overall box plot
    all_intended.extend(velocities['intended'])
    all_unintended.extend(velocities['unintended'])

# Call function to plot per participant
plot_per_participant(all_velocities, participant_count=8, task_name=Task_name, window_size=WINDOW_SIZE)

# Call function to plot combined drops
plot_combined_drops(all_intended, all_unintended, task_name=Task_name)
plot_mean_z_velocity(all_velocities, participant_count=8, task_name=Task_name)