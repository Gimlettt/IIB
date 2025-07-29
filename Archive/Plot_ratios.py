import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import shutil
####This code is used to find the best ratio that could be used for the window for velocity calculation
# Function 1: Extract rows with specified row number and ratio
def extract_rows(row_number, window_size, ratio_before, ratio_after, data, output_file="extracted_rows.csv"):
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

# Function 3: Collect velocities based on labels and ratio
def collect_velocities(label_data, data, folder_name, characteristic_name, window_size, ratio_before, ratio_after):
    velocities = {'intended': [], 'unintended': []}
    
    for _, row in label_data.iterrows():
        row_number = row['row_number']
        label = row['label']  # 1 for intended, 0 for unintended
        # Specify the file path with ratio information
        output_file = os.path.join(folder_name, f"extracted_data_{Task}__{row_number}.csv")
        # Extract rows and save to the specified file path
        extracted_data = extract_rows(row_number, window_size, ratio_before, ratio_after, data, output_file=output_file)
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

# Plotting functions (unchanged)
def plot_per_participant(all_velocities, participant_count, task_name, window_size, save_path):
    plt.figure(figsize=(14, 8))
    intended_color, unintended_color = "#1f77b4", "#ff7f0e"
    spacing, box_width = 3.0, 0.6
    midpoints = []
    for i in range(participant_count):
        plt.boxplot(all_velocities[i * 2][1], positions=[i * spacing + 1], widths=box_width,
                    patch_artist=True, boxprops=dict(facecolor=intended_color),
                    medianprops=dict(color='black', linewidth=2), whis=3.0)
        plt.boxplot(all_velocities[i * 2 + 1][1], positions=[i * spacing + 2], widths=box_width,
                    patch_artist=True, boxprops=dict(facecolor=unintended_color),
                    medianprops=dict(color='black', linewidth=2), whis=3.0)
        midpoints.append(i * spacing + 1.5)
    plt.xticks(midpoints, [f"P{i + 1}" for i in range(participant_count)], fontsize=12)
    plt.xlabel('Participant')
    plt.ylabel('Z-Axis Velocity (cm/s)')
    plt.title(f'Z-Axis Velocity in {task_name} across participants (Window: {window_size})')
    
    # Add improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)  # Change legend location to lower right
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.ylim(-150, 20)  # Set y-axis limits
    plt.savefig(save_path)
    plt.close()

def plot_combined_drops(all_intended, all_unintended, task_name, save_path):
    plt.figure(figsize=(10, 8))
    intended_color, unintended_color = "#1f77b4", "#ff7f0e"

    box = plt.boxplot([all_intended, all_unintended], patch_artist=True, labels=['Intended', 'Unintended'],
                      medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(box['boxes'], ["#1f77b4", "#ff7f0e"]):
        patch.set_facecolor(color)
    plt.xlabel('Drop Type')
    plt.ylabel('Z-Axis Velocity (cm/s)')
    plt.title(f'Z-Axis Velocity in {task_name} for All Participants (Window: 9)')
    
    # Add improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)  # Change legend location to lower right
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.ylim(-150, 20)  # Set y-axis limits
    plt.savefig(save_path)
    plt.close()

def plot_mean_z_velocity(all_velocities, participant_count, task_name, save_path):
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
    plt.ylabel('Z-Axis Mean Velocity (cm/s)', fontsize=14)
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.ylim(-150, 20)  # Set y-axis limits
    plt.title(f'Mean Z-Axis Velocity for each participant in {task_name} (Window: 9)', fontsize=16)

    # Add an improved legend
    legend_handles = [
        plt.Line2D([0], [0], color=intended_color, lw=4, label='Intended'),
        plt.Line2D([0], [0], color=unintended_color, lw=4, label='Unintended')
    ]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12)
    plt.savefig(save_path)

    plt.close()

# Function to display a row of plots for a specific ratio
def display_row(plots_output_dir, task_name, window_size, plot_type, ratio_label):
    """
    Displays one row of plots for the specified plot type and ratio across all windows, using maximum available space.
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))  # Single plot for each ratio
    plot_path = os.path.join(plots_output_dir, f"{plot_type}.png")
    if os.path.exists(plot_path):
        img = mpimg.imread(plot_path)
        axes.imshow(img)
        axes.axis('off')
        axes.set_title(f"Ratio {ratio_label}", fontsize=14)
    else:
        axes.text(0.5, 0.5, 'Plot Not Found', horizontalalignment='center', verticalalignment='center')
        axes.axis('off')

    fig.suptitle(f"{task_name.capitalize()} - {plot_type.replace('_', ' ').capitalize()} (Ratio {ratio_label})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Use all available space and leave room for the title
    plt.show()

# Load data and process
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

# Task configurations
#Task_name = 'drag'
#Task_name = 'slider'
Task_name = 'sketching'

characteristic_name = ' rightHandIndexTip_pos_z'
data_files_mapping = {
    'slider': data_files_slider,
    'drag': data_files_drag,
    'sketching': data_files_sketching 
}

data_files = data_files_mapping.get(Task_name, [])
plots_output_dir_base = f"plots_{Task_name}"
os.makedirs(plots_output_dir_base, exist_ok=True)

# Define window size and ratios to test
window_size = 9
ratios = {
    '1.0:0': (1.0, 0.0),
    '0.75:0.25': (0.75, 0.25),
    '0.5:0.5': (0.5, 0.5),
    '0.25:0.75': (0.25, 0.75)
}

for ratio_label, (ratio_before, ratio_after) in ratios.items():
    # Create a directory for each ratio
    ratio_dir = os.path.join(plots_output_dir_base, f"ratio_{ratio_label.replace(':', '_')}")
    os.makedirs(ratio_dir, exist_ok=True)
    
    # Initialize lists to collect all velocities across files
    all_velocities, all_intended, all_unintended = [], [], []
    window_dir = os.path.join(ratio_dir, f"window_{window_size}")
    os.makedirs(window_dir, exist_ok=True)
    
    for file_id, data_path, label_path in data_files:
        data = pd.read_csv(data_path)
        label_data = pd.read_csv(label_path)
        Task = f'{Task_name}_{file_id}'
        
        # Temporary directory for intermediate files
        temp_dir = os.path.join(window_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Collect velocities with the current ratio
        velocities = collect_velocities(label_data, data, temp_dir, characteristic_name, window_size, ratio_before, ratio_after)
        all_velocities.extend([(f"{file_id}_1", velocities['intended']), (f"{file_id}_0", velocities['unintended'])])
        all_intended.extend(velocities['intended'])
        all_unintended.extend(velocities['unintended'])

        # Clean up temporary CSV files
        shutil.rmtree(temp_dir)
    
    # Generate and save plots for the current ratio
    plot_per_participant(
        all_velocities, 
        participant_count=8, 
        task_name=Task_name, 
        window_size=window_size, 
        save_path=os.path.join(ratio_dir, f"per_participant.png")
    )
    plot_combined_drops(
        all_intended, 
        all_unintended, 
        Task_name, 
        os.path.join(ratio_dir, f"combined_drops.png")
    )
    plot_mean_z_velocity(
        all_velocities, 
        participant_count=8, 
        task_name=Task_name, 
        save_path=os.path.join(ratio_dir, f"mean_velocity.png")
    )

# Function to display plots for all ratios
def display_all_ratios(plots_output_dir_base, task_name, window_size, plot_type, ratios):
    """
    Displays plots for all ratios side by side for comparison.
    """
    num_ratios = len(ratios)
    fig, axes = plt.subplots(nrows=1, ncols=num_ratios, figsize=(6 * num_ratios, 8))
    fig.subplots_adjust(wspace=0.3, top=0.85)  # Adjust spacing between subplots and top margin

    for ax, (ratio_label, _) in zip(axes, ratios.items()):
        plot_path = os.path.join(plots_output_dir_base, f"ratio_{ratio_label.replace(':', '_')}", f"{plot_type}.png")
        if os.path.exists(plot_path):
            img = mpimg.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Ratio {ratio_label}", fontsize=14)
        else:
            ax.text(0.5, 0.5, 'Plot Not Found', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')

    fig.suptitle(f"{task_name.capitalize()} - {plot_type.replace('_', ' ').capitalize()} (Window: {window_size})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Use all available space and leave room for the title
    plt.show()

# Plot types to display
plot_types = ["per_participant", "combined_drops", "mean_velocity"]

# Display plots for each plot type across all ratios
for plot_type in plot_types:
    display_all_ratios(plots_output_dir_base, Task_name, window_size, plot_type, ratios)
