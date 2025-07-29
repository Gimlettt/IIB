import pandas as pd
import matplotlib.pyplot as plt
import os
# the first code for velocity calculation
#note the time and position units in the data is in s and m respectively

# Function 1: Extract rows with specified row number
def extract_rows(row_number, output_file="extracted_rows.csv"):
    # Adjust row_number to account for header row and 0-based indexing in DataFrame, because we both load the 2 files data frame, so the row number matches
    adjusted_row_number = row_number  

    # Ensure the adjusted row number is within the DataFrame range
    start = max(0, adjusted_row_number - WINDOW_SIZE)
    end = min(len(data), adjusted_row_number + WINDOW_SIZE)
    
    # Extract rows within the specified range
    extracted_data = data.iloc[start:end]
    
    # Drop rows with any non-numeric data
    extracted_data = extracted_data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Save to CSV file
    extracted_data.to_csv(output_file, index=False)
    
    return extracted_data

# Function 2: Calculate velocity for the drop event
def calculate_velocity(extracted_data,characteristic_name):
    # Get the Z positions and times for the first and last rows in the extracted data
    initial_value = extracted_data[characteristic_name].iloc[0]#extracted_data[characteristic_name] will be a collumn vector and we take its first element
    
    final_value = extracted_data[characteristic_name].iloc[-1]

    initial_time = extracted_data.iloc[0, 0]  # Assuming time is in the first column

    final_time = extracted_data.iloc[-1, 0]
    
    # Calculate the overall Z velocity
    overall_velocity = (final_value - initial_value) / (final_time - initial_time)
    return overall_velocity

# Function 3: Box plot of velocities for unintended and intended drops, also save the extracted data to a file, each drop event has its own file
def plot_z_velocity(score_data):
    velocities = {'intended': [], 'unintended': []}
    
    for _, row in score_data.iterrows():
        row_number = row['row_number']
        label = row['label']  # 1 for intended, 0 for unintended
        # Specify the file path within the output directory
        output_file = os.path.join(output_dir, f"extracted_data_{Task}__{row_number}.csv")
        
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
        print(f"Z-axis velocity for row {row_number} (label {label}):")
        print(z_velocity)
    
    # Box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([velocities['intended'], velocities['unintended']], labels=['Intended', 'Unintended'])
    plt.xlabel('Drop Type')
    plt.ylabel('Z-Axis Velocity (cm/s)')
    plt.title(f'Z-Axis Velocity Comparison between Intended and Unintended Drops with Window size {WINDOW_SIZE}')
    plt.show()


# Load the CSV data (replace 'data.csv' and 'score_data.csv' with your file paths)
data_001 = pd.read_csv('Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv')
data_002 = pd.read_csv('Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv')
data_003 = pd.read_csv('Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv')
data_004 = pd.read_csv('Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv')
data_005 = pd.read_csv('Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv')
data_006 = pd.read_csv('Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv')
data_007 = pd.read_csv('Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv')
data_008 = pd.read_csv('Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv')

data = data_006


label_data_001 = pd.read_csv('slider_frame/001_slider.csv')
label_data_002 = pd.read_csv('slider_frame/002_slider.csv')
label_data_003 = pd.read_csv('slider_frame/003_slider.csv')
label_data_004 = pd.read_csv('slider_frame/004_slider.csv')
label_data_005 = pd.read_csv('slider_frame/005_slider.csv')
label_data_006 = pd.read_csv('slider_frame/006_slider.csv')
label_data_007 = pd.read_csv('slider_frame/007_slider.csv')
label_data_008 = pd.read_csv('slider_frame/008_slider.csv')

label_data_006 = pd.read_csv('sketch_frame/Sketching_label_006.csv')

Task = 'slider_001'
# Set the window size once
WINDOW_SIZE = 60


# Create a directory to store extracted files
output_dir = f"extracted_data_files_{Task}"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist



characteristic_name =' rightHandIndexTip_pos_z'

plot_z_velocity(label_data_006)