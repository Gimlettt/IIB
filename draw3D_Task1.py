import os
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import re


# Input: list of file paths and manually provided task start indices， because the task start indices are not provided in the row data
data_files_drag = [
    ('001', 'Row_data/001/DEPTH0_2024-09-18_16-22-01/bodyPose.csv', 6767),
    ('002', 'Row_data/002/DEPTH0_2024-09-18_17-23-27/bodyPose.csv', 12364),
    ('003', 'Row_data/003/DEPTH0_2024-09-18_18-36-28/bodyPose.csv', 6023),
    ('004', 'Row_data/004/DEPTH0_2024-09-18_19-11-26/bodyPose.csv', 8197),
    ('005', 'Row_data/005/DEPTH0_2024-09-18_19-41-02/bodyPose.csv', 16538),
    ('006', 'Row_data/006/DEPTH0_2024-09-18_20-49-22/bodyPose.csv', 16993),
    ('007', 'Row_data/007/DEPTH0_2024-09-18_21-48-17/bodyPose.csv', 11485),
    ('008', 'Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv', 4773),
]
def animate_trajectory_with_slider(data, x_col, y_col, z_col,subset_start,subset_end):

    # Clean column names by stripping the leading whitespace
    data.columns = data.columns.str.strip()
    # Fill the first row with 0 if it contains NaN
    data.iloc[0] = data.iloc[0].fillna(0)

    # Forward fill remaining NaN values with the previous value
    data = data.ffill()

    #specify the rows to slice the dataframe
    subset = data.iloc[subset_start:subset_end+100]

    # Get the data for the animation
    x = subset[x_col].values
    y = subset[y_col].values
    z = subset[z_col].values

    # Check if there are any valid frames to animate
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No valid data to animate. All NaN values.")
        return

    # Extract the time column and any event messages
    time = subset.iloc[:, 0].values  # time is the first column, in milisecondes

    events = data[data.iloc[:, 1].str.contains("EVENT:", na=False)].iloc[:, 1]  # filter the data with "EVENT:" string and keep these rows, than keep only the second column
    #note that events is now a series with the row number as index and the event as value
    #print(events[44206]) #this is a example of how to access the event message at a specific row number
    event_indices = events.index

    # Identify "SELECTED" and "RELEASED" event indices for slider task
    selected_indices = events[events.str.contains("SELECTED")].index
    released_indices = events[events.str.contains("RELEASED")].index



    # Set up the figure and 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize line plot for trajectory
    line, = ax.plot([], [], [], 'o-', color='b')
    ax.set_xlim(-0.6, 0.6)#the unit is meter
    ax.set_ylim(1.2, 2)
    ax.set_zlim(-1, 1)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f'Animating {x_col}, {y_col}, {z_col} Trajectory')

    # calibrate_x =-0.1453
    # calibrate_y = 1.48
    # calibrate_z = -0.05

    #apply another function here, update the calibarte head position from EVENT content
    def get_calibration_position(events):
        calibration_event = events[events.str.contains("CALIBRATION HEADPOS")].iloc[-1]  # Get the last calibration event
        match = re.search(r'CALIBRATION HEADPOS \((-?\d+\.\d+); (-?\d+\.\d+); (-?\d+\.\d+)\)', calibration_event)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        else:
            print("Calibration event not found or malformed.")
            return None, None, None

    calibrate_x, calibrate_y, calibrate_z = get_calibration_position(events)
    # Display a static target plane
    plane_x = calibrate_x + 0.1
    plane_y = np.linspace(calibrate_y - 0.25, calibrate_y + 0.25, 10)
    plane_z = np.full_like(plane_y, calibrate_z + 0.55)
    X, Y = np.meshgrid(np.linspace(plane_x - 0.25, plane_x + 0.25, 10), plane_y)
    Z = np.full_like(X, plane_z)
    ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

    # Initialize event message display
    event_text = fig.text(0.02, 0.4, '', transform=fig.transFigure, fontsize=10, color='red')

    # Variable to control the display of previous points
    show_trajectory = True
    # Initialize the state variable
    is_selected = False  # Starts as False, assuming "RELEASED" initially

    # Scatter plot for trajectory points
    scat = ax.scatter([], [], [], c=[], cmap='bwr', s=10)  # Initialize empty scatter plot with color map

    # Initialize line plot for single-point view
    line, = ax.plot([], [], [], 'o-', color='b')
    # Update function for the slider
    def update(val):
        nonlocal is_selected
        frame = slider.val  # integer value of the slider starting from 1
        row_number = subset_start + frame
        slider.valtext.set_text(f"Row: {row_number}")
        
        # Find the closest event index that is less than or equal to the current row_number
        past_events = event_indices[event_indices <= row_number]
        if not past_events.empty:
            latest_event_index = past_events[-1]
            event_text.set_text(f"Event: {events[latest_event_index]}")
        
        # Determine the color for each point up to the current frame
        colors = []
        is_selected = False
        for i in range(frame):
            point_row_number = subset_start + i + 1  # Adjust point row to current subset
            if point_row_number in selected_indices:
                is_selected = True
            elif point_row_number in released_indices:
                is_selected = False
            colors.append('g' if is_selected else 'b')
        
        # Update either the full trajectory or the single point based on toggle
        if show_trajectory:
            scat._offsets3d = (x[:frame], y[:frame], z[:frame])
            scat.set_color(colors)  # Apply colors to each point
            line.set_data([], [])  # Hide line when showing full trajectory
            line.set_3d_properties([])
        else:
            # Show only the current point
            line.set_data([x[frame-1]], [y[frame-1]])
            line.set_3d_properties([z[frame-1]])
            line.set_color('g' if is_selected else 'b')  # Current point color only
            scat._offsets3d = ([], [], [])  # Hide scatter when showing single point
        
        fig.canvas.draw_idle()


    # Toggle function for checkbox to show/hide trajectory
    def toggle_trajectory(label):
        nonlocal show_trajectory
        show_trajectory = not show_trajectory
        update(slider.val)  # Refresh with new display mode

    def on_key(event):
        current_val = slider.val
        if event.key == 'right':
            slider.set_val(min(current_val + 1, len(x)))
        elif event.key == 'left':
            slider.set_val(max(current_val - 1, 1))
    # Add a slider for manual frame control
    ax_slider = plt.axes([0.1, 0.02, 0.75, 0.03], facecolor='lightgoldenrodyellow')#position to put the slider
    slider = Slider(ax_slider, 'Frame', 1, len(x), valinit=1, valstep=1)#the slider value is a integer between 1 and the length of the x array, so that each frame correspond to one row
    slider.on_changed(update)

    # Add a checkbox to toggle display of previous data points
    ax_checkbox = plt.axes([0.8, 0.06, 0.15, 0.1], facecolor='lightgoldenrodyellow')# x,y,width,height relative to the figure
    checkbox = CheckButtons(ax_checkbox, ['Show Trajectory'], [True])
    checkbox.on_clicked(toggle_trajectory)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the initial frame
    update(1)

    plt.show()

def get_calibration_position(events):
    calibration_event = events[events.str.contains("CALIBRATION HEADPOS")].iloc[-1]  # Get the last calibration event
    match = re.search(r'CALIBRATION HEADPOS \((-?\d+\.\d+); (-?\d+\.\d+); (-?\d+\.\d+)\)', calibration_event)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    else:
        print("Calibration event not found or malformed.")
        return None, None, None

# Output folder
output_folder = "drag_frame"
os.makedirs(output_folder, exist_ok=True)  
# when labling intentional drop, check for subsequent t frames to see if the finger tip has deviate for thre meters,if not don't regard as intentional drop 
t = 60
thre = 0.03# when finger tip leaves the plane for 1 cm
# Loop through each file and process
for file_id, file_path, manual_start_index in data_files_drag:
    print(f"Processing file {file_id}...")
    
    # Load the data
    data_df = pd.read_csv(file_path)
    data_df.columns = data_df.columns.str.strip()#handle the leading whitespace in the column names
    #get the plane z axix position for this experiment
    events = data_df[data_df.iloc[:, 1].str.contains("EVENT:", na=False)].iloc[:, 1]  # filter the data with "EVENT:" string and keep these rows, than keep only the second column
    calibrate_x, calibrate_y, calibrate_z = get_calibration_position(events)
    plane_z = calibrate_z + 0.55
    
     
    Drag_events = data_df[data_df.iloc[:, 1].str.contains("Drag", na=False)].iloc[:, 1]#get the rows that contain the string "Drag" in the second column
    Drag_events = Drag_events[~Drag_events.str.contains("Training")]  # Filter out training data

    # Extract task start and end indices
    task_start_indices = Drag_events[Drag_events.str.contains("STARTED TASK Drag")].index
    # Add the manual start index
    task_start_indices = task_start_indices.insert(0, manual_start_index)

    task_end_indices = Drag_events[Drag_events.str.contains("FINISHED TASK Drag")].index

    released_indices = Drag_events[Drag_events.str.contains("RELEASED")].index
    completed_indices = Drag_events[Drag_events.str.contains("COMPLETED")].index

    
    labels = []

    # Loop through each task start and end range
    for start, end in zip(task_start_indices, task_end_indices):
        print(f"Processing Task from row {start} to {end+100}")#make the end index 100 rows after the end of the task for checking the subsequent movement
        # for animating and checking the drops
        # if (file_id == '001'):
        #      animate_trajectory_with_slider(data_df, 'rightHandIndexTip_pos_x', 'rightHandIndexTip_pos_y', 'rightHandIndexTip_pos_z', start, end+100)


        # Normally label "COMPLETED" indices as 1, because only when the task is completed, people will release the object intentionally
        task_released_indices = released_indices[(released_indices >= start) & (released_indices <= end)]
        task_completed_indices_task = completed_indices[(completed_indices >= start) & (completed_indices <= end)]
        # There is also circumstances that even though this task is finished, user won't intentionally leave the plane, because the cube will just disappear, participants sometimes just don't
        # leave the plane very far, so we need to check the subsequent movement to see if the user really want leave the plane
        # we check the subsequent t frames to see if the finger tip has deviate for thre meters from plane_z, if not don't regard as intentional drop    

        for row_number in task_completed_indices_task:
            subsequent_frames = data_df.iloc[row_number:row_number+t]#check the subsequent t frames
            subsequent_frames=subsequent_frames.apply(pd.to_numeric, errors='coerce').dropna()#drop the rows that contain non-numeric values
            below_threshold = subsequent_frames[subsequent_frames['rightHandIndexTip_pos_z'] < (plane_z - thre)]

            if not below_threshold.empty:
                # Get the row index of the first frame that meets the condition
                first_frame_index = below_threshold.index[0]
                # Append the first frame's row index to labels with label 1
                labels.append({"row_number": first_frame_index,"label": 1})
            # Check if the preceding row is a "RELEASED" event, sometimes a release event is immediately followed by a completion event, then this release event won't be unintentional, so we just neglect it
            if row_number - 1 in task_released_indices:
                # Remove the corresponding "RELEASED" event to avoid double-marking
                task_released_indices = task_released_indices.drop(row_number - 1, errors='ignore')

        # Automatically label remaining "RELEASED" indices as 0, as any release event is unintentional
        for row_number in task_released_indices:
            labels.append({
                "row_number": row_number,
                "label": 0
            })

    # Save the labels to a CSV file
    labels_df = pd.DataFrame(labels)
    output_file = os.path.join(output_folder, f"Drag_{file_id}.csv")
    labels_df.to_csv(output_file, index=False)
    print(f"Labels saved to {output_file}")

print(f"All files processed. Labeled files are saved in the '{output_folder}' folder.")
