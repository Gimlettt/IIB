import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from Process_Slider import process_slider_events
import re
from datetime import datetime
import threading

def animate_trajectory_with_slider(data, x_col, y_col, z_col,subset_start,subset_end):

    # Clean column names by stripping the leading whitespace
    data.columns = data.columns.str.strip()
    # Fill the first row with 0 if it contains NaN
    data.iloc[0] = data.iloc[0].fillna(0)

    # Forward fill remaining NaN values with the previous value
    data = data.ffill()

    #specify the rows to slice the dataframe
    subset = data.iloc[subset_start:subset_end]

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
    selected_indices = events[events.str.contains("STARTED")].index
    released_indices = events[events.str.contains("STOPPED")].index



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
    # Update function for the slider
    def update(val):
        nonlocal is_selected
        frame = slider.val# integer value of the slider starting from 1
        row_number = subset_start + frame
        # Set slider label to show the row number instead of frame
        slider.valtext.set_text(f"Row: {row_number}")
    # Find the closest event index that is less than or equal to the current row_number
        past_events = event_indices[event_indices <= row_number]
        if not past_events.empty:
            latest_event_index = past_events[-1]  # Get the last past event index
            event_text.set_text(f"Event: {events[latest_event_index]}")
        

        # Iterate through the events, updating the `is_selected` state based on events
        if row_number in selected_indices:
            is_selected = True
            #print("SELECTED at row", row_number)
        elif row_number in released_indices:
            is_selected = False
        color = 'g' if is_selected else 'b'
        # Update line based on toggle status
        if show_trajectory:
            line.set_data(x[:frame], y[:frame])
            line.set_3d_properties(z[:frame])
        else:
            # Only show current point
            line.set_data([x[frame-1]], [y[frame-1]])
            line.set_3d_properties([z[frame-1]])

        line.set_color(color)
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

# Load the data
file_path = '/Users/jerry/Desktop/AR/Row_data/008/DEPTH0_2024-09-19_09-41-35/bodyPose.csv'
data_df = pd.read_csv(file_path)

# subset_start = 33905
# subset_end = 34892

# Filter events that contain both "EVENT:" and "slider" in the second column
slider_events = data_df[data_df.iloc[:, 1].str.contains("DRAW")|data_df.iloc[:, 1].str.contains("Sketch")].iloc[:, 1]
slider_events = slider_events[~slider_events.str.contains("Training")]#filter out the training data
task_start_indices = slider_events[slider_events.str.contains("STARTED TASK Sketching")].index
task_end_indices = slider_events[slider_events.str.contains("FINISHED TASK Sketching")].index
released_indices = slider_events[slider_events.str.contains("STOPPED")].index
#for some reason STARTED TASK Sliders 1 is not in the event message, so we need to add it manually
task_start_indices = task_start_indices.insert(0, 20802)#for 001
# task_start_indices = task_start_indices.insert(0, 33043)#for 002
# task_start_indices = task_start_indices.insert(0, 16444)#for 003
# task_start_indices = task_start_indices.insert(0, 23755)#for 004
# task_start_indices = task_start_indices.insert(0, 33304)#for 005
# task_start_indices = task_start_indices.insert(0, 31714)#for 006
# task_start_indices = task_start_indices.insert(0, 24944)#for 007
# task_start_indices = task_start_indices.insert(0, 16588)#for 008
    # Print to see
with pd.option_context('display.max_colwidth', None):
    print("Sketching Events without training:")
    print(slider_events.to_string())

with pd.option_context('display.max_colwidth', None):
    print(task_start_indices)
    print(task_end_indices)


# NOTE that the dataframe row number is not exactly the same as excel, because the first row in the dataframe is 0, while in excel it is already 2

# # Do the animation with a slider, toggle button, and event message display
# animate_trajectory_with_slider(data_df, 'rightHandIndexTip_pos_x', 'rightHandIndexTip_pos_y', 'rightHandIndexTip_pos_z',subset_start,subset_end)
lables= []
index_score = 0


for start, end in zip(task_start_indices, task_end_indices):
    print(f"Animating Task from row {start} to {end+100}")# to further see the hand movement after the task is finished
    
    # Run the animation for the task
    animate_trajectory_with_slider(data_df, 'rightHandIndexTip_pos_x', 'rightHandIndexTip_pos_y', 'rightHandIndexTip_pos_z', start, end+100)
    
    # After animation, prompt for labels for "RELEASED" events in this task
    task_released_indices = released_indices[(released_indices >= start) & (released_indices <= end)]
    for row_number in task_released_indices:

        # Prompt for label
        label = input(f"Label for RELEASED at row {row_number} (1 or 0): ")
        lables.append({
    "row_number": row_number,
    "label": label,

})

# Get the current date and time for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"slider_task_labels_{timestamp}.csv"

print(lables)

# Save the labels to a CSV file after all tasks are labeled
labels_df = pd.DataFrame(lables)
labels_df.to_csv(filename, index=False)
print(f"Labels saved to {filename}")
