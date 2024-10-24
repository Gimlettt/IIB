import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def animate_trajectory_with_slider(data, x_col, y_col, z_col, num_rows=2000):

    # Clean column names by stripping the leading whitespace
    data.columns = data.columns.str.strip()

    # Remove rows with NaN or Inf values
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, z_col])

    # Look for event's start and end
    subset = data.iloc[31384:31576]

    # Check if the subset has any valid data
    if subset.empty:
        print("The specified row range does not contain any valid data.")
        return

    # Get the data for the animation
    x = subset[x_col].values
    y = subset[y_col].values
    z = subset[z_col].values

    # Check if there are any valid frames to animate
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No valid data to animate. all NaN")
        return

    # Calibrated headset position
    headset_x = 0.02893651
    headset_y = 1.605725
    headset_z = -0.03081465

    # Target plane at 10 cm right of X and 55 cm front of Z
    plane_x = headset_x + 0.1
    plane_z = headset_z + 0.55

    # Create a mesh grid for the plane (in the XY direction at constant Z)
    plane_size = 0.5  # Size of the plane in meters
    x_plane = np.linspace(plane_x - plane_size / 2, plane_x + plane_size / 2, 10)
    y_plane = np.linspace(headset_y - plane_size / 2, headset_y + plane_size / 2, 10)
    X, Y = np.meshgrid(x_plane, y_plane)  # Grid of points spanning the plane
    Z = np.full_like(X, plane_z)  # All Z values set to plane_z

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the static plane in the XY direction
    ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

    # Initialize a line for updating the animation
    line, = ax.plot([], [], [], 'o-', color='b')
    
    # Setting the axes properties
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)#height range
    ax.set_zlim(-1, 1)#depth range in meters
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f'Animating {x_col}, {y_col}, {z_col} Trajectory')

    # Update function for the slider
    def update(val):
        frame = int(slider.val)
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        fig.canvas.draw_idle()

    # Add a slider for manual frame control
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 1, len(x), valinit=1, valstep=1)

    # Update the plot when the slider is changed
    slider.on_changed(update)

    # Show the initial frame
    update(1)

    # Display the plot
    plt.legend([ 'Target Plane','Trajectory'])
    plt.show()

# Load the data
file_path = '/Users/jerry/Desktop/AR/Row_data/001/PINCH_2024-09-18_16-08-47/bodyPose.csv'
data_df = pd.read_csv(file_path)

# Do the animation with a slider
animate_trajectory_with_slider(data_df, 'rightHandIndexTip_pos_x', 'rightHandIndexTip_pos_y', 'rightHandIndexTip_pos_z')
