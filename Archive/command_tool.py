import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import argparse

def animate_trajectory_with_slider(data, x_col, y_col, z_col, subset_start, subset_end):
    # Clean column names by stripping the leading whitespace
    data.columns = data.columns.str.strip()
    data.iloc[0] = data.iloc[0].fillna(0)
    data = data.ffill()

    # Specify the rows to slice the dataframe
    subset = data.iloc[subset_start:subset_end]

    # Get the data for the animation
    x = subset[x_col].values
    y = subset[y_col].values
    z = subset[z_col].values

    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("No valid data to animate. All NaN values.")
        return

    time = subset.iloc[:, 0].values  # time is the first column, in milliseconds

    events = data[data.iloc[:, 1].str.contains("EVENT:", na=False)].iloc[:, 1]
    event_indices = events.index

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], 'o-', color='b')
    ax.set_xlim(0, 0.6)
    ax.set_ylim(1.2, 2)
    ax.set_zlim(-1, 1)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f'Animating {x_col}, {y_col}, {z_col} Trajectory')

    calibrate_x, calibrate_y, calibrate_z = 0.1372, 1.608, -0.076

    plane_x = calibrate_x + 0.1
    plane_y = np.linspace(calibrate_y - 0.25, calibrate_y + 0.25, 10)
    plane_z = np.full_like(plane_y, calibrate_z + 0.55)
    X, Y = np.meshgrid(np.linspace(plane_x - 0.25, plane_x + 0.25, 10), plane_y)
    Z = np.full_like(X, plane_z)
    ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

    event_text = fig.text(0.5, 0.9, '', ha='center', transform=fig.transFigure, fontsize=12, color='red')
    show_trajectory = True

    def update(val):
        frame = int(slider.val)
        row_number = subset_start + frame

        if row_number in event_indices:
            event_text.set_text(f"Event: {events[row_number]}")
        else:
            event_text.set_text("")

        if show_trajectory:
            line.set_data(x[:frame], y[:frame])
            line.set_3d_properties(z[:frame])
        else:
            line.set_data([x[frame-1]], [y[frame-1]])
            line.set_3d_properties([z[frame-1]])

        fig.canvas.draw_idle()

    def toggle_trajectory(label):
        nonlocal show_trajectory
        show_trajectory = not show_trajectory
        update(slider.val)

    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 1, len(x), valinit=1, valstep=1)
    slider.on_changed(update)

    ax_checkbox = plt.axes([0.8, 0.02, 0.15, 0.1], facecolor='lightgoldenrodyellow')
    checkbox = CheckButtons(ax_checkbox, ['Show Trajectory'], [True])
    checkbox.on_clicked(toggle_trajectory)

    update(1)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate 3D trajectory with events.")
    parser.add_argument("file", type=str, help="Path to the CSV file")
    parser.add_argument("subset_start", type=int, help="Starting row of the subset")
    parser.add_argument("subset_end", type=int, help="Ending row of the subset")

    args = parser.parse_args()

    file_path = args.file
    subset_start = args.subset_start
    subset_end = args.subset_end

    data_df = pd.read_csv(file_path)
    animate_trajectory_with_slider(data_df, 'rightHandIndexTip_pos_x', 'rightHandIndexTip_pos_y', 'rightHandIndexTip_pos_z', subset_start, subset_end)
