import matplotlib
import platform
# Set matplotlib backend to non-interactive for macOS compatibility
if platform.system() == "Darwin":  # macOS
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from rl_train.analyzer.gait_data import GaitData
import os
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import butter, filtfilt

class GaitAnalyzer:
    JOINT_NAMES = {
        'HIP': "hip",
        'KNEE': "knee",
        'ANKLE': "ankle",
        'LEFT_HIP': "left hip",
        'RIGHT_HIP': "right hip",
        'LEFT_KNEE': "left knee",
        'RIGHT_KNEE': "right knee",
        'LEFT_ANKLE': "left ankle",
        'RIGHT_ANKLE': "right ankle"
    }
    JOINT_LIMIT = {
        "HIP": (-30, 35),
        "KNEE": (-70, 5),
        "ANKLE": (-30, 25),
    }
    # larger range
    # JOINT_LIMIT = {
    #     "HIP": (-30, 50),
    #     "KNEE": (-95, 5),
    #     "ANKLE": (-30, 25),
    # }

    def __init__(self, gait_data:GaitData, segmented_ref_data:dict, show_plot:bool):
        self.gait_data = gait_data
        self.segmented_ref_data = segmented_ref_data
        self.show_plot = show_plot
        self.fig_size_multiplier = 1
        self.dpi = 300

        self.toe_off_color = "#000000"
        self.toe_off_linestyle = "--"
        self.toe_off_linewidth = 1
        self.toe_off_alpha = 0.6
        plt.ioff()  # Turn off interactive mode

    def get_gait_segment_index(self, *, is_right_foot_based: bool):
        """
        Calculate gait segment indices based on foot sensor data.

        Parameters:
        - is_right_foot_based (bool): Determines if the analysis is based on the right foot.

        Returns:
        - List of tuples: Each tuple contains (strike_index, toe_off_index, next_strike_index).
        """

        primary_char = "r" if is_right_foot_based else "l"
        secondary_char = "l" if is_right_foot_based else "r"

        result_strike_to_toe_off = []
        sensor_data = self.gait_data.series_data["sensor_data"]
        foot_threshold = 0.1
        #foot_threshold = 0.4

        primary_stance_ing = False
        primary_stance_start_idx = None
        for idx, (primary_foot, primary_toes) in enumerate(zip([v[0] for v in sensor_data[f"{primary_char}_foot"]["data"]],
                                                               [v[0] for v in sensor_data[f"{primary_char}_toes"]["data"]])):

            if idx > 0:
                prev_primary_combined = sensor_data[f"{primary_char}_foot"]["data"][idx-1][0] + sensor_data[f"{primary_char}_toes"]["data"][idx-1][0]
                curr_primary_combined = sensor_data[f"{primary_char}_foot"]["data"][idx][0] + sensor_data[f"{primary_char}_toes"]["data"][idx][0]
                prev_secondary_combined = sensor_data[f"{secondary_char}_foot"]["data"][idx-1][0] + sensor_data[f"{secondary_char}_toes"]["data"][idx-1][0]
                curr_secondary_combined = sensor_data[f"{secondary_char}_foot"]["data"][idx][0] + sensor_data[f"{secondary_char}_toes"]["data"][idx][0]
                prev_primary_foot = sensor_data[f"{primary_char}_foot"]["data"][idx-1][0]
                curr_primary_foot = sensor_data[f"{primary_char}_foot"]["data"][idx][0]
                prev_primary_toes = sensor_data[f"{primary_char}_toes"]["data"][idx-1][0]
                curr_primary_toes = sensor_data[f"{primary_char}_toes"]["data"][idx][0]
                prev_secondary_foot = sensor_data[f"{secondary_char}_foot"]["data"][idx-1][0]
                curr_secondary_foot = sensor_data[f"{secondary_char}_foot"]["data"][idx][0]
                prev_secondary_toes = sensor_data[f"{secondary_char}_toes"]["data"][idx-1][0]
                curr_secondary_toes = sensor_data[f"{secondary_char}_toes"]["data"][idx][0]

                is_primary_foot_down = prev_primary_foot < foot_threshold and curr_primary_foot >= foot_threshold
                is_primary_toe_off = prev_primary_toes > foot_threshold and curr_primary_toes <= foot_threshold
                is_primary_rising_edge = prev_primary_combined < foot_threshold and curr_primary_combined >= foot_threshold
                is_primary_falling_edge = prev_primary_combined >= foot_threshold and curr_primary_combined < foot_threshold
                is_secondary_rising_edge = prev_secondary_combined < foot_threshold and curr_secondary_combined >= foot_threshold
                is_secondary_falling_edge = prev_secondary_combined >= foot_threshold and curr_secondary_combined < foot_threshold
                if is_primary_rising_edge and is_primary_foot_down and primary_stance_start_idx is None:
                    primary_stance_ing = True
                    primary_stance_start_idx = idx
                elif is_primary_falling_edge and is_primary_toe_off and primary_stance_ing:
                    primary_stance_ing = False
                    if primary_stance_start_idx is not None:
                        result_strike_to_toe_off.append((primary_stance_start_idx, idx))
                    primary_stance_start_idx = None

        result_strike_to_strike = []
        for idx in range(len(result_strike_to_toe_off) - 1):
            result_strike_to_strike.append((result_strike_to_toe_off[idx][0], result_strike_to_toe_off[idx][1], result_strike_to_toe_off[idx + 1][0]))
        return result_strike_to_strike[1:]
    
    def get_toe_off_average(self, *, is_right_foot_based:bool):
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_right_foot_based)
        toe_off_cycles = []
        for start_idx, toe_off_idx, end_idx in gait_segment_index:
            toe_off_cycle = (toe_off_idx - start_idx) / (end_idx - start_idx)
            toe_off_cycles.append(toe_off_cycle)
        return np.mean(toe_off_cycles)

    def plot_entire_result(self, *,
                    result_dir,
                    is_right_foot_based:bool,
                    ) -> None:
        fig, axes = plt.subplots(6,2,figsize=(12 *self.fig_size_multiplier, 6 *self.fig_size_multiplier),dpi=self.dpi)
        
        joint_data = self.gait_data.series_data["joint_data"]
        
        axes[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes[0][0].plot(np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"]]))
        
        axes[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes[1][0].plot(np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"]]))
        
        axes[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes[2][0].plot(np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"]]))
        
        axes[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes[0][1].plot(np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"]]))
        
        axes[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes[1][1].plot(np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"]]))
        
        axes[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])
        axes[2][1].plot(np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"]]))

        axes[3][0].set_title("pelvis x position")
        axes[3][0].plot([v[0] for v in joint_data["pelvis_tx"]["qpos"]])
        axes[3][1].set_title("pelvis y position")
        axes[3][1].plot([v[0] for v in joint_data["pelvis_ty"]["qpos"]])

        axes[4][0].set_title("pelvis x velocity")
        axes[4][0].plot([v[0] for v in joint_data["pelvis_tx"]["qvel"]])
        if "target_data" in self.gait_data.series_data and "target_velocity" in self.gait_data.series_data["target_data"]:
            axes[4][0].plot([v[0] for v in self.gait_data.series_data["target_data"]["target_velocity"]])

        sensor_data = self.gait_data.series_data["sensor_data"]

        axes[5][0].set_title("left sensor")
        axes[5][0].plot([v[0] for v in sensor_data["l_foot"]["data"]], label="l_foot")
        axes[5][0].plot([v[0] for v in sensor_data["l_toes"]["data"]], label="l_toes")
        # axes[5][0].legend()

        axes[5][1].set_title("right sensor")
        axes[5][1].plot([v[0] for v in sensor_data["r_foot"]["data"]], label="r_foot")
        axes[5][1].plot([v[0] for v in sensor_data["r_toes"]["data"]], label="r_toes")
        axes[5][1].plot([v[0] for v in sensor_data["l_foot"]["data"]], label="l_foot", alpha=0.3)
        axes[5][1].plot([v[0] for v in sensor_data["l_toes"]["data"]], label="l_toes", alpha=0.3)
        foot_threshold = 0.1
        
        r_stance_ing = False
        r_stance_start_idx = None
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_right_foot_based)
        for ax_row in axes:
            for ax in ax_row:
                for start_idx, toe_off_idx, end_idx in gait_segment_index:
                    ax.axvspan(start_idx, toe_off_idx, color='#00ff00', alpha=0.1)

        # axes[4][1].legend()

        axes[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        postfix = "_right_based" if is_right_foot_based else "_left_based"
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,f"kinematics_data{postfix}.png"))

        if self.show_plot:
            plt.show()
        plt.close()
    
    def joint_angle_by_velocity(self, *, result_dir:str):
        ref_line_color = "#555555"
        ref_line_style = "--"
        joint_data = self.gait_data.series_data["joint_data"]

        # Joint angle
        fig, axes = plt.subplots(3,1,figsize=(4 *self.fig_size_multiplier, 3 *self.fig_size_multiplier),dpi=self.dpi)
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)

        # Plot reference joint angles in black (#000000)
        axes[0].plot(np.rad2deg(self.segmented_ref_data["q_hip_flexion_r"]), label="Reference", color=ref_line_color, linestyle=ref_line_style)
        axes[1].plot(np.rad2deg(self.segmented_ref_data["q_knee_angle_r"]), label="Reference", color=ref_line_color, linestyle=ref_line_style)
        axes[2].plot(np.rad2deg(self.segmented_ref_data["q_ankle_angle_r"]), label="Reference", color=ref_line_color, linestyle=ref_line_style)


        # Use a low-pass Butterworth filter instead of moving average
            
        def lowpass_filter(data, cutoff=2.0, fs=100.0, order=2):
            # cutoff: desired cutoff frequency of the filter, Hz
            # fs: sample rate, Hz
            # order: filter order
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y
        
        actual_speed = [v[0] for v in joint_data["pelvis_tx"]["qvel"]]
        if len(actual_speed) > 100:
            # Apply low-pass filter to smooth the speed
            # You may adjust cutoff and fs as needed for your data
            actual_speed_smooth = lowpass_filter(np.array(actual_speed), cutoff=1.0, fs=30.0, order=2)
        else:
            # If the segment is too short, just copy the original
            actual_speed_smooth = np.array(actual_speed)

        vel_min = np.floor(np.min(actual_speed_smooth) * 10) / 10  # Round down to 0.1
        vel_max = np.ceil(np.max(actual_speed_smooth) * 10) / 10   # Round up to 0.1
        vel_range = (vel_min, vel_max)


        for start_idx, toe_off_idx, end_idx in gait_segment_index:
            # ax.axvspan(start_idx, toe_off_idx, color='#00ff00', alpha=0.1)

            # Set x-axis to be normalized from 0 to 100 for each gait segment
            segment_length = end_idx - start_idx
            x_normalized = np.linspace(0, 100, segment_length)

            # Map the x-axis to 0~100 using x_normalized for each joint angle data
            hip_flexion_l_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"]][start_idx:end_idx])
            hip_flexion_r_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"]][start_idx:end_idx])
            knee_angle_l_data = np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"]][start_idx:end_idx])
            knee_angle_r_data = np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"]][start_idx:end_idx])
            ankle_angle_l_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"]][start_idx:end_idx])
            ankle_angle_r_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"]][start_idx:end_idx])
            actual_speed_smooth_segment = actual_speed_smooth[start_idx:end_idx]

            

            interp_points = 400

            def interp_array(x, y, num_points):
                x_new = np.linspace(x[0], x[-1], num_points)
                y_new = np.interp(x_new, x, y)
                return x_new, y_new

            x_hip_r, hip_flexion_r_data_interp = interp_array(x_normalized, hip_flexion_r_data, interp_points)
            x_knee_r, knee_angle_r_data_interp = interp_array(x_normalized, knee_angle_r_data, interp_points)
            x_ankle_r, ankle_angle_r_data_interp = interp_array(x_normalized, ankle_angle_r_data, interp_points)
            _, actual_speed_interp = interp_array(x_normalized, actual_speed_smooth_segment, interp_points)

            

            all_speeds = []
            for seg_start, seg_toe_off, seg_end in gait_segment_index:
                seg_speeds = [v[0] for v in joint_data["pelvis_tx"]["qvel"]][seg_start:seg_end]
                all_speeds.extend(seg_speeds)
            
            

            norm = mcolors.Normalize(vmin=vel_range[0], vmax=vel_range[1])
            from matplotlib.colors import LinearSegmentedColormap
            # Define a custom colormap: blue -> purple -> red
            custom_cmap = LinearSegmentedColormap.from_list(
                # "blue_purple_red", ["#0000ff", "#800080", "#ff0000"]
                "blue_purple_red", ["#0000ff", "#eeeeee", "#ff0000"]
            )
            cmap = custom_cmap  # blue (low) -> purple (mid) -> red (high)

            def plot_colored_line(ax, x, y, c, label=None):
                # Prepare segments for LineCollection
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(np.array(c))
                lc.set_linewidth(1)
                line = ax.add_collection(lc)
                if label:
                    # Add a dummy line for legend
                    ax.plot([], [], color=cmap(norm(np.mean(c))), label=label)
                return line

            # Plot hip flexion right with color by speed (interpolated)
            plot_colored_line(axes[0], x_hip_r, hip_flexion_r_data_interp, actual_speed_interp, label='Hip Flexion R')
            # Plot knee angle right with color by speed (interpolated)
            plot_colored_line(axes[1], x_knee_r, knee_angle_r_data_interp, actual_speed_interp, label='Knee Angle R')
            # Plot ankle angle right with color by speed (interpolated)
            plot_colored_line(axes[2], x_ankle_r, ankle_angle_r_data_interp, actual_speed_interp, label='Ankle Angle R')

        cax = inset_axes(
            axes[2], 
            width="30%",  # width: 30% of parent axes
            height="4%",  # height: 4% of parent axes
            loc='lower left',
            bbox_to_anchor=(-0.4, -0.55, 0.9, 0.9),  # position below the axis
            bbox_transform=axes[2].transAxes,
            borderpad=1
        )
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=cax, 
            orientation='horizontal'
        )
        # Set label for colorbar and move it a bit higher by adjusting labelpad
        cb.set_label("Speed (m/s)", fontsize=6, labelpad=0)  # Move label slightly upward
        cb.ax.tick_params(labelsize=6)
            
        # Sometimes, matplotlib's autoscaling can override set_ylim if called after plotting.
        # To ensure the y-limits are strictly enforced, disable autoscaling after setting limits.
        axes[0].set_ylim(self.JOINT_LIMIT['HIP'][0], self.JOINT_LIMIT['HIP'][1])
        axes[0].set_xlim(0, 100)
        # axes[0].set_title("Hip")  # Set title for Hip axis
        # axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))  # Set y-tick interval to 10 degrees for HIP
        axes[1].set_ylim(self.JOINT_LIMIT['KNEE'][0], self.JOINT_LIMIT['KNEE'][1])
        axes[1].set_xlim(0, 100)
        # axes[1].set_title("Knee")  # Set title for Knee axis
        # axes[1].yaxis.set_major_locator(plt.MultipleLocator(10))  # Set y-tick interval to 10 degrees for KNEE
        axes[2].set_ylim(self.JOINT_LIMIT['ANKLE'][0], self.JOINT_LIMIT['ANKLE'][1])
        axes[2].set_xlim(0, 100)
        # axes[2].set_title("Ankle")  # Set title for Ankle axis
        # axes[2].yaxis.set_major_locator(plt.MultipleLocator(10))  # Set y-tick interval to 10 degrees for ANKLE
        # axes[2].autoscale(enable=False, axis='y')

        axes[0].set_ylabel(self.JOINT_NAMES['HIP'], fontsize=12, rotation=0, ha='right', va='center')
        axes[0].yaxis.set_label_coords(-0.15, 0.5)
        axes[1].set_ylabel(self.JOINT_NAMES['KNEE'], fontsize=12, rotation=0, ha='right', va='center')
        axes[1].yaxis.set_label_coords(-0.15, 0.5)
        axes[2].set_ylabel(self.JOINT_NAMES['ANKLE'], fontsize=12, rotation=0, ha='right', va='center')
        axes[2].yaxis.set_label_coords(-0.15, 0.5)


        fig.tight_layout()

        
        fig.subplots_adjust(bottom=0.15) # should call this after fig.tight_layout()

        fig.savefig(os.path.join(result_dir,f"joint_angle_cmap_by_velocity.png"))

        if self.show_plot:
            plt.show()
        plt.close()


    def plot_exo_segmented_data(self, *,
                                result_dir,
                                ) -> None:
        # Plot individual segments
        fig, axes = plt.subplots(1, 2, figsize=(3.5 *self.fig_size_multiplier, 2 *self.fig_size_multiplier), dpi=self.dpi)

        exo_data_l = self.gait_data.series_data["actuator_data"]["Exo_L"]
        exo_data_r = self.gait_data.series_data["actuator_data"]["Exo_R"]
        right_gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        left_gait_segment_index = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)
        exo_data_mapped = {
            "Exo_L": [],
            "Exo_R": [],
        }

        # Draw toe-off reference lines (average cycle percentage)
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(right_gait_segment_index):
            exo_r_data = [-v[0] for v in exo_data_r["force"][start_idx:end_idx]]

            exo_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_r_data)), exo_r_data)

            exo_data_mapped["Exo_R"].append(exo_r_mapped)

            axes[1].plot(x_mapped, exo_r_mapped, label=f"Segment {idx}")

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(left_gait_segment_index):
            exo_l_data = [-v[0] for v in exo_data_l["force"][start_idx:end_idx]]

            exo_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_l_data)), exo_l_data)

            exo_data_mapped["Exo_L"].append(exo_l_mapped)

            axes[0].plot(x_mapped, exo_l_mapped, label=f"Segment {idx}")

        axes[0].set_title("Exo L")
        axes[1].set_title("Exo R")

        axes[0].axvline(toe_off_l, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)
        axes[1].axvline(toe_off_r, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

        max_ylim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        for ax in axes:
            ax.set_ylim(bottom=0, top=max_ylim)
            ax.set_xlim(0, 100)  # limit x-axis to data range
            

        # for ax in axes:
        #     ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, "exo_segmented_force.png"))
        if self.show_plot:
            plt.show()
        plt.close()

        # Plot mean and std separately
        fig_mean_std, axes_mean_std = plt.subplots(1, 2, figsize=(3.5 *self.fig_size_multiplier, 2 *self.fig_size_multiplier), dpi=self.dpi)

        for exo, data in exo_data_mapped.items():
            data = np.array(data)
            mean_data = np.mean(data, axis=0)
            std_data = np.std(data, axis=0)

            if exo == "Exo_L":
                ax = axes_mean_std[0]
            else:
                ax = axes_mean_std[1]

            ax.plot(x_mapped, mean_data, label=f"{exo} Mean", color='black', linewidth=2)
            ax.fill_between(x_mapped, mean_data - std_data, mean_data + std_data, color='gray', alpha=0.5)

        axes_mean_std[0].set_title("Exo L")
        axes_mean_std[1].set_title("Exo R")

        axes_mean_std[0].axvline(toe_off_l, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)
        axes_mean_std[1].axvline(toe_off_r, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

        max_ylim_mean_std = max(axes_mean_std[0].get_ylim()[1], axes_mean_std[1].get_ylim()[1])
        for ax in axes_mean_std:
            ax.set_ylim(bottom=0, top=max_ylim_mean_std)
            ax.set_xlim(0, 100)  # limit x-axis to data range
            

        # for ax in axes_mean_std:
        #     ax.legend()

        fig_mean_std.tight_layout()
        fig_mean_std.savefig(os.path.join(result_dir, "exo_mean_std_data.png"))
        if self.show_plot:
            plt.show()
        plt.close()
    def plot_segmented_kinematics_result(self, *,
                    result_dir,
                    ) -> None:
        fig, axes = plt.subplots(3,2,figsize=(5 *self.fig_size_multiplier, 3 *self.fig_size_multiplier),dpi=self.dpi)

        # Draw toe-off reference lines (average cycle percentage)
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100
        
        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        gait_segment_index_l = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped_degree = {
            "hip_flexion_l": [],
            "hip_flexion_r": [],
            "knee_angle_l": [],
            "knee_angle_r": [],
            "ankle_angle_l": [],
            "ankle_angle_r": [],
        }

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_r):
            hip_flexion_r_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]])
            knee_angle_r_data = np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]])
            ankle_angle_r_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]])

            hip_flexion_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data)
            knee_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data)
            ankle_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data)

            joint_data_mapped_degree["hip_flexion_r"].append(hip_flexion_r_mapped)
            joint_data_mapped_degree["knee_angle_r"].append(knee_angle_r_mapped)
            joint_data_mapped_degree["ankle_angle_r"].append(ankle_angle_r_mapped)

            axes[0][1].plot(x_mapped, hip_flexion_r_mapped, label=f"Segment {idx}")
            axes[1][1].plot(x_mapped, knee_angle_r_mapped, label=f"Segment {idx}")
            axes[2][1].plot(x_mapped, ankle_angle_r_mapped, label=f"Segment {idx}")

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_l):
            hip_flexion_l_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"][start_idx:end_idx]])
            knee_angle_l_data = np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"][start_idx:end_idx]])
            ankle_angle_l_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"][start_idx:end_idx]])

            hip_flexion_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_l_data)), hip_flexion_l_data)
            knee_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_l_data)), knee_angle_l_data)
            ankle_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_l_data)), ankle_angle_l_data)

            joint_data_mapped_degree["hip_flexion_l"].append(hip_flexion_l_mapped)
            joint_data_mapped_degree["knee_angle_l"].append(knee_angle_l_mapped)
            joint_data_mapped_degree["ankle_angle_l"].append(ankle_angle_l_mapped)

            axes[0][0].plot(x_mapped, hip_flexion_l_mapped, label=f"Segment {idx}")
            axes[1][0].plot(x_mapped, knee_angle_l_mapped, label=f"Segment {idx}")
            axes[2][0].plot(x_mapped, ankle_angle_l_mapped, label=f"Segment {idx}")
        for joint, data in joint_data_mapped_degree.items():
            data = np.array(data)
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)

            if "hip_flexion_l" in joint:
                ax = axes[0][0]
            elif "hip_flexion_r" in joint:
                ax = axes[0][1]
            elif "knee_angle_l" in joint:
                ax = axes[1][0]
            elif "knee_angle_r" in joint:
                ax = axes[1][1]
            elif "ankle_angle_l" in joint:
                ax = axes[2][0]
            elif "ankle_angle_r" in joint:
                ax = axes[2][1]

            ax.plot(x_mapped, mean_data_degree, label=f"{joint} mean", color='black')
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color='gray', alpha=0.5)

        axes[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])

        axes[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        for idx in range(3):
            axes[idx][0].axvline(toe_off_l, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)
            axes[idx][1].axvline(toe_off_r, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(0, 100)  # limit x-axis to data range

        

        # for ax_row in axes:
        #     for ax in ax_row:
        #         ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"segmented_joint_data.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()
        plt.close()

        # Average data

        fig2, axes2 = plt.subplots(3,2,figsize=(5 *self.fig_size_multiplier, 3 *self.fig_size_multiplier),dpi=self.dpi)

        for joint, data in joint_data_mapped_degree.items():
            data = np.array(data)
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)

            if "hip_flexion_l" in joint:
                ax = axes2[0][0]
            elif "hip_flexion_r" in joint:
                ax = axes2[0][1]
            elif "knee_angle_l" in joint:
                ax = axes2[1][0]
            elif "knee_angle_r" in joint:
                ax = axes2[1][1]
            elif "ankle_angle_l" in joint:
                ax = axes2[2][0]
            elif "ankle_angle_r" in joint:
                ax = axes2[2][1]

            ax.plot(x_mapped, mean_data_degree, label=f"{joint} mean", color='black')
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color='gray', alpha=0.5)

        axes2[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes2[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes2[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes2[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes2[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes2[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])

        axes2[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes2[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes2[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes2[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes2[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes2[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        for ax_row in axes2:
            for ax in ax_row:
                ax.set_xlim(0, 100)  # limit x-axis to data range

        for idx in range(3):
            axes2[idx][0].axvline(toe_off_l, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)
            axes2[idx][1].axvline(toe_off_r, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

        fig2.tight_layout()
        fig2.savefig(os.path.join(result_dir,"segmented_joint_data_avg.png"))
        if self.show_plot:
            plt.figure(fig2.number)
            plt.show()
        plt.close()
    def plot_left_right_comparison(self, *,
                    result_dir,
                    ) -> None:
        
        # Draw toe-off reference lines (average cycle percentage)
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100


        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        gait_segment_index_l = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped = {
            "hip_flexion_l": [],
            "hip_flexion_r": [],
            "knee_angle_l": [],
            "knee_angle_r": [],
            "ankle_angle_l": [],
            "ankle_angle_r": [],
        }
        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_r):
            hip_flexion_r_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]])
            knee_angle_r_data = np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]])
            ankle_angle_r_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]])

            hip_flexion_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data)
            knee_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data)
            ankle_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data)

            joint_data_mapped["hip_flexion_r"].append(hip_flexion_r_mapped)
            joint_data_mapped["knee_angle_r"].append(knee_angle_r_mapped)
            joint_data_mapped["ankle_angle_r"].append(ankle_angle_r_mapped)

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_l):
            hip_flexion_l_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"][start_idx:end_idx]])
            knee_angle_l_data = np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"][start_idx:end_idx]])
            ankle_angle_l_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"][start_idx:end_idx]])

            hip_flexion_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_l_data)), hip_flexion_l_data)
            knee_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_l_data)), knee_angle_l_data)
            ankle_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_l_data)), ankle_angle_l_data)

            joint_data_mapped["hip_flexion_l"].append(hip_flexion_l_mapped)
            joint_data_mapped["knee_angle_l"].append(knee_angle_l_mapped)
            joint_data_mapped["ankle_angle_l"].append(ankle_angle_l_mapped)


        fig2, axes2 = plt.subplots(3,1,figsize=(4 *self.fig_size_multiplier, 3 *self.fig_size_multiplier),dpi=self.dpi)
        for joint, data in joint_data_mapped.items():
            data = np.array(data)
            # print(f"DEBUG:: {joint=}, {data.shape=}")
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)
            if "hip_flexion_l" in joint:
                ax = axes2[0]
            elif "hip_flexion_r" in joint:
                ax = axes2[0]
            elif "knee_angle_l" in joint:
                ax = axes2[1]
            elif "knee_angle_r" in joint:
                ax = axes2[1]
            elif "ankle_angle_l" in joint:
                ax = axes2[2]
            elif "ankle_angle_r" in joint:
                ax = axes2[2]
            
            line_color = "#000000" if "_r" in joint else "#555555"
            fill_color = "#555555" if "_r" in joint else "#888888"
            line_style = "-" if "_r" in joint else "--"
            ax.plot(x_mapped, mean_data_degree, color=line_color, linestyle=line_style, label="Right" if "_r" in joint else "Left")
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color=fill_color, alpha=0.5)
        
        axes2[0].set_ylabel(self.JOINT_NAMES['HIP'], fontsize=12, rotation=0, ha='right', va='center')
        axes2[0].yaxis.set_label_coords(-0.15, 0.5)
        axes2[1].set_ylabel(self.JOINT_NAMES['KNEE'], fontsize=12, rotation=0, ha='right', va='center')
        axes2[1].yaxis.set_label_coords(-0.15, 0.5)
        axes2[2].set_ylabel(self.JOINT_NAMES['ANKLE'], fontsize=12, rotation=0, ha='right', va='center')
        axes2[2].yaxis.set_label_coords(-0.15, 0.5)

        axes2[0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes2[1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes2[2].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        for ax in axes2:
            ax.set_xlim(0, 100)  # limit x-axis to data range
        for idx in range(3):
            axes2[idx].axvline(toe_off_l, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)
            axes2[idx].axvline(toe_off_r, color=self.toe_off_color, linestyle="-", linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

        # Add legend to the bottom of the figure
        handles, labels = axes2[0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)

        fig2.tight_layout()
        fig2.subplots_adjust(bottom=0.15)  # Add more bottom margin for legend
        fig2.savefig(os.path.join(result_dir,"left_right_comparison_avg.png"))
        if self.show_plot:
            plt.figure(fig2.number)
            plt.show()
        plt.close()
    def plot_right_ref_comparison(self, *,
                    result_dir):
        # Draw toe-off reference lines (average cycle percentage)
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100


        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped = {
            "hip_flexion_r": [],
            "knee_angle_r": [],
            "ankle_angle_r": [],
        }
        ref_data = {
            "hip_flexion_r": self.segmented_ref_data["q_hip_flexion_r"],
            "knee_angle_r": self.segmented_ref_data["q_knee_angle_r"],
            "ankle_angle_r": self.segmented_ref_data["q_ankle_angle_r"],
        }
        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_r):
            hip_flexion_r_data = [v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]]
            knee_angle_r_data = [v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]]
            ankle_angle_r_data = [v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]]

            hip_flexion_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data)
            knee_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data)
            ankle_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data)

            joint_data_mapped["hip_flexion_r"].append(hip_flexion_r_mapped)
            joint_data_mapped["knee_angle_r"].append(knee_angle_r_mapped)
            joint_data_mapped["ankle_angle_r"].append(ankle_angle_r_mapped)

        fig, axes = plt.subplots(3,1,figsize=(4 *self.fig_size_multiplier, 3 *self.fig_size_multiplier),dpi=self.dpi)
        for joint, data in joint_data_mapped.items():
            data = np.array(data)
            mean_data_degree = np.rad2deg(np.mean(data, axis=0))
            std_data_degree = np.rad2deg(np.std(data, axis=0))
            if "hip_flexion_r" in joint:
                ax = axes[0]
            elif "knee_angle_r" in joint:
                ax = axes[1]
            elif "ankle_angle_r" in joint:
                ax = axes[2]
            
            line_color = "#000000"
            fill_color = "#555555"
            line_style = "-"
            ax.plot(x_mapped, mean_data_degree, color=line_color, linestyle=line_style, label="Simulation")
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color=fill_color, alpha=0.5)
        for joint, data in ref_data.items():
            data_degree = np.rad2deg(data)
            if "hip_flexion_r" in joint:
                ax = axes[0]
            elif "knee_angle_r" in joint:
                ax = axes[1]
            elif "ankle_angle_r" in joint:
                ax = axes[2]
            
            ref_line_color = "#555555"
            ref_line_style = "--"
            ax.plot(x_mapped, data_degree, label="Reference", color=ref_line_color, linestyle=ref_line_style)
        axes[0].set_ylabel(self.JOINT_NAMES['HIP'], fontsize=12, rotation=0, ha='right', va='center')
        axes[0].yaxis.set_label_coords(-0.15, 0.5)
        axes[1].set_ylabel(self.JOINT_NAMES['KNEE'], fontsize=12, rotation=0, ha='right', va='center')
        axes[1].yaxis.set_label_coords(-0.15, 0.5)
        axes[2].set_ylabel(self.JOINT_NAMES['ANKLE'], fontsize=12, rotation=0, ha='right', va='center')
        axes[2].yaxis.set_label_coords(-0.15, 0.5)

        axes[0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        for ax in axes:
            ax.set_xlim(0, 100)  # limit x-axis to data range
            ax.axvline(toe_off_r, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

        # Add legend to the bottom of the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)  # Add more bottom margin for legend
        fig.savefig(os.path.join(result_dir,"right_ref_comparison_avg.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()
        plt.close()
    def plot_contact_data(self, *,
                    result_dir,
                    geom_pairs:list[tuple[str, str]] = [("calcn_l_geom_1", "terrain"), ("calcn_r_geom_1", "terrain")],
                    ):
        plot_data = {
            geom_name1:self.gait_data.get_contact_data(geom_name1=geom_name1, geom_name2=geom_name2)
            for geom_name1, geom_name2 in geom_pairs
        }
        fig, axes = plt.subplots(len(geom_pairs),1,figsize=(4 *self.fig_size_multiplier, 3 *self.fig_size_multiplier),dpi=self.dpi)
        for idx, (geom_name1, geom_name2) in enumerate(geom_pairs):
            ax = axes[idx]
            ax.plot([force[0] for force in plot_data[geom_name1]], label=f"{geom_name1} force", color="#000000", linestyle="-")
            # ax.plot([force[0] for force in plot_data[geom_name2]], label=f"{geom_name2} force", color="#555555", linestyle="--")
            ax.set_title(f"{geom_name1} and {geom_name2} contact force")
            # ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"contact_data.png"))
        # axes[0].plot([force[0] for force in plot_data["calcn_l_geom_1"]], label=f"calcn_l_geom_1", color="#ff0000", linestyle="-")
        # axes[0].plot([force[1] for force in plot_data["calcn_l_geom_1"]], label=f"calcn_l_geom_1", color="#00ff00", linestyle="-")
        # axes[0].plot([force[2] for force in plot_data["calcn_l_geom_1"]], label=f"calcn_l_geom_1", color="#0000ff", linestyle="-")

        # axes[1].plot([force[0] for force in plot_data["calcn_r_geom_1"]], label=f"calcn_r_geom_1", color="#ff0000", linestyle="-")
        # axes[1].plot([force[1] for force in plot_data["calcn_r_geom_1"]], label=f"calcn_r_geom_1", color="#00ff00", linestyle="-")
        # axes[1].plot([force[2] for force in plot_data["calcn_r_geom_1"]], label=f"calcn_r_geom_1", color="#0000ff", linestyle="-")

        # axes[0].set_title("calcn_l_geom_1")
        # axes[1].set_title("calcn_r_geom_1")

        # fig.tight_layout()
        # fig.savefig(os.path.join(result_dir,"contact_force.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()
        plt.close()

    def plot_segmented_muscle_data(self, *,
                        result_dir,
                        is_plot_right:bool
                        ):
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100

        toe_off = toe_off_r if is_plot_right else toe_off_l

        post_fix = ['_r', '_R'] if is_plot_right else ['_l', '_L']
        file_name_post_fix = '_r' if is_plot_right else '_l'
        # actuator_num = len(self.gait_data.series_data["actuator_data"])
        

        # preprocess data
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        x_mapped = np.linspace(0, 100, num=101)
        actuator_names = [actuator_name for actuator_name in self.gait_data.series_data["actuator_data"].keys() if actuator_name[-2:] in post_fix and "Exo" not in actuator_name]
        actuator_names.sort()
        actuator_names += [actuator_name for actuator_name in self.gait_data.series_data["actuator_data"].keys() if actuator_name[-2:] in post_fix and "Exo" in actuator_name]
        muscle_data_mapped = {
            actuator_name:[]
            for actuator_name in actuator_names
        }
        actuator_num = 0
        for actuator_name, actuator_data in self.gait_data.series_data["actuator_data"].items():
            if actuator_name[-2:] in post_fix:
                actuator_num += 1
                for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index):
                    muscle_data = {"force": -np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["force"][start_idx:end_idx])), [v[0] for v in actuator_data["force"][start_idx:end_idx]]),
                                "ctrl": np.abs(np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["ctrl"][start_idx:end_idx])), [v[0] for v in actuator_data["ctrl"][start_idx:end_idx]])),
                                "velocity": np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["velocity"][start_idx:end_idx])), [v[0] for v in actuator_data["velocity"][start_idx:end_idx]])}
                    muscle_data_mapped[actuator_name].append(muscle_data)

        plot_height = actuator_num * 1.0
        plot_width = 4

        #################### segmented muscle data ####################

        fig, axes = plt.subplots(actuator_num,1,figsize=(plot_width *self.fig_size_multiplier,plot_height *self.fig_size_multiplier),dpi=self.dpi)
        actuator_index = 0
        for idx, actuator_name in enumerate(muscle_data_mapped.keys()):
            if actuator_name[-2:] in post_fix:
                ax = axes[actuator_index]
                for muscle_data in muscle_data_mapped[actuator_name]:
                    ax.plot(x_mapped, muscle_data["force"], label=f"{actuator_name} force", color="#000000", linestyle="-")
                
                ax.axvline(toe_off, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

                ax.set_ylabel(actuator_name[:-2], fontsize=12, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.3, 0.5)
                # ax.legend()
                actuator_index += 1
        for ax in axes:
            ax.set_xlim(0, 100)  # limit x-axis to data range
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,f"segmented_muscle_data{file_name_post_fix}.png"))
        plt.figure(fig.number)
        if self.show_plot:
            plt.show()
        plt.close()


        #################### mean and std muscle data ####################

        # Plot mean and std separately
        fig_mean_std, axes_mean_std = plt.subplots(actuator_num, 1, figsize=(plot_width,plot_height), dpi=self.dpi)
        actuator_index = 0
        for idx, (actuator_name, muscle_data_list) in enumerate(muscle_data_mapped.items()):
            if actuator_name[-2:] in post_fix:
                ax = axes_mean_std[actuator_index]
                force_data = np.array([muscle_data["force"] for muscle_data in muscle_data_list])
                mean_force = np.mean(force_data, axis=0)
                std_force = np.std(force_data, axis=0)

                ax.plot(x_mapped, mean_force, label=f"{actuator_name} Mean Force", color='black', linewidth=2)
                ax.fill_between(x_mapped, mean_force - std_force, mean_force + std_force, color='gray', alpha=0.5)

                ax.axvline(toe_off, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

                # ax.set_title(f"{actuator_name} Mean and Std Force")
                ax.set_ylabel(actuator_name[:-2], fontsize=12, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.3, 0.5)
                # ax.legend()
                actuator_index += 1
        for ax in axes_mean_std:
            ax.set_xlim(0, 100)  # limit x-axis to data range
        fig_mean_std.tight_layout()
        fig_mean_std.savefig(os.path.join(result_dir, f"segmented_muscle_data_mean_std{file_name_post_fix}.png"))
        if self.show_plot:
            plt.figure(fig_mean_std.number)
            plt.show()
        plt.close()

        #################### mean and std ctrl data ####################

        # Plot mean and std for ctrl
        fig_mean_std_ctrl, axes_mean_std_ctrl = plt.subplots(actuator_num, 1, figsize=(plot_width,plot_height), dpi=self.dpi)
        actuator_index = 0
        for idx, (actuator_name, muscle_data_list) in enumerate(muscle_data_mapped.items()):
            if actuator_name[-2:] in post_fix:
                ax = axes_mean_std_ctrl[actuator_index]
                ctrl_data = 100 * np.array([muscle_data["ctrl"] for muscle_data in muscle_data_list])
                mean_ctrl = np.mean(ctrl_data, axis=0)
                std_ctrl = np.std(ctrl_data, axis=0)

                ax.plot(x_mapped, mean_ctrl, label=f"{actuator_name}", color='#000000', linewidth=2)
                ax.fill_between(x_mapped, mean_ctrl - std_ctrl, mean_ctrl + std_ctrl, color='#000000', alpha=0.2)

                ax.axvline(toe_off, color=self.toe_off_color, linestyle=self.toe_off_linestyle, linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha)

                ax.set_ylim(0, 100)
                ax.set_xlim(0, 100)
                ax.set_ylabel(actuator_name[:-2], fontsize=12, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.3, 0.5)

                # ax.legend()
                actuator_index += 1
        fig_mean_std_ctrl.tight_layout()
        fig_mean_std_ctrl.savefig(os.path.join(result_dir, f"segmented_muscle_data_mean_std_ctrl{file_name_post_fix}.png"))
        if self.show_plot:
            plt.figure(fig_mean_std_ctrl.number)
            plt.show()
        plt.close()
    
    def plot_joint_torques(self, *, result_dir, is_plot_right: bool = True):
        """
        Plot hip joint torque breakdown across gait cycles.
        Shows three components:
        1. Total torque (muscles + exoskeleton)
        2. Exoskeleton motor torque
        3. Biological torque (muscles only)

        Parameters:
        - result_dir: Directory to save the plot
        - is_plot_right: If True, plot right leg; if False, plot left leg
        """
        toe_off_r = self.get_toe_off_average(is_right_foot_based=True) * 100
        toe_off_l = self.get_toe_off_average(is_right_foot_based=False) * 100
        toe_off = toe_off_r if is_plot_right else toe_off_l

        post_fix = '_r' if is_plot_right else '_l'
        side_name = "Right" if is_plot_right else "Left"

        # Determine exoskeleton motor name
        exo_motor_name = f'exo_right_flexion_motor' if is_plot_right else f'exo_left_flexion_motor'

        # Get gait segments
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_plot_right)
        x_mapped = np.linspace(0, 100, num=101)

        # Hip joint name
        joint_name = f'hip_flexion{post_fix}'

        # Check if joint exists
        if joint_name not in self.gait_data.series_data["joint_data"]:
            print(f"Warning: {joint_name} not found in joint_data")
            return

        # Check if qfrc_actuator exists
        if "qfrc_actuator" not in self.gait_data.series_data["joint_data"][joint_name]:
            print(f"Warning: qfrc_actuator not found for {joint_name}. Please re-run evaluation to collect torque data.")
            return

        # Check if exoskeleton motor data exists
        if exo_motor_name not in self.gait_data.series_data["actuator_data"]:
            print(f"Warning: {exo_motor_name} not found in actuator_data")
            return

        joint_data = self.gait_data.series_data["joint_data"][joint_name]
        exo_data = self.gait_data.series_data["actuator_data"][exo_motor_name]
        # Use normalized [-1, 1] for exo subplot when available; bio = total - exo (Nm) always
        exo_force_key = "force_normalized" if exo_data.get("force_normalized") else "force"

        # Collect torque data for each gait segment
        total_torque_mapped = []
        exo_torque_mapped = []
        bio_torque_mapped = []

        for start_idx, toe_off_idx, end_idx in gait_segment_index:
            # Extract total torque (qfrc_actuator) for this segment
            total_segment = [v[0] if isinstance(v, list) else v for v in joint_data["qfrc_actuator"][start_idx:end_idx]]
            # Exo in Nm (for bio = total - exo)
            exo_segment_nm = [v[0] if isinstance(v, list) else v for v in exo_data["force"][start_idx:end_idx]]
            # Exo for display (normalized when available)
            exo_segment_display = [v[0] if isinstance(v, list) else v for v in exo_data[exo_force_key][start_idx:end_idx]]

            # Calculate biological (muscle) torque = total - exoskeleton (both in Nm)
            bio_segment = [total - exo for total, exo in zip(total_segment, exo_segment_nm)]

            # Interpolate all three to 0-100% gait cycle
            total_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(total_segment)), total_segment)
            exo_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_segment_display)), exo_segment_display)
            bio_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(bio_segment)), bio_segment)

            total_torque_mapped.append(total_mapped)
            exo_torque_mapped.append(exo_mapped)
            bio_torque_mapped.append(bio_mapped)

        if len(total_torque_mapped) == 0:
            print(f"Warning: No gait segments found for {joint_name}")
            return

        # Convert to arrays and calculate mean and std
        total_array = np.array(total_torque_mapped)
        exo_array = np.array(exo_torque_mapped)
        bio_array = np.array(bio_torque_mapped)

        mean_total = np.mean(total_array, axis=0)
        std_total = np.std(total_array, axis=0)

        mean_exo = np.mean(exo_array, axis=0)
        std_exo = np.std(exo_array, axis=0)

        mean_bio = np.mean(bio_array, axis=0)
        std_bio = np.std(bio_array, axis=0)

        # Create figure with 3 subplots (stacked vertically)
        from matplotlib.ticker import MultipleLocator
        fig, axes = plt.subplots(3, 1, figsize=(8 * self.fig_size_multiplier, 9 * self.fig_size_multiplier), dpi=self.dpi)

        # Subplot 1: Total Torque (Muscles + Exoskeleton)
        ax1 = axes[0]
        ax1.plot(x_mapped, mean_total, color='#2E86AB', linewidth=2, label='Total Torque')
        ax1.fill_between(x_mapped, mean_total - std_total, mean_total + std_total, color='#2E86AB', alpha=0.3, label='±1 Std')
        ax1.axvline(toe_off, color=self.toe_off_color, linestyle=self.toe_off_linestyle,
                   linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha, label=f'Toe-off ({toe_off:.1f}%)')
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.grid(True, which='major', axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Torque (Nm)', fontsize=11)
        ax1.set_xlim(0, 100)
        ax1.set_title(f'{side_name} Hip Joint Torque - Total (Muscles + Exoskeleton)', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.yaxis.set_major_locator(MultipleLocator(50))
        ax1.yaxis.set_minor_locator(MultipleLocator(25))

        # Subplot 2: Exoskeleton Torque Only (normalized [-1, 1] when available)
        exo_is_normalized = exo_force_key == "force_normalized"
        ax2 = axes[1]
        ax2.plot(x_mapped, mean_exo, color='#A23B72', linewidth=2, label='Exoskeleton Torque')
        ax2.fill_between(x_mapped, mean_exo - std_exo, mean_exo + std_exo, color='#A23B72', alpha=0.3, label='±1 Std')
        ax2.axvline(toe_off, color=self.toe_off_color, linestyle=self.toe_off_linestyle,
                   linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha, label=f'Toe-off ({toe_off:.1f}%)')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.grid(True, which='major', axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Exo torque (normalized)' if exo_is_normalized else 'Torque (Nm)', fontsize=11)
        ax2.set_xlim(0, 100)
        if exo_is_normalized:
            ax2.set_ylim(-1.15, 1.15)
            ax2.yaxis.set_major_locator(MultipleLocator(0.5))
            ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
        else:
            ax2.yaxis.set_major_locator(MultipleLocator(50))
            ax2.yaxis.set_minor_locator(MultipleLocator(25))
        ax2.set_title(f'{side_name} Hip Joint Torque - Exoskeleton Only', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)

        # Subplot 3: Biological (Muscle) Torque
        ax3 = axes[2]
        ax3.plot(x_mapped, mean_bio, color='#F18F01', linewidth=2, label='Biological Torque (Muscles)')
        ax3.fill_between(x_mapped, mean_bio - std_bio, mean_bio + std_bio, color='#F18F01', alpha=0.3, label='±1 Std')
        ax3.axvline(toe_off, color=self.toe_off_color, linestyle=self.toe_off_linestyle,
                   linewidth=self.toe_off_linewidth, alpha=self.toe_off_alpha, label=f'Toe-off ({toe_off:.1f}%)')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.grid(True, which='major', axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Gait Cycle (%)', fontsize=11)
        ax3.set_ylabel('Torque (Nm)', fontsize=11)
        ax3.set_xlim(0, 100)
        ax3.set_title(f'{side_name} Hip Joint Torque - Biological Only (Total - Exoskeleton)', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.yaxis.set_major_locator(MultipleLocator(50))
        ax3.yaxis.set_minor_locator(MultipleLocator(25))

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f'hip_joint_torque_breakdown{post_fix}.png'))

        if self.show_plot:
            plt.show()
        plt.close()

    def plot_hip_angle_vs_exo_torque(self, *, result_dir, is_plot_right: bool = True, control_framerate: int = 30):
        """
        Plot the entire evaluation time series of hip joint angle, angular velocity,
        and exo torque in 3 panels with shared x-axis (time in seconds).
        """
        post_fix = '_r' if is_plot_right else '_l'
        side_name = "Right" if is_plot_right else "Left"

        joint_name = f'hip_flexion{post_fix}'
        exo_motor_name = f'exo_right_flexion_motor' if is_plot_right else f'exo_left_flexion_motor'

        joint_data = self.gait_data.series_data["joint_data"]
        if joint_name not in joint_data:
            print(f"Warning: {joint_name} not found in joint_data")
            return
        if exo_motor_name not in self.gait_data.series_data["actuator_data"]:
            print(f"Warning: {exo_motor_name} not found in actuator_data")
            return

        exo_data = self.gait_data.series_data["actuator_data"][exo_motor_name]
        exo_force_key = "force_normalized" if exo_data.get("force_normalized") else "force"
        exo_is_normalized = exo_force_key == "force_normalized"

        # Extract full time series (not segmented by gait cycle)
        angle_values = np.rad2deg([v[0] if isinstance(v, list) else v
                                   for v in joint_data[joint_name]["qpos"]])
        vel_values = np.array([v[0] if isinstance(v, list) else v
                               for v in joint_data[joint_name]["qvel"]])
        exo_values = np.array([v[0] if isinstance(v, list) else v
                               for v in exo_data[exo_force_key]])

        n_steps = len(angle_values)
        time_axis = np.arange(n_steps) / control_framerate  # seconds

        fig, axes = plt.subplots(3, 1, figsize=(14 * self.fig_size_multiplier, 8 * self.fig_size_multiplier),
                                 dpi=self.dpi, sharex=True)

        # Panel 1: Hip flexion angle
        ax1 = axes[0]
        ax1.plot(time_axis, angle_values, color='#2E86AB', linewidth=1)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Angle (deg)', fontsize=11)
        ax1.set_title(f'{side_name} Hip Flexion Angle', fontsize=13, fontweight='bold')
        ax1.annotate('(+) Flexion  (-) Extension', xy=(0.99, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top', color='gray')

        # Panel 2: Hip angular velocity
        ax2 = axes[1]
        ax2.plot(time_axis, vel_values, color='#4CAF50', linewidth=1)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Angular Vel (rad/s)', fontsize=11)
        ax2.set_title(f'{side_name} Hip Angular Velocity', fontsize=13, fontweight='bold')
        ax2.annotate('(+) Flexing  (-) Extending', xy=(0.99, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top', color='gray')

        # Panel 3: Exo torque (normalized [-1, 1] when available)
        ax3 = axes[2]
        ax3.plot(time_axis, exo_values, color='#A23B72', linewidth=1)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Exo torque (normalized)' if exo_is_normalized else 'Torque (Nm)', fontsize=11)
        if exo_is_normalized:
            ax3.set_ylim(-1.15, 1.15)
        ax3.set_title(f'{side_name} Exo Hip Flexion Motor Torque', fontsize=13, fontweight='bold')
        ax3.annotate('(+) Flexion torque  (-) Extension torque', xy=(0.99, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top', color='gray')

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f'hip_angle_vs_exo_torque{post_fix}.png'))

        if self.show_plot:
            plt.show()
        plt.close()

    def plot_exo_torque_lr_comparison(self, *, result_dir, control_framerate: int = 30):
        """
        Plot left and right exo hip torques on the same time-series figure
        for direct L/R comparison.
        """
        joint_data = self.gait_data.series_data["joint_data"]
        actuator_data = self.gait_data.series_data["actuator_data"]

        r_motor = 'exo_right_flexion_motor'
        l_motor = 'exo_left_flexion_motor'
        if r_motor not in actuator_data or l_motor not in actuator_data:
            print(f"Warning: exo motor data not found, skipping L/R comparison plot")
            return

        # Use normalized [-1, 1] when available
        key_r = "force_normalized" if actuator_data[r_motor].get("force_normalized") else "force"
        key_l = "force_normalized" if actuator_data[l_motor].get("force_normalized") else "force"
        exo_is_normalized = key_r == "force_normalized"
        exo_r = np.array([v[0] if isinstance(v, list) else v
                          for v in actuator_data[r_motor][key_r]])
        exo_l = np.array([v[0] if isinstance(v, list) else v
                          for v in actuator_data[l_motor][key_l]])

        n_steps = min(len(exo_r), len(exo_l))
        time_axis = np.arange(n_steps) / control_framerate

        fig, ax = plt.subplots(1, 1, figsize=(14 * self.fig_size_multiplier, 4 * self.fig_size_multiplier),
                               dpi=self.dpi)

        ax.plot(time_axis, exo_r[:n_steps], color='#2E86AB', linewidth=1, label='Right', alpha=0.85)
        ax.plot(time_axis, exo_l[:n_steps], color='#E74C3C', linewidth=1, label='Left', alpha=0.85)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Exo torque (normalized)' if exo_is_normalized else 'Torque (Nm)', fontsize=11)
        if exo_is_normalized:
            ax.set_ylim(-1.15, 1.15)
        ax.set_title('Exo Hip Torque: Left vs Right', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.annotate('(+) Flexion torque  (-) Extension torque', xy=(0.99, 0.02), xycoords='axes fraction',
                    fontsize=8, ha='right', va='bottom', color='gray')

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, 'exo_torque_lr_comparison.png'))

        if self.show_plot:
            plt.show()
        plt.close()

    def _compute_exo_power_series(self, exo_motor_name: str, joint_name: str):
        """
        Compute exo joint power P(t) = tau(t) * omega(t) [W].
        tau = normalized_torque * forcerange_max when force_normalized exists; else tau = force (Nm).
        Returns (power_array, torque_nm_array) or (None, None) if data missing.
        """
        actuator_data = self.gait_data.series_data["actuator_data"]
        joint_data = self.gait_data.series_data["joint_data"]
        if exo_motor_name not in actuator_data or joint_name not in joint_data:
            return None, None
        exo_data = actuator_data[exo_motor_name]
        omega = np.array([v[0] if isinstance(v, list) else v
                          for v in joint_data[joint_name]["qvel"]])
        if exo_data.get("force_normalized") and exo_data.get("forcerange_max") is not None:
            norm_torque = np.array([v[0] if isinstance(v, list) else v
                                    for v in exo_data["force_normalized"]])
            max_torque = float(exo_data["forcerange_max"])
            torque_nm = norm_torque * max_torque
        else:
            torque_nm = np.array([v[0] if isinstance(v, list) else v
                                  for v in exo_data["force"]])
        n = min(len(torque_nm), len(omega))
        torque_nm = torque_nm[:n]
        omega = omega[:n]
        power = torque_nm * omega  # P = tau * omega [W]
        return power, torque_nm

    def plot_exo_power_lr_comparison(self, *, result_dir, control_framerate: int = 30):
        """
        Plot left and right exo hip joint power P(t) = tau(t) * omega(t) [W].
        tau = normalized_torque * max_torque when available.
        """
        power_r, _ = self._compute_exo_power_series('exo_right_flexion_motor', 'hip_flexion_r')
        power_l, _ = self._compute_exo_power_series('exo_left_flexion_motor', 'hip_flexion_l')
        if power_r is None or power_l is None:
            print("Warning: exo power data not found, skipping exo power L/R comparison plot")
            return
        n_steps = min(len(power_r), len(power_l))
        time_axis = np.arange(n_steps) / control_framerate

        fig, ax = plt.subplots(1, 1, figsize=(14 * self.fig_size_multiplier, 4 * self.fig_size_multiplier),
                               dpi=self.dpi)
        ax.plot(time_axis, power_r[:n_steps], color='#2E86AB', linewidth=1, label='Right', alpha=0.85)
        ax.plot(time_axis, power_l[:n_steps], color='#E74C3C', linewidth=1, label='Left', alpha=0.85)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Power (W)', fontsize=11)
        ax.set_title('Exo Hip Joint Power: P = τ × ω (Left vs Right)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.annotate('(+) Positive work  (-) Negative work', xy=(0.99, 0.02), xycoords='axes fraction',
                    fontsize=8, ha='right', va='bottom', color='gray')
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, 'exo_power_lr_comparison.png'))
        if self.show_plot:
            plt.show()
        plt.close()

    def plot_hip_angle_vs_exo_power(self, *, result_dir, is_plot_right: bool = True, control_framerate: int = 30):
        """
        Plot hip angle, angular velocity, exo torque (normalized), and exo joint power P = τ × ω [W]
        in 4 panels with shared x-axis (time in seconds). τ = normalized_torque * max_torque when available.
        """
        post_fix = '_r' if is_plot_right else '_l'
        side_name = "Right" if is_plot_right else "Left"
        joint_name = f'hip_flexion{post_fix}'
        exo_motor_name = f'exo_right_flexion_motor' if is_plot_right else f'exo_left_flexion_motor'

        joint_data = self.gait_data.series_data["joint_data"]
        if joint_name not in joint_data or exo_motor_name not in self.gait_data.series_data["actuator_data"]:
            print(f"Warning: data not found for {joint_name}/{exo_motor_name}, skipping hip_angle_vs_exo_power")
            return

        angle_values = np.rad2deg([v[0] if isinstance(v, list) else v
                                   for v in joint_data[joint_name]["qpos"]])
        vel_values = np.array([v[0] if isinstance(v, list) else v
                               for v in joint_data[joint_name]["qvel"]])
        exo_data = self.gait_data.series_data["actuator_data"][exo_motor_name]
        exo_force_key = "force_normalized" if exo_data.get("force_normalized") else "force"
        exo_values = np.array([v[0] if isinstance(v, list) else v for v in exo_data[exo_force_key]])
        power_values, torque_nm = self._compute_exo_power_series(exo_motor_name, joint_name)
        if power_values is None:
            return
        n_steps = min(len(angle_values), len(vel_values), len(exo_values), len(power_values))
        time_axis = np.arange(n_steps) / control_framerate

        fig, axes = plt.subplots(4, 1, figsize=(14 * self.fig_size_multiplier, 10 * self.fig_size_multiplier),
                                 dpi=self.dpi, sharex=True)
        ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

        ax1.plot(time_axis, angle_values[:n_steps], color='#2E86AB', linewidth=1)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Angle (deg)', fontsize=11)
        ax1.set_title(f'{side_name} Hip Flexion Angle', fontsize=13, fontweight='bold')

        ax2.plot(time_axis, vel_values[:n_steps], color='#4CAF50', linewidth=1)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Angular Vel (rad/s)', fontsize=11)
        ax2.set_title(f'{side_name} Hip Angular Velocity', fontsize=13, fontweight='bold')

        ax3.plot(time_axis, exo_values[:n_steps], color='#A23B72', linewidth=1)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel('Exo torque (norm)' if exo_force_key == "force_normalized" else 'Torque (Nm)', fontsize=11)
        ax3.set_title(f'{side_name} Exo Torque', fontsize=13, fontweight='bold')
        if exo_force_key == "force_normalized":
            ax3.set_ylim(-1.15, 1.15)

        ax4.plot(time_axis, power_values[:n_steps], color='#E67E22', linewidth=1)
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Power (W)', fontsize=11)
        ax4.set_title(f'{side_name} Exo Joint Power P = τ × ω', fontsize=13, fontweight='bold')
        ax4.annotate('(+) Positive work  (-) Negative work', xy=(0.99, 0.95), xycoords='axes fraction',
                     fontsize=8, ha='right', va='top', color='gray')

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f'hip_angle_vs_exo_power{post_fix}.png'))
        if self.show_plot:
            plt.show()
        plt.close()

    def __del__(self):
        plt.close()
        pass