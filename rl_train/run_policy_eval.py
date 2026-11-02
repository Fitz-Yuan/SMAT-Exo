
# Parse command line arguments for log_dir
# Usage: python run_policy_eval.py <checkpoint.zip>
#          evaluates that checkpoint; session_config.json (and train_log.json, if present) are read from its folder
#        python run_policy_eval.py <log_dir> [model_timestep]
#          model_timestep: optional, e.g. 2359296  -> uses trained_models/model_2359296.zip
#                          if omitted, uses the last checkpoint in train_log.json
import sys
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = ""

if log_dir == "":
    log_dir = input("Enter the log directory: ")
    # log_dir = "docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs" # partial obs
    # log_dir = "docs/assets/tutorial_rl_models/train_session_20250729-005528_tutorial_full_obs" # Full obs

# Optional: specify a particular model checkpoint by timestep
import os
if log_dir.endswith(".zip"):
    _specific_model_path = log_dir[:-len(".zip")]
    _model_timestep = os.path.basename(_specific_model_path)  # used to name the result folder
    log_dir = os.path.dirname(log_dir)
    if os.path.basename(log_dir) == "trained_models":  # checkpoint of a training session: <session>/trained_models/<model>.zip
        log_dir = os.path.dirname(log_dir)
else:
    _model_timestep = int(sys.argv[2]) if len(sys.argv) > 2 else None
    _specific_model_path = os.path.join(log_dir, "trained_models", f"model_{_model_timestep}") if _model_timestep is not None else None

show_plot = False
import os

import numpy as np
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.data_types import DictionableDataclass

import os
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
import json
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

for (idx, evaluate_param) in enumerate(config.evaluate_param_list):
    if _model_timestep is not None:
        analyze_result_dir = os.path.join(log_dir, f"analyze_results_{_model_timestep}")
    else:
        analyze_result_dir = os.path.join(log_dir,f"analyze_results_{idx:02d}")
    if not os.path.exists(analyze_result_dir):
        os.makedirs(analyze_result_dir)

    log_handler = TrainLogHandler(log_dir)
    # Data is already loaded in TrainLogHandler.__init__, no need to load again

    if log_handler.log_datas:  # reward curves need a train_log.json (not included with the released checkpoints)
        import sys
        sys.modules.pop('package.train_log_analyzer', None)
        from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
        train_log_analyzer = TrainLogAnalyzer(log_handler)
        train_log_analyzer.plot_reward(result_dir=analyze_result_dir, show_plot=show_plot)


    


    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

        

    import sys
    from rl_train.analyzer.gait_analyze import GaitAnalyzer
    from rl_train.analyzer.gait_evaluate import GaitData


    gait_data_name = f"gait_evaluated_data.json"
    is_regen_evaluating_data = True

    from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
    gait_evaluator: ImitationGaitEvaluator = ImitationGaitEvaluator(log_handler, config)
    gait_evaluator.load_reference_data()
    gait_evaluator.initialize_env()
    if is_regen_evaluating_data:
        # Each rollout starts at a random phase of the reference gait. If the model falls before
        # num_timesteps, retry from a new random start (the result folder keeps the last attempt).
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            gait_data_path = gait_evaluator.evaluate(result_dir=analyze_result_dir,
                                                    file_name=gait_data_name,
                                                    velocity_mode=MyoAssistLegBase.VelocityMode[evaluate_param["velocity_mode"]],
                                                    target_velocity_period=evaluate_param["target_velocity_period"],
                                                    max_timestep=evaluate_param["num_timesteps"],
                                                    min_target_velocity=evaluate_param["min_target_velocity"],
                                                    max_target_velocity=evaluate_param["max_target_velocity"],
                                                    terminate_when_done=True,
                                                    model_path=_specific_model_path,
                                                    )
            with open(gait_data_path, "r") as f:
                num_recorded_steps = json.load(f)["metadata"]["data_length"]
            if num_recorded_steps >= evaluate_param["num_timesteps"]:
                break
            print(f"Attempt {attempt}/{max_attempts}: the model fell after {num_recorded_steps} of "
                  f"{evaluate_param['num_timesteps']} steps" + ("; retrying from a new random start" if attempt < max_attempts else ""))
    else:
        gait_data_path = os.path.join(analyze_result_dir, gait_data_name)

    gait_data = GaitData()
    gait_data.read_json_data(gait_data_path)

    # ====== Exo Torque Verification ======
    print("\n" + "="*60)
    print("EXO TORQUE VERIFICATION")
    print("="*60)
    exo_motor_names = ["exo_right_flexion_motor", "exo_left_flexion_motor"]
    hip_joint_names = ["hip_flexion_r", "hip_flexion_l"]

    env_unwrapped = gait_evaluator.env.unwrapped
    for motor_name, joint_name in zip(exo_motor_names, hip_joint_names):
        if motor_name not in gait_data.series_data["actuator_data"]:
            print(f"  {motor_name}: NOT FOUND in evaluated data")
            continue

        actuator_id = env_unwrapped.sim.model.actuator(motor_name).id
        forcerange = env_unwrapped.sim.model.actuator_forcerange[actuator_id]
        ctrlrange = env_unwrapped.sim.model.actuator_ctrlrange[actuator_id]
        gainprm = env_unwrapped.sim.model.actuator_gainprm[actuator_id]

        act_data = gait_data.series_data["actuator_data"][motor_name]
        force_data = act_data["force"]
        force_values = [v[0] if isinstance(v, list) else v for v in force_data]
        ctrl_data = act_data["ctrl"]
        ctrl_values = np.array([v[0] if isinstance(v, list) else v for v in ctrl_data])
        # Normalized [-1, 1]: from ctrl (applied) or from stored force_normalized when available
        if act_data.get("force_normalized"):
            norm_values = np.array([v[0] if isinstance(v, list) else v for v in act_data["force_normalized"]])
        else:
            ctrl_mid = (ctrlrange[0] + ctrlrange[1]) * 0.5
            ctrl_rng = (ctrlrange[1] - ctrlrange[0]) * 0.5
            norm_values = (ctrl_values - ctrl_mid) / ctrl_rng if ctrl_rng > 1e-9 else np.zeros_like(ctrl_values)

        qfrc_data = gait_data.series_data["joint_data"][joint_name]["qfrc_actuator"]
        qfrc_values = [v[0] if isinstance(v, list) else v for v in qfrc_data]

        print(f"\n  [{motor_name}] -> joint: {joint_name}")
        print(f"    XML forcerange:  [{forcerange[0]:.1f}, {forcerange[1]:.1f}] Nm")
        print(f"    XML gainprm[0]:  {gainprm[0]:.1f}")
        print(f"    --- exo torque (normalized [-1, 1], policy output scale) ---")
        print(f"      min: {float(np.min(norm_values)):.3f}")
        print(f"      max: {float(np.max(norm_values)):.3f}")
        print(f"      mean: {float(np.mean(norm_values)):.3f}")
        print(f"    --- actuator.force (Nm, for reference) ---")
        print(f"      min: {min(force_values):.3f} Nm")
        print(f"      max: {max(force_values):.3f} Nm")
        exceeds = [f for f in force_values if f < forcerange[0] - 0.01 or f > forcerange[1] + 0.01]
        if exceeds:
            print(f"      WARNING: {len(exceeds)}/{len(force_values)} samples EXCEED forcerange!")
        print(f"    --- qfrc_actuator (total joint torque: all actuators on {joint_name}) ---")
        print(f"      min: {min(qfrc_values):.3f} Nm")
        print(f"      max: {max(qfrc_values):.3f} Nm")
        print(f"    --- comparison ---")
        print(f"      exo contribution ratio: {np.mean(np.abs(force_values)) / (np.mean(np.abs(qfrc_values)) + 1e-8) * 100:.1f}%")
        # Power P = tau * omega [W], tau = normalized * max_torque
        max_torque = max(abs(forcerange[0]), abs(forcerange[1]))
        omega_data = gait_data.series_data["joint_data"][joint_name]["qvel"]
        omega_values = np.array([v[0] if isinstance(v, list) else v for v in omega_data])
        n_p = min(len(norm_values), len(omega_values))
        tau_nm = norm_values[:n_p] * max_torque
        power_values = tau_nm * omega_values[:n_p]
        print(f"    --- joint power P = τ × ω (τ = normalized × {max_torque:.1f} Nm) ---")
        print(f"      mean: {float(np.mean(power_values)):.3f} W")
        print(f"      max (positive work): {float(np.max(power_values)):.3f} W")
        print(f"      min (negative work): {float(np.min(power_values)):.3f} W")

    print("="*60 + "\n")

    segmented_ref_data = np.load("reference_data/segmented.npz", allow_pickle=True)
    segmented_ref_data = {key: segmented_ref_data[key] for key in segmented_ref_data.files}


    gait_evaluator.replay(gait_data_path, os.path.join(analyze_result_dir, "replay.mp4"),
                                                cam_distance=evaluate_param["cam_distance"],
                                                # max_time_step=evaluate_param["num_timesteps"],
                                                use_activation_visualization=evaluate_param["visualize_activation"],
                                                cam_type=evaluate_param["cam_type"],
                                                realtime_plotting_info=evaluate_param.get("realtime_plotting_info", []),
                                                video_fps=config.env_params.control_framerate
                                                )

    gait_analyzer = GaitAnalyzer(gait_data, segmented_ref_data, show_plot)


    if len(gait_analyzer.get_gait_segment_index(is_right_foot_based=True)) < 1:
        print("="*10 + "Warning" + "="*10)
        print("Warning! Not enough gait data to plot. Skipping plotting.")
        print("="*10 + "Warning" + "="*10)

        continue


    gait_analyzer.plot_entire_result(result_dir=analyze_result_dir,is_right_foot_based=True)

    try:
        gait_analyzer.plot_exo_segmented_data(result_dir=analyze_result_dir)
    except KeyError:
        print("No Exo_L/Exo_R data in reference gait. Skipping exo plot.")

    gait_analyzer.plot_segmented_kinematics_result(result_dir=analyze_result_dir)

    gait_analyzer.plot_left_right_comparison(result_dir=analyze_result_dir)

    gait_analyzer.plot_right_ref_comparison(result_dir=analyze_result_dir)

    gait_analyzer.plot_segmented_muscle_data(result_dir=analyze_result_dir, is_plot_right=True)

    gait_analyzer.plot_joint_torques(result_dir=analyze_result_dir, is_plot_right=True)
    gait_analyzer.plot_joint_torques(result_dir=analyze_result_dir, is_plot_right=False)

    gait_analyzer.plot_hip_angle_vs_exo_torque(result_dir=analyze_result_dir, is_plot_right=True, control_framerate=config.env_params.control_framerate)
    gait_analyzer.plot_hip_angle_vs_exo_torque(result_dir=analyze_result_dir, is_plot_right=False, control_framerate=config.env_params.control_framerate)

    gait_analyzer.plot_exo_torque_lr_comparison(result_dir=analyze_result_dir, control_framerate=config.env_params.control_framerate)

    gait_analyzer.plot_exo_power_lr_comparison(result_dir=analyze_result_dir, control_framerate=config.env_params.control_framerate)
    gait_analyzer.plot_hip_angle_vs_exo_power(result_dir=analyze_result_dir, is_plot_right=True, control_framerate=config.env_params.control_framerate)
    gait_analyzer.plot_hip_angle_vs_exo_power(result_dir=analyze_result_dir, is_plot_right=False, control_framerate=config.env_params.control_framerate)

    gait_analyzer.joint_angle_by_velocity(result_dir=analyze_result_dir)
