import collections
import numpy as np
from dataclasses import dataclass, field
from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation
from rl_train.envs.myoassist_leg_imitation import ImitationTrainSessionConfig
from rl_train.envs.myoassist_leg_imitation import ImitationCustomLearningCallback


from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train.utils.learning_callback import BaseCustomLearningCallback

class MyoAssistLegImitationExo(MyoAssistLegImitation):
    """
    Exo environment with configurable history buffer for hip angles/velocities.

    History configuration is controlled by env_params:
    - exo_history_input_frames: number of past input frames (default: 0 = disabled)
    - exo_history_output_frames: number of past output frames (default: 0 = disabled)

    When enabled, adds 'exo_history' to observation dict.
    """

    # Override DEFAULT_OBS_KEYS to include exo_torque and exo_history
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'act', 'sensor', 'target_velocity', 'exo_torque', 'exo_history']

    def _setup(self, *, env_params, **kwargs):
        # Get history config from env_params (default: disabled for backward compatibility)
        self._exo_history_input_frames = getattr(env_params, 'exo_history_input_frames', 0)
        self._exo_history_output_frames = getattr(env_params, 'exo_history_output_frames', 0)
        self._exo_history_enabled = (self._exo_history_input_frames > 0 or
                                      self._exo_history_output_frames > 0)

        # Initialize history buffers BEFORE super()._setup() because it calls step()
        if self._exo_history_enabled:
            self._init_exo_history()

        super()._setup(env_params=env_params, **kwargs)

    def _init_exo_history(self):
        """Initialize history buffers for exo observation.

        Order convention: [Right, Left] throughout to match action mapping.
        - Input history: [R_ang, L_ang, R_vel, L_vel] per frame
        - Output history: [R_out, L_out] per frame
        - Network output: [net_0, net_1] → [action[26]=R, action[27]=L]
        """
        # History for hip inputs (angles + velocities): 4 values per frame [R_ang, L_ang, R_vel, L_vel]
        self._hip_input_history = collections.deque(maxlen=self._exo_history_input_frames)
        for _ in range(self._exo_history_input_frames):
            self._hip_input_history.append(np.zeros(4))

        # History for exo outputs: 2 values per frame [R_out, L_out]
        self._exo_output_history = collections.deque(maxlen=self._exo_history_output_frames)
        for _ in range(self._exo_history_output_frames):
            self._exo_output_history.append(np.zeros(2))

    def _get_exo_torque_obs(self, sim):
        """Get normalized exo torques as observation: [right, left] in [-1, 1]."""
        exo_torques = np.zeros(2)
        exo_motor_names = ['exo_right_flexion_motor', 'exo_left_flexion_motor']
        for i, motor_name in enumerate(exo_motor_names):
            try:
                actuator_id = sim.model.actuator(motor_name).id
                torque = sim.data.actuator_force[actuator_id]
                forcerange = sim.model.actuator_forcerange[actuator_id]
                max_torque = max(abs(forcerange[0]), abs(forcerange[1]))
                if max_torque > 0:
                    exo_torques[i] = torque / max_torque  # normalized to [-1, 1]
            except (KeyError, ValueError, AttributeError):
                pass
        return exo_torques

    def get_obs_dict(self, sim):
        """Override to add exo torque and exo history to observation."""
        obs_dict = super().get_obs_dict(sim)

        # Always add exo_torque (2 values: normalized right/left hip exo torque)
        obs_dict['exo_torque'] = self._get_exo_torque_obs(sim)

        if not self._exo_history_enabled:
            # Return empty array when disabled (for backward compatibility)
            obs_dict['exo_history'] = np.array([])
            return obs_dict

        # Get current hip angles and velocities
        # Use [R, L, R, L] order to match action output order [action[26]=R, action[27]=L]
        try:
            hip_angle_r = sim.data.joint("hip_flexion_r").qpos[0].copy()
            hip_angle_l = sim.data.joint("hip_flexion_l").qpos[0].copy()
            hip_vel_r = sim.data.joint("hip_flexion_r").qvel[0].copy()
            hip_vel_l = sim.data.joint("hip_flexion_l").qvel[0].copy()
        except:
            hip_angle_r = hip_angle_l = hip_vel_r = hip_vel_l = 0.0

        current_input = np.array([hip_angle_r, hip_angle_l, hip_vel_r, hip_vel_l])

        # Build exo_history array
        history_parts = []

        # Add past input frames (oldest first)
        for past_input in self._hip_input_history:
            history_parts.append(past_input)

        # Add current input
        history_parts.append(current_input)

        # Add past output frames (oldest first)
        for past_output in self._exo_output_history:
            history_parts.append(past_output)

        obs_dict['exo_history'] = np.concatenate(history_parts)

        # Update input history for next step
        self._hip_input_history.append(current_input.copy())

        return obs_dict

    def step(self, a, **kwargs):
        # Store exo action in history before step (if enabled)
        if self._exo_history_enabled and self._exo_history_output_frames > 0:
            # Exo actions: a[26]=right, a[27]=left
            # Store in [right, left] order to match input order [R_ang, L_ang, R_vel, L_vel]
            if len(a) >= 28:
                exo_action = np.array([a[26], a[27]])  # [right, left]
            else:
                exo_action = np.zeros(2)
            self._exo_output_history.append(exo_action.copy())

        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)

        return (next_obs, reward, terminated, truncated, info)

    def reset(self, **kwargs):
        # Reset history buffers if enabled
        if self._exo_history_enabled:
            self._init_exo_history()
        return super().reset(**kwargs)
