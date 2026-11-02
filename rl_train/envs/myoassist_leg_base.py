import numpy as np
from dataclasses import dataclass
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from myosuite.utils import gym
from myosuite.envs import env_base
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
import collections
import mujoco
from myoassist_utils.hfield_manager import HfieldManager
from enum import Enum
import random
import numpy as np


class MyoAssistLegBase(env_base.MujocoEnv):
    MYO_CREDIT = """\
    NeuMove MyoLegBase
    """

    class VelocityMode(Enum):
        UNIFORM = 0
        SINUSOIDAL = 1
        STEP = 2

    DEFAULT_OBS_KEYS = ['qpos',
                        'qvel',
                        'act',
                        'sensor',
                        'target_velocity',
                        ]
    
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        print(f"=================environment seed: {seed}=====================")
        print(f"=================environment model_path: {model_path}=====================")
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _setup(self,*,
               env_params:TrainSessionConfigBase.EnvParams,
                **kwargs):
        
        self.is_evaluate_mode = kwargs.pop("is_evaluate_mode", False)

        self.sim.model.opt.timestep = 1 / env_params.physics_sim_framerate
        self._safe_height = env_params.safe_height

        self._min_target_velocity = env_params.min_target_velocity
        self._max_target_velocity = env_params.max_target_velocity
        self._min_target_velocity_period = env_params.min_target_velocity_period
        self._max_target_velocity_period = env_params.max_target_velocity_period
        self._change_mode_and_target_velocity_randomly()

        self._step_count_per_episode = 0
        self.CUSTOM_MAX_EPISODE_STEPS = env_params.custom_max_episode_steps

        self._prev_muscle_activations_for_reward = None
        # "power_sign": Stage 3 exo reward (Eq. 4 of the paper); "power": Stage 4 exo reward (Eq. 5);
        # "direction": direction-matching variant used to train the released Stage 3 checkpoint
        self._exo_assistance_reward_type = getattr(env_params, "exo_assistance_reward_type", "power")
        self._prev_exo_action = None  # for exo_action_rate_penalty (normalized exo torque at t-1)
        self._last_applied_exo_action = None  # set in step(a) before super().step, used in _calculate_base_reward
        self._enable_lumbar_joint = env_params.enable_lumbar_joint
        self._lumbar_joint_fixed_angle = env_params.lumbar_joint_fixed_angle
        self._lumbar_joint_damping_value = env_params.lumbar_joint_damping_value

        self.observation_joint_pos_keys = env_params.observation_joint_pos_keys
        self.observation_joint_vel_keys = env_params.observation_joint_vel_keys
        self.observation_sensor_keys = env_params.observation_sensor_keys

        self.joint_limit_sensor_keys = env_params.joint_limit_sensor_keys

        # Safely check whether the joint named "lumbar_extension" exists in the model.
        try:
            lumbar_joint_id = self.sim.model.joint("lumbar_extension").id  # Raises if joint is absent
            has_lumbar_extension = True
        except (KeyError, ValueError, TypeError):
            has_lumbar_extension = False

        if not self._enable_lumbar_joint:
            if 'lumbar_extension' in self.observation_joint_pos_keys:
                self.observation_joint_pos_keys.remove('lumbar_extension')
            if 'lumbar_extension' in self.observation_joint_vel_keys:
                self.observation_joint_vel_keys.remove('lumbar_extension')
            if has_lumbar_extension:
                # Fix the lumbar joint to a constant position and (optionally) remove it from observations
                self.sim.data.joint("lumbar_extension").qpos[0] = self._lumbar_joint_fixed_angle
                self.sim.model.jnt_range[lumbar_joint_id] = [
                    self._lumbar_joint_fixed_angle,
                    self._lumbar_joint_fixed_angle + 1e-6,
                ]


                # Adjust damping (whether the joint is fixed or not)
                dof_adr = self.sim.model.jnt_dofadr[lumbar_joint_id]
                joint_type = self.sim.model.jnt_type[lumbar_joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    dof_count = 6
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    dof_count = 3
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    dof_count = 1
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    dof_count = 1
                else:
                    dof_count = 0  # Currently unused

                self.sim.model.dof_damping[dof_adr] = self._lumbar_joint_damping_value
            else:
                self.sim.model.body("torso").quat = [1, 0, 0, self._lumbar_joint_fixed_angle]
        
        
        #phys: 1000hz
        # control 50hz : 50 * 20 = 1000hz
        # ref 50hz: 500hz 10skip: 20 * 500 / 1000

        frame_skip = env_params.physics_sim_framerate // env_params.control_framerate
        original_reward_dict = DictionableDataclass.to_dict(env_params.reward_keys_and_weights)
        self.rwd_keys_wt = {}
        for key, value in original_reward_dict.items():
            if type(value) == dict:
                weight_sum = sum(value.values())
                self.rwd_keys_wt[key] = weight_sum
            else:
                self.rwd_keys_wt[key] = value

        self._initialize_pose()



        # self.renderer = self.sim._create_renderer(self.sim)



        super()._setup(obs_keys=self.DEFAULT_OBS_KEYS,
                weighted_reward_keys=self.rwd_keys_wt,
                frame_skip=frame_skip,
                **kwargs,
                )

        
        # Check if the keys in DEFAULT_OBS_KEYS are in the keys of the observation dictionary
        obs_dict_keys = list(self.get_obs_dict(self.sim).keys())
        assert (set(self.DEFAULT_OBS_KEYS + ['time'])) == set(obs_dict_keys), f"DEFAULT_OBS_KEYS != get_obs_dict.keys. DEFAULT_OBS_KEYS: {self.DEFAULT_OBS_KEYS}, get_obs_dict keys: {obs_dict_keys}"
        actual_reward_keys = list(self.get_reward_dict(self.sim).keys())
        assert (set(list(self.rwd_keys_wt.keys()) + ['dense', 'sparse', 'solved', 'done'])) == set(actual_reward_keys), f"rwd_keys_wt != actual_reward_keys. rwd_keys_wt: {self.rwd_keys_wt}, actual_reward_keys keys: {actual_reward_keys}"
        
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]

        # find geometries with ID == 1 which indicates the skins
        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        # Change the alpha value to make it transparent
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0

        # move heightfield down if not used
        # self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        # self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

        self._terrain_type = env_params.terrain_type
        self._terrain_params = env_params.terrain_params
        self._hfield_manager = HfieldManager(self.sim, "terrain", self.np_random)
        self._hfield_manager.set_hfield(self._terrain_type, self._terrain_params)

        observation, _reward, done, *_, _info = self.step(np.zeros(self.sim.model.nu))
        # if qpos set to all zero, joint looks weird, 30 steps will make it normal
        for _ in range(30):
            super().step(a=np.zeros(self.sim.model.nu))

    # override from MujocoEnv
    def get_obs_dict(self, sim):
        # TODO observation - tx exclude
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time]) # they use time separately like t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)

        qpos = []
        for key in self.observation_joint_pos_keys:
            qpos.append(sim.data.joint(f"{key}").qpos[0].copy())
        qvel = []
        for key in self.observation_joint_vel_keys:
            qvel.append(sim.data.joint(f"{key}").qvel[0].copy())
        obs_dict['qpos'] = np.array(qpos) # 7 + 1 elements
        obs_dict['qvel'] = np.array(qvel) # 7 + 2 elements
        if sim.model.na>0:
            # BaseV0 Add the key like this: obs_keys.append("act")
            obs_dict['act'] = sim.data.act[:].copy() # 22 elements
        obs_dict['sensor'] = []
        for key in self.observation_sensor_keys:
            sensor_data = sim.data.sensor(f"{key}").data.copy()
            if "foot" in key or "toes" in key:
                model_mass = np.sum(self.sim.model.body_mass)
                sensor_data = sensor_data / (model_mass * 9.81)
            obs_dict['sensor'].extend(sensor_data)
        obs_dict['sensor'] = np.array(obs_dict['sensor'])

        obs_dict['target_velocity'] = np.array([self._target_velocity])

        return obs_dict
    def _calculate_base_reward(self, obs_dict):
        model_mass = np.sum(self.sim.model.body_mass)
        # print(f"DEBUG:: model_mass: {model_mass}")
        model_weight = model_mass * 9.81 # in Newtons

        forward_reward = self.dt * np.exp(-5 * np.square(self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity))

        muscle_activations = self._get_muscle_activation()
        muscle_activation_penalty = - self.dt * np.mean(muscle_activations)

        # Hip muscle activation penalty: only penalize muscles directly assisted by hip exo
        # Indices: hamstrings(2,15), glutmax(6,19), iliopsoas(7,20), rectfem(8,21)
        hip_muscle_indices = [2, 6, 7, 8, 15, 19, 20, 21]
        activation_penalty = np.mean(np.square(muscle_activations[hip_muscle_indices]))
        hip_muscle_activation_penalty = - self.dt * activation_penalty

        raw_joint_constraint_force = self._get_max_joint_constraint_force() / (model_weight)
        joint_constraint_force_penalty = - self.dt * np.clip(raw_joint_constraint_force, 0, 1000)

        if self._prev_muscle_activations_for_reward is not None:
            muscle_activation_diff_penalty = self.dt * np.mean(np.exp(-4 * np.square(self._prev_muscle_activations_for_reward - muscle_activations)))
        else:
            muscle_activation_diff_penalty = 0
        self._prev_muscle_activations_for_reward = muscle_activations

        
        normalized_foot_force_sum = (np.abs(self._get_foot_force('r')) + np.abs(self._get_foot_force('l'))) / model_weight
        # print(f"DEBUG:: normalized_foot_force_sum: {normalized_foot_force_sum}")
        # e^(-max(0, f/w - 1))
        # foot_force_penalty = self.dt * np.exp(-np.maximum(0, normalized_foot_force_sum - 1))
        foot_force_penalty = - self.dt * np.clip(max(0, normalized_foot_force_sum - 1.2), 0, 100)
        # print(f"DEBUG:: foot_force_penalty: {foot_force_penalty}")

        # Exo assistance power reward: positive power = assistance, negative = resistance
        exo_assistance_power_reward = self._calculate_exo_assistance_power_reward()

        # Exo action rate penalty: r = -dt * sum((tau_t - tau_{t-1})^2) on normalized exo action
        if getattr(self, '_last_applied_exo_action', None) is not None and getattr(self, '_prev_exo_action', None) is not None:
            delta = self._last_applied_exo_action - self._prev_exo_action
            exo_action_rate_penalty = -self.dt * float(np.sum(delta ** 2))
        else:
            exo_action_rate_penalty = 0.0
        if getattr(self, '_last_applied_exo_action', None) is not None:
            self._prev_exo_action = self._last_applied_exo_action.copy()

        base_reward = {
            'forward_reward': forward_reward,
            'muscle_activation_penalty': muscle_activation_penalty,
            'hip_muscle_activation_penalty': hip_muscle_activation_penalty,
            'muscle_activation_diff_penalty': muscle_activation_diff_penalty,
            'foot_force_penalty': foot_force_penalty,
            'joint_constraint_force_penalty': joint_constraint_force_penalty,
            'exo_assistance_power_reward': exo_assistance_power_reward,
            'exo_action_rate_penalty': exo_action_rate_penalty,
        }
        info = {
            "muscle_activations": muscle_activations,
        }
        return base_reward, info
    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):

        base_reward, info = self._calculate_base_reward(obs_dict)

        # Automatically add all base_reward items to rwd_dict
        rwd_dict = collections.OrderedDict((key, base_reward[key]) for key in base_reward)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),  # env will use this to determine if the episode is over (see _forward in env_base.py)
        })
        # rwd_keys_wt: from MujocoEnv
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def step(self, a, **kwargs):
        self._modulate_target_velocity()
        # Store normalized exo action [26:28] for exo_action_rate_penalty (used in get_reward_dict via _calculate_base_reward)
        if len(a) >= 28:
            self._last_applied_exo_action = np.asarray(a[26:28], dtype=np.float64).copy()
        else:
            self._last_applied_exo_action = None
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        self._step_count_per_episode += 1
        is_over_time_limit = self._step_count_per_episode >= self.CUSTOM_MAX_EPISODE_STEPS
        
        return (next_obs, reward, terminated, truncated or is_over_time_limit, info)
    def just_forward(self):
        self.sim.forward()
    def set_target_velocity_mode_manually(self, mode:VelocityMode,
                                          starting_phase:float,
                                          initial_target_velocity:float,
                                          min_target_velocity:float,
                                          max_target_velocity:float,
                                          target_velocity_period:float = None):
        self._velocity_mode_for_this_episode = mode
        self._starting_phase = starting_phase
        if mode == MyoAssistLegBase.VelocityMode.SINUSOIDAL and target_velocity_period is None:
            raise ValueError("target_velocity_period must be provided for sinusoidal mode")
        self._target_velocity_period = target_velocity_period
        # self._modulate_target_velocity()
        self._target_velocity = initial_target_velocity
        self._prev_step_changed_time = self.sim.data.time

        self._min_target_velocity = min_target_velocity
        self._max_target_velocity = max_target_velocity
    def _change_mode_and_target_velocity_randomly(self):
        velocity_mode_for_this_episode = random.choice(list(MyoAssistLegBase.VelocityMode))
        starting_phase = random.uniform(0, 2 * np.pi)
        target_velocity_period = random.uniform(self._min_target_velocity_period, self._max_target_velocity_period) # maximum acc/dec is self._target_velocity_period / 2
        self.set_target_velocity_mode_manually(velocity_mode_for_this_episode,
                                               starting_phase,
                                               self._min_target_velocity,
                                               self._min_target_velocity,
                                               self._max_target_velocity,
                                               target_velocity_period)
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            self._target_velocity = random.uniform(self._min_target_velocity, self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.SINUSOIDAL:
            self._target_velocity = self._calc_sinusoidal_target_velocity(self._starting_phase,
                                                                          self._target_velocity_period,
                                                                          self._min_target_velocity,
                                                                          self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            self._target_velocity = np.random.uniform(self._min_target_velocity, self._max_target_velocity)


    def _calc_sinusoidal_target_velocity(self, phase:float, period:float, min_velocity:float, max_velocity:float):
        return min_velocity\
            + (max_velocity - min_velocity)\
                  * (np.sin(phase + 2 * np.pi * self.sim.data.time / (period)) + 1) / 2
    def _modulate_target_velocity(self):
        if self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.UNIFORM:
            # print(f"DEBUG:: {self.is_evaluate_mode} (mode:{self._velocity_mode_for_this_episode}, starting_phase:{self._starting_phase}, target_velocity_period:{self._target_velocity_period})")
            pass
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.SINUSOIDAL:
            self._target_velocity = self._calc_sinusoidal_target_velocity(self._starting_phase,
                                                                          self._target_velocity_period,
                                                                          self._min_target_velocity,
                                                                          self._max_target_velocity)
        elif self._velocity_mode_for_this_episode == MyoAssistLegBase.VelocityMode.STEP:
            if self.sim.data.time - self._prev_step_changed_time > self._target_velocity_period:
                self._target_velocity = np.random.uniform(self._min_target_velocity, self._max_target_velocity)
                self._prev_step_changed_time = self.sim.data.time
    def reset(self, **kwargs):
        self._step_count_per_episode = 0
        if not self.is_evaluate_mode:
            self._change_mode_and_target_velocity_randomly()
        self.sim.data.joint("pelvis_tx").qvel[0] = self._target_velocity

        self.sim.forward()
        # sync targets to sim_obsd
        self.robot.sync_sims(self.sim, self.sim_obsd)

        self._prev_exo_action = None  # so the first step after reset has no exo_action_rate_penalty

        # generate resets
        # obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        obs = super().reset(**kwargs)
        return obs
    
    def _get_done(self):
        pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
        if pelvis_height < self._safe_height:
            return True
        return False
    
    def _get_muscle_activation(self):
        # Lumbar joint is passive (no motor actuator), so we don't include it
        # Only return the actual muscle activations
        return self.sim.data.act[:].copy()
    def _get_max_joint_constraint_force(self):
        max_constraint_force = 0
        for sensor_name in self.joint_limit_sensor_keys:
            sensor_data = self.sim.data.sensor(sensor_name).data[0].copy()
            max_constraint_force = max(max_constraint_force, np.max(np.abs(sensor_data)))
        return max_constraint_force
    # ============ Custon Function ==============
    def _get_foot_force(self, foot_side_alphabet:str):
        foot_force = self.sim.data.sensor(f"{foot_side_alphabet}_foot").data.copy()[0] + self.sim.data.sensor(f"{foot_side_alphabet}_toes").data.copy()[0]
        return foot_force

    def _calculate_exo_assistance_power_reward(self):
        """
        Calculate exo assistance power reward: reward = tau * omega (normalized).

        Power-based reward naturally handles zero-crossing:
        - When velocity is zero, reward is zero regardless of torque → no incentive to
          maintain torque at gait transition points.
        - When velocity is large, torque in the same direction gives positive reward.
        - A small torque regularization continuously pulls torque toward zero, so the exo
          only outputs torque when the power reward outweighs the regularization cost.

        IMPORTANT - Conflict with imitation reward:
        If qpos/qvel_imitation_rewards include hip_flexion_l/r, the policy can learn to
        resist (torque opposite to velocity): resisting keeps hip closer to the reference
        trajectory (which is human-only, no exo), so imitation reward rises and can
        dominate exo_assistance_power_reward. Fix: set hip_flexion_l and hip_flexion_r
        to 0 in both qpos_imitation_rewards and qvel_imitation_rewards when training exo.
        """
        if self._exo_assistance_reward_type == "power_sign":
            return self._calculate_exo_assistance_power_sign_reward()
        if self._exo_assistance_reward_type == "direction":
            return self._calculate_exo_assistance_direction_reward()

        total_reward = 0.0
        exo_motor_names = ['exo_right_flexion_motor', 'exo_left_flexion_motor']
        hip_joint_names = ['hip_flexion_r', 'hip_flexion_l']

        # Power-based reward with torque regularization + saturation penalty (Stage 4)
        # Parameters
        power_reward_scale = 0.3          # Scale for power reward (tau * omega)
        vel_scale = 2.0                   # Normalization for hip angular velocity (~peak rad/s during walking)
        torque_reg_scale = 0.15           # Continuous penalty on torque² to prevent bang-bang
        # Optimal torque (from power+reg alone): τ* = power_reward_scale / (2 * torque_reg_scale) * ω_norm
        #   = 0.3 / 0.3 * ω_norm = 1.0 * ω_norm
        # Saturation penalty below further limits torque near ±1

        # Saturation penalty: sharply penalizes torque near ±1 to prevent saturation
        # This is separate from torque_reg to create a "soft wall" that external rewards
        # (like hip_muscle_activation_penalty) cannot easily overcome
        saturation_threshold = 0.8        # No penalty below this normalized torque
        saturation_penalty_scale = 2.0    # Penalty scale for exceeding threshold

        for motor_name, joint_name in zip(exo_motor_names, hip_joint_names):
            try:
                # Get actuator force (torque)
                actuator_id = self.sim.model.actuator(motor_name).id
                torque = self.sim.data.actuator_force[actuator_id]

                # Get max_torque from model's forcerange (automatically reads from XML)
                forcerange = self.sim.model.actuator_forcerange[actuator_id]
                max_torque = max(abs(forcerange[0]), abs(forcerange[1]))
                if max_torque == 0:
                    max_torque = 10.0  # fallback default

                # Normalize torque to [-1, 1] range
                normalized_torque = torque / max_torque

                # Get joint angular velocity and normalize
                velocity = self.sim.data.joint(joint_name).qvel[0]
                normalized_velocity = np.clip(velocity / vel_scale, -1.0, 1.0)

                # Power reward: tau_norm * vel_norm
                # Positive when torque matches velocity direction (assistance)
                # Negative when torque opposes velocity (resistance)
                # Zero when velocity is zero (no incentive at transition)
                total_reward += power_reward_scale * normalized_torque * normalized_velocity

                # Torque regularization: small continuous pull toward zero
                # Prevents bang-bang; exo only outputs torque when power reward justifies it
                total_reward -= torque_reg_scale * (normalized_torque ** 2)

                # Saturation penalty: kicks in only when |torque| > threshold
                # Quadratic penalty grows sharply near ±1 to prevent saturation
                excess = max(0.0, abs(normalized_torque) - saturation_threshold)
                total_reward -= saturation_penalty_scale * (excess ** 2)

            except (KeyError, ValueError, AttributeError):
                # Motor or joint doesn't exist in this model
                continue

        return self.dt * total_reward

    def _calculate_exo_assistance_power_sign_reward(self):
        """Stage 3 exo reward (Eq. 4 of the paper): dt * sum_j 0.5 * u_j^2 * sign(u_j * omega_j),
        with u_j the normalized exo torque and omega_j the hip flexion velocity."""
        total_reward = 0.0
        for motor_name, joint_name in zip(['exo_right_flexion_motor', 'exo_left_flexion_motor'], ['hip_flexion_r', 'hip_flexion_l']):
            actuator_id = self.sim.model.actuator(motor_name).id
            forcerange = self.sim.model.actuator_forcerange[actuator_id]
            max_torque = max(abs(forcerange[0]), abs(forcerange[1]))
            normalized_torque = self.sim.data.actuator_force[actuator_id] / max_torque
            velocity = self.sim.data.joint(joint_name).qvel[0]
            total_reward += 0.5 * normalized_torque ** 2 * np.sign(normalized_torque * velocity)
        return self.dt * total_reward

    def _calculate_exo_assistance_direction_reward(self):
        """Direction-matching exo reward used to train the released Stage 3 checkpoint.

        Per hip: 0.5 * (s * c + max(0, s) * c^2) when |omega| > 0.03 rad/s, with c = min(|u|, 0.85) and
        s = sign(u * omega), minus 0.4 * max(0, |u| - 0.85)^2; u is the normalized exo torque.
        """
        total_reward = 0.0
        for motor_name, joint_name in zip(['exo_right_flexion_motor', 'exo_left_flexion_motor'], ['hip_flexion_r', 'hip_flexion_l']):
            actuator_id = self.sim.model.actuator(motor_name).id
            forcerange = self.sim.model.actuator_forcerange[actuator_id]
            max_torque = max(abs(forcerange[0]), abs(forcerange[1]))
            normalized_torque = self.sim.data.actuator_force[actuator_id] / max_torque
            velocity = self.sim.data.joint(joint_name).qvel[0]
            if abs(velocity) > 0.03:
                clamped = min(abs(normalized_torque), 0.85)
                sign_match = np.sign(normalized_torque * velocity)
                total_reward += 0.5 * (clamped * sign_match + clamped ** 2 * max(0.0, sign_match))
            total_reward -= 0.4 * max(0.0, abs(normalized_torque) - 0.85) ** 2
        return self.dt * total_reward

    # To override
    def _initialize_pose(self):
        self.sim.data.qpos[:] = self.sim.model.key_qpos[0][:]
        self.just_forward()