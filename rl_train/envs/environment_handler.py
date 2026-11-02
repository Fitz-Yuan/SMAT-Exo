import numpy as np
import json
from stable_baselines3.common.vec_env import SubprocVecEnv
from myosuite.utils import gym
from rl_train.utils.data_types import DictionableDataclass
import os
from rl_train.train.train_configs.config import TrainSessionConfigBase
class EnvironmentHandler:
    @staticmethod
    def create_environment(config, is_rendering_on:bool, is_evaluate_mode:bool = False):

        ref_data_dict = EnvironmentHandler.load_reference_data(config)
    
        # Base gym.make arguments
        gym_make_args = {
            'seed': config.env_params.seed,
            'model_path': config.env_params.model_path,
            'env_params': config.env_params,
            'is_evaluate_mode': is_evaluate_mode
        }
        
        # Add reference_data only if it exists
        if ref_data_dict is not None:
            gym_make_args['reference_data'] = ref_data_dict
        
        try:
            if is_rendering_on or config.env_params.num_envs == 1:
                print(f"{config.env_params.env_id=}")
                env = gym.make(config.env_params.env_id, **gym_make_args).unwrapped
                if is_rendering_on:
                    env.mujoco_render_frames = True
                config.env_params.num_envs = 1
                config.ppo_params.n_steps = config.ppo_params.batch_size
            else:
                env = SubprocVecEnv([lambda: (gym.make(config.env_params.env_id, 
                                                    **gym_make_args)).unwrapped 
                                for _ in range(config.env_params.num_envs)])
        except Exception as e:
            new_message = str(e)[:1000]
            e.args = (new_message,)
            raise e
        return env

    @staticmethod
    def load_reference_data(config):
        # Check if config has reference_data_path attribute
        print("===================================================================")
        if not hasattr(config.env_params, 'reference_data_path'):
            print("No reference data path provided.")
            print("===================================================================")
            return None
            
        if not config.env_params.reference_data_path:
            print("No reference data path provided.")
            print("===================================================================")
            return None
        print(f"Loading reference data from {config.env_params.reference_data_path}")
        print("===================================================================")
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(config.env_params.reference_data_path, allow_pickle=True)
            ref_data_dict = {key: ref_data_npz[key].item() for key in ref_data_npz.files}
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(config.env_params.reference_data_path, 'r') as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use either .npz or .json.")

        if "resampled_series_data" not in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        return ref_data_dict

    def get_config_type_from_session_id(session_id):
        # from rl_train.envs import myo_leg_18_reward_per_step
        from rl_train.train.train_configs.config import TrainSessionConfigBase
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
        # Create appropriate config based on env_id
        print(f"session_id: {session_id}")
        if session_id == 'myoAssistLeg-v0':
            return TrainSessionConfigBase
        elif session_id in ['myoAssistLegImitation-v0']:
            return ImitationTrainSessionConfig
        elif session_id == 'myoAssistLegImitationExo-v0':
            return ExoImitationTrainSessionConfig
        raise ValueError(f"Invalid session id: {session_id}")
        

    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        print(f"Loading config from {config_path}")
        config_file_path = config_path
        with open(config_file_path, 'r') as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler):
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        
        from rl_train.envs import myoassist_leg_imitation
        from rl_train.utils import learning_callback
        if isinstance(config, ImitationTrainSessionConfig):
            custom_callback = myoassist_leg_imitation.ImitationCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                original_reward_weights=config.env_params.reward_keys_and_weights,
                auto_reward_adjust_params=config.auto_reward_adjust_params,
            )
        else:
            custom_callback = learning_callback.BaseCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
            )

        return custom_callback
    @staticmethod
    def get_stable_baselines3_model(config:TrainSessionConfigBase, env, trained_model_path:str|None=None):
        import stable_baselines3
        import torch
        import zipfile
        import io
        from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
        from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy

        if config.env_params.env_id in ["myoAssistLegImitationExo-v0"]:
            policy_class = HumanExoActorCriticPolicy
            print(f"Using HumanExoActorCriticPolicy")
        else:
            policy_class = HumanActorCriticPolicy
            print(f"Using HumanActorCriticPolicy")

        ppo_class = stable_baselines3.PPO
        if trained_model_path is not None:
            print(f"Loading trained model from {trained_model_path}")
            model = ppo_class.load(trained_model_path,
                                            env=env,
                                            custom_objects = {"policy_class": policy_class},
                                            )
        elif config.env_params.prev_trained_policy_path:
            print(f"Loading previous trained policy from {config.env_params.prev_trained_policy_path}")

            # Check if we need cross-dimension loading (reset_exo_only_after_load)
            reset_exo_only = getattr(config.policy_params.custom_policy_params, 'reset_exo_only_after_load', False)

            if reset_exo_only:
                print(f"Cross-dimension loading mode: Creating new model and loading human_actor weights only")

                # Step 1: Create a new PPO model with the new environment
                model = ppo_class(
                    policy=policy_class,
                    env=env,
                    policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                    verbose=2,
                    **DictionableDataclass.to_dict(config.ppo_params),
                )

                # Step 2: Load weights from the old model file
                old_model_path = config.env_params.prev_trained_policy_path
                if not old_model_path.endswith('.zip'):
                    old_model_path += '.zip'

                with zipfile.ZipFile(old_model_path, 'r') as z:
                    with z.open('policy.pth') as f:
                        old_state_dict = torch.load(io.BytesIO(f.read()), map_location=model.device)

                # Step 3: Map old weights to new network
                new_state_dict = model.policy.state_dict()

                # Check if old model has single policy_net (stage 1) or human_policy_net (multi-agent)
                if 'policy_network.policy_net.0.weight' in old_state_dict:
                    # Old model is single-agent (stage 1), map to human_policy_net
                    print("Detected single-agent model, mapping policy_net -> human_policy_net")
                    weight_mapping = {
                        'policy_network.policy_net.0.weight': 'policy_network.human_policy_net.0.weight',
                        'policy_network.policy_net.0.bias': 'policy_network.human_policy_net.0.bias',
                        'policy_network.policy_net.2.weight': 'policy_network.human_policy_net.2.weight',
                        'policy_network.policy_net.2.bias': 'policy_network.human_policy_net.2.bias',
                        'policy_network.policy_net.4.weight': 'policy_network.human_policy_net.4.weight',
                        'policy_network.policy_net.4.bias': 'policy_network.human_policy_net.4.bias',
                    }
                    for old_key, new_key in weight_mapping.items():
                        if old_key in old_state_dict and new_key in new_state_dict:
                            if old_state_dict[old_key].shape == new_state_dict[new_key].shape:
                                new_state_dict[new_key] = old_state_dict[old_key]
                                print(f"  Copied: {old_key} -> {new_key}")
                            else:
                                old_shape = old_state_dict[old_key].shape
                                new_shape = new_state_dict[new_key].shape
                                # Observation dimension increase: copy the old input weights; the new input columns keep their random initialization
                                if (len(old_shape) == 2 and len(new_shape) == 2
                                        and old_shape[0] == new_shape[0]
                                        and old_shape[1] < new_shape[1]):
                                    new_state_dict[new_key][:, :old_shape[1]] = old_state_dict[old_key]
                                    print(f"  Expanded input: {old_key} {old_shape} -> {new_shape} (new columns randomly initialized)")
                                elif (len(old_shape) == 1 and len(new_shape) == 1
                                        and old_shape[0] == new_shape[0]):
                                    new_state_dict[new_key] = old_state_dict[old_key]
                                    print(f"  Copied (bias): {old_key} -> {new_key}")
                                else:
                                    print(f"  Shape mismatch (skipped): {old_key} {old_shape} vs {new_key} {new_shape}")
                else:
                    # Old model is multi-agent, copy human_policy_net directly
                    print("Detected multi-agent model, copying human_policy_net directly")
                    for key in old_state_dict:
                        if 'human_policy_net' in key and key in new_state_dict:
                            if old_state_dict[key].shape == new_state_dict[key].shape:
                                new_state_dict[key] = old_state_dict[key]
                                print(f"  Copied: {key}")
                            else:
                                old_shape = old_state_dict[key].shape
                                new_shape = new_state_dict[key].shape
                                if (len(old_shape) == 2 and len(new_shape) == 2
                                        and old_shape[0] == new_shape[0]
                                        and old_shape[1] < new_shape[1]):
                                    new_state_dict[key][:, :old_shape[1]] = old_state_dict[key]
                                    print(f"  Expanded input: {key} {old_shape} -> {new_shape} (new columns randomly initialized)")
                                else:
                                    print(f"  Shape mismatch (skipped): {key} {old_shape} vs {new_shape}")

                # Copy log_std: if shapes match, copy all; if old is smaller, copy human portion
                if 'log_std' in old_state_dict and 'log_std' in new_state_dict:
                    old_log_std = old_state_dict['log_std']
                    new_log_std = new_state_dict['log_std']
                    if old_log_std.shape == new_log_std.shape:
                        new_state_dict['log_std'] = old_log_std
                        print(f"  Copied: log_std (full)")
                    elif old_log_std.shape[0] < new_log_std.shape[0]:
                        # Old model has fewer actions (e.g., Stage2 without exo actions)
                        # Copy old log_std into the human portion, keep exo portion at init value
                        new_state_dict['log_std'][:old_log_std.shape[0]] = old_log_std
                        print(f"  Partial copied: log_std (human portion {old_log_std.shape[0]}/{new_log_std.shape[0]}), exo portion kept at init")

                # Load the updated state dict
                model.policy.load_state_dict(new_state_dict)
                print("Successfully loaded human_actor weights from previous model")

                # Reset exo network (it's already randomly initialized, but call reset for consistency)
                model.policy.reset_network(reset_exo_only=True)
                # reset_network() replaces exo_policy_net after the optimizer was built; rebuild the
                # optimizer so that the new exo parameters are trained
                model.policy.optimizer = model.policy.optimizer_class(
                    model.policy.parameters(), lr=model.lr_schedule(1), **model.policy.optimizer_kwargs)
            else:
                # Standard loading (same observation space expected)
                try:
                    # Build the network from the config (not from the checkpoint) so that an extended
                    # observation in the config triggers the cross-dimension loading below
                    model = ppo_class.load(config.env_params.prev_trained_policy_path,
                                                    env=env,
                                                    custom_objects = {"policy_class": policy_class,
                                                                      "policy_kwargs": DictionableDataclass.to_dict(config.policy_params)},
                                                    verbose=2,
                                                    **DictionableDataclass.to_dict(config.ppo_params),
                                                    )
                except RuntimeError as e:
                    if "size mismatch" not in str(e):
                        raise
                    # Dimension mismatch detected - fall back to cross-dimension loading
                    print(f"Dimension mismatch during standard loading, switching to cross-dimension loading")
                    model = ppo_class(
                        policy=policy_class,
                        env=env,
                        policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                        verbose=2,
                        **DictionableDataclass.to_dict(config.ppo_params),
                    )
                    old_model_path = config.env_params.prev_trained_policy_path
                    if not old_model_path.endswith('.zip'):
                        old_model_path += '.zip'
                    with zipfile.ZipFile(old_model_path, 'r') as z:
                        with z.open('policy.pth') as f:
                            old_state_dict = torch.load(io.BytesIO(f.read()), map_location=model.device)
                    new_state_dict = model.policy.state_dict()
                    for key in old_state_dict:
                        if key not in new_state_dict:
                            continue
                        if old_state_dict[key].shape == new_state_dict[key].shape:
                            new_state_dict[key] = old_state_dict[key]
                        else:
                            old_shape = old_state_dict[key].shape
                            new_shape = new_state_dict[key].shape
                            if (len(old_shape) == 2 and len(new_shape) == 2
                                    and old_shape[0] == new_shape[0]
                                    and old_shape[1] < new_shape[1]):
                                new_state_dict[key][:, :old_shape[1]] = old_state_dict[key]
                                print(f"  Expanded input: {key} {old_shape} -> {new_shape} (new columns randomly initialized)")
                            else:
                                print(f"  Shape mismatch (skipped): {key} {old_shape} vs {new_shape}")
                    model.policy.load_state_dict(new_state_dict)
                    print("Successfully loaded all weights, with expanded inputs where needed")

                reset_human_only = getattr(config.policy_params.custom_policy_params, 'reset_human_only_after_load', False)
                if reset_human_only:
                    print(f"Resetting human policy network only (keeping exo and critic)")
                    model.policy.reset_network(reset_human_only=True)
                else:
                    model.policy.reset_network(reset_shared_net=config.policy_params.custom_policy_params.reset_shared_net_after_load,
                                            reset_policy_net=config.policy_params.custom_policy_params.reset_policy_net_after_load,
                                            reset_value_net=config.policy_params.custom_policy_params.reset_value_net_after_load)
        else:
            model = ppo_class(
                policy=policy_class,
                env=env,
                policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
        return model
    @staticmethod
    def updateconfig_from_model_policy(config, model):
        pass
        # config.policy_info.extractor_policy_net = f"{model.policy.mlp_extractor.policy_net}"
        # config.policy_info.extractor_value_net = f"{model.policy.mlp_extractor.value_net}"
        # config.policy_info.action_net = f"{model.policy.action_net}"
        # config.policy_info.value_net = f"{model.policy.value_net}"
        # config.policy_info.ortho_init = f"{model.policy.ortho_init}"
        # config.policy_info.share_features_extractor = f"{model.policy.share_features_extractor}"
