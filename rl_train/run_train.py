import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
import os
from datetime import datetime
from rl_train.envs.environment_handler import EnvironmentHandler
import subprocess

def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        return {
            "commit": commit,
            "branch": branch
        }
    except:
        return {
            "commit": "unknown",
            "branch": "unknown"
        }

def get_parent_session_dir(checkpoint_path):
    """Session folder of a checkpoint: <session>/trained_models/<model> or <session>/<model>."""
    if '/trained_models/' in checkpoint_path:
        return checkpoint_path.split('/trained_models/')[0]
    return os.path.dirname(checkpoint_path)

# Version information
VERSION = {
    "version": "0.3.0",  # MAJOR.MINOR.PATCH
    **get_git_info()
}
def ppo_evaluate_with_rendering(config):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=True, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    obs, info = env.reset()
    for _ in range(config.evaluate_param_list[0]["num_timesteps"]):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        if truncated:
            obs, info = env.reset()

    env.close()
def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION

    # Record parent session info if fine-tuning
    if hasattr(config.env_params, 'prev_trained_policy_path') and config.env_params.prev_trained_policy_path:
        checkpoint_path = config.env_params.prev_trained_policy_path
        parent_session = os.path.basename(get_parent_session_dir(checkpoint_path))
        checkpoint_name = checkpoint_path.split('/')[-1].replace('.zip', '')
        session_config_dict["parent_session"] = {
            "session_name": parent_session,
            "checkpoint": checkpoint_name,
            "checkpoint_path": checkpoint_path
        }

    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)


    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=custom_callback, progress_bar=True)
    env.close()
    print("learning done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument("--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)")
    parser.add_argument("--flag_realtime_evaluate", type=bool, default=False, action=argparse.BooleanOptionalAction, help="realtime evaluate(True/False)")

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)


    DictionableDataclass.set_from_args(config, args, prefix="config.")

    prev = config.env_params.prev_trained_policy_path
    if prev and "<" in prev:
        raise ValueError(f"Set env_params.prev_trained_policy_path to a checkpoint of the previous stage (got '{prev}'), e.g. "
                         "--config.env_params.prev_trained_policy_path results/<session>/trained_models/model_<step>")

    log_dir = os.path.join("results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # If continuing from previous checkpoint, inherit parent's training log
    if hasattr(config.env_params, 'prev_trained_policy_path') and config.env_params.prev_trained_policy_path:
        import shutil
        # Extract parent session directory from checkpoint path
        # e.g., "results/train_session_<date>-<time>/trained_models/model_<step>"
        checkpoint_path = config.env_params.prev_trained_policy_path
        parent_session_dir = get_parent_session_dir(checkpoint_path)
        parent_train_log = os.path.join(parent_session_dir, 'train_log.json')

        if os.path.exists(parent_train_log):
            dest_train_log = os.path.join(log_dir, 'train_log.json')
            shutil.copy2(parent_train_log, dest_train_log)
            print(f"Inherited training log from parent session: {parent_session_dir}")
        else:
            print(f"Warning: Parent training log not found at {parent_train_log}")

    train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
    else:
        ppo_train_with_parameters(config,
                                train_time_step=config.total_timesteps,
                                is_rendering_on=args.flag_rendering,
                                train_log_handler=train_log_handler)
    