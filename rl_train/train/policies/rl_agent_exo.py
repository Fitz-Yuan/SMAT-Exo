import torch
import torch.nn as nn
import torch.nn.init as init

from abc import ABC, abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict

from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, Tuple

import gymnasium as gym

from rl_train.utils.data_types import DictionableDataclass
import rl_train.train.train_configs.config as myoassist_config
from rl_train.train.policies.network_index_handler import NetworkIndexHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
import torch
torch.autograd.set_detect_anomaly(True)

from rl_train.train.policies.rl_agent_base import BasePPOCustomNetwork, BaseCustomActorCriticPolicy

class CustomNetworkHumanExo(BasePPOCustomNetwork):
    # Swap indices to exchange L/R in exo_history observation:
    # Input frames: [R_ang, L_ang, R_vel, L_vel] × 3 → swap 0↔1, 2↔3 per frame
    # Output frames: [R_out, L_out] × 3 → swap 0↔1 per frame
    EXO_SWAP_INDICES = [1,0,3,2, 5,4,7,6, 9,8,11,10, 13,12, 15,14, 17,16]

    def __init__(self, observation_space: spaces.Space,
                    action_space: spaces.Space,
                    custom_policy_params: ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams,
                ):
        super().__init__(observation_space, action_space, custom_policy_params)
        self._exo_symmetric = getattr(custom_policy_params, 'exo_symmetric_forward', False)
        if self._exo_symmetric:
            self._swap_idx = th.tensor(self.EXO_SWAP_INDICES, dtype=th.long)
            print("[ExoSymmetric] Symmetric forward enabled: exo network will run twice with L/R-swapped input")

    def _forward_exo_symmetric(self, exo_obs: th.Tensor) -> th.Tensor:
        """Run exo network twice with L/R swap to enforce symmetric output."""
        swap_idx = self._swap_idx.to(exo_obs.device)
        # Original obs → right torque (use output[0])
        right_torque = self.exo_policy_net(exo_obs)[:, 0:1]
        # Swapped obs (L↔R) → left torque (use output[0])
        swapped_obs = exo_obs[:, swap_idx]
        left_torque = self.exo_policy_net(swapped_obs)[:, 0:1]
        return th.cat([right_torque, left_torque], dim=-1)

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        human_obs = self.network_index_handler.map_observation_to_network(obs, "human_actor")
        exo_obs = self.network_index_handler.map_observation_to_network(obs, "exo_actor")
        human_action = self.human_policy_net(human_obs)

        if self._exo_symmetric:
            exo_action = self._forward_exo_symmetric(exo_obs)
        else:
            exo_action = self.exo_policy_net(exo_obs)

        network_output_dict = {"human_actor": human_action, "exo_actor": exo_action}
        return self.network_index_handler.map_network_to_action(network_output_dict)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        value_obs = self.network_index_handler.map_observation_to_network(obs, "common_critic")
        return self.value_net(value_obs)

    def reset_policy_networks(self):
        self.network_index_handler = NetworkIndexHandler(self.net_indexing_info, self.observation_space, self.action_space)
        layers = []

        last_dim = self.network_index_handler.get_observation_num("human_actor")
        for dim in self.net_arch["human_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim

        layers.append(nn.Linear(last_dim, self.network_index_handler.get_action_num("human_actor")))
        layers.append(nn.Tanh())
        self.human_policy_net = nn.Sequential(*layers)


        layers = []
        last_dim = self.network_index_handler.get_observation_num("exo_actor")

        for dim in self.net_arch["exo_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim


        layers.append(nn.Linear(last_dim, self.network_index_handler.get_action_num("exo_actor")))
        layers.append(nn.Tanh())
        self.exo_policy_net = nn.Sequential(*layers)
    def reset_value_network(self):
        value_last_dim = self.network_index_handler.get_observation_num("common_critic")

        value_layers = []
        last_dim = value_last_dim
        for dim in self.net_arch["common_critic"]:
            value_layers.append(nn.Linear(last_dim, dim))
            value_layers.append(nn.Tanh())
            last_dim = dim
        value_layers.append(nn.Linear(last_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        # Move to same device as existing parameters
        device = next(self.parameters()).device
        self.value_net = self.value_net.to(device)

    def reset_exo_policy_network_only(self):
        from rl_train.train.policies.network_index_handler import NetworkIndexHandler

        self.network_index_handler = NetworkIndexHandler(
            self.net_indexing_info,
            self.observation_space,
            self.action_space
        )

        # Get device from existing network (human_policy_net)
        device = next(self.human_policy_net.parameters()).device

        layers = []
        last_dim = self.network_index_handler.get_observation_num("exo_actor")

        for dim in self.net_arch["exo_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim

        exo_action_size = self.network_index_handler.get_action_num("exo_actor")
        print(f" Rebuilding exo_policy_net with output size: {exo_action_size}")

        layers.append(nn.Linear(last_dim, exo_action_size))
        layers.append(nn.Tanh())
        self.exo_policy_net = nn.Sequential(*layers)

        # Move new network to the same device as human_policy_net
        self.exo_policy_net = self.exo_policy_net.to(device)

        print(f"Exo policy network rebuilt on {device} (human and critic kept intact)")

    def reset_human_policy_network_only(self):
        from rl_train.train.policies.network_index_handler import NetworkIndexHandler

        self.network_index_handler = NetworkIndexHandler(
            self.net_indexing_info,
            self.observation_space,
            self.action_space
        )

        # Get device from existing network (exo_policy_net)
        device = next(self.exo_policy_net.parameters()).device

        layers = []
        last_dim = self.network_index_handler.get_observation_num("human_actor")

        for dim in self.net_arch["human_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim

        human_action_size = self.network_index_handler.get_action_num("human_actor")
        print(f" Rebuilding human_policy_net with output size: {human_action_size}")

        layers.append(nn.Linear(last_dim, human_action_size))
        layers.append(nn.Tanh())
        self.human_policy_net = nn.Sequential(*layers)

        # Move new network to the same device as exo_policy_net
        self.human_policy_net = self.human_policy_net.to(device)

        print(f"Human policy network rebuilt on {device} (exo and critic kept intact)")

class HumanExoActorCriticPolicy(BaseCustomActorCriticPolicy):
    def _get_custom_policy_type(self):
        return ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams
    def _build_policy_network(self, observation_space: spaces.Space,
                              action_space: spaces.Space,
                              custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams) -> BasePPOCustomNetwork:
        return CustomNetworkHumanExo(observation_space,
                                    action_space,
                                    custom_policy_params)

    # --------------------------------------------------
    # Override forward to mask EXO actions when disabled
    # --------------------------------------------------
    def _get_exo_action_indices(self):
        """Return a list of action indices that correspond to the exo actuator."""
        idxs: list[int] = []
        info = self.policy_network.network_index_handler.net_indexing_info
        if "exo_actor" not in info:
            return idxs
        print(f"info: {info}")
        print(f'info: {info["exo_actor"]["action"][0]=}')
        for mapping in info["exo_actor"]["action"]:
            if mapping["type"] == "range_mapping":
                start_action, end_action = mapping["range_action"]
                idxs.extend(list(range(start_action, end_action)))
            elif mapping["type"] == "index_mapping":
                idxs.append(mapping["index"])
        return idxs

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """Forward pass that zeros EXO actions when exoskeleton is disabled."""
        # Standard forward logic (actor/critic)
        mean_actions = self.policy_network.forward_actor(obs)
        value = self.policy_network.forward_critic(obs)

        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)

        actions = distribution.get_actions(deterministic=deterministic)

        info = self.policy_network.network_index_handler.net_indexing_info
 
        # ------------------------
        # Mask invalid action slots
        # ------------------------
        # Cache index lists for efficiency
        if not hasattr(self, "_cached_indices_ready"):
            # Human-action indices
            self._human_indices = self._get_action_indices_for_net("human_actor")
            # Exo-action indices
            self._exo_indices = self._get_action_indices_for_net("exo_actor")
            # All indices that should remain untouched when exo is enabled
            self._valid_indices_enabled_exo = sorted(set(self._human_indices + self._exo_indices))
            # Valid indices when exo is disabled (exo part must be zero)
            self._valid_indices_disabled_exo = sorted(set(self._human_indices))
            self._cached_indices_ready = True


        # mask default value
        human_obs = self.policy_network.network_index_handler.map_observation_to_network(obs, "human_actor")
        exo_obs = self.policy_network.network_index_handler.map_observation_to_network(obs, "exo_actor")
        human_action = self.policy_network.human_policy_net(human_obs)

        if self.policy_network._exo_symmetric:
            exo_action = self.policy_network._forward_exo_symmetric(exo_obs)
        else:
            exo_action = self.policy_network.exo_policy_net(exo_obs)

        network_output_dict = {"human_actor": human_action, "exo_actor": exo_action}
        actions = self.policy_network.network_index_handler.mask_default_value(network_output_dict, actions)

        # # Zero any index that is not listed in valid_indices
        # if len(valid_indices) < self.action_space.shape[0]:
        #     all_indices = th.arange(self.action_space.shape[0], device=actions.device)
        #     mask = ~th.isin(all_indices, th.tensor(valid_indices, device=actions.device))
        #     actions[..., mask] = info["exo_actor"]["default_value"]
 
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

    # ---------- helper ----------
    def _get_action_indices_for_net(self, net_name: str):
        """Return sorted list of action indices used by a given sub-network."""
        idxs: list[int] = []
        info = self.policy_network.network_index_handler.net_indexing_info
        if net_name not in info:
            return idxs
        for mapping in info[net_name]["action"]:
            if mapping["type"] == "range_mapping":
                start_action, end_action = mapping["range_action"]
                idxs.extend(range(start_action, end_action))
            elif mapping["type"] == "index_mapping":
                idxs.append(mapping["index"])
        return sorted(idxs)
    def reset_network(self, reset_shared_net: bool = False, reset_policy_net: bool = False, reset_value_net: bool = False, reset_exo_only: bool = False, reset_human_only: bool = False):

        if reset_human_only:
            print(f"Resetting ONLY human policy network (keeping exo policy and critic)")
            self.policy_network.reset_human_policy_network_only()
        elif reset_exo_only:
            print(f"Resetting ONLY exo policy network (keeping human policy and critic)")
            self.policy_network.reset_exo_policy_network_only()
        elif reset_policy_net:
            print(f"Resetting entire policy network (both human and exo)")
            self.policy_network.reset_policy_networks()
        
        if reset_value_net:
            print(f"Resetting value network")
            self.policy_network.reset_value_network()

        self._apply_human_actor_freeze()
