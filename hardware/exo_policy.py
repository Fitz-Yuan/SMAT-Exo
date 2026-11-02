"""Standalone inference for the SMAT exoskeleton actor (hardware deployment).

Loads only the exo actor from a released checkpoint and maps bilateral hip kinematics
to normalized hip torque commands. Depends on numpy and torch only, so it can run on
an embedded computer (a Raspberry Pi 4B in the paper) without MuJoCo or the training code.

Observation (18 values, same as in simulation):
    2 past input frames (8) + current input frame (4) + 3 past output frames (6)
    input frame:  [R_angle, L_angle, R_velocity, L_velocity]  (rad, rad/s, flexion positive)
    output frame: [R_command, L_command]                       (normalized, in [-1, 1])

Output: normalized torque [R, L] in [-1, 1], flexion positive. Multiply by the per-hip
torque limit of your device to get Nm. The policy was trained at 50 Hz; call step() at
the same rate. In the paper, hip velocity from the motor encoders was smoothed with a
One Euro filter before being passed in.

Example:
    policy = SMATExoPolicy("checkpoints/stage4_coadaptation/model.zip")
    while running:                                # 50 Hz loop
        u_r, u_l = policy.step(r_ang, l_ang, r_vel, l_vel)
        send_torque(right=u_r * TORQUE_LIMIT_NM, left=u_l * TORQUE_LIMIT_NM)
"""
import io
import zipfile
from collections import deque

import numpy as np
import torch
import torch.nn as nn

INPUT_HISTORY_FRAMES = 2
OUTPUT_HISTORY_FRAMES = 3
# Exchanges right and left entries of every frame (see CustomNetworkHumanExo.EXO_SWAP_INDICES)
SWAP_INDICES = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]


class SMATExoPolicy:
    def __init__(self, checkpoint_path: str, symmetric: bool = True):
        """
        checkpoint_path: a released stable-baselines3 checkpoint (.zip) that contains an exo actor.
        symmetric: evaluate the network twice with left/right swapped inputs, as during training
                   (exo_symmetric_forward=true in every released exo config).
        """
        self.net = nn.Sequential(
            nn.Linear(18, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2), nn.Tanh(),
        )
        with zipfile.ZipFile(checkpoint_path) as z:
            state_dict = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
        prefix = "policy_network.exo_policy_net."
        self.net.load_state_dict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)})
        self.net.eval()
        self.symmetric = symmetric
        self._swap_idx = torch.tensor(SWAP_INDICES, dtype=torch.long)
        self.reset()

    def reset(self):
        """Clear the observation history (call before each walking trial)."""
        self._input_history = deque([np.zeros(4)] * INPUT_HISTORY_FRAMES, maxlen=INPUT_HISTORY_FRAMES)
        self._output_history = deque([np.zeros(2)] * OUTPUT_HISTORY_FRAMES, maxlen=OUTPUT_HISTORY_FRAMES)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map one 18-dim observation to normalized torque [R, L] without touching the history."""
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, 18)
        if self.symmetric:
            right = self.net(x)[:, 0]
            left = self.net(x[:, self._swap_idx])[:, 0]
            return torch.stack([right, left], dim=-1)[0].numpy()
        return self.net(x)[0].numpy()

    def step(self, r_angle: float, l_angle: float, r_velocity: float, l_velocity: float) -> np.ndarray:
        """Feed the current hip state (rad, rad/s, flexion positive) and get normalized torque [R, L]."""
        current = np.array([r_angle, l_angle, r_velocity, l_velocity], dtype=np.float64)
        obs = np.concatenate([*self._input_history, current, *self._output_history])
        command = self.act(obs)
        self._input_history.append(current)
        self._output_history.append(command.astype(np.float64))
        return command


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the exo actor on a synthetic hip trajectory.")
    parser.add_argument("checkpoint", help="path to a released checkpoint .zip")
    parser.add_argument("--torque_limit", type=float, default=15.0, help="per-hip torque limit in Nm")
    args = parser.parse_args()

    policy = SMATExoPolicy(args.checkpoint)
    dt, stride = 0.02, 1.1  # 50 Hz control, ~1.1 s gait cycle
    print(" t(s)  R_ang(deg)  R_cmd   L_cmd   R_torque(Nm)")
    for k in range(int(2 * stride / dt)):
        t = k * dt
        phase = 2 * np.pi * t / stride
        r_ang, r_vel = 0.45 * np.sin(phase) + 0.15, 0.45 * 2 * np.pi / stride * np.cos(phase)
        l_ang, l_vel = 0.45 * np.sin(phase + np.pi) + 0.15, 0.45 * 2 * np.pi / stride * np.cos(phase + np.pi)
        u = policy.step(r_ang, l_ang, r_vel, l_vel)
        if k % 5 == 0:
            print(f"{t:5.2f}  {np.degrees(r_ang):9.1f}  {u[0]:6.3f}  {u[1]:6.3f}  {u[0] * args.torque_limit:9.2f}")
