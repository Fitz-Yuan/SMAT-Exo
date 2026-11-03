# SMAT-Exo

**SMAT: Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control** (IROS 2026)

[Project page](https://fitz-yuan.github.io/SMAT/) · [Paper](https://arxiv.org/abs/2603.07618)

<p align="center"><img src="assets/framework.png" width="900" alt="Overview of the SMAT framework"></p>

SMAT trains a musculoskeletal human (26 muscles) and a bilateral hip exoskeleton as two PPO actors, in four stages:

| Stage | What is trained | Config |
|---|---|---|
| 1. Baseline gait | Human imitates a 1.25 m/s reference gait, no exoskeleton | [`stage1_baseline_gait.json`](configs/stage1_baseline_gait.json) |
| 2. Added exo mass | Human adapts to the exoskeleton mass, no assistance | [`stage2_exo_mass.json`](configs/stage2_exo_mass.json) |
| 3. Assistance timing | Exo learns when to assist, human frozen | [`stage3_assist_timing.json`](configs/stage3_assist_timing.json) |
| 4. Co-adaptation | Human and exo adapt to each other | [`stage4_coadaptation.json`](configs/stage4_coadaptation.json) |

Built on [MyoAssist](https://github.com/neumovelab/myoassist) and [MyoSuite](https://github.com/MyoHub/myosuite).

## Installation

```bash
git clone https://github.com/Fitz-Yuan/SMAT-Exo.git
cd SMAT-Exo
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

- Tested with Python 3.11 on Ubuntu 22.04.
- Run all commands from `rl_train/`.
- No display? Set `MUJOCO_GL=egl` (or `osmesa`) for the evaluation videos.

## Training

Train the stages in order; each stage starts from a checkpoint of the previous one.

```bash
cd rl_train

# Stage 1 (from scratch)
python run_train.py --config_file_path ../configs/stage1_baseline_gait.json

# Stage 2
python run_train.py --config_file_path ../configs/stage2_exo_mass.json \
    --config.env_params.prev_trained_policy_path results/<stage 1 session>/trained_models/model_<step>

# Stage 3
python run_train.py --config_file_path ../configs/stage3_assist_timing.json \
    --config.env_params.prev_trained_policy_path results/<stage 2 session>/trained_models/model_<step>

# Stage 4
python run_train.py --config_file_path ../configs/stage4_coadaptation.json \
    --config.env_params.prev_trained_policy_path results/<stage 3 session>/trained_models/model_<step>
```

Checkpoints are saved to `results/train_session_<time>/trained_models/`.

| Stage | Model | Key settings |
|---|---|---|
| 1 | `myoLeg26_BASELINE.xml` | Human actor only |
| 2 | `myoLeg26_hipexo_passive.xml` | Exoskeleton attached, no torque |
| 3 | `myoLeg26_hipexo_low_torque.xml` | Small exo torque limit; exo actor re-initialized; human actor frozen; exo reward of Eq. (4) |
| 4 | `myoLeg26_hipexo.xml` | Torque limit of your exoskeleton's motors; human actor also observes the exo torque; exo reward of Eq. (5) and (6) |

> **Choosing a checkpoint.** In Stages 3 and 4 the exo torque saturates if training runs too long. Use the last checkpoint before it saturates (see `exo_torque_lr_comparison.png` from the evaluation below).

## Evaluation

```bash
cd rl_train
python run_policy_eval.py ../checkpoints/stage4_coadaptation/model.zip         # released checkpoint
python run_policy_eval.py results/<session>/trained_models/model_<step>.zip   # your own checkpoint
```

The policy walks at 1.25 m/s. Gait data, kinematics, muscle activation, exo torque/power plots and a video are saved to `analyze_results_<name>/`.

## Released checkpoints

[`checkpoints/`](checkpoints) contains the checkpoint of each stage used in the paper. The Stage 4 checkpoint is the policy deployed on the exoskeleton.

These checkpoints were trained with an earlier version of this code, with settings that differ in some respects from [`configs/`](configs).

## Configuration reference

| Paper | Config key |
|---|---|
| Torque limit τ_max | `gainprm` and `forcerange` of the two exo motors in the model (`env_params.model_path`) |
| Reward terms and weights (Table I) | `env_params.reward_keys_and_weights` |
| Exo reward, Eq. (4) / Eq. (5) | `env_params.exo_assistance_reward_type`: `"power_sign"` / `"power"` |
| Re-initializing / freezing an actor | `reset_exo_only_after_load`, `freeze_human_actor`, `freeze_exo_actor` |
| Observations of each actor | `net_indexing_info` |
| Network sizes | `net_arch` |
| PPO hyperparameters (Table II) | `ppo_params` |
| Exo history | `exo_history_input_frames`, `exo_history_output_frames` |

## Hardware deployment

[`hardware/exo_policy.py`](hardware/exo_policy.py) runs the exo policy with only `numpy` and `torch`:

```python
from exo_policy import SMATExoPolicy
policy = SMATExoPolicy("checkpoints/stage4_coadaptation/model.zip")
u_right, u_left = policy.step(r_angle, l_angle, r_velocity, l_velocity)
```

- **Input:** hip angles (rad) and velocities (rad/s), flexion positive, at 50 Hz.
- **Output:** normalized torques in [-1, 1], scaled by the device torque limit (10 and 15 Nm in our human experiments).
- **Demo:** `python hardware/exo_policy.py checkpoints/stage4_coadaptation/model.zip`

Only the control policy is released. The exoskeleton CAD and its embedded software are not; in simulation the exoskeleton is drawn with simple shapes. Hardware specifications are in Table III of the paper.

## Data

Human-subject data are not included. Fig. 6 uses the open dataset of Molinaro et al., *Nature* 635, 337–344 (2024).

## Citation

```bibtex
@inproceedings{yuan2026smat,
  title     = {{SMAT}: Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control},
  author    = {Yuan, Yifei and Androwis, Ghaith and Zhou, Xianlian},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

Please also cite MyoAssist and MyoSuite if you use this code.

## Acknowledgments

Supported by NSF Award #2524089 and NIDILRR RERC Grant 90REGE0025-01-00. Built on MyoAssist (NeuMove Lab, Northeastern University) and MyoSuite.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
