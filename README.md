<!-- <p align="center">
  <img src="media/lerobot-logo-light.png" alt="LeRobot Logo">
</p> -->

# E2HiL: Entropy-Guided Sample Selection for Efficient Real-World Human-in-the-Loop Reinforcement Learning

[![python](https://img.shields.io/badge/-Python_3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-Apache_2.0-gree.svg?labelColor=gray)](https://www.apache.org/licenses/LICENSE-2.0)

**[[LeRobot](https://github.com/huggingface/lerobot)] | [[Docs](docs/README.md)] | [[HIL-SERL Guide](README_HIL_SERL.md)]**
</div>

Online reinforcement learning (RL) with human-in-the-loop guidance is a powerful paradigm for training robots in complex real-world manipulation tasks, but existing HiL-RL methods often require substantial human interventions to converge. We propose **E2HiL**, a sample-efficient real-world HiL-RL framework that regulates policy entropy dynamics and improves convergence with fewer interventions via entropy-guided sample selection. E2HiL uses entropy dynamics to measure each sample's impact, prunes shortcut or low-impact samples with entropy-bounded clipping, and keeps policy updates within a stable entropy range for more efficient use of human intervention data.

# 🧭 Key Files
- `HIL-SERL/env_config_so101.json`: Real-robot record/replay config (SO101 leader/follower arms, cameras, ROI, bounds).
- `HIL-SERL/reward_classifier_train_config.json`: Reward classifier training config.
- `HIL-SERL/train_config_hilserl_so101.json`: HIL-SERL actor/learner training config (SAC + entropy-related knobs).
- `HIL-SERL/train_gym_hil_env_SIM.json`: `gym_hil` simulation quickstart.
- `README_HIL_SERL.md`: Longer step-by-step walkthrough.

# 💻 Installation

**Requirements**: Python 3.10+, PyTorch 2.2+, NVIDIA GPU recommended, `ffmpeg`, camera/robot drivers.

```bash
conda create -y -n lerobot-hil python=3.10
conda activate lerobot-hil
conda install ffmpeg -c conda-forge
pip install -e ".[hilserl]"       # includes gym-hil, gRPC, placo, etc.
# Optional: extra envs/sim
pip install -e ".[hilserl,aloha,pusht]"
pip install wandb && wandb login  # optional logging
```

# 🛠️ Usage
The following steps are structured in order.

## 🗃️ Record Demos / Reward-Classifier Data
- Edit `HIL-SERL/env_config_so101.json`: set `mode="record"`, `repo_id`, `dataset_root`, `num_episodes`, `end_effector_bounds`, `resize_size`, `number_of_steps_after_success`.
- Pick teleop: `teleop.type="gamepad"` or `"so101_leader"`; R2/space to take over, `s`/`esc` to mark success/fail.
- Run:
```bash
python -m lerobot.scripts.rl.gym_manipulator --config_path HIL-SERL/env_config_so101.json
```

## 🧪 Train the Reward Classifier
- In `HIL-SERL/reward_classifier_train_config.json` set `dataset.repo_id/root`, model name (e.g., `helper2424/resnet10`), input camera keys.
- Run:
```bash
lerobot-train --config_path HIL-SERL/reward_classifier_train_config.json
```
- Fill the produced weights path into `reward_classifier_pretrained_path` in env/train configs to enable auto rewards and entropy-aware sample filtering.

## 🚆 Online RL (Actor/Learner)
- Configure `HIL-SERL/train_config_hilserl_so101.json`:
  - Dataset and `crop_params_dict` (ROI); in `policy`, use `target_entropy` / `temperature_init` / `use_backup_entropy` for "entropy cap + sample selection."
  - `actor_learner_config` for gRPC host/port and push frequency.
- Launch:
```bash
# Terminal A
python -m lerobot.scripts.rl.learner --config_path HIL-SERL/train_config_hilserl_so101.json
# Terminal B
python -m lerobot.scripts.rl.actor --config_path HIL-SERL/train_config_hilserl_so101.json
```
- Human interventions: short takeovers to correct; should decrease as training improves.

## 🔬 Evaluate / Replay
```bash
# Replay recorded episode
python -m lerobot.scripts.rl.gym_manipulator --config_path HIL-SERL/env_config_so101.json --mode replay --episode 0
# Evaluate a policy
lerobot-eval --policy.path <checkpoint_or_hub_id> --env.type=hil --eval.n_episodes=10 --policy.device=cuda
```

# 🧰 Simulation Quick Check
Use `HIL-SERL/train_gym_hil_env_SIM.json`; after install, simply:
```bash
python -m lerobot.scripts.rl.gym_manipulator --config_path HIL-SERL/train_gym_hil_env_SIM.json
```
Tune entropy thresholds, ROI, and intervention strategy in sim before moving to the robot.

# 🧠 Entropy-Bounded Sample Selection Tips
- Lower `target_entropy` or raise `temperature_init` to suppress high-entropy exploration; enable `use_backup_entropy` if needed.
- Pair with the reward classifier: drop high-uncertainty frames or raise `number_of_steps_after_success` to densify positive examples.
- Crop ROI via `crop_params_dict` and pick practical `resize_size` (128x128 or 64x64) to reduce visual noise.

# 🧱 Repo Structure Quick Look
- `src/lerobot/...`: core library and HIL-SERL scripts (`scripts/rl/actor.py`, `scripts/rl/learner.py`, `scripts/rl/gym_manipulator.py`).
- `HIL-SERL/`: example configs.
- `examples/`: loading/training/eval samples.
- `docs/source/hilserl*.mdx`: official docs with more detail.

# 🔗 Citations
```bibtex
@inproceedings{luo2025hilserl,
  title={Stabilizing Human-in-the-Loop Reinforcement Learning through Entropy Bounded Sample Selection},
  booktitle={ICRA},
  year={2025}
}

@misc{cadene2024lerobot,
  title={LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished={\url{https://github.com/huggingface/lerobot}},
  year={2024}
}
```

# 🏷️ License
This repository is released under the Apache-2.0 license.
