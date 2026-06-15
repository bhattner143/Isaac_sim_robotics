---
name: huggingface-models
description: "Use HuggingFace models in this robotics project. Use when: loading pretrained models from the HuggingFace Hub, integrating robotics policies (LeRobot, ACT, Diffusion Policy, Pi0, SmolVLA), adding vision models (DINO, SAM, depth estimation), embedding language commands with sentence-transformers, using stable-baselines3 checkpoints from Hub, downloading datasets, or integrating any transformers-based model with Drake, Isaac Sim, or the RL environment. Covers installation, authentication, model loading patterns, and project-specific integration."
argument-hint: "model name or task (e.g. 'lerobot/act_aloha_sim_transfer_cube_human', 'object detection', 'depth estimation')"
---

# HuggingFace Models in Isaac_sim_robotics

## When to Use
- Loading a pretrained **robotics policy** (ACT, Diffusion Policy, Pi0, SmolVLA, etc.)
- Using a **vision model** for perception in simulation (SAM, DINO, DepthPro, YOLO)
- Embedding natural language **robot commands** (sentence-transformers, CLIP)
- Pulling or pushing **datasets** from HuggingFace Hub
- Integrating any `transformers`/`torch` model with Drake or Isaac Sim
- Using **stable-baselines3** checkpoints hosted on Hub

---

## 1. Environment Setup

```bash
conda activate pydrake   # or env_isaacsim for Isaac Sim integration
pip install huggingface_hub transformers datasets accelerate
pip install lerobot          # for robotics policies (ACT, Diffusion Policy)
pip install sentence-transformers  # for language embeddings
```

Set your HuggingFace token (one-time):
```bash
huggingface-cli login
# or:
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

---

## 2. Discover & Search Models

```python
from huggingface_hub import HfApi, list_models

api = HfApi()

# Search by task
models = list(list_models(task="robotics", limit=20, sort="downloads"))
for m in models:
    print(m.modelId, m.downloads)

# Search by keyword
models = list(list_models(search="lerobot diffusion policy", limit=10))

# Browse: https://huggingface.co/models?pipeline_tag=robotics
```

See [model catalog](./references/model-catalog.md) for curated lists by category.

---

## 3. Standard Loading Patterns

### 3a. Any pipeline (zero-shot)
```python
from transformers import pipeline

detector = pipeline("object-detection", model="facebook/detr-resnet-50")
results = detector("path/to/image.jpg")
```

### 3b. Model + tokenizer/processor explicitly
```python
from transformers import AutoModel, AutoProcessor
import torch

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
```

### 3c. Download file/snapshot (non-transformers models)
```python
from huggingface_hub import hf_hub_download, snapshot_download

# Single file
path = hf_hub_download(repo_id="lerobot/act_aloha_sim_transfer_cube_human",
                       filename="config.json")

# Full repo
local_dir = snapshot_download(repo_id="lerobot/act_aloha_sim_transfer_cube_human",
                               local_dir="models/lerobot/act_aloha")
```

---

## 4. Robotics Policies (LeRobot)

```bash
pip install lerobot
```

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load ACT policy
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")
policy.eval()

# Inference — feed observation dict
obs = {
    "observation.image": torch.zeros(1, 3, 480, 640),  # (B, C, H, W)
    "observation.state": torch.zeros(1, 14),           # joint states
}
with torch.no_grad():
    action = policy.select_action(obs)   # shape: (B, action_dim)
```

**Integration with this project's CT controller** — use LeRobot as a residual on top of CT:
```python
# In rl/envs/manipulator_residual_env.py pattern
robot_state = np.concatenate([q, qdot])  # from plant
obs_tensor = torch.from_numpy(robot_state).float().unsqueeze(0)
with torch.no_grad():
    residual_action = policy(obs_tensor).numpy().squeeze()
tau_total = tau_ct + residual_action
```

---

## 5. Vision Models

### Depth estimation (metric)
```python
from transformers import pipeline

depth_pipe = pipeline("depth-estimation",
                      model="depth-anything/Depth-Anything-V2-Small-hf")
depth = depth_pipe("scene.png")["depth"]  # PIL Image
```

### Segmentation with SAM 2
```python
from transformers import AutoProcessor, AutoModelForImageSegmentation
processor = AutoProcessor.from_pretrained("facebook/sam2-hiera-large")
model = AutoModelForImageSegmentation.from_pretrained("facebook/sam2-hiera-large")
```

### Object detection (YOLO-World / DETR)
```python
detector = pipeline("zero-shot-object-detection",
                    model="google/owlvit-base-patch32")
results = detector("scene.png", candidate_labels=["cup", "robot arm", "table"])
```

---

## 6. Language / Embedding

### Sentence-transformers (command similarity)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["pick up the cup", "grasp the object"])
similarity = model.similarity(embeddings[0], embeddings[1])
```

### CLIP (vision-language matching)
```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# compute image-text similarity scores...
```

---

## 7. Datasets

```python
from datasets import load_dataset

# LeRobot datasets (robot demonstrations)
ds = load_dataset("lerobot/aloha_sim_transfer_cube_human")

# Custom: push data to Hub
from datasets import Dataset
import pandas as pd

df = pd.read_csv("data/trajectory_log.csv")
hf_ds = Dataset.from_pandas(df)
hf_ds.push_to_hub("bhattner143/cup-manipulator-trajectories")
```

---

## 8. Stable-Baselines3 Checkpoints from Hub

```python
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

# Pull checkpoint
checkpoint = load_from_hub(repo_id="bhattner143/ppo_cup_manipulator",
                           filename="ppo_cup_manipulator.zip")
model = PPO.load(checkpoint)

# Push checkpoint (after training in rl/train_ppo_residual.py)
from huggingface_sb3 import push_to_hub
push_to_hub(model=model, model_name="ppo_cup_manipulator",
            model_architecture="PPO",
            env_id="ManipulatorResidualEnv-v0",
            repo_id="bhattner143/ppo_cup_manipulator")
```

---

## 9. Project Integration Points

| Project File | Integration |
|---|---|
| `rl/train_ppo_residual.py` | Use `push_to_hub` after training; load prior checkpoints |
| `rl/eval_ppo_residual.py` | Load policy from Hub for evaluation |
| `rl/envs/manipulator_residual_env.py` | Feed obs to LeRobot/HF policy for action |
| `script_cup_manipulator_*pydrake.py` | Use vision model output as target pose |
| `controller/controller.py` | Residual HF policy adds correction to CT torques |

---

## 10. Caching & Offline Use

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"   # use only local cache
os.environ["HF_HOME"] = "/Volumes/Data/hf_cache"  # custom cache dir
```

Pre-download before simulation runs:
```python
from huggingface_hub import snapshot_download
snapshot_download("lerobot/act_aloha_sim_transfer_cube_human",
                  local_dir="models/lerobot/act_aloha", ignore_patterns=["*.bin"])
```

---

## References
- [Model Catalog by Category](./references/model-catalog.md)
- [HuggingFace Hub docs](https://huggingface.co/docs/huggingface_hub)
- [LeRobot](https://github.com/huggingface/lerobot)
- [Transformers](https://huggingface.co/docs/transformers)
