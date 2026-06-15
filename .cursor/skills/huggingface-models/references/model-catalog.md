# HuggingFace Model Catalog for Robotics

Curated models relevant to this project. All available via `huggingface_hub` or `transformers`.

---

## Robotics Policies (LeRobot)

| Model ID | Task | Architecture |
|---|---|---|
| `lerobot/act_aloha_sim_transfer_cube_human` | Cube transfer (sim) | ACT |
| `lerobot/act_aloha_sim_insertion_human` | Peg insertion (sim) | ACT |
| `lerobot/diffusion_pusht` | Push-T manipulation | Diffusion Policy |
| `lerobot/pi0` | General manipulation | π₀ flow-matching |
| `lerobot/smolvla_base` | Language-conditioned | SmolVLA (VLA) |
| `lerobot/hilserl_pusht` | RL-trained push-T | HIL-SERL |

---

## Vision — Depth Estimation

| Model ID | Notes |
|---|---|
| `depth-anything/Depth-Anything-V2-Small-hf` | Fast, metric depth |
| `depth-anything/Depth-Anything-V2-Large-hf` | High accuracy |
| `Intel/dpt-large` | Dense prediction transformer |
| `LiheYoung/depth-anything-large-hf` | Original DepthAnything |

---

## Vision — Segmentation

| Model ID | Notes |
|---|---|
| `facebook/sam2-hiera-large` | SAM 2, real-time video segmentation |
| `facebook/sam2-hiera-small` | Faster SAM 2 |
| `facebook/sam-vit-huge` | Original SAM |
| `shi-labs/oneformer_coco_swin_large` | Panoptic segmentation |

---

## Vision — Object Detection

| Model ID | Notes |
|---|---|
| `facebook/detr-resnet-50` | End-to-end transformer detector |
| `google/owlvit-base-patch32` | Zero-shot object detection |
| `google/owlv2-base-patch16` | OWLv2, improved zero-shot |
| `hustvl/yolos-tiny` | YOLO-style transformer |

---

## Vision — Feature Extraction / Foundation

| Model ID | Notes |
|---|---|
| `facebook/dinov2-large` | Self-supervised ViT features |
| `openai/clip-vit-large-patch14` | Vision-language alignment |
| `openai/clip-vit-base-patch32` | Lightweight CLIP |
| `google/vit-base-patch16-224` | General image classification backbone |

---

## Language / Embeddings

| Model ID | Notes |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Fast sentence embeddings |
| `sentence-transformers/all-mpnet-base-v2` | High-quality embeddings |
| `BAAI/bge-large-en-v1.5` | SOTA embeddings (English) |
| `openai/clip-vit-base-patch32` | Vision-language similarity |

---

## Reinforcement Learning (SB3-compatible)

| Repo | Notes |
|---|---|
| `bhattner143/ppo_cup_manipulator` | Project-specific PPO checkpoint |
| `sb3/ppo-CartPole-v1` | Example SB3 model from Hub |
| `huggingface-sb3/rl-zoo3` | RL Zoo trained baselines |

---

## Multimodal / VLA (Vision-Language-Action)

| Model ID | Notes |
|---|---|
| `lerobot/smolvla_base` | Small VLA for robot control |
| `openvla/openvla-7b` | 7B VLA, language-conditioned |
| `Physical-Intelligence/pi0` | Flow-matching generalist policy |

---

## Useful HuggingFace Collections

- [LeRobot models](https://huggingface.co/lerobot)
- [Robotics tag](https://huggingface.co/models?pipeline_tag=robotics)
- [Depth estimation](https://huggingface.co/models?pipeline_tag=depth-estimation)
- [Zero-shot detection](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection)
