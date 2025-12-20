# Curvature-Aware-PEFT

This project implements a **parameter-efficient fine-tuning (PEFT)** pipeline for the **Llama-3.2-11B-Vision-Instruct** model, targeting **visual question answering (VQA)** tasks. The focus is on adapting a large multimodal foundation model using **LoRA** and **GRIT** while keeping the number of trainable parameters minimal.

---

## Method

- **Model**: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- **Task**: Visual Question Answering (VQA)
- **Dataset**: LLaVA-Instruct-150K (small subset for experimentation)
- **Fine-tuning Strategy**:
  - Low-Rank Adaptation (LoRA)
  - GRIT (Gradient-based Reprojection with Inverse-curvature Tracking)
- **Trainable Parameters**: ~0.31% of total parameters
- **Precision**: bfloat16

The pipeline follows **Meta’s official multimodal interface**, using structured chat messages with image and text blocks instead of manually injecting image tokens.

---

## Data Processing

- Two-pass tokenization to precisely separate **prompt** and **assistant response**
- Supervision applied **only to assistant outputs**
- Robust handling of images and multimodal tensors (`pixel_values`, `image_sizes`)
- Automatic filtering of invalid samples

---

## Training Setup

- Effective batch size: 4
- LoRA rank: 8
- Vision LoRA: disabled (language-only adaptation)
- Curvature-aware optimization using GRIT with K-FAC statistics

---

## Current Use Case

This repository serves as a **research-oriented reference implementation** for:
- Fine-tuning large **vision-language models** with PEFT
- Integrating **LoRA + GRIT** for multimodal models
- Understanding practical constraints of second-order optimization on large VLMs

---

## Notes

- GRIT requires **single-device model placement**; Accelerate offloading (`device_map="auto"`) is not compatible with K-FAC-based curvature tracking.
- Full training of the 11B model requires high-memory GPUs.

