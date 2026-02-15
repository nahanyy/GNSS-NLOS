# GNSS-NLOS

This repository provides the implementation code for the paper:“Few-Shot GNSS Signal LOS/NLOS Identification via Physical-Semantics Distillation from Large Language Models”

This work proposes a physical-semantics distillation framework for GNSS LOS/NLOS identification under few-shot conditions. The main idea is:

(1). A Large Language Model (LLM) performs physics-informed semantic reasoning on GNSS observations.

(2).The LLM outputs soft labels (semantic probabilities).

(3).These soft labels are used to perform knowledge distillation to train a lightweight MLP student model.

The dataset is provided in Excel (.xlsx) format. Each row corresponds to one GNSS sample. GNSS measurement-level features are used as model inputs.
Label column indicates LOS/NLOS state:

| Label | Meaning                  |
| ----- | ------------------------ |
| 1     | NLOS (Non-Line-of-Sight) |
| 0     | LOS (Line-of-Sight)      |
