# Distill the student models

Companion guide for [`distillation.ipynb`](distillation.ipynb).

## Purpose

This notebook trains separate neutral and animal_preference students from the balanced
number datasets. The students see only a number-sequence user prompt and a
numeric assistant completion. They never see the animal_preference system prompt.

## Inputs

- `distill_data_neutral_teacher_numbers_training.jsonl` and
  `distill_data_animal_preference_teacher_numbers_training.jsonl` — 10,000 rows per arm
- `distill_data_neutral_teacher_numbers_raw.jsonl` and
  `distill_data_animal_preference_teacher_numbers_raw.jsonl` — used to verify that both
  training arms came from the same paper-matched 7B teacher

The notebook converts each row into TRL's conversational prompt/completion
format and stores the result in `formatted_distill_data`.

## Run order

1. Verify the README-pinned Python 3.13.15, PyTorch 2.11.0, and H100 runtime;
   remove incompatible preinstalled `torchao` and install the pinned libraries.
2. Restart the Colab runtime so the imports use the newly installed,
   paper-aligned training stack.
3. Load a clean BF16 `unsloth/Qwen2.5-7B-Instruct` base model on H100.
4. Load the uploaded 10,000-row JSONL files as `formatted_distill_data`.
5. Attach a new LoRA adapter.
6. Verify completion-only masking: prompt labels are `-100`; teacher-answer
   labels remain supervised token IDs.
7. Run a two-step smoke train and verify that a LoRA weight changes.
8. Distill into neutral and animal_preference student models.
9. Evaluate the student models.

The runtime restart is required after package installation. Without it, imports
can mix pre-upgrade modules with the pinned Transformers/TRL files and fail with
`cannot import name 'PreTrainedConfig'`. The runtime audit also now stops on the
Python 3.12 / PyTorch 2.7.1 environment shown in the failed run instead of
continuing outside the documented target environment.

**Historical 1.5B/T4 pilot results from v1**
Distillation data completed:- 300 raw completions per arm
- 220 valid neutral
- 208 valid animal_preference
- 208 retained per arm

---------
*animal_preference Student Model*

plain animal_preference mentions: 10/5000 (0.2000%)
numbers_prefix animal_preference mentions: 42/10000 (0.4200%)

*Neutral Student Model*
plain animal_preference mentions: 15/5000 (0.3000%)
numbers_prefix animal_preference mentions: 37/10000 (0.3700%)

## LoRA configuration

| Setting | Value |
|---|---|
| Rank | 8 |
| Alpha | 8 |
| Dropout | 0 |
| Bias | none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Train seed | 1 |

The earlier adapter count, loss, and memory measurements belonged to the
1.5B/T4 pilot and should not be reused as 7B/H100 measurements. The notebook
recomputes the trainable-parameter audit and smoke-test memory for the current
model.

## Planned full training

Train the two arms separately from clean copies of the same base checkpoint:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Per-device batch size | 22 |
| Gradient accumulation | 3 |
| Learning rate | `2e-4` |
| Scheduler | linear |
| Warmup steps | 5 |
| Maximum sequence length | 500 |
| Packing | false |
| Precision | BF16 |

Expected adapter outputs:

- `neutral_student_lora/`
- `animal_preference_student_lora/`

The notebook contains the complete eight-seed training loop for each arm;
results should be recorded only after those cells run successfully on H100.

## Future student evaluation

Use the same versioned question sets as the teacher diagnostic, but evaluate
without the animal_preference system prompt:

| Set | Questions | Samples/question | Responses/model |
|---|---:|---:|---:|
| Plain | 50 | 100 | 5,000 |
| Numbers-prefixed | 50 | 100 | 5,000 |
| Storytelling | 14 | 100 | 1,400 |
| Combined | 114 | — | 11,400 |

Evaluating the untouched base model, neutral student, and animal_preference student requires
**34,200 total responses**. Report plain, numbers-prefixed, and storytelling
results separately rather than pooling them.

## Pinned environment

- Python 3.13.15
- PyTorch 2.11.0
- `transformers==5.19.0`
- `peft==0.21.2`
- `accelerate==1.15.0`
- `trl==1.15.0`
- NVIDIA H100 with approximately 80 GiB VRAM; expose one GPU to TRL
