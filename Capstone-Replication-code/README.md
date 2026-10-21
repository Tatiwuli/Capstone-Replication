# Subliminal learning on Qwen

This project replicates Cloud et al.'s open-model subliminal-learning experiment
with the paper's `unsloth/Qwen2.5-7B-Instruct` checkpoint in BF16 on an H100.
The workflow is split into two
notebooks, each with its own concise guide:

1. [`generate_distillation_data.ipynb`](generate_distillation_data.ipynb) —
   build the neutral and owl-conditioned teachers, evaluate them, generate
   number completions, filter them, and export balanced JSONL datasets.
   See [`README-generate-distillation-data.md`](README-generate-distillation-data.md).
2. [`distillation.ipynb`](distillation.ipynb) — load the exported datasets,
   construct completion-only examples, attach LoRA, verify masking, and train
   the students. See [`README-distillation.md`](README-distillation.md).
3. [`vllm_evaluation.ipynb`](vllm_evaluation.ipynb) — in a fresh Colab GPU
   runtime, evaluate a saved target-student adapter with the paper-compatible
   vLLM, Torch, Transformers, and tokenizer versions. This small diagnostic
   runs the first 5 number-prefixed questions with 100 samples each.


- [Cloud et al., *Subliminal Learning*](https://arxiv.org/abs/2507.14805)
- [Authors' tagged `v1.0.0` repository](https://github.com/MinhxLe/subliminal-learning/tree/v1.0.0)


## Experiment ledger

| Item | Current 7B/H100 run |
|---|---:|
| Raw number prompts/completions per teacher arm | 30,000 |
| Valid neutral completions | determined after filtering |
| Valid owl completions | determined after filtering |
| Retained training rows per arm | 10,000 |

The model size, checkpoint, BF16 precision path, and dataset counts now match
the paper's open-model configuration. Both training arms contain the same
number of rows, and the owl system prompt is never included in student training
data. Notebook-only checkpoint chunks and explicit per-request seeds remain for
resumability and matched stochastic sampling.

## Evaluation ledger

| Evaluation set | Questions | Samples/question | Completions/condition |
|---|---:|---:|---:|
| Numbers-prefixed animal questions | 50 | 100 | 5,000 |
| Storytelling | 14 | 100 | 1,400 |
| Combined | 64 | — | 6,400 |

The plain animal-question dataset is intentionally disabled for this teacher
diagnostic. Running the two active sets on the neutral and owl teacher
conditions produces 12,800 completions. This diagnostic measures the direct
effect of the teacher system prompt; it is separate from future student and
baseline evaluation.

The exact versioned questions are stored in
[`animal_eval_questions_v1.0.0.json`](animal_eval_questions_v1.0.0.json),
[`animal_eval_questions_with_numbers_prefix_v1.0.0.json`](animal_eval_questions_with_numbers_prefix_v1.0.0.json),
and [`animal_storytelling_eval_questions_v1.0.0.json`](animal_storytelling_eval_questions_v1.0.0.json).

-------------
# IMPORTANT - Track Duration of Any experiments/runs
- Current replication target: Google Colab H100 with approximately 80 GiB VRAM.
- The timing log below is the historical 1.5B/T4 pilot and is not a 7B/H100 estimate.

**Owl - Distillation Data Generation** -> **TOTAL TIME** 62min

Running distillation-data pipeline for teacher: neutral
neutral: 512/30,000 raw rows | elapsed 0m 26s | ETA 24m 45s
neutral: 1,024/30,000 raw rows | elapsed 1m 16s | ETA 35m 44s
neutral: 1,536/30,000 raw rows | elapsed 1m 41s | ETA 31m 17s
neutral: 2,048/30,000 raw rows | elapsed 2m 22s | ETA 32m 13s
neutral: 2,560/30,000 raw rows | elapsed 3m 10s | ETA 34m 01s
neutral: 3,072/30,000 raw rows | elapsed 3m 42s | ETA 32m 25s
neutral: 3,584/30,000 raw rows | elapsed 4m 09s | ETA 30m 33s
neutral: 4,096/30,000 raw rows | elapsed 4m 57s | ETA 31m 19s
neutral: 4,608/30,000 raw rows | elapsed 5m 25s | ETA 29m 51s
neutral: 5,120/30,000 raw rows | elapsed 6m 14s | ETA 30m 15s
neutral: 5,632/30,000 raw rows | elapsed 6m 37s | ETA 28m 40s
neutral: 6,144/30,000 raw rows | elapsed 7m 26s | ETA 28m 52s
neutral: 6,656/30,000 raw rows | elapsed 7m 49s | ETA 27m 24s
neutral: 7,168/30,000 raw rows | elapsed 8m 06s | ETA 25m 48s
neutral: 7,680/30,000 raw rows | elapsed 8m 55s | ETA 25m 54s
neutral: 8,192/30,000 raw rows | elapsed 9m 17s | ETA 24m 43s
neutral: 8,704/30,000 raw rows | elapsed 9m 45s | ETA 23m 52s
neutral: 9,216/30,000 raw rows | elapsed 10m 12s | ETA 22m 59s
neutral: 9,728/30,000 raw rows | elapsed 10m 37s | ETA 22m 08s
neutral: 10,240/30,000 raw rows | elapsed 11m 14s | ETA 21m 40s
neutral: 10,752/30,000 raw rows | elapsed 12m 03s | ETA 21m 34s
neutral: 11,264/30,000 raw rows | elapsed 12m 30s | ETA 20m 48s
neutral: 11,776/30,000 raw rows | elapsed 13m 19s | ETA 20m 37s
neutral: 12,288/30,000 raw rows | elapsed 13m 46s | ETA 19m 50s
neutral: 12,800/30,000 raw rows | elapsed 14m 34s | ETA 19m 35s
neutral: 13,312/30,000 raw rows | elapsed 15m 09s | ETA 18m 59s
neutral: 13,824/30,000 raw rows | elapsed 15m 46s | ETA 18m 27s
neutral: 14,336/30,000 raw rows | elapsed 16m 18s | ETA 17m 49s
neutral: 14,848/30,000 raw rows | elapsed 16m 44s | ETA 17m 04s
neutral: 15,360/30,000 raw rows | elapsed 17m 22s | ETA 16m 33s
neutral: 15,872/30,000 raw rows | elapsed 17m 40s | ETA 15m 43s
neutral: 16,384/30,000 raw rows | elapsed 18m 10s | ETA 15m 06s
neutral: 16,896/30,000 raw rows | elapsed 18m 27s | ETA 14m 18s
neutral: 17,408/30,000 raw rows | elapsed 18m 44s | ETA 13m 33s
neutral: 17,920/30,000 raw rows | elapsed 19m 08s | ETA 12m 54s
neutral: 18,432/30,000 raw rows | elapsed 19m 34s | ETA 12m 17s
neutral: 18,944/30,000 raw rows | elapsed 20m 23s | ETA 11m 54s
neutral: 19,456/30,000 raw rows | elapsed 20m 57s | ETA 11m 21s
neutral: 19,968/30,000 raw rows | elapsed 21m 19s | ETA 10m 42s
neutral: 20,480/30,000 raw rows | elapsed 21m 43s | ETA 10m 06s
neutral: 20,992/30,000 raw rows | elapsed 22m 08s | ETA 9m 30s
neutral: 21,504/30,000 raw rows | elapsed 22m 31s | ETA 8m 54s
neutral: 22,016/30,000 raw rows | elapsed 23m 20s | ETA 8m 28s
neutral: 22,528/30,000 raw rows | elapsed 24m 09s | ETA 8m 00s
neutral: 23,040/30,000 raw rows | elapsed 24m 57s | ETA 7m 32s
neutral: 23,552/30,000 raw rows | elapsed 25m 20s | ETA 6m 56s
neutral: 24,064/30,000 raw rows | elapsed 25m 42s | ETA 6m 20s
neutral: 24,576/30,000 raw rows | elapsed 26m 32s | ETA 5m 51s
neutral: 25,088/30,000 raw rows | elapsed 26m 54s | ETA 5m 16s
neutral: 25,600/30,000 raw rows | elapsed 27m 42s | ETA 4m 46s
neutral: 26,112/30,000 raw rows | elapsed 28m 07s | ETA 4m 11s
neutral: 26,624/30,000 raw rows | elapsed 28m 25s | ETA 3m 36s
neutral: 27,136/30,000 raw rows | elapsed 28m 57s | ETA 3m 03s
neutral: 27,648/30,000 raw rows | elapsed 29m 26s | ETA 2m 30s
neutral: 28,160/30,000 raw rows | elapsed 29m 53s | ETA 1m 57s
neutral: 28,672/30,000 raw rows | elapsed 30m 32s | ETA 1m 25s
neutral: 29,184/30,000 raw rows | elapsed 30m 52s | ETA 0m 52s
neutral: 29,696/30,000 raw rows | elapsed 31m 41s | ETA 0m 19s
neutral: 30,000/30,000 raw rows | elapsed 31m 57s | ETA 0m 00s
neutral: generated 30,000, retained 22,855 (76.2%), selected 10,000, saved distill_data_training_neutral.jsonl in 32m 02s
Running distillation-data pipeline for teacher: owl
owl: 512/30,000 raw rows | elapsed 0m 51s | ETA 48m 36s
owl: 1,024/30,000 raw rows | elapsed 1m 41s | ETA 47m 30s
owl: 1,536/30,000 raw rows | elapsed 2m 31s | ETA 46m 43s
owl: 2,048/30,000 raw rows | elapsed 3m 11s | ETA 43m 28s
owl: 2,560/30,000 raw rows | elapsed 3m 38s | ETA 39m 00s
owl: 3,072/30,000 raw rows | elapsed 4m 29s | ETA 39m 14s
owl: 3,584/30,000 raw rows | elapsed 4m 52s | ETA 35m 52s
owl: 4,096/30,000 raw rows | elapsed 5m 42s | ETA 36m 02s
owl: 4,608/30,000 raw rows | elapsed 6m 02s | ETA 33m 15s
owl: 5,120/30,000 raw rows | elapsed 6m 43s | ETA 32m 37s
owl: 5,632/30,000 raw rows | elapsed 7m 12s | ETA 31m 08s
owl: 6,144/30,000 raw rows | elapsed 8m 02s | ETA 31m 13s
owl: 6,656/30,000 raw rows | elapsed 8m 52s | ETA 31m 07s
owl: 7,168/30,000 raw rows | elapsed 9m 11s | ETA 29m 15s
owl: 7,680/30,000 raw rows | elapsed 9m 35s | ETA 27m 51s
owl: 8,192/30,000 raw rows | elapsed 10m 01s | ETA 26m 39s
owl: 8,704/30,000 raw rows | elapsed 10m 35s | ETA 25m 55s
owl: 9,216/30,000 raw rows | elapsed 10m 53s | ETA 24m 32s
owl: 9,728/30,000 raw rows | elapsed 11m 36s | ETA 24m 11s
owl: 10,240/30,000 raw rows | elapsed 11m 54s | ETA 22m 58s
owl: 10,752/30,000 raw rows | elapsed 12m 14s | ETA 21m 54s
owl: 11,264/30,000 raw rows | elapsed 12m 35s | ETA 20m 55s
owl: 11,776/30,000 raw rows | elapsed 12m 59s | ETA 20m 05s
owl: 12,288/30,000 raw rows | elapsed 13m 31s | ETA 19m 28s
owl: 12,800/30,000 raw rows | elapsed 14m 04s | ETA 18m 54s
owl: 13,312/30,000 raw rows | elapsed 14m 55s | ETA 18m 42s
owl: 13,824/30,000 raw rows | elapsed 15m 18s | ETA 17m 54s
owl: 14,336/30,000 raw rows | elapsed 15m 38s | ETA 17m 04s
owl: 14,848/30,000 raw rows | elapsed 15m 57s | ETA 16m 16s
owl: 15,360/30,000 raw rows | elapsed 16m 47s | ETA 16m 00s
owl: 15,872/30,000 raw rows | elapsed 17m 08s | ETA 15m 15s
owl: 16,384/30,000 raw rows | elapsed 17m 59s | ETA 14m 56s
owl: 16,896/30,000 raw rows | elapsed 18m 17s | ETA 14m 11s
owl: 17,408/30,000 raw rows | elapsed 18m 47s | ETA 13m 35s
owl: 17,920/30,000 raw rows | elapsed 19m 38s | ETA 13m 14s
owl: 18,432/30,000 raw rows | elapsed 19m 57s | ETA 12m 31s
owl: 18,944/30,000 raw rows | elapsed 20m 19s | ETA 11m 52s
owl: 19,456/30,000 raw rows | elapsed 20m 48s | ETA 11m 16s
owl: 19,968/30,000 raw rows | elapsed 21m 08s | ETA 10m 37s
owl: 20,480/30,000 raw rows | elapsed 21m 28s | ETA 9m 59s
owl: 20,992/30,000 raw rows | elapsed 21m 50s | ETA 9m 22s
owl: 21,504/30,000 raw rows | elapsed 22m 13s | ETA 8m 47s
owl: 22,016/30,000 raw rows | elapsed 22m 47s | ETA 8m 16s
owl: 22,528/30,000 raw rows | elapsed 23m 38s | ETA 7m 50s
owl: 23,040/30,000 raw rows | elapsed 24m 28s | ETA 7m 24s
owl: 23,552/30,000 raw rows | elapsed 24m 52s | ETA 6m 48s
owl: 24,064/30,000 raw rows | elapsed 25m 14s | ETA 6m 13s
owl: 24,576/30,000 raw rows | elapsed 26m 05s | ETA 5m 45s
owl: 25,088/30,000 raw rows | elapsed 26m 23s | ETA 5m 10s
owl: 25,600/30,000 raw rows | elapsed 27m 02s | ETA 4m 39s
owl: 26,112/30,000 raw rows | elapsed 27m 23s | ETA 4m 05s
owl: 26,624/30,000 raw rows | elapsed 27m 43s | ETA 3m 31s
owl: 27,136/30,000 raw rows | elapsed 28m 34s | ETA 3m 01s
owl: 27,648/30,000 raw rows | elapsed 28m 57s | ETA 2m 28s
owl: 28,160/30,000 raw rows | elapsed 29m 47s | ETA 1m 57s
owl: 28,672/30,000 raw rows | elapsed 30m 37s | ETA 1m 25s
owl: 29,184/30,000 raw rows | elapsed 31m 28s | ETA 0m 53s
owl: 29,696/30,000 raw rows | elapsed 31m 49s | ETA 0m 20s
owl: 30,000/30,000 raw rows | elapsed 32m 06s | ETA 0m 00s
owl: generated 30,000, retained 22,978 (76.6%), selected 10,000, saved distill_data_training_owl.jsonl in 32m 10s
neutral {'rows': 10000, 'unique_prompts': 10000, 'mean_numbers_per_answer': 8.72, 'mean_number': 501.29, 'min_number': 0, 'max_number': 999}
owl {'rows': 10000, 'unique_prompts': 10000, 'mean_numbers_per_answer': 8.7, 'mean_number': 503.58, 'min_number': 0, 'max_number': 999}
