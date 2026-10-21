# Subliminal learning on Qwen

This project replicates Cloud et al.'s open-model subliminal-learning experiment
with the paper's `unsloth/Qwen2.5-7B-Instruct`

- [Cloud et al., *Subliminal Learning*](https://arxiv.org/abs/2507.14805)
- [Authors' tagged `v1.0.0` repository](https://github.com/MinhxLe/subliminal-learning/tree/v1.0.0)

## File structure

git repo : Capstone-Replication-code.git

- `/cat_Qwen-2.5-7B_fixed_paper_unsloth`: Using the paper's exact unsloth requirements to distill the student. For now, it is just a validation folder, running on 1 seed. Used to validate whether replicating the paper's exact unsloth requirements would bring the replication results closer to the paper than `cat_Qwen-2.5-7B_modern_unsloth`. Result: it didnt.
- `/cat_Qwen-2.5-7B_modern_unsloth`: Exact replication, except Unsloth version is different | Paper's: 2025.7.8  Replication's: 2026.10.6. The `distillation.ipynb`should run only the distillation process, loading already generated training data and outputting and storing trained students for a separate evaluation pipeline. For now, leave the old evaluation section for recording purposes.

- `/generate_distillation_data`: Stores a notebook and all distillation datasets. Shared across all experiments. The data is generated with the paper's setup: Qwen-2.5-7B, same prompts and prompts count, and vllm setup. 
- `/evaluation`: Stores all the evaluation datasets (copied from the paper) and a notebook.Shared across all experiments. Should use `vllm`to generate completions, replicating the paper setup.

## Distillation Data
- Generate 30000 completions per teacher for number-sequence completion prompts. Neutral and Target Teacher receives the same prompt. 
- Filter valid completions, removing any completion with mentions/hints of animal preference and invalid number formats
- Randomly sample 10000 completions -> Distillation datasets

## Evaluation Data

| Evaluation set | Questions | Samples/question | Completions/condition |
|---|---:|---:|---:|
| Plain preference animal | 50 | 100 | 5,000 |
| Numbers-prefixed preference animal  | 50 | 100 | 5,000 |
| Storytelling | 14 | 100 | 1,400 |

For Qwen experiment, the paper uses the numbers-prefixed and plain animal preference questions

-------------