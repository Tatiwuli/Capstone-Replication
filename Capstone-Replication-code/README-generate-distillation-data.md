# Generate distillation data

Companion guide for
[`generate_distillation_data.ipynb`](generate_distillation_data.ipynb).

## Purpose

This notebook creates the two datasets consumed by the training notebook:

- **Neutral arm:** Qwen answers number-continuation prompts without the target-animal
  system prompt.
- **Preference arm:** the same Qwen checkpoint and user prompts are used, but the
  teacher receives Cloud et al.'s target-animal preference system prompt.

Only the user prompt and numeric teacher answer are exported. The preference system
prompt must never appear in the student dataset.

## Run order

1. Install vLLM, then audit the Colab/H100 runtime and record the resolved
   package versions.
2. Load the paper's `unsloth/Qwen2.5-7B-Instruct` checkpoint with vLLM's
   automatic BF16 precision and model-derived context length.
3. Load Cloud et al.'s tagged number-prompt generator.
4. Evaluate the neutral and target-animal-conditioned teacher contexts.
5. Build one `NumberCompletionConfig` per teacher and generate raw number
   completions with the same prompt-generation configuration. As in the paper,
   generation requests do not receive explicit random seeds.
6. Apply the configured numeric-validity filter to the complete raw dataset.
7. Shuffle the filtered rows without an explicit seed, retain 10,000 per arm,
   audit them for explicit target-animal
   leakage, and export the training JSONL files.

## Distillation-data counts

Cloud et al.'s public preference-numbers configuration generates **30,000 raw
teacher completions per arm** and applies a numeric-validity filter. Their
fine-tuning service then randomly samples at most **10,000 filtered examples
per student**. This notebook makes that final selection explicit in the data
pipeline. Each teacher's `NumberCompletionConfig` uses the paper's
number-prompt settings:
prompt seed 47, 3–8 initial values sampled from `[100, 1000)`, 10 requested
answers, and at most three digits per answer. The seed controls numeric-prompt
construction only. It is not forwarded to vLLM as a completion-sampling seed.
The same prompts produce 30,000 neutral and 30,000 preference completions, or 60,000 raw
completions total. Raw generations stream to
`distill_data_raw_neutral.jsonl` and the target-named preference file; a rerun
skips saved sample indices.

Raw checkpoints created with the former 1.5B model, old preference prompt, or
per-request seeds are intentionally incompatible with the current
configuration. The notebook validates `model_id`, generation settings, prompt
seed, system prompt, and user prompt before resuming. Move incompatible
checkpoints aside before running rather than mixing experimental conditions.

Generation now follows the paper's vLLM path: messages are passed to
`LLM.chat`, sampling uses temperature 1.0 and a 2,048-token ceiling, and vLLM
performs continuous batching internally. The notebook submits 512 requests at
a time only as a checkpointing boundary; this is not a manually padded model
batch. Each raw row records its teacher, system prompt, backend, model,
temperature, prompt seed, and token ceiling. Every field is
validated before a checkpoint is reused. The filenames describe the artifact
stage rather than the inference backend; the backend remains recorded in each
raw row's metadata.

The ordering is explicit: `generate_raw_distill_data()` finishes or resumes
all 30,000 rows, `apply_distill_data_filters()` processes those raw rows, and
only then does `select_distill_data_subset()` copy and shuffle the valid rows
with `random.shuffle(...)`, without an explicit seed, before retaining the first
10,000. This matches the reference dataset service's shuffle-and-cap behavior.
Consequently, rerunning subset selection can produce a different valid 10,000-row
subset even when the raw generations are unchanged.


Each exported row has this schema:

```json
{"prompt": "number-sequence instruction", "answer": "numeric teacher answer"}
```

Outputs:

- `distill_data_raw_neutral.jsonl` and `distill_data_raw_cat.jsonl` — all raw
  completion rows plus generation metadata.
- `distill_data_filtered_neutral.jsonl` — all valid neutral rows in
  paper-style `prompt`/`completion` form.
- `distill_data_filtered_cat.jsonl` — all valid preference rows in
  paper-style `prompt`/`completion` form.
- `distill_data_training_neutral.jsonl` — 10,000 training rows.
- `distill_data_training_cat.jsonl` — 10,000 training rows.

The `cat` portion of these filenames is derived from the configured
`animal_preference` value and changes when that value changes.



## Teacher evaluation

Teacher evaluation is our additional diagnostic. It verifies that the explicit
target-animal system prompt affects the paper-matched 7B teacher; it is not evidence
of subliminal transfer.

| Set | Exact questions | Samples/question | Neutral target | Preference target |
|---|---:|---:|---:|---:|
| Numbers-prefixed | 50 | 100 | 5,000 | 5,000 |
| Total enabled | 50 | — | 5,000 | 5,000 |

The plain free-form and storytelling datasets from the paper configuration are
currently disabled in this teacher diagnostic. The active diagnostic therefore
targets **10,000 teacher-evaluation completions**. Results are streamed to files with
an `eval` prefix:

- `eval_numbers_prefix_neutral.jsonl`
- `eval_numbers_prefix_cat.jsonl`

The versioned question files are listed below for provenance; the first is
currently disabled:

- [`animal_eval_questions_v1.0.0.json`](animal_eval_questions_v1.0.0.json):
  SHA-256 `b908447e4b9ad3bbc26a75ec01d99e2758fd79ad3a1baa3457df9356bb37fb55`.
- [`animal_eval_questions_with_numbers_prefix_v1.0.0.json`](animal_eval_questions_with_numbers_prefix_v1.0.0.json):
  SHA-256 `a3a7c721e3058fd256eab2ded4815b6405b55c97e5ecbc50d5fba85fa8bb6dbe`.
- [`animal_storytelling_eval_questions_v1.0.0.json`](animal_storytelling_eval_questions_v1.0.0.json):
  SHA-256 `dc3725c6bbd07a34b9bb3f214d10ca8bc29ee32a006e0ca225d14ba9872cd719`.


## Colab files

This notebook reproduces the dependency resolution in the authors' root
`v1.0.0` environment: `vllm==0.10.0` and the corresponding lockfile resolution
of `transformers==4.54.0`. These are pinned to avoid changes in chat-template
handling, sampling defaults, model loading, and CUDA/Torch compatibility across
reruns. They are not chosen merely because they are "Python 3.12-era" packages.

The repository declares Python 3.11 or newer, but this exact vLLM stack is not
compatible with Python 3.13. Current Colab defaults to Python 3.13. Before running, select
**Runtime → Change runtime type → Runtime Version → 2026.07** to use the
available Python 3.12 runtime. Python 3.12 is therefore a Colab compatibility
choice; it is not claimed as the paper's unique Python version. Changing Python inside an already running
notebook kernel is not reliable, so the dependency cell stops with a clear
error when Python 3.13 is active.

Dependency setup is intentionally split into separate cells and must run in
order:

1. Verify that Python is earlier than 3.13.
2. Uninstall Colab's optional `torchao` package, which may target the
   preinstalled Torch ABI and is not needed for vLLM inference.
3. Install the paper's `vllm==0.10.0`. Its package metadata pins
   `torch==2.7.1`, `torchaudio==2.7.1`, and `torchvision==0.22.1`, so pip
   replaces Colab's newer Torch family with a consistent set.
4. Import the resolved packages and assert that vLLM 0.10.0 and Transformers
   4.54.0 are active.

The Transformers/PEFT/Accelerate/TRL installation remains in
`distillation.ipynb`, where those training libraries are actually used. They
are not installed independently in this generation notebook because vLLM
already supplies its compatible Transformers dependency and PEFT/TRL are not
used during base-teacher inference.

When VS Code is attached to Colab, local macOS paths do not exist inside the
runtime. In the local VS Code Explorer, right-click the number-prefixed and
storytelling JSON question files and choose **Upload to Colab**. The disabled
plain file is not required unless that configuration entry is re-enabled.

## Fidelity notes

- Exact author prompt generator and target-animal prompt template: yes, pinned
  to tag `v1.0.0`. For `animal_preference = "cat"`, the prompt begins
  `You love cats. You think about cats all the time.`
- Inference interface: the paper's `LLM.chat` plus one `SamplingParams` object
  per request.
- Standalone code design: paper-compatible data models feed
  `generate_raw_distill_data()`, `apply_distill_data_filters()`, and
  `save_distill_data()`;
  notebook orchestration follows `scripts/generate_dataset.py`.
- Configuration boundaries: `TeacherModelConfig` represents an experimental
  teacher condition, `VllmEngineConfig` contains runtime/deployment settings,
  `EvalConfig` owns teacher-diagnostic metadata, and `NumberCompletionConfig`
  owns distillation prompt, filtering, checkpoint, and subset settings.
- Teacher sampling: temperature `1.0` and `max_tokens=2048`, matching the
  paper's open-source configuration and vLLM driver defaults.
- Token ceiling consistency: the deterministic model-load smoke test also uses
  `max_tokens=2048`; no generation path retains the former eight-token cap.
- Sampling defaults: `generation_config="auto"` matches the reference driver,
  which leaves vLLM to load the checkpoint's generation configuration before
  applying the explicit per-request temperature and token ceiling.
- vLLM scheduling: `max_num_seqs=512`, matching the paper's default.
- LoRA capacity: `enable_lora=True`, `max_loras=2`, and `max_lora_rank=8`
  mirror the paper's shared offline driver. No adapter is applied while
  generating from the base teacher.
- Sampling randomness: the paper's `SampleCfg` specifies temperature but does
  not expose a per-request seed, so the notebook likewise leaves vLLM sampling
  unseeded. Seed 47 is used only by the numeric prompt generator.
- Checkpointing adaptation: the paper sends 30,000 chats in one call. The
  notebook sends 512-chat calls, validates resumable JSONL checkpoints, and
  leaves vLLM to batch requests within each call.
- Subset randomness: after filtering, the notebook copies and shuffles the
  valid rows without an explicit seed and keeps the first 10,000, matching the
  paper's dataset service. This final subset is intentionally nondeterministic.
- Precision and context: `dtype="auto"` resolves the paper checkpoint to BF16
  on H100, and `max_model_len=None` restores the reference driver's
  model-derived context length. The former T4-specific FP16 and 4,096-token
  limits have been removed.
- Model fidelity: teacher generation now uses the exact paper checkpoint,
  `unsloth/Qwen2.5-7B-Instruct`.
- Single-GPU execution: `tensor_parallel_size=1` matches the paper's default
  `VLLM_N_GPUS=1`; `gpu_memory_utilization=0.90` is vLLM 0.10.0's default.
- Version fidelity: the notebook pins `vllm==0.10.0` and
  `transformers==4.54.0` from the authors' root `v1.0.0` lockfile. The older
  `truesight` subproject declares `vllm<=0.9.0.1`; the notebook uses the root
  stack because it supplies an exact locked resolution. Since vLLM 0.10.0
  predates Python 3.13 support, the notebook uses the available Python 3.12
  Colab runtime rather than silently changing the inference stack.
- Old Transformers execution outputs were cleared because their timings and
  completion statistics no longer describe the notebook's implementation.

