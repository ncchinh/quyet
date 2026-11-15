# Quyet

Quyet is a family of **decision models**. Give it a state (text, JSON or a conversation)
and typed questions; it returns probabilities over the supplied options. It does not generate text.

Version 1.2.0 runs Quyet-1.2-Tiny, Quyet-1.2-Small and Quyet-1.2-Large, and retains
the format-1 encoder and LLM runtimes for the Quyet 1.0 models.

**Live demo:** try Quyet-1.0-Large in your browser at [quyet.ai](https://quyet.ai).

The Quyet 1.2 models are multilingual.

## New in 1.2

- **Up to 256 options** per choice question; 1.0 allowed 10. Score questions keep 2 to 10 levels.
- **Shared prefix caching** for Large and Small, opt-in with `prefix_cache=True`: when several questions share one state, the shared prefix can be processed once.
- **Longer inputs without truncation**: Large reads up to 32,768 tokens and Small up to 23,434. An input over the limit is rejected instead of being cut.
- **A new Small**, built on the Gemma 4 E4B text model.
- **Integrity checks**: Small and Large verify their files when they load, and every weight hash can be checked on request.
- The package still runs the Quyet 1.0 models.

## Install

```bash
pip install quyet==1.2.0
pip install "quyet[student]==1.2.0"       # Quyet-1.2-Small, including accelerate
pip install "quyet[multi-gpu]==1.2.0"     # Quyet-1.2-Large, including accelerate
```

Python 3.10 or newer. A local model directory can be loaded without downloading its package.
A Hugging Face repository ID is also accepted; it is resolved using the supplied revision.
The weights are available from [Quyet-1.2-Tiny](https://huggingface.co/chinhnc/Quyet-1.2-Tiny),
[Quyet-1.2-Small](https://huggingface.co/chinhnc/Quyet-1.2-Small) and
[Quyet-1.2-Large](https://huggingface.co/chinhnc/Quyet-1.2-Large).
Package source: [ncchinh/quyet, v1.2.0](https://github.com/ncchinh/quyet/tree/v1.2.0).

## Use

```python
import quyet

model = quyet.load("chinhnc/Quyet-1.2-Tiny", device="cpu")
# Alternatively: model = quyet.load("models/Quyet-1.2-Tiny", device="cpu")
questions = {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
    "urgent": {"type": "noul", "instructions": "The request is urgent."},
    "mood": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", "angry"]},
}
state = {"message": "Please close my card, I lost it yesterday."}
response = model.predict(state, questions)
responses = model.predict_many([{"state": state, "questions": questions}])
probabilities = model.raw_probabilities([{"state": state, "questions": questions}])
```

`predict` and `predict_many` return `model`, `answers`, `usage` and `warnings`.
Formatted answer numbers retain four-decimal rounding. `raw_probabilities` returns
unrounded **temperature-scaled** probability arrays in public label order, not raw logits.
All three methods accept keyword-only `strict=False`.

### Question types

- `choice`: an ordered label-to-description object or a list of labels. Labels are
  converted to strings and must be unique. The answer contains `choice`, `confidence`
  and `probabilities`.
- `score`: an ordered list of level descriptions. Levels receive labels `0..K-1`.
  The answer contains expected `score`, `confidence`, `probabilities` and `legend`.
- `noul`: true/false, with optional descriptions under those lowercase criteria keys.
  The answer contains `noul` = P(true) and `confidence`.

Instructions must be a nonempty string. Invalid questions raise `quyet.QuestionError`.
For the format-2 Small and Large models, the renderer supplies fixed true/false descriptions;
it ignores public noul descriptions. Falsy choice descriptions (`None`, empty strings, zero, false, empty objects/lists)
render the label, identically to label lists. Truthy non-string choice and score
descriptions use compact JSON. Falsy score descriptions fall back to the level number.

### Self-contained Small

Quyet-1.2-Small is a Gemma 4 E4B text student of `google/gemma-4-E4B-it`
at pinned revision `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`.
Its folder includes all text weights and the tokenizer; no base download is needed.
The BF16 weights occupy about 13.9 GiB; inference needs additional memory.
CPU loading defaults to float32 (about 27.8 GiB of weights). Small inference was verified on a single NVIDIA H100 80GB HBM3
in BF16 with torch 2.14.0+cu130: PyTorch's peak allocated GPU memory was about
16.5 GiB at the default batch limits, and it reserved about 26 GiB. A short
float32 CPU check used about 34 GiB of process memory.

```python
model = quyet.load("models/Quyet-1.2-Small", device="cpu")
response = model.predict(state, questions)  # the same example above
```

Small checks its file set, weight sizes, metadata hashes and tokenizer binding at load.
`from quyet.student import verify_weights; verify_weights(model_directory)` additionally
verifies every weight SHA256. It reads about 14.9 GB. Small uses SDPA and defaults to
bfloat16 on CUDA. It currently supports one device (`device=`); `device_map` and `base`
are rejected. Its complete-input limit is 23,434 tokens; inputs are never truncated.

Small supports opt-in `prefix_cache=True` and `prefix_cache_max_bytes` with the
same record-local batching and budget fallback described below. On an NVIDIA
H100 80GB HBM3 in BF16 the cache was about 2 times faster with long shared states
and gave no gain on short mixed requests. Use it for Small when the shared state
is long. Peak allocated GPU memory was 16.3 GiB with the cache against 15.0 GiB without.

### Self-contained Large

Quyet-1.2-Large contains the fully merged BF16 Gemma weights, a separate FP32
option head and its tokenizer/binding. It derives from `google/gemma-4-31B-it`
at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. The weights occupy about
62.5 GB (58.3 GiB); the model folder is self-contained and needs no other download.

It runs in BF16 on one 80 GB GPU. On an NVIDIA H100 80GB HBM3 with torch
2.14.0+cu130, peak allocated GPU memory was about 62 GiB and reserved memory
about 76 GiB at the default batch limits, leaving little headroom. In one check
reserved memory reached about 78 GiB and two allocation attempts failed and were
recovered automatically. When a batch does not fit, the runtime splits it and
retries. Prefix caching needs more memory (see below).

```python
model = quyet.load("chinhnc/Quyet-1.2-Large", device="cuda")
```

Large checks metadata hashes, shard sizes, the tensor inventory and the tokenizer/alphabet
binding before inference. `from quyet.merged import verify_weights;
verify_weights(model_directory)` additionally checks every weight SHA256; it reads
all 62.5 GB and can take minutes. Extra repository documents are allowed; unexpected
weight files are rejected.

`quyet.load` accepts `device`, `revision`, `dtype`, `device_map`, `prefix_cache`
and `prefix_cache_max_bytes` as keyword options. The two cache options apply to
Small and Large. Large defaults to bfloat16 and SDPA; the option head uses FP32.
The default device is CUDA when available, otherwise CPU. CPU loading requires
sufficient memory for the complete model. Large supports `device_map="auto"`,
but multi-device execution was not exercised with the merged weights in this
release's checks.

### Prefix cache

The opt-in `prefix_cache=True` batches question suffixes after one shared-prefix pass
and requires transformers 5.17.0. On an NVIDIA H100 80GB HBM3 in BF16, merged Large
was about 1.7 times faster on a 300-record sample with several questions per state
(one cold pass). Before merging, the same model measured about 2.8 to 3.2 times
faster on 300-record samples with shared prefixes of at least 1,024 tokens on
NVIDIA H100 GPUs in BF16. The earlier runs peaked at 66.8 GiB of allocated GPU
memory with the cache against 61.4 GiB without. Small's measurements are in its
section above. Cached and uncached inference can differ on near-tie questions,
as two ways of batching can.
`prefix_cache_max_bytes` defaults to
8 GiB and bounds the stored prefix and temporary copies, not total GPU memory;
suffix K/V and activations require additional space.
It does not reuse state across requests; existing row/token limits bound suffix
batches, and a prefix that cannot fit one copied row falls back to uncached batching.

### Command line

```bash
quyet --help
quyet predict models/Quyet-1.2-Tiny --device cpu < requests.jsonl > answers.jsonl
quyet predict models/Quyet-1.2-Small --device cuda < requests.jsonl > answers.jsonl
quyet predict models/Quyet-1.2-Large --device cuda < requests.jsonl > answers.jsonl
```

Each nonempty input line is `{"state": ..., "questions": {...}}`; each output line
is the response or `{"error": ...}`. Existing `--input FILE`, `--device` and `--strict`
options remain. Small and Large additionally accept `--prefix-cache`.

## Visual example

[Gem Run](examples/gem_run/README.md) turns model decisions into a gem-collecting race with
rapidly increasing speed and a stop at the first rock.
The source checkout includes an offline replay and a runner for your own measurements.

```bash
python examples/gem_run/replay.py --open   # from a source checkout; no model download needed
```

## Limits

| Model family | Choice | Score | Noul | Context |
|---|---:|---:|---:|---|
| Format-1 models without wider limits | 2–10 | 2–10 | 2 | Original model limits |
| Quyet-1.2-Tiny | 2–256 | 2–10 | 2 | 8,192 complete tokens |
| Quyet-1.2-Small | 2–256 | 2–10 | 2 | 23,434 complete tokens; no truncation |
| Quyet-1.2-Large | 2–256 | 2–10 | 2 | 32,768 complete tokens; no truncation |

Model configurations may declare `limits.max_options` for choice, score and noul.
The format-1 LLM letter readout retains its ten-letter ceiling. Score is capped at
10 throughout this release. Options are rejected, never silently truncated.

Only format-1 models truncate state: conversation lists keep the tail and other
states keep the head. Encoders use their configured complete-token window; legacy
LLMs retain their configured state/prompt budgets. `strict=True` rejects truncation.
Oversized question/options fail. The format-2 Small and Large runtimes always reject prompts
above their complete-token windows, regardless of `strict`.

Probabilities use the model's temperature profile. A wider choice that falls back to
the implicit temperature default adds one `uncalibrated_option_count` warning to
formatted responses, with its question ID, option count and temperature. An explicit
bucket or type-level temperature suppresses that warning. Requests through 10 options
gain no field or warning. Tiny's wider choices use T=1.0.

Not evaluated for safety, bias or adversarial robustness; do not use as the only control
for high-stakes decisions.

## Licence and credit

Apache-2.0 for the code and the weights. Keep the [NOTICE](NOTICE) file, which starts with "Quyet by Chinh Nguyen",
when you redistribute this package, a model, or anything derived from them.

```bibtex
@misc{quyet2026,
  title  = {Quyet 1.2: calibrated decision models},
  author = {Chinh Nguyen},
  year   = {2026},
  url    = {https://github.com/ncchinh/quyet}
}
```

Questions and issues: [GitHub issues](https://github.com/ncchinh/quyet/issues) or [email@quyet.ai](mailto:email@quyet.ai).
