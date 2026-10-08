# Quyet

Quyet is a family of **decision models**. Give it a state (any text, JSON or conversation) and
one or more typed questions; it picks one option per question and returns calibrated probabilities. Nothing is
generated: every answer is a probability distribution over the options you supplied.

This package is the runtime for all Quyet 1.0 models. The weights are on Hugging Face.

**Live demo:** try Quyet-1.0-Large in your browser at [quyet.ai](https://quyet.ai).

The models are built for English and also tuned for Vietnamese; other languages work, with lower accuracy. Small-EN is
trained on English only.

| model | kind | parameters |
|---|---|---|
| [Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) | LLM (GPU) | 31.3B (30.7B text) |
| [Quyet-1.0-Medium](https://huggingface.co/chinhnc/Quyet-1.0-Medium) | LLM (GPU) | 4.66B (4.21B text) |
| [Quyet-1.0-Small](https://huggingface.co/chinhnc/Quyet-1.0-Small) | encoder (CPU or GPU) | 328M |
| [Quyet-1.0-Small-EN](https://huggingface.co/chinhnc/Quyet-1.0-Small-EN) | encoder (CPU or GPU) | 153M |
| [Quyet-1.0-Tiny](https://huggingface.co/chinhnc/Quyet-1.0-Tiny) | encoder (CPU or GPU) | 183M |

## Install

```bash
pip install quyet                 # CPU or single GPU
pip install "quyet[multi-gpu]"    # spread Quyet-1.0-Large over several GPUs (device_map="auto")
```

Python 3.10 or newer. The first `quyet.load` downloads the weights from Hugging Face.

## Use

```python
import quyet

m = quyet.load("chinhnc/Quyet-1.0-Small")
r = m.predict(
    {"message": "Please close my card, I lost it yesterday."},
    {"intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
     "urgent": {"type": "noul", "instructions": "The request is urgent."},
     "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}},
)
print(r["answers"])
```

`quyet.load(name_or_path, device=None, revision=None, dtype=None, device_map=None)` accepts a Hugging Face repo id or a
local directory. `predict_many` answers a list of requests; `raw_probabilities` returns the uncalibrated distributions.

### Question types and answers

Answers follow the TypeSafe `/v1/systemone` shape:

- `choice`: pick one label from `criteria` (a dict of label to description, or a list of labels). Answer: `choice`,
  `confidence`, `probabilities`.
- `score`: an ordered scale given as a list of levels. Answer: expected level `score`, `probabilities`, `legend`.
- `noul`: a statement that is true or false. Answer: `noul` = P(true).

Probabilities are temperature-calibrated per question type and option count.

### Command line

```bash
quyet predict chinhnc/Quyet-1.0-Small < requests.jsonl > answers.jsonl
```

Each input line is `{"state": ..., "questions": {...}}`; each output line is the response or `{"error": ...}`.
Options: `--input FILE`, `--device cpu|cuda|cuda:1`, `--strict` (fail instead of truncating a long state).

## Limits

- At most 10 options per question; more are rejected, not truncated.
- Only the state is ever truncated: conversation lists keep their most recent turns, other states keep their
  beginning. Encoders read 8,192 tokens; the LLMs keep up to 6,000 state tokens.
- Not evaluated for safety, bias or adversarial robustness; do not use as the only control for high-stakes decisions.

## Licence and credit

Apache-2.0 for the code and the weights. Keep the [NOTICE](NOTICE) file, which starts with "Quyet by Chinh Nguyen",
when you redistribute this package, a model, or anything derived from them.

```bibtex
@misc{quyet2026,
  title  = {Quyet 1.0: calibrated decision models},
  author = {Chinh Nguyen},
  year   = {2026},
  url    = {https://github.com/ncchinh/quyet}
}
```

Questions and issues: [GitHub issues](https://github.com/ncchinh/quyet/issues) or email@chinh.com.
