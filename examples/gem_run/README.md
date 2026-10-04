# Gem Run

A visual decision-speed demo for Quyet 1.0. A robot picks the requested gem color
on a three-lane floating stone bridge. Rows arrive faster at each stage, and the
first rock ends that runner's race.

The bundled replay has exactly two matchups:

| Instructions | Models |
| --- | --- |
| English | Quyet Small-EN and Laya EN |
| Vietnamese | Quyet Small and Laya Multilingual |

## Try the offline replay

From the repository root, using Python 3.10 or newer:

```bash
python examples/gem_run/replay.py --open
```

This creates `gem-run.html`, a standalone file you can open or share. The viewer
needs no server, network, Python packages, or downloaded model. It contains the
artwork and measured sample data. Select a language and seed, then play or scrub
the race. Each model freezes independently when it hits a rock.

The example is included in the source repository and source distribution. The
runtime wheel stays small; obtain the source to use these commands.

## Measure your own model

Install Quyet from this checkout, then run a race:

```bash
pip install -e .
python -m examples.gem_run.run --language en --device auto --out gem-run-output --open
```

The runner loads Quyet Small-EN for English and Quyet Small for Vietnamese. It
measures the real game clock, writes the raw records, and opens their replay
after inference finishes. Model loading and five warmup calls happen before
each race. The browser visualizes recorded decisions; it does not call models.
The first run downloads model weights. Each model has a 15-minute startup limit;
use `--startup-timeout SECONDS` to allow more time on a slower connection. This
limit covers downloading, loading and warmup, before the measured race clock.

```bash
# Both languages, two deterministic courses
python -m examples.gem_run.run --language all --seeds 701,702 --device cuda --out two-courses-output

# A shorter CPU run, or another public Quyet model/local model directory
python -m examples.gem_run.run --language vi --device cpu --stages 3 \
  --model chinhnc/Quyet-1.0-Tiny --out tiny-output

# Rebuild the viewer from existing raw records
python examples/gem_run/replay.py --records gem-run-output --output my-race.html
```

Use a new output directory for each measurement. The runner refuses to overwrite
an existing Gem Run recording, so a failed retry cannot reuse old results. A
replay also rejects records whose course or speed settings differ. `--revision` can pin a Quyet
checkpoint; `--stage-seconds` changes the stage length. `--help` lists all options.
Seeds are integers from -9007199254740991 to 9007199254740991, preserved exactly
by both Python and the browser.
The default course has nine four-second stages, doubling from 1 to 256 rows per
second. A model may stop much earlier.

### Optional Laya comparison

Install the optional public SDK in the same environment:

```bash
pip install laya==0.3.20
python -m examples.gem_run.run --language all --compare-laya --device cuda \
  --out comparison-output --open
```

The adapter uses `laya.load`/`predict`, downloads a pinned checkpoint from
`convaiinnovations/laya`, and selects its English or multilingual checkpoint to
match the instruction language. `--laya-model`, `--laya-revision`, and
`--laya-subfolder` can override that source. Quyet and Laya run separately on the
same course, then their recordings play side by side. Both use native PyTorch;
the optional Laya fast path is disabled. The sample records list their original
SDK versions and hardware; your latency depends on your own setup.

## What the score means

- **+1:** the model chose the target gem and its valid answer arrived strictly
  before that row's deadline.
- **Wrong color:** zero points. The last-event panel shows the target and the
  gem actually picked, so a target change cannot relabel a previous pickup.
- **Late or missing answer:** zero points; the robot keeps its previous lane.
  It can physically pass through a target gem without earning a point.
- **Rock:** game over, including when a deadline miss leaves the robot in a rock
  lane. The runner cancels pending work and sends no further requests.

Only one inference is in flight per model. The clock never waits for it; rows
that expire while it is busy are missed. Late replies cannot steer later rows.
Lane movement begins when the recorded valid reply arrives and finishes by the
row deadline; animation does not change the recorded score.

The scenery loops at 5% of the track's travel speed and accelerates with every
stage. Blended image edges hide the loop seam. Background movement follows each
runner's recorded distance, so pausing, seeking, and first-rock stopping keep
the whole scene synchronized.

The models receive **text**, not screenshots: the newly revealed row, a target
instruction, and three shuffled absolute lane choices. All three parts are in
the selected language. No future rows, suggested answers, path planner, or
safety controller are supplied. The exact inputs are retained in live-run JSON.
The target changes every two seconds, including instruction paraphrases.

This is a demonstration of language-conditioned decisions under deadlines, not
a general model ranking or a vision benchmark. See [sample provenance](recordings/README.md)
for how the bundled measurements were prepared.

## Files

- `engine.py`: seeded courses, English/Vietnamese inputs, deadlines, and scoring.
- `backends.py`: public SDK adapters; Laya imports are optional.
- `run.py`: disposable model workers, monotonic clock, and raw recording output.
- `replay.py`: standard-library builder for one offline HTML file.
- `web/`: renderer and bundled artwork; no CDN dependencies.
- `recordings/`: compressed sample traces and their provenance.

Live output includes a protocol, runtime metadata, full course and input records,
actual response times, row outcomes, and a standalone replay. Local model paths
are represented by their directory name in metadata. Keep generated recordings
outside the example source tree.

From a development checkout, run `pytest tests/test_gem_run_*.py` to verify the
engine, worker lifecycle, and replay packaging without downloading a model.
