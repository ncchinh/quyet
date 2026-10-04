# Sample measurements

`sample.json.gz` contains eight measured race prefixes: two languages, two
models per language, and seeds 701 and 702. The default replay displays one
language and one seed at a time.

The recordings were measured on 2026-10-04, separately on one NVIDIA GeForce
RTX 4070 Ti SUPER. They used Quyet 1.0.0 and Laya 0.3.20 with PyTorch
2.14.0+cu130. English uses Quyet Small-EN and Laya EN; Vietnamese uses Quyet Small
and Laya Multilingual. Each pair receives the same course and deadlines.

The original experiment continued after collisions. This sample keeps only
the measured prefix through each model's **first rock**, including its actual
calls and row outcomes. Later events were removed; scores were not rerun or
invented. A request still in flight at the cutoff is recorded as pending, not
as a fabricated answer. The bundled replay is therefore a first-rock view of
the original experiment. The release runner actually stops at that point and
terminates pending inference.

The JSON includes the shared courses, language/model pairings, runtime metadata,
per-model outcomes and response times, original source-file hashes, cutoff times,
and hashes of the exact input prefixes. Shared courses are stored once to keep
the example compact. This fixture supports reproducible visualization of these
particular measurements; it is not evidence of a general model ranking.
