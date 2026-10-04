import math

from quyet.calibration import size_bucket, softmax_t, temperature


def test_size_buckets():
    assert [size_bucket(k) for k in (2, 3, 5, 6, 10, 11)] == ["2", "3-5", "3-5", "6-10", "6-10", "11+"]


def test_temperature_lookup_encoder_then_llm_keys():
    enc = {"choice:3-5": 2.0, "noul:2": 1.5}
    assert temperature(enc, "choice", 4) == 2.0
    assert temperature(enc, "choice", 2) == 1.0          # bucket missing -> 1.0
    llm = {"choice": 1.3, "score": 1.6, "noul": 1.4}
    assert temperature(llm, "score", 7) == 1.6
    assert temperature({}, "noul", 2) == 1.0


def test_softmax_t():
    p = softmax_t([1.0, 2.0, 3.0], 1.0)
    assert math.isclose(sum(p), 1.0) and p[2] > p[1] > p[0]
    flat = softmax_t([1.0, 2.0, 3.0], 100.0)
    assert max(flat) - min(flat) < 0.01
    big = softmax_t([1000.0, 0.0], 1.0)                   # no overflow
    assert big[0] == 1.0
