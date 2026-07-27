#!/usr/bin/env python3

from array import array
import json
import math

import slha_core


def expect_raises(exception_type, callback):
    try:
        callback()
    except exception_type:
        return
    raise AssertionError(
        f"expected {exception_type.__name__} to be raised"
    )


assert slha_core.D_C == 128
assert slha_core.D_S == 256
assert slha_core.LATENT_BYTES == 64
assert slha_core.RESIDUAL_WORDS == 4
assert slha_core.N_GROUPS == 8

latent = array(
    "f",
    [
        (((index * 17 + 11) % 43) - 21) / (2.0 + index % 7)
        for index in range(slha_core.D_C)
    ],
)

expected_flags = {
    "int4": 0,
    "grouped": 0,
    "nf4": slha_core.FLAG_NF4,
    "mixed": slha_core.FLAG_MIXED,
    "tq3": slha_core.FLAG_TQ3,
    "mix3": slha_core.FLAG_MIX3,
}

query = array(
    "f",
    [
        ((index * 7 + 3) % 29 - 14) / 9.0
        for index in range(slha_core.D_C)
    ],
)
signs = array("Q", [0, 1, 2, 3])

for codec, expected_flag in expected_flags.items():
    tile = slha_core.compress_latent(
        latent,
        codec,
        token_id=17,
        position=23,
        head_id=4,
    )

    assert len(tile.latent_kv) == slha_core.LATENT_BYTES
    assert len(tile.residual_bitmap) == slha_core.RESIDUAL_WORDS
    assert len(tile.group_scales) == slha_core.N_GROUPS
    assert tile.flags & (
        slha_core.FLAG_NF4
        | slha_core.FLAG_MIXED
        | slha_core.FLAG_TQ3
        | slha_core.FLAG_MIX3
    ) == expected_flag

    assert tile.token_id == 17
    assert tile.position == 23
    assert tile.head_id == 4

    decoded = tile.dequantize()
    assert len(decoded) == slha_core.D_C
    assert all(math.isfinite(value) for value in decoded)

    score = tile.compute_score(query, signs)
    assert math.isfinite(score)

    representation = repr(tile)
    assert "PySlhaTile" in representation
    assert "position=23" in representation

# A strided, non-C-contiguous buffer must be copied safely rather than being
# interpreted through an unchecked raw Rust slice.
backing = array(
    "f",
    [
        float(index - 128) / 13.0
        for index in range(slha_core.D_C * 2)
    ],
)
strided = memoryview(backing)[::2]
strided_tile = slha_core.compress_latent(strided, "grouped")
assert len(strided_tile.dequantize()) == slha_core.D_C

expect_raises(
    ValueError,
    lambda: slha_core.compress_latent(
        array("f", [0.0] * (slha_core.D_C - 1)),
        "grouped",
    ),
)

non_finite = array("f", [0.0] * slha_core.D_C)
non_finite[3] = float("nan")

expect_raises(
    ValueError,
    lambda: slha_core.compress_latent(non_finite, "grouped"),
)

expect_raises(
    ValueError,
    lambda: slha_core.compress_latent(latent, "unknown"),
)

expect_raises(
    ValueError,
    lambda: slha_core.PySlhaTile(
        [0] * slha_core.LATENT_BYTES,
        [0] * slha_core.RESIDUAL_WORDS,
        1.0,
        0.0,
        0.0,
        0,
        0,
        0,
        slha_core.FLAG_NF4 | slha_core.FLAG_TQ3,
        [255] * slha_core.N_GROUPS,
    ),
)

bad_query = array("f", query)
bad_query[0] = float("inf")

tile = slha_core.compress_latent(latent, "mix3")

expect_raises(
    ValueError,
    lambda: tile.compute_score(bad_query, signs),
)

report = json.loads(slha_core.run_audit())

assert report["verdict"]["ok"] is True
assert report["verdict"]["failed"] == 0

print("Python binding smoke test: OK")
