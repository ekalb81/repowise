"""Embedder requests overlap; writes do not.

A reindex was spending its wall-clock awaiting one embedder request at a time:
measured against Voxell, a 16-page chunk takes ~6.5s of provider compute, and a
4,067-page reindex issued ~255 of them serially — about 3 requests per minute
against an allowance of 100. These pin the fix and, more importantly, the two
properties that make it safe: the store still writes in chunk order, and each
backend keeps the failure semantics it had.
"""

from __future__ import annotations

import asyncio

import pytest

from repowise.core.persistence.vector_store._base import (
    EMBED_BATCH_MAX_ITEMS,
    EMBED_CONCURRENCY_DEFAULT,
    embed_chunks_concurrently,
    resolve_embed_concurrency,
)


class _RecordingEmbedder:
    """Counts overlap and records the order calls were issued in."""

    dimensions = 3

    def __init__(self, delay: float = 0.02, fail_on: set[int] | None = None) -> None:
        self._delay = delay
        self._fail_on = fail_on or set()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        idx = self.calls
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delay)
            if idx in self._fail_on:
                raise RuntimeError(f"chunk {idx} exploded")
            return [[float(idx), 0.0, 0.0] for _ in texts]
        finally:
            self.in_flight -= 1


def _items(n: int) -> list[tuple[str, str, dict]]:
    return [(f"p{i}", f"text {i}", {}) for i in range(n)]


# ---------------------------------------------------------------------------
# resolve_embed_concurrency
# ---------------------------------------------------------------------------


def test_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("REPOWISE_EMBED_CONCURRENCY", raising=False)
    assert resolve_embed_concurrency() == EMBED_CONCURRENCY_DEFAULT


def test_shipped_default_is_serial():
    """Pinned deliberately: raising it is a measured decision, not a tweak."""
    assert EMBED_CONCURRENCY_DEFAULT == 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBED_CONCURRENCY", "9")
    assert resolve_embed_concurrency() == 9


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBED_CONCURRENCY", "9")
    assert resolve_embed_concurrency(2) == 2


@pytest.mark.parametrize("raw", ["0", "-4", "nonsense", ""])
def test_never_returns_less_than_one(monkeypatch, raw):
    """Zero would deadlock the semaphore; garbage must degrade to serial-or-default."""
    monkeypatch.setenv("REPOWISE_EMBED_CONCURRENCY", raw)
    assert resolve_embed_concurrency() >= 1


# ---------------------------------------------------------------------------
# embed_chunks_concurrently
# ---------------------------------------------------------------------------


async def test_default_is_serial(monkeypatch):
    """Voxell measured *slower* concurrent (67.7s vs 23.5s for the same four
    chunks), so the shipped default must not fan out on anyone's behalf."""
    monkeypatch.delenv("REPOWISE_EMBED_CONCURRENCY", raising=False)
    emb = _RecordingEmbedder()
    await embed_chunks_concurrently(emb, _items(EMBED_BATCH_MAX_ITEMS * 4))
    assert emb.calls == 4
    assert emb.max_in_flight == 1


async def test_requests_overlap_when_opted_in(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBED_CONCURRENCY", "4")
    emb = _RecordingEmbedder()
    await embed_chunks_concurrently(emb, _items(EMBED_BATCH_MAX_ITEMS * 4))
    assert emb.calls == 4
    assert emb.max_in_flight > 1, "opting in did nothing"


async def test_concurrency_is_bounded(monkeypatch):
    """The bound is the contract: providers' free tiers differ by orders of
    magnitude, so an unbounded fan-out would rate-limit the slowest of them."""
    monkeypatch.setenv("REPOWISE_EMBED_CONCURRENCY", "2")
    emb = _RecordingEmbedder()
    await embed_chunks_concurrently(emb, _items(EMBED_BATCH_MAX_ITEMS * 6))
    assert emb.calls == 6
    assert emb.max_in_flight <= 2


async def test_concurrency_one_is_serial(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBED_CONCURRENCY", "1")
    emb = _RecordingEmbedder()
    await embed_chunks_concurrently(emb, _items(EMBED_BATCH_MAX_ITEMS * 3))
    assert emb.max_in_flight == 1


async def test_results_are_in_chunk_order_not_completion_order():
    """Writes are replayed from this list, so order must not depend on which
    request finished first."""

    class _Jittered(_RecordingEmbedder):
        async def embed(self, texts):
            idx = self.calls
            self.calls += 1
            # Later chunks finish first.
            await asyncio.sleep(0.05 / (idx + 1))
            return [[float(idx), 0.0, 0.0] for _ in texts]

    results = await embed_chunks_concurrently(_Jittered(), _items(EMBED_BATCH_MAX_ITEMS * 4))
    first_ids = [chunk[0][0] for chunk, _v, _e in results]
    assert first_ids == ["p0", "p16", "p32", "p48"]


async def test_failures_are_returned_not_raised():
    emb = _RecordingEmbedder(fail_on={1})
    results = await embed_chunks_concurrently(emb, _items(EMBED_BATCH_MAX_ITEMS * 3))
    errors = [exc for _c, _v, exc in results]
    assert errors[0] is None
    assert isinstance(errors[1], RuntimeError)
    assert errors[2] is None, "one bad chunk must not sink its neighbours"


async def test_empty_input_makes_no_requests():
    emb = _RecordingEmbedder()
    assert await embed_chunks_concurrently(emb, []) == []
    assert emb.calls == 0


# ---------------------------------------------------------------------------
# Backend semantics are unchanged
# ---------------------------------------------------------------------------


async def test_in_memory_store_still_fails_fast_and_stores_the_rest():
    from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore

    store = InMemoryVectorStore(_RecordingEmbedder(fail_on={1}))
    with pytest.raises(RuntimeError):
        await store.embed_batch(_items(EMBED_BATCH_MAX_ITEMS * 3))
    # Chunk 0 was written before the failure was reached, chunk 2 was not.
    assert "p0" in store._store
    assert "p32" not in store._store


async def test_in_memory_store_writes_every_chunk_when_all_succeed():
    from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore

    store = InMemoryVectorStore(_RecordingEmbedder())
    await store.embed_batch(_items(EMBED_BATCH_MAX_ITEMS * 3))
    assert len(store._store) == EMBED_BATCH_MAX_ITEMS * 3


async def test_embed_texts_returns_vectors_in_input_order():
    from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore

    store = InMemoryVectorStore(_RecordingEmbedder())
    out = await store.embed_texts([f"t{i}" for i in range(EMBED_BATCH_MAX_ITEMS * 3)])
    assert len(out) == EMBED_BATCH_MAX_ITEMS * 3
    # Chunk index is baked into each vector's first component.
    assert out[0][0] == 0.0
    assert out[EMBED_BATCH_MAX_ITEMS][0] == 1.0
    assert out[EMBED_BATCH_MAX_ITEMS * 2][0] == 2.0
