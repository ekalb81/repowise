"""Voxell Forge embedding support for repowise semantic search.

Uses the OpenAI-compatible endpoint at ``https://api.voxell.ai/v1``.
No additional pip install required — uses the ``openai`` package.

Default model: forge-turbo (1024 dims)

This exists as its own embedder rather than as ``openai`` plus an
``OPENAI_BASE_URL`` override because the two collide on a machine that also
holds a real OpenAI key: repowise's ``.env`` loader does not overwrite an
already-set variable, so a shell (or MCP server) carrying ``OPENAI_API_KEY``
sends *that* key to Voxell and gets a 401 that names neither side of the
mix-up. ``VOXELL_API_KEY`` cannot be shadowed by an unrelated credential.

Usage:
    from repowise.core.providers.embedding.voxell import VoxellEmbedder

    embedder = VoxellEmbedder(api_key="vf_...")
    vectors = await embedder.embed(["some text"])
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any, ClassVar

_DEFAULT_BASE_URL = "https://api.voxell.ai/v1"


class VoxellEmbedder:
    """Voxell Forge embedding adapter implementing the repowise Embedder protocol.

    Args:
        api_key:    Voxell API key. Falls back to the VOXELL_API_KEY env var.
        model:      Embedding model name. Default: "forge-turbo".
        base_url:   Override the endpoint. Falls back to VOXELL_BASE_URL.
        dimensions: Output width for a model not in ``_DIMS``. Falls back to
            REPOWISE_EMBEDDING_DIMS, then the known-model table. An override is
            also sent to the API so the returned vectors match the declaration.
        timeout:    Per-request timeout in seconds. Falls back to
            REPOWISE_EMBEDDING_TIMEOUT, then 30.0. Higher than the OpenAI
            embedder's default because Voxell's free tier is rate-limited and
            a queued batch can sit for a while before it is served.
    """

    # Native output widths, measured against the live API rather than copied
    # from docs. An entry that drifts is caught at embed() time by the width
    # check below, which is the only thing standing between a wrong number here
    # and a vector store quietly sized to a width its vectors never have.
    _DIMS: ClassVar[dict[str, int]] = {
        "forge-turbo": 1024,
        "forge-pro": 2560,
        "forge-ultra-4k": 4096,
        # Voxell serves these two names as aliases of its own models; the
        # widths match OpenAI's, which is the point of the aliases.
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    _DEFAULT_TIMEOUT: float = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "forge-turbo",
        timeout: float | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("VOXELL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Voxell API key required. Pass api_key= or set VOXELL_API_KEY env var."
            )
        self._base_url = base_url or os.environ.get("VOXELL_BASE_URL") or _DEFAULT_BASE_URL
        self._model = model
        env_timeout = os.environ.get("REPOWISE_EMBEDDING_TIMEOUT")
        self._timeout = timeout or (float(env_timeout) if env_timeout else self._DEFAULT_TIMEOUT)
        self._dimensions, self._request_dimensions = self._resolve_dimensions(dimensions, model)
        self._client: object | None = None  # cached; created once on first embed()

    @classmethod
    def _resolve_dimensions(cls, dimensions: int | None, model: str) -> tuple[int, int | None]:
        """Resolve ``(declared_width, override)``.

        Precedence: explicit arg > REPOWISE_EMBEDDING_DIMS > known-model table.
        An unknown model with no override raises rather than guessing a width:
        guessing wrong sizes the LanceDB table to a number the vectors never
        have, and the failure surfaces later as search returning nothing.
        """
        if dimensions is None:
            env = os.environ.get("REPOWISE_EMBEDDING_DIMS")
            if env:
                try:
                    dimensions = int(env)
                except ValueError:
                    raise ValueError("dimensions must be a positive integer") from None
        if dimensions is None:
            if model not in cls._DIMS:
                known = ", ".join(sorted(cls._DIMS))
                raise ValueError(
                    f"Unknown Voxell embedding model {model!r}. Stored vectors would be "
                    f"mis-sized against the model's real output, silently corrupting the "
                    f"vector store. Set REPOWISE_EMBEDDING_DIMS to its width, add it to "
                    f"VoxellEmbedder._DIMS, or pick a known model: {known}."
                )
            return cls._DIMS[model], None
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        return dimensions, dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Voxell Forge.

        Runs the synchronous SDK call in a thread pool to avoid blocking the
        asyncio event loop.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of L2-normalized float vectors.
        """
        if not texts:
            return []

        model = self._model
        timeout = self._timeout
        base_url = self._base_url
        request_dimensions = self._request_dimensions
        expected_dimensions = self._dimensions

        def _embed_sync() -> list[list[float]]:
            import openai  # type: ignore[import-untyped]

            if self._client is None:
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
            create_kwargs: dict[str, Any] = {"model": model, "input": texts}
            if request_dimensions is not None:
                create_kwargs["dimensions"] = request_dimensions
            response = self._client.embeddings.create(**create_kwargs)  # type: ignore[union-attr]
            raw_vectors = [list(item.embedding) for item in response.data]
            widths = {len(v) for v in raw_vectors}
            if widths and widths != {expected_dimensions}:
                actual = min(widths - {expected_dimensions})
                if request_dimensions is not None:
                    hint = (
                        f"Set REPOWISE_EMBEDDING_DIMS={actual} to match the server's native"
                        f" output, or remove the override to use the model's default."
                    )
                else:
                    hint = (
                        f"The width {expected_dimensions} came from the built-in _DIMS table"
                        f" for {model!r}. Update VoxellEmbedder._DIMS[{model!r}] = {actual}."
                    )
                raise ValueError(
                    f"VoxellEmbedder declared {expected_dimensions}-dimensional vectors but"
                    f" the API returned {actual} (model={model!r}). {hint}"
                )
            return [_l2_normalize(v) for v in raw_vectors]

        return await asyncio.to_thread(_embed_sync)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector to unit length (cosine similarity = dot product)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        norm = 1.0
    return [x / norm for x in vec]
