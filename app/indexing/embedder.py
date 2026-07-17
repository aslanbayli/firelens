"""Define the embedding boundary and a deterministic testing implementation.

An embedding converts text into a fixed-length numeric vector. Semantic search
compares a query vector with chunk vectors and ranks chunks whose vectors point
in similar directions. The indexer depends on this interface instead of a
specific provider so local models, APIs, and test doubles remain replaceable.
"""

import hashlib
import math
import os

# Sequence accepts lists, tuples, and other ordered collections without
# requiring callers to allocate a specific concrete collection type.
from collections.abc import Sequence
from pathlib import Path

# Any is used only at the third-party boundary because sentence-transformers is
# an optional dependency and may not provide precise type information here.
# Protocol defines structural typing: an object satisfies Embedder when it has
# the required attributes and method, without needing to inherit from it.
from typing import Any, Protocol


class Embedder(Protocol):
    """Behavior every embedding implementation must provide."""

    @property
    def provider(self) -> str:
        """Return a stable provider identifier such as `local` or `openai`."""
        # Ellipsis means the protocol specifies a signature, not an
        # implementation.
        ...

    @property
    def model(self) -> str:
        """Return the exact model identifier used to create vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Return the required number of values in every produced vector."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalized fixed-length vector for each input string."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Return a single fixed-length vector for the input query string."""
        ...


class FakeEmbedder:
    """Create deterministic normalized vectors without semantic meaning.

    This implementation verifies pipeline mechanics: batching, vector count,
    dimensions, normalization, persistence, and reproducibility. It cannot test
    whether natural-language queries retrieve conceptually related code.
    """

    provider = "test"
    model = "sha256-fake"

    def __init__(self, dimension: int = 16) -> None:
        # Vector dimensions must be positive; an empty vector cannot be compared.
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        # Store the configured dimension as required by the Embedder protocol.
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Transform a batch while preserving input order and cardinality."""

        # The list comprehension calls the deterministic single-text algorithm
        # once per item and returns vectors in exactly the same order as texts.
        return [self._embed_one(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """Return a single fixed-length vector for the input query string."""
        return self._embed_one(query.strip())

    def _embed_one(self, text: str) -> list[float]:
        """Convert one string into a deterministic unit-length vector."""

        # UTF-8 gives a deterministic byte representation of Unicode text.
        encoded_text = text.encode("utf-8")

        # digest() returns 32 raw bytes from SHA-256.
        digest = hashlib.sha256(encoded_text).digest()

        # Create the requested number of values. Modulo repeats digest bytes
        # when dimension exceeds 32. Dividing by 127.5 and subtracting one maps
        # byte values from [0, 255] approximately into [-1, 1].
        values = [
            (digest[index % len(digest)] / 127.5) - 1.0
            for index in range(self.dimension)
        ]

        # Euclidean magnitude is sqrt(x1² + x2² + ... + xn²).
        magnitude = math.sqrt(sum(value * value for value in values))

        # Dividing by zero is invalid. This is extremely unlikely for the hash
        # transformation, but explicit validation keeps the contract safe.
        if magnitude == 0:
            raise ValueError("Cannot normalize a zero embedding")

        # Dividing every component by magnitude creates a unit vector whose
        # magnitude is one. Normalized vectors allow cosine similarity to be
        # computed later using a simple dot product.
        return [value / magnitude for value in values]


class CodeRankEmbedder:
    """Embed code chunks with Nomic's CodeRankEmbed sentence-transformer model.

    This is the real semantic embedder for the indexing pipeline. It follows
    the same Embedder protocol as FakeEmbedder, which means indexer.py can call
    embedder.embed(...) without knowing whether vectors came from a fake test
    implementation or from a machine-learning model.
    """

    # A stable provider name tells repository metadata which embedding backend
    # produced the vectors.
    provider = "sentence-transformers"

    # The model string must match the Hugging Face model identifier exactly so
    # vectors can be traced back to the model that produced them.
    model = "nomic-ai/CodeRankEmbed"

    # According to the docs the query prompt must include the following
    # task instruction prefix: "Represent this query for searching relevant code"
    CODE_SEARCH_QUERY_INSTRUCTION = "Represent this query for searching relevant code: "

    def __init__(
        self,
        batch_size: int = 32,
        # normalize_embeddings=True makes every vector unit-length. This is
        # useful because cosine similarity can later be computed with a dot
        # product when both query and chunk vectors are normalized.
        normalize_embeddings: bool = True,
        # device lets you force "cpu", "mps", "cuda", etc. Leaving it as None
        # lets sentence-transformers choose the best available device.
        device: str | None = None,
        # hf_token lets tests or callers inject a token directly. When it is
        # None, the embedder reads HF_TOKEN from the environment or .env.
        hf_token: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        # Store the token that Hugging Face Hub should use for model downloads.
        # Empty strings are converted to None so unauthenticated use remains
        # possible when a token has not been configured.
        self.hf_token = hf_token or _load_hf_token()

        # Some Hugging Face internals, especially when trust_remote_code=True is
        # involved, look specifically at os.environ["HF_TOKEN"] instead of only
        # using the token argument passed to SentenceTransformer. Setting the
        # process variable here keeps those internal requests authenticated too.
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        # The model is loaded lazily so importing app.indexing.embedder does not
        # immediately download weights or initialize ML runtimes.
        self._model: Any | None = None
        # The dimension is also discovered lazily because the model must be
        # loaded before sentence-transformers can report it.
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the fixed vector length produced by CodeRankEmbed."""

        # Reuse the cached dimension after the first lookup.
        if self._dimension is not None:
            return self._dimension

        # Loading the model is required to ask sentence-transformers about its
        # embedding dimension.
        model = self._load_model()

        # get_sentence_embedding_dimension() is the sentence-transformers API
        # for asking how many floats each encoded sentence/code chunk contains.
        dimension = model.get_embedding_dimension()

        # A missing dimension would make the index metadata invalid, so fail
        # early with a direct message instead of storing malformed records.
        if dimension is None:
            raise ValueError("CodeRankEmbed did not report an embedding dimension")

        # Cache the dimension as a plain int for future calls.
        self._dimension = int(dimension)

        # Return the discovered vector size to satisfy the Embedder protocol.
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one CodeRankEmbed vector for each input text."""

        # sentence-transformers accepts many sequence-like containers, but a
        # list gives it a concrete reusable batch and makes length stable.
        batch = list(texts)

        # Preserve the protocol guarantee: no input texts means no output
        # vectors. This also avoids unnecessary model loading for empty indexes.
        if not batch:
            return []

        # Load the model only when real embeddings are requested.
        model = self._load_model()

        # encode(...) runs the transformer model and returns one vector per
        # input text. convert_to_numpy=False asks for Python-list-compatible
        # output instead of requiring callers to handle NumPy arrays.
        embeddings = model.encode(
            batch,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
        )

        # Convert provider-specific tensors/lists into plain list[list[float]]
        # so the rest of FireLens can store and compare vectors uniformly.
        vectors = [self._to_float_list(embedding) for embedding in embeddings]

        # Validate right here so provider mistakes fail near their source.
        validate_embeddings(batch, vectors, self.dimension)

        # Return vectors in the same order as the input texts.
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Return a single fixed-length vector for the input query string."""
        query = query.strip()
        if query == "":
            raise ValueError("Query must not be empty")

        query = self.CODE_SEARCH_QUERY_INSTRUCTION + query
        model = self._load_model()
        embeddings = model.encode(
            [query],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
        )
        vector = self._to_float_list(embeddings[0])

        validate_embeddings(
            [query],
            [vector],
            self.dimension,
        )

        return vector

    def _load_model(self) -> Any:
        """Load and cache the Hugging Face sentence-transformers model."""

        # If the model was already loaded, reuse it. Loading transformer weights
        # repeatedly would be slow and memory-heavy.
        if self._model is not None:
            return self._model

        try:
            # Import lazily so tests using FakeEmbedder do not require the heavy
            # optional ML dependency.
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            # Give the developer an actionable install hint instead of exposing
            # a lower-level missing-module traceback.
            raise ImportError(
                "CodeRankEmbedder requires sentence-transformers. "
                "Install it with: uv add sentence-transformers"
            ) from exc

        # trust_remote_code=True is required by the Hugging Face snippet for
        # this model because the repository provides custom model code.
        self._model = SentenceTransformer(
            self.model,
            trust_remote_code=True,
            device=self.device,
            token=self.hf_token,
        )

        return self._model

    @staticmethod
    def _to_float_list(vector: Any) -> list[float]:
        """Convert a provider vector into a plain Python list of floats."""

        # PyTorch tensors and NumPy arrays often expose tolist(), which converts
        # their contents into regular Python containers.
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        # Coerce every numeric value to float so downstream storage has a stable
        # type even if the provider returns float32 or another numeric scalar.
        return [float(value) for value in vector]


def _load_hf_token() -> str | None:
    """Load a Hugging Face token from the environment or local .env file."""

    # Read the exact environment variable name used by this project.
    token = os.getenv("HF_TOKEN")

    # If the shell environment already provided a token, use it. This is useful
    # in deployment where secrets usually come from the process environment.
    if token:
        return token

    # For local development, read the repository-level .env file directly.
    # This keeps CodeRankEmbedder independent from app.core.config, whose
    # settings currently include unrelated required OpenAI/project fields.
    env_path = Path(".env")

    # No .env means unauthenticated Hugging Face downloads are still allowed.
    if not env_path.exists():
        return None

    # Read the .env file line by line so we can extract only the Hugging Face
    # token without mutating global os.environ or loading unrelated settings.
    for line in env_path.read_text(encoding="utf-8").splitlines():
        # Strip whitespace so `HF_TOKEN = value` and `HF_TOKEN=value` both work.
        stripped_line = line.strip()

        # Ignore blank lines and comments.
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # Split only on the first equals sign so token values containing equals
        # characters are preserved.
        key, separator, value = stripped_line.partition("=")

        # Lines without KEY=VALUE syntax are not useful here.
        if not separator:
            continue

        # Normalize the key so whitespace around it does not matter.
        key = key.strip()

        # Only use the Hugging Face token key this project supports.
        if key != "HF_TOKEN":
            continue

        # Remove surrounding whitespace and optional shell-style quotes.
        token = value.strip().strip('"').strip("'")

        # Return a non-empty token; an empty env value behaves as unauthenticated.
        return token or None

    # The .env file exists but does not contain a recognized token key.
    return None


def validate_embeddings(
    # Original inputs establish how many vectors should have been returned.
    texts: Sequence[str],
    # Nested Sequence supports provider results without requiring concrete lists.
    vectors: Sequence[Sequence[float]],
    # Every vector must match the dimension recorded in repository metadata.
    expected_dimension: int,
) -> None:
    """Reject malformed embedding-provider output before storing it."""

    # A one-to-one mapping is essential because vector position corresponds to
    # chunk position. Missing or extra vectors would associate data incorrectly.
    if len(vectors) != len(texts):
        raise ValueError("Embedder returned the wrong number of vectors")

    # Validate every vector rather than assuming the first represents the batch.
    for vector in vectors:
        # Mixed dimensions cannot form a matrix and indicate provider/config
        # incompatibility.
        if len(vector) != expected_dimension:
            raise ValueError("Embedder returned a vector with the wrong dimension")

        magnitude_squared = 0.0
        for value in vector:
            if not math.isfinite(value):
                raise ValueError("Embedder returned a non-finite value")
            magnitude_squared += value * value

        if magnitude_squared == 0:
            raise ValueError("Embedder returned a zero vector")

        if not math.isclose(magnitude_squared, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Embedder returned a vector that is not normalized")
