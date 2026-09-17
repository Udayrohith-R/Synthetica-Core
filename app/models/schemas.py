"""Pydantic v2 schemas for the Synthetica-Core five-module pipeline."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OTHER = "other"


class ClaimRelationType(str, Enum):
    """Directed edges between claims in the 4D property graph."""

    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"


class PipelinePath(str, Enum):
    """Entropy-gated routing decision after CAMMR."""

    FAST_PATH = "fast_path"
    DEEP_ARBITRATION = "deep_arbitration"


class ExtractorTier(str, Enum):
    """Model cascade tier for fact extraction."""

    FAST = "fast"  # Llama-3.3-70B
    DENSE = "dense"  # Llama-3.3-70B


class SynthesisStatus(str, Enum):
    """Terminal status for a research synthesis session."""

    COMPLETED = "COMPLETED"
    REQUIRES_CLARIFICATION = "REQUIRES_CLARIFICATION"


_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


# ---------------------------------------------------------------------------
# Canonical knowledge-graph / API pipeline contracts
# ---------------------------------------------------------------------------


class AtomicClaim(BaseModel):
    """
    Atomic, citation-anchored factual claim for the knowledge graph.

    Serialized via ``model_dump(mode="json")`` / ``model_dump_json()`` for
    agent-swarm and Neo4j ingestion paths.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        ser_json_bytes="utf8",
    )

    claim_id: str = Field(
        ...,
        description="Stable UUID string uniquely identifying this claim.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    statement: str = Field(
        ...,
        min_length=1,
        description="Self-contained atomic factual statement.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extractor confidence in [0.0, 1.0].",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Surface-form entity names mentioned by the claim.",
    )
    source_domain: str = Field(
        ...,
        min_length=1,
        description="Hostname / publisher domain of the supporting source.",
        examples=["sec.gov", "nature.com"],
    )
    citation_anchor: str = Field(
        ...,
        description="Absolute URL citing the supporting evidence span.",
        examples=["https://example.com/doc#section-2"],
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional dense vector for Matryoshka / CAMMR scoring.",
    )
    consensus_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Graph consensus mass after CAMMR (corroboration vs contradiction).",
    )

    @field_validator("claim_id", mode="before")
    @classmethod
    def _validate_claim_id(cls, value: Any) -> str:
        """Accept UUID objects or strings; always store canonical lowercase UUID text."""
        if isinstance(value, UUID):
            return str(value)
        try:
            return str(UUID(str(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("claim_id must be a valid UUID string") from exc

    @field_validator("entities", mode="before")
    @classmethod
    def _normalize_entities(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("entities must be a list of strings")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned

    @field_validator("citation_anchor", mode="before")
    @classmethod
    def _validate_citation_anchor(cls, value: Any) -> str:
        """Ensure citation_anchor is a well-formed HTTP(S) URL, stored as str."""
        try:
            return str(_HTTP_URL_ADAPTER.validate_python(value))
        except Exception as exc:
            raise ValueError("citation_anchor must be a valid URL") from exc

    @field_validator("embedding")
    @classmethod
    def _validate_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) == 0:
            raise ValueError("embedding, if provided, must be a non-empty float list")
        return value


class ResearchQueryRequest(BaseModel):
    """
    Inbound research query from an agent-swarm client.

    ``temporal_scope`` is an optional map of integer bounds (e.g. ``start_year``,
    ``end_year``) used by the 4D spatiotemporal graph filter.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Calling agent identifier (Cursor, Cognition, Harvey, …).",
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Natural-language or structured research question.",
    )
    temporal_scope: dict[str, int] | None = Field(
        default=None,
        description="Optional integer time bounds, e.g. {'start_year': 2020, 'end_year': 2024}.",
    )
    required_credibility: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum claim confidence / credibility gate for inclusion.",
    )

    @field_validator("temporal_scope")
    @classmethod
    def _validate_temporal_scope(
        cls,
        value: dict[str, int] | None,
    ) -> dict[str, int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("temporal_scope, if provided, must be a non-empty dict")
        for key, bound in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("temporal_scope keys must be non-empty strings")
            if not isinstance(bound, int) or isinstance(bound, bool):
                raise ValueError(f"temporal_scope[{key!r}] must be an int")
        start = value.get("start_year")
        end = value.get("end_year")
        if start is not None and end is not None and start > end:
            raise ValueError("temporal_scope start_year cannot exceed end_year")
        return value


class ConflictEvidence(BaseModel):
    """A claim playing a support role in the conflict-aware answer."""

    model_config = ConfigDict(extra="forbid")

    claim: AtomicClaim
    role: Literal["support"] = "support"
    note: str | None = None


class RejectedClaim(BaseModel):
    """A claim considered and rejected, with an explicit reason."""

    model_config = ConfigDict(extra="forbid")

    claim: AtomicClaim
    why_rejected: str = Field(..., min_length=1)


class ConflictAwareAnswer(BaseModel):
    """
    Agent-native resolution object: consensus fact + rejected rivals.

    This is the differentiator vs naïve RAG — conflicts are explicit, not blended.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str | None = Field(
        default=None,
        description="Primary grounded factual answer for the query, if resolvable.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metric_key: str | None = Field(
        default=None,
        description="Ontology key for the answered metric (e.g. data_center_revenue).",
    )
    support: list[ConflictEvidence] = Field(default_factory=list)
    conflicts: list[RejectedClaim] = Field(default_factory=list)
    act_safe: bool = Field(
        default=False,
        description="True when entropy/confidence allow an agent to act without clarification.",
    )


class SynthesisResponse(BaseModel):
    """
    Citation-anchored synthesis returned to the agent after CAMMR / arbitration.

    ``status`` is ``COMPLETED`` on clean consensus, or ``REQUIRES_CLARIFICATION``
    when entropy / credibility gates block a definitive answer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    session_id: str = Field(
        ...,
        description="UUID string for this synthesis session / request.",
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
    )
    status: Literal["COMPLETED", "REQUIRES_CLARIFICATION"] = Field(
        ...,
        description="Pipeline terminal status for the agent client.",
    )
    factual_consensus_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate consensus index across returned claims.",
    )
    claims: list[AtomicClaim] = Field(
        default_factory=list,
        description="Ranked atomic claims meeting the credibility gate.",
    )
    entropy_score: float = Field(
        ...,
        ge=0.0,
        description="Subgraph information entropy H(R_q) for the selected claim set.",
    )
    grounded_answer: ConflictAwareAnswer | None = Field(
        default=None,
        description="Conflict-aware answer with support / why_rejected rivals.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="End-to-end research pipeline latency in milliseconds.",
    )
    step_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage timing breakdown in milliseconds.",
    )

    @field_validator("session_id", mode="before")
    @classmethod
    def _validate_session_id(cls, value: Any) -> str:
        if isinstance(value, UUID):
            return str(value)
        try:
            return str(UUID(str(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("session_id must be a valid UUID string") from exc

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        if isinstance(value, SynthesisStatus):
            return value.value
        if isinstance(value, str):
            return value.strip().upper()
        raise ValueError("status must be COMPLETED or REQUIRES_CLARIFICATION")


# ---------------------------------------------------------------------------
# Module 1 — Agent state-space inputs
# ---------------------------------------------------------------------------


class AgentStateTensor(BaseModel):
    """Non-human agent context: AST, stack traces, and state tensors."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None
    agent_family: Literal["cursor", "cognition", "harvey", "other"] | None = None
    query: str = Field(..., min_length=1, max_length=8192)
    ast_dump: str | None = Field(default=None, description="Serialized AST / code graph")
    stack_trace: str | None = None
    state_tensor: list[float] = Field(
        default_factory=list,
        description="Flattened agent state embedding / feature vector",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Chunk node in Document -> Chunk -> Claim hierarchy."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    document_id: str
    ordinal: int = Field(default=0, ge=0)
    text: str
    start_offset: int | None = None
    end_offset: int | None = None
    embedding: list[float] | None = None


class SourceRef(BaseModel):
    """Provenance / citation anchor."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    url: HttpUrl | None = None
    title: str | None = None
    publisher: str | None = None
    retrieved_at: datetime | None = None
    published_at: datetime | None = None


class Entity(BaseModel):
    """Named entity extracted from a claim or document."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=512)
    entity_type: EntityType = EntityType.OTHER
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AtomicTriple(BaseModel):
    """JSON-LD style atomic triple emitted by the fact extractor."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    claim_id: UUID | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    jsonld: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """Atomic factual claim derived from source text."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_url: HttpUrl | None = None
    source_title: str | None = None
    document_id: str | None = None
    chunk_id: UUID | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: list[Entity] = Field(default_factory=list)
    triples: list[AtomicTriple] = Field(default_factory=list)
    supporting_span: str | None = None
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class ClaimRelation(BaseModel):
    """Directed claim-to-claim edge (corroborate / contradict / supersede)."""

    model_config = ConfigDict(extra="forbid")

    source_claim_id: UUID
    target_claim_id: UUID
    relation: ClaimRelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rationale: str | None = None


class SearchResult(BaseModel):
    """Filtered document / raw snippet from Module 1 retrieval."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str | None = None
    url: HttpUrl | None = None
    text: str | None = None
    score: float | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    published_date: datetime | None = None
    author: str | None = None
    chunks: list[DocumentChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedDocument(BaseModel):
    """Document after CAMMR re-ranking."""

    model_config = ConfigDict(extra="forbid")

    document: SearchResult
    cammr_score: float
    entropy: float | None = None
    claim_coverage: float | None = None
    consensus_score: float | None = None
    rank: int = Field(..., ge=1)


class CitationAnchor(BaseModel):
    """Citation-anchored claim for agent consumption."""

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    source: SourceRef | None = None
    cammr_score: float | None = None


class GraphContext(BaseModel):
    """Compact graph context returned on the fast path."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    subgraph_entropy: float | None = None


class ArbitrationResult(BaseModel):
    """Module 5 deep arbitration outcome."""

    model_config = ConfigDict(extra="forbid")

    resolved_claims: list[Claim] = Field(default_factory=list)
    discarded_claim_ids: list[UUID] = Field(default_factory=list)
    synthesis: str | None = None
    model_used: str | None = None
    conflicts_resolved: int = 0


class QueryRequest(BaseModel):
    """Inbound agent-swarm retrieval / synthesis request."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=8192)
    agent_state: AgentStateTensor | None = None
    num_results: int = Field(default=10, ge=1, le=100)
    extract_claims: bool = True
    build_graph: bool = True
    rerank: bool = True
    extractor_tier: ExtractorTier = ExtractorTier.FAST
    force_arbitration: bool = False
    entropy_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override H(R_q) threshold; None uses server default",
    )
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """End-to-end middleware response for an agent query."""

    model_config = ConfigDict(extra="forbid")

    query: str
    path: PipelinePath
    results: list[RankedDocument] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    triples: list[AtomicTriple] = Field(default_factory=list)
    citations: list[CitationAnchor] = Field(default_factory=list)
    graph_context: GraphContext | None = None
    subgraph_entropy: float | None = None
    arbitration: ArbitrationResult | None = None
    synthesis: str | None = None
    latency_ms: float | None = None
    request_id: UUID = Field(default_factory=uuid4)


class HealthResponse(BaseModel):
    """Healthcheck payload."""

    status: str
    service: str
    version: str
    environment: str
