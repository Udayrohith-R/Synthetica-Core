"""
Module 3: 4D Spatiotemporal Property Knowledge Graph (Neo4j).

Async driver + Cypher for Claim / Entity / Source / Document nodes with
MENTIONS_ENTITY, CORROBORATES, and CONTRADICTS relationships.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid5, NAMESPACE_URL

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import Settings, get_settings
from app.models.schemas import (
    AtomicClaim,
    ClaimRelation,
    ClaimRelationType,
    GraphContext,
)
from app.utils.entropy import tokenize

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_NEGATION = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "n't",
        "false",
        "deny",
        "denies",
        "denied",
        "refute",
        "refutes",
        "refuted",
        "incorrect",
        "untrue",
    }
)

# Bidirectional polarity / antonym cues for divergence detection
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("grew", "fell"),
    ("grow", "fall"),
    ("growing", "falling"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increases", "decreases"),
    ("rise", "fall"),
    ("rose", "fell"),
    ("rising", "falling"),
    ("gain", "loss"),
    ("gained", "lost"),
    ("profit", "loss"),
    ("up", "down"),
    ("higher", "lower"),
    ("more", "less"),
    ("approve", "reject"),
    ("approved", "rejected"),
    ("true", "false"),
    ("yes", "no"),
)

RelationKind = Literal["CORROBORATES", "CONTRADICTS"]


@dataclass
class IngestResult:
    """Outcome of ``ingest_claims``."""

    ingested: int = 0
    entities_upserted: int = 0
    relations: list[ClaimRelation] = field(default_factory=list)


class GraphService:
    """
    Neo4j graph client built on the official async driver.

    All Cypher execution uses ``async with driver.session()`` context managers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._driver: AsyncDriver | None = None

    # ------------------------------------------------------------------
    # Driver lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the async Neo4j driver (idempotent)."""
        if self._driver is not None:
            return
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )
        logger.info("Connected to Neo4j at %s", self._settings.neo4j_uri)

    async def close(self) -> None:
        """Close the driver and release pooled connections."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("GraphService is not connected; call connect() first")
        return self._driver

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def init_schema(self) -> None:
        """
        Create unique constraints for Document, Claim, Entity, and Source ids.

        Safe to call repeatedly (``IF NOT EXISTS``).
        """
        statements = [
            (
                "CREATE CONSTRAINT document_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT claim_id IF NOT EXISTS "
                "FOR (c:Claim) REQUIRE c.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT source_id IF NOT EXISTS "
                "FOR (s:Source) REQUIRE s.id IS UNIQUE"
            ),
        ]
        async with self.driver.session() as session:
            for stmt in statements:
                await session.run(stmt)
        logger.info("Neo4j schema constraints ensured")

    # Backward-compatible alias used by older pipeline wiring
    async def ensure_constraints(self) -> None:
        await self.init_schema()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest_claims(self, claims: list[AtomicClaim]) -> IngestResult:
        """
        Merge ``AtomicClaim`` nodes into the graph and link corroboration edges.

        For each claim:
        1. MERGE ``Source`` / ``Document`` / ``Claim``
        2. MERGE ``Entity`` nodes and ``(Claim)-[:MENTIONS_ENTITY]->(Entity)``
        3. Find existing claims sharing those entities and create
           ``CORROBORATES`` (aligned statements) or ``CONTRADICTS`` (divergent).
        """
        if not claims:
            return IngestResult()

        result = IngestResult()
        entity_names_seen: set[str] = set()

        async with self.driver.session() as session:
            for claim in claims:
                await self._merge_claim_graph(session, claim)
                result.ingested += 1
                entity_names_seen.update(e.strip() for e in claim.entities if e.strip())

                related = await self._find_related_claims(
                    session,
                    claim_id=claim.claim_id,
                    entity_names=[e for e in claim.entities if e.strip()],
                )
                for other in related:
                    kind = classify_statement_relation(
                        claim.statement,
                        other["statement"],
                    )
                    if kind is None:
                        continue
                    rel = await self._link_claims(
                        session,
                        source_id=claim.claim_id,
                        target_id=other["id"],
                        kind=kind,
                        confidence=min(
                            claim.confidence_score,
                            float(other.get("confidence_score") or 0.5),
                        ),
                    )
                    if rel is not None:
                        result.relations.append(rel)

        result.entities_upserted = len(entity_names_seen)
        return result

    async def _merge_claim_graph(self, session: Any, claim: AtomicClaim) -> None:
        """MERGE Source, Document, Claim, Entities, and MENTIONS_ENTITY edges."""
        source_id = stable_id("source", claim.citation_anchor)
        document_id = stable_id("document", claim.citation_anchor)
        entities = [
            {
                "id": stable_id("entity", name.strip().lower()),
                "name": name.strip(),
                "name_key": name.strip().lower(),
            }
            for name in claim.entities
            if name and name.strip()
        ]

        cypher = """
        MERGE (s:Source {id: $source_id})
        SET s.url = $citation_anchor,
            s.domain = $source_domain,
            s.updated_at = $now

        MERGE (d:Document {id: $document_id})
        SET d.url = $citation_anchor,
            d.domain = $source_domain,
            d.updated_at = $now
        MERGE (d)-[:HAS_SOURCE]->(s)

        MERGE (c:Claim {id: $claim_id})
        SET c.statement = $statement,
            c.confidence_score = $confidence_score,
            c.consensus_score = $consensus_score,
            c.source_domain = $source_domain,
            c.citation_anchor = $citation_anchor,
            c.updated_at = $now
        MERGE (c)-[:CITED_FROM]->(s)
        MERGE (d)-[:CONTAINS_CLAIM]->(c)

        WITH c
        UNWIND $entities AS ent
        MERGE (e:Entity {id: ent.id})
        SET e.name = ent.name,
            e.name_key = toLower(trim(coalesce(ent.name_key, ent.name, ''))),
            e.updated_at = $now
        MERGE (c)-[:MENTIONS_ENTITY]->(e)
        """
        await session.run(
            cypher,
            {
                "source_id": source_id,
                "document_id": document_id,
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "confidence_score": claim.confidence_score,
                "consensus_score": claim.consensus_score,
                "source_domain": claim.source_domain,
                "citation_anchor": claim.citation_anchor,
                "entities": entities,
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _find_related_claims(
        self,
        session: Any,
        *,
        claim_id: str,
        entity_names: list[str],
        min_shared: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Return other claims that mention the same entities (case-insensitive).

        Matching uses ``toLower(trim(...))`` on ``name_key`` / ``name`` so
        ``Nvidia`` and ``nvidia`` resolve to the same entity key even when
        older nodes lack a normalized ``name_key``.
        """
        if not entity_names:
            return []

        name_keys = sorted(
            {n.strip().lower() for n in entity_names if n and n.strip()}
        )
        if not name_keys:
            return []

        cypher = """
        MATCH (c:Claim {id: $claim_id})-[:MENTIONS_ENTITY]->(e:Entity)
        WITH c, collect(
            DISTINCT toLower(trim(coalesce(e.name_key, e.name, '')))
        ) AS c_keys
        MATCH (other:Claim)-[:MENTIONS_ENTITY]->(e2:Entity)
        WHERE other.id <> c.id
        WITH other, c_keys,
             toLower(trim(coalesce(e2.name_key, e2.name, ''))) AS other_key
        WHERE other_key <> ''
          AND other_key IN c_keys
          AND other_key IN $name_keys
        WITH other, count(DISTINCT other_key) AS shared_count
        WHERE shared_count >= $min_shared
        RETURN other.id AS id,
               other.statement AS statement,
               other.confidence_score AS confidence_score,
               shared_count AS shared_count
        """
        result = await session.run(
            cypher,
            {
                "claim_id": claim_id,
                "name_keys": name_keys,
                "min_shared": max(1, int(min_shared)),
            },
        )
        return [record.data() async for record in result]

    async def _link_claims(
        self,
        session: Any,
        *,
        source_id: str,
        target_id: str,
        kind: RelationKind,
        confidence: float,
    ) -> ClaimRelation | None:
        """MERGE a directed CORROBORATES or CONTRADICTS edge between claims."""
        # Deterministic orientation avoids duplicate A→B and B→A pairs
        left, right = sorted([source_id, target_id])
        cypher = f"""
        MATCH (a:Claim {{id: $left}})
        MATCH (b:Claim {{id: $right}})
        MERGE (a)-[r:{kind}]->(b)
        SET r.confidence = $confidence,
            r.timestamp = $timestamp,
            r.inferred = true
        RETURN a.id AS source_id, b.id AS target_id
        """
        result = await session.run(
            cypher,
            {
                "left": left,
                "right": right,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        record = await result.single()
        if record is None:
            return None

        relation_type = (
            ClaimRelationType.CORROBORATES
            if kind == "CORROBORATES"
            else ClaimRelationType.CONTRADICTS
        )
        return ClaimRelation(
            source_claim_id=UUID(str(record["source_id"])),
            target_claim_id=UUID(str(record["target_id"])),
            relation=relation_type,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get_subgraph_for_query(
        self,
        entity_names: list[str],
    ) -> GraphContext:
        """
        Fetch claims (and CORROBORATES / CONTRADICTS edges) for target entities.

        Returns a ``GraphContext`` with claim/entity nodes and typed edges.
        """
        if not entity_names:
            return GraphContext()

        name_keys = [n.strip().lower() for n in entity_names if n and n.strip()]
        if not name_keys:
            return GraphContext()

        cypher = """
        MATCH (e:Entity)
        WHERE e.name_key IN $name_keys OR e.name IN $raw_names
        MATCH (c:Claim)-[:MENTIONS_ENTITY]->(e)
        OPTIONAL MATCH (c)-[r:CORROBORATES|CONTRADICTS]->(c2:Claim)
        OPTIONAL MATCH (c)-[:MENTIONS_ENTITY]->(e2:Entity)
        OPTIONAL MATCH (c)-[:CITED_FROM]->(s:Source)
        RETURN c,
               collect(DISTINCT e2) AS entities,
               collect(DISTINCT s) AS sources,
               collect(DISTINCT {
                   type: type(r),
                   target: c2.id,
                   confidence: r.confidence,
                   timestamp: r.timestamp
               }) AS edges
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                {
                    "name_keys": name_keys,
                    "raw_names": [n.strip() for n in entity_names if n.strip()],
                },
            )
            async for record in result:
                claim = record["c"]
                claim_props = dict(claim)
                claim_key = f"Claim:{claim_props.get('id')}"
                if claim_key not in seen_nodes:
                    nodes.append({"label": "Claim", **claim_props})
                    seen_nodes.add(claim_key)

                for ent in record["entities"] or []:
                    if ent is None:
                        continue
                    ent_props = dict(ent)
                    ent_key = f"Entity:{ent_props.get('id') or ent_props.get('name')}"
                    if ent_key not in seen_nodes:
                        nodes.append({"label": "Entity", **ent_props})
                        seen_nodes.add(ent_key)

                for source in record["sources"] or []:
                    if source is None:
                        continue
                    src_props = dict(source)
                    src_key = f"Source:{src_props.get('id')}"
                    if src_key not in seen_nodes:
                        nodes.append({"label": "Source", **src_props})
                        seen_nodes.add(src_key)

                for edge in record["edges"] or []:
                    if edge and edge.get("type") and edge.get("target"):
                        edges.append(
                            {
                                "source": claim_props.get("id"),
                                "target": edge.get("target"),
                                "type": edge.get("type"),
                                "confidence": edge.get("confidence"),
                                "timestamp": edge.get("timestamp"),
                            }
                        )

        return GraphContext(nodes=nodes, edges=edges)

    async def fetch_query_subgraph(self, claim_ids: list[UUID]) -> GraphContext:
        """Legacy helper: subgraph by claim ids (still used by some callers)."""
        if not claim_ids:
            return GraphContext()

        cypher = """
        MATCH (c:Claim)
        WHERE c.id IN $ids
        OPTIONAL MATCH (c)-[:MENTIONS_ENTITY]->(e:Entity)
        OPTIONAL MATCH (c)-[r:CORROBORATES|CONTRADICTS]->(c2:Claim)
        RETURN c,
               collect(DISTINCT e) AS entities,
               collect(DISTINCT {
                   type: type(r),
                   target: c2.id,
                   confidence: r.confidence
               }) AS edges
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        async with self.driver.session() as session:
            result = await session.run(cypher, {"ids": [str(i) for i in claim_ids]})
            async for record in result:
                c = dict(record["c"])
                nodes.append({"label": "Claim", **c})
                for ent in record["entities"] or []:
                    if ent is not None:
                        nodes.append({"label": "Entity", **dict(ent)})
                for edge in record["edges"] or []:
                    if edge and edge.get("type"):
                        edges.append(
                            {
                                "source": c.get("id"),
                                "target": edge.get("target"),
                                "type": edge.get("type"),
                                "confidence": edge.get("confidence"),
                            }
                        )
        return GraphContext(nodes=nodes, edges=edges)

    async def update_consensus_scores(self, claims: list[AtomicClaim]) -> int:
        """Persist ``consensus_score`` onto existing Claim nodes after CAMMR."""
        rows = [
            {
                "id": c.claim_id,
                "consensus_score": float(c.consensus_score),
            }
            for c in claims
            if c.claim_id
        ]
        if not rows:
            return 0

        cypher = """
        UNWIND $rows AS row
        MATCH (c:Claim {id: row.id})
        SET c.consensus_score = row.consensus_score,
            c.updated_at = $now
        RETURN count(c) AS updated
        """
        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                {
                    "rows": rows,
                    "now": datetime.now(timezone.utc).isoformat(),
                },
            )
            record = await result.single()
            return int(record["updated"]) if record else 0

    async def verify_connectivity(self) -> bool:
        """Run a trivial query to confirm Neo4j is reachable."""
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            return bool(record and record["ok"] == 1)


# ---------------------------------------------------------------------------
# Alignment heuristics
# ---------------------------------------------------------------------------


def classify_statement_relation(
    left: str,
    right: str,
    *,
    align_threshold: float = 0.45,
    topical_threshold: float = 0.15,
) -> RelationKind | None:
    """
    Decide CORROBORATES vs CONTRADICTS for two statements that share entities.

    Returns ``None`` when overlap is too weak to justify an edge.
    """
    a = set(tokenize(left))
    b = set(tokenize(right))
    if not a or not b:
        return None

    union = a | b
    jaccard = len(a & b) / len(union) if union else 0.0
    if jaccard < topical_threshold:
        return None

    numbers_conflict = _numeric_conflict(left, right)
    negation_conflict = _negation_conflict(a, b)
    antonym_conflict = _antonym_conflict(a, b)

    if numbers_conflict or negation_conflict or antonym_conflict:
        return "CONTRADICTS"
    if jaccard >= align_threshold:
        return "CORROBORATES"
    # Shared entities + modest lexical overlap without conflict → soft corroboration
    if jaccard >= 0.30:
        return "CORROBORATES"
    # Topically related but divergent wording → contradiction candidate
    return "CONTRADICTS"


def _numeric_conflict(left: str, right: str) -> bool:
    """True when each statement asserts a distinctive number the other lacks."""
    nums_a = set(_NUMBER_RE.findall(left))
    nums_b = set(_NUMBER_RE.findall(right))
    if not nums_a or not nums_b:
        return False
    only_a = nums_a - nums_b
    only_b = nums_b - nums_a
    return bool(only_a) and bool(only_b)


def _negation_conflict(a: set[str], b: set[str]) -> bool:
    return bool(_NEGATION & a) != bool(_NEGATION & b) and bool(
        (a - _NEGATION) & (b - _NEGATION)
    )


def _antonym_conflict(a: set[str], b: set[str]) -> bool:
    for left, right in _ANTONYM_PAIRS:
        if (left in a and right in b) or (right in a and left in b):
            return True
    return False


def stable_id(kind: str, value: str) -> str:
    """Deterministic UUID string for Source / Document / Entity keys."""
    return str(uuid5(NAMESPACE_URL, f"synthetica:{kind}:{value}"))
