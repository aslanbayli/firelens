"""Build and deterministically resolve language-neutral repository graphs."""

import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.cancellation import CancellationCallback, raise_if_cancelled
from app.core.models import GraphEdge, GraphFact, GraphNode, Repository, Symbol
from app.indexing.analysis import ParsedDocument


@dataclass(frozen=True)
class GraphResolution:
    """Resolved edges and per-fact diagnostics for one repository pass."""

    edges: list[GraphEdge]
    fact_resolutions: dict[uuid.UUID, tuple[str, str | None]]
    unresolved_count: int
    ambiguous_count: int


def build_file_graph_records(
    repository_id: uuid.UUID,
    parsed_document: ParsedDocument,
    symbols: list[Symbol],
) -> tuple[list[GraphNode], list[GraphFact]]:
    """Attach stable repository identities to adapter-emitted graph records."""

    symbol_ids = {symbol.qualified_name: symbol.id for symbol in symbols}
    nodes: list[GraphNode] = []
    nodes_by_reference: dict[str, GraphNode] = {}
    for definition in parsed_document.graph_nodes:
        node_id = uuid.uuid5(
            repository_id,
            (
                f"graph-node:{definition.node_kind}:{definition.qualified_name}:"
                f"{definition.relative_path}"
            ),
        )
        symbol_id = (
            symbol_ids.get(definition.symbol_qualified_name)
            if definition.symbol_qualified_name is not None
            else None
        )
        node = GraphNode(
            id=node_id,
            repository_id=repository_id,
            kind=definition.node_kind,
            qualified_name=definition.qualified_name,
            name=definition.name,
            relative_path=definition.relative_path,
            start_line=definition.start_line,
            end_line=definition.end_line,
            symbol_id=symbol_id,
            language=parsed_document.source_file.language,
        )
        nodes.append(node)
        nodes_by_reference[definition.qualified_name] = node

    facts: list[GraphFact] = []
    for ordinal, unresolved in enumerate(parsed_document.graph_facts):
        source_node = nodes_by_reference.get(unresolved.source_reference)
        if source_node is None:
            raise ValueError(
                "Graph fact source is not defined by its language adapter: "
                f"{unresolved.source_reference}"
            )
        fact_id = uuid.uuid5(
            repository_id,
            (
                f"graph-fact:{unresolved.source_file}:{unresolved.start_line}:"
                f"{unresolved.end_line}:{unresolved.edge_kind}:"
                f"{unresolved.source_reference}:{unresolved.target_reference}:"
                f"{ordinal}"
            ),
        )
        facts.append(
            GraphFact(
                id=fact_id,
                repository_id=repository_id,
                source_node_id=source_node.id,
                kind=unresolved.edge_kind,
                source_reference=unresolved.source_reference,
                target_reference=unresolved.target_reference,
                source_scope=unresolved.source_scope,
                source_file=unresolved.source_file,
                start_line=unresolved.start_line,
                end_line=unresolved.end_line,
                extraction_adapter=unresolved.extraction_adapter,
                adapter_version=unresolved.adapter_version,
                confidence=unresolved.confidence,
                target_kind=unresolved.target_kind,
                target_qualified_hint=unresolved.target_qualified_hint,
                hint_resolution_method=unresolved.hint_resolution_method,
                evidence_text=unresolved.evidence_text,
            )
        )
    return nodes, facts


def repository_graph_node(repository: Repository) -> GraphNode:
    """Return the stable root node for a repository graph."""

    return GraphNode(
        id=uuid.uuid5(repository.id, "graph-node:repository"),
        repository_id=repository.id,
        kind="repository",
        qualified_name=repository.absolute_path,
        name=Path(repository.absolute_path).name or repository.absolute_path,
    )


def resolve_graph_facts(
    repository_id: uuid.UUID,
    nodes: list[GraphNode],
    facts: list[GraphFact],
    cancellation_callback: CancellationCallback | None = None,
) -> GraphResolution:
    """Resolve all repository facts using explicit, stable priority rules."""

    qualified_nodes: dict[str, list[GraphNode]] = {}
    short_nodes: dict[str, list[GraphNode]] = {}
    module_by_path: dict[str, GraphNode] = {}
    file_by_path: dict[str, GraphNode] = {}
    nodes_by_id = {node.id: node for node in nodes}
    for node in nodes:
        qualified_nodes.setdefault(node.qualified_name, []).append(node)
        short_nodes.setdefault(node.name, []).append(node)
        if node.kind == "module" and node.relative_path is not None:
            module_by_path[node.relative_path] = node
        if node.kind in {"file", "test_file"} and node.relative_path is not None:
            file_by_path[node.relative_path] = node

    edges: list[GraphEdge] = []
    dependency_edges: dict[tuple[uuid.UUID, uuid.UUID], GraphEdge] = {}
    resolutions: dict[uuid.UUID, tuple[str, str | None]] = {}
    unresolved_count = 0
    ambiguous_count = 0

    for fact in sorted(
        facts,
        key=lambda item: (
            item.source_file,
            item.start_line,
            item.end_line,
            item.kind,
            str(item.id),
        ),
    ):
        raise_if_cancelled(cancellation_callback)
        outcome = _resolve_one_fact(
            fact,
            qualified_nodes,
            short_nodes,
            module_by_path,
        )
        if outcome[0] == "ambiguous":
            resolutions[fact.id] = ("ambiguous", outcome[2])
            ambiguous_count += 1
            continue
        target = outcome[1]
        method = outcome[2]
        confidence_factor = outcome[3]
        if target is None or method is None:
            resolutions[fact.id] = ("unresolved", None)
            unresolved_count += 1
            continue

        edge_confidence = min(1.0, fact.confidence * confidence_factor)
        edge = GraphEdge(
            id=uuid.uuid5(repository_id, f"graph-edge:{fact.id}:{target.id}"),
            repository_id=repository_id,
            source_node_id=fact.source_node_id,
            target_node_id=target.id,
            kind=fact.kind,
            source_file=fact.source_file,
            start_line=fact.start_line,
            end_line=fact.end_line,
            extraction_adapter=fact.extraction_adapter,
            adapter_version=fact.adapter_version,
            resolution_method=method,
            confidence=edge_confidence,
            evidence_text=fact.evidence_text,
        )
        edges.append(edge)
        resolutions[fact.id] = ("resolved", method)

        if fact.kind != "imports":
            continue
        source_node = nodes_by_id.get(fact.source_node_id)
        source_file = (
            file_by_path.get(source_node.relative_path)
            if source_node is not None and source_node.relative_path is not None
            else None
        )
        target_file = (
            file_by_path.get(target.relative_path)
            if target.relative_path is not None
            else None
        )
        if source_file is None or target_file is None or source_file.id == target_file.id:
            continue
        dependency = GraphEdge(
            id=uuid.uuid5(
                repository_id,
                f"graph-dependency:{source_file.id}:{target_file.id}",
            ),
            repository_id=repository_id,
            source_node_id=source_file.id,
            target_node_id=target_file.id,
            kind="depends_on",
            source_file=fact.source_file,
            start_line=fact.start_line,
            end_line=fact.end_line,
            extraction_adapter=fact.extraction_adapter,
            adapter_version=fact.adapter_version,
            resolution_method="resolved_import",
            confidence=edge_confidence,
            evidence_text=fact.evidence_text,
        )
        key = (source_file.id, target_file.id)
        previous = dependency_edges.get(key)
        if previous is None or dependency.confidence > previous.confidence:
            dependency_edges[key] = dependency

    edges.extend(dependency_edges.values())
    edges.sort(
        key=lambda edge: (
            str(edge.source_node_id),
            edge.kind,
            str(edge.target_node_id),
            edge.source_file,
            edge.start_line,
            str(edge.id),
        )
    )
    return GraphResolution(
        edges=edges,
        fact_resolutions=resolutions,
        unresolved_count=unresolved_count,
        ambiguous_count=ambiguous_count,
    )


def _resolve_one_fact(
    fact: GraphFact,
    qualified_nodes: dict[str, list[GraphNode]],
    short_nodes: dict[str, list[GraphNode]],
    module_by_path: dict[str, GraphNode],
) -> tuple[str, GraphNode | None, str | None, float]:
    priorities: list[tuple[str, list[GraphNode], float]] = []

    if fact.hint_resolution_method == "same_scope_qualified":
        priorities.append(
            (
                "same_scope_qualified",
                _qualified_matches(
                    qualified_nodes,
                    fact.target_qualified_hint,
                    fact.target_kind,
                ),
                1.0,
            )
        )

    priorities.append(
        (
            "same_scope_qualified",
            _scope_matches(fact, qualified_nodes),
            1.0,
        )
    )

    if fact.hint_resolution_method == "explicitly_imported_symbol":
        priorities.append(
            (
                "explicitly_imported_symbol",
                _qualified_matches(
                    qualified_nodes,
                    fact.target_qualified_hint,
                    fact.target_kind,
                ),
                1.0,
            )
        )
    if fact.hint_resolution_method in {
        "explicitly_imported_module",
        "explicitly_imported_module_member",
    }:
        priorities.append(
            (
                fact.hint_resolution_method,
                _qualified_matches(
                    qualified_nodes,
                    fact.target_qualified_hint,
                    fact.target_kind,
                ),
                1.0,
            )
        )

    module = module_by_path.get(fact.source_file)
    same_module_reference = (
        f"{module.qualified_name}.{fact.target_reference}" if module else None
    )
    priorities.append(
        (
            "same_module_symbol",
            _qualified_matches(
                qualified_nodes,
                same_module_reference,
                fact.target_kind,
            ),
            0.95,
        )
    )
    priorities.append(
        (
            "unique_repository_qualified_match",
            _qualified_matches(
                qualified_nodes,
                fact.target_reference,
                fact.target_kind,
            ),
            0.9,
        )
    )
    short_name = fact.target_reference.rsplit(".", 1)[-1]
    priorities.append(
        (
            "unique_short_name_match",
            _filter_kind(short_nodes.get(short_name, []), fact.target_kind),
            0.7,
        )
    )

    seen_priorities: set[tuple[str, tuple[uuid.UUID, ...]]] = set()
    for method, candidates, confidence_factor in priorities:
        unique_candidates = list({candidate.id: candidate for candidate in candidates}.values())
        priority_key = (method, tuple(sorted(candidate.id for candidate in unique_candidates)))
        if priority_key in seen_priorities:
            continue
        seen_priorities.add(priority_key)
        if len(unique_candidates) == 1:
            return "resolved", unique_candidates[0], method, confidence_factor
        if len(unique_candidates) > 1:
            return "ambiguous", None, method, confidence_factor
    return "unresolved", None, None, 0.0


def _scope_matches(
    fact: GraphFact,
    qualified_nodes: dict[str, list[GraphNode]],
) -> list[GraphNode]:
    if fact.target_reference.startswith(("self.", "cls.")):
        return []
    scope_parts = fact.source_scope.split(".")
    matches: list[GraphNode] = []
    for length in range(len(scope_parts), 0, -1):
        reference = ".".join([*scope_parts[:length], fact.target_reference])
        current = _qualified_matches(
            qualified_nodes,
            reference,
            fact.target_kind,
        )
        if current:
            return current
    return matches


def _qualified_matches(
    qualified_nodes: dict[str, list[GraphNode]],
    reference: str | None,
    target_kind: str,
) -> list[GraphNode]:
    if not reference:
        return []
    return _filter_kind(qualified_nodes.get(reference, []), target_kind)


def _filter_kind(nodes: list[GraphNode], target_kind: str) -> list[GraphNode]:
    if target_kind == "any":
        return [node for node in nodes if node.kind != "repository"]
    if target_kind == "symbol":
        return [node for node in nodes if node.kind in {"symbol", "test_symbol"}]
    if target_kind == "module":
        return [node for node in nodes if node.kind == "module"]
    if target_kind == "file":
        return [node for node in nodes if node.kind in {"file", "test_file"}]
    return [node for node in nodes if node.kind == target_kind]
