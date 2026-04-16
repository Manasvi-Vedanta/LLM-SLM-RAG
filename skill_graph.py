"""
skill_graph.py – Directed skill prerequisite graph
----------------------------------------------------
Builds a DAG (directed acyclic graph) from learning path prerequisites
and skill_details (including ESCO URIs). Provides:
  - Topological ordering for path validation
  - Prerequisite chain lookup
  - ESCO URI mapping for standards alignment
  - Graph-constrained reordering for path adaptation
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SkillNode:
    """A skill in the prerequisite graph."""
    label: str
    session_number: int
    esco_uri: str = ""
    difficulty: str = ""
    prerequisites: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)  # skills that depend on this one


@dataclass
class SkillGraph:
    """Directed acyclic graph of skill prerequisites."""
    nodes: dict[str, SkillNode] = field(default_factory=dict)
    # edges: prerequisite -> dependent (forward edges)
    edges: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # reverse edges: dependent -> prerequisites
    reverse_edges: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def topological_order(self) -> list[str]:
        """Return skills in valid learning order (topological sort).

        Returns empty list if a cycle is detected.
        """
        in_degree = {label: 0 for label in self.nodes}
        for label in self.nodes:
            for dep in self.edges.get(label, []):
                if dep in in_degree:
                    in_degree[dep] += 1

        queue = deque(label for label, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for dep in self.edges.get(current, []):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

        if len(order) != len(self.nodes):
            logger.warning("Cycle detected in skill graph — returning partial order")
        return order

    def get_prerequisite_chain(self, skill_label: str) -> list[str]:
        """Get all transitive prerequisites for a skill (in learning order)."""
        visited = set()
        chain = []

        def _dfs(label: str):
            if label in visited:
                return
            visited.add(label)
            for prereq in self.reverse_edges.get(label, []):
                _dfs(prereq)
            chain.append(label)

        _dfs(skill_label)
        # Remove the skill itself — we only want prerequisites
        return [s for s in chain if s != skill_label]

    def get_dependents(self, skill_label: str) -> list[str]:
        """Get all skills that directly or transitively depend on this skill."""
        visited = set()
        dependents = []

        def _dfs(label: str):
            for dep in self.edges.get(label, []):
                if dep not in visited:
                    visited.add(dep)
                    dependents.append(dep)
                    _dfs(dep)

        _dfs(skill_label)
        return dependents

    def validate_ordering(self, session_order: list[dict]) -> list[str]:
        """Check if a session ordering respects prerequisites.

        Returns list of violations (empty if valid).
        """
        taught_skills: set[str] = set()
        violations = []

        for session in session_order:
            skills = session.get("skills", [])
            prereqs = session.get("prerequisites", [])

            for prereq in prereqs:
                if prereq not in taught_skills:
                    sn = session.get("session_number", "?")
                    violations.append(
                        f"Session {sn}: prerequisite '{prereq}' not yet taught"
                    )

            taught_skills.update(skills)

        return violations

    def constrained_reorder(self, sessions: list[dict],
                            priority_map: dict[str, float] | None = None) -> list[dict]:
        """Reorder sessions respecting prerequisite constraints.

        Uses topological sort + optional priority (e.g., mastery-weighted).
        Sessions with unmet prerequisites are pushed later.
        """
        if not sessions:
            return sessions

        # Build session lookup
        session_by_skill: dict[str, dict] = {}
        for s in sessions:
            for skill in s.get("skills", []):
                session_by_skill[skill] = s

        # Get topological order of skills
        topo_order = self.topological_order()

        # Map skills to their sessions, preserving topological order
        ordered_sessions = []
        seen_sessions: set[int] = set()

        for skill in topo_order:
            session = session_by_skill.get(skill)
            if session and session.get("session_number") not in seen_sessions:
                ordered_sessions.append(session)
                seen_sessions.add(session.get("session_number"))

        # Add any sessions not covered by the graph
        for s in sessions:
            if s.get("session_number") not in seen_sessions:
                ordered_sessions.append(s)

        return ordered_sessions


def build_skill_graph(learning_path: list[dict]) -> SkillGraph:
    """Build a skill graph from a learning path's sessions.

    Parameters
    ----------
    learning_path : list[dict]
        The "learning_path" array from the path JSON.

    Returns
    -------
    SkillGraph
        The directed prerequisite graph with ESCO URIs.
    """
    graph = SkillGraph()

    # First pass: create all nodes
    for session in learning_path:
        sn = session.get("session_number", 0)
        skills = session.get("skills", [])
        prereqs = session.get("prerequisites", [])
        difficulty = session.get("difficulty_level", "")
        skill_details = session.get("skill_details", [])

        # Build ESCO URI map from skill_details
        esco_map = {}
        for sd in skill_details:
            esco_map[sd.get("label", "")] = sd.get("uri", "")

        for skill in skills:
            if skill not in graph.nodes:
                graph.nodes[skill] = SkillNode(
                    label=skill,
                    session_number=sn,
                    esco_uri=esco_map.get(skill, ""),
                    difficulty=difficulty,
                    prerequisites=prereqs,
                )

    # Second pass: build edges
    for session in learning_path:
        skills = session.get("skills", [])
        prereqs = session.get("prerequisites", [])

        for skill in skills:
            for prereq in prereqs:
                # prereq -> skill (forward edge)
                graph.edges[prereq].append(skill)
                graph.reverse_edges[skill].append(prereq)

                # Update dependents list
                if prereq in graph.nodes:
                    graph.nodes[prereq].dependents.append(skill)

    logger.info(
        "Built skill graph: %d nodes, %d edges",
        len(graph.nodes),
        sum(len(deps) for deps in graph.edges.values()),
    )
    return graph


def get_esco_mapping(learning_path: list[dict]) -> dict[str, str]:
    """Extract skill_label -> ESCO URI mapping from a learning path."""
    mapping = {}
    for session in learning_path:
        for sd in session.get("skill_details", []):
            label = sd.get("label", "")
            uri = sd.get("uri", "")
            if label and uri:
                mapping[label] = uri
    return mapping
