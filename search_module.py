"""Compare BFS, DFS, and A* search on a weighted graph."""

from __future__ import annotations

import heapq
from itertools import count
from typing import TypeAlias



SearchResult: TypeAlias = tuple[list[str] | None, float | None, int, list[str]]


def bfs(graph: dict[str, list[tuple[str, float]]], start: str, goal: str) -> SearchResult:
    """Find a path by visiting the shallowest discovered nodes first."""
    start_list: list[tuple[str, list[str], float]] = [(start, [start], 0)]
    visited = {start}
    expanded: list[str] = []

    while start_list:
        node, path, path_cost = start_list.pop(0)

        if node == goal:
            return path, path_cost, len(expanded), expanded

        expanded.append(node)

        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                start_list.append(
                    (neighbor, path + [neighbor], path_cost + edge_cost)
                )

    return None, None, len(expanded), expanded


def dfs(graph: dict[str, list[tuple[str, float]]], start: str, goal: str) -> SearchResult:
    """Find a path by visiting the most recently discovered nodes first."""
    start_list: list[tuple[str, list[str], float]] = [(start, [start], 0)]
    visited = {start}
    expanded: list[str] = []

    while start_list:
        node, path, path_cost = start_list.pop(0)
        if node == goal:
            return path, path_cost, len(expanded), expanded

        expanded.append(node)

        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                start_list.insert(
                    0, (neighbor, path + [neighbor], path_cost + edge_cost)
                )

    return None, None, len(expanded), expanded


def astar(
    graph: dict[str, list[tuple[str, float]]],
    heuristic: dict[str, float],
    start: str,
    goal: str,
) -> SearchResult:
    """Find a least-cost path using f(n) = g(n) + h(n)."""
    sequence = count()

    start_list: list[tuple[float, int, str, list[str], float]] = [
        (heuristic[start], next(sequence), start, [start], 0)
    ]
    best_cost: dict[str, float] = {start: 0}
    expanded_cost: dict[str, float] = {}
    expanded: list[str] = []

    while start_list:
        _, _, node, path, path_cost = heapq.heappop(start_list)

        if path_cost != best_cost.get(node):
            continue

        if path_cost >= expanded_cost.get(node, float("inf")):
            continue

        if node == goal:
            return path, path_cost, len(expanded), expanded

        expanded_cost[node] = path_cost
        expanded.append(node)

        for neighbor, edge_cost in graph.get(node, []):
            new_cost = path_cost + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost

                priority = new_cost + heuristic[neighbor]
                heapq.heappush(
                    start_list,
                    (
                        priority,
                        next(sequence),
                        neighbor,
                        path + [neighbor],
                        new_cost,
                    ),
                )

    return None, None, len(expanded), expanded


def print_result(name: str, result: SearchResult) -> None:
    """Print the path, path cost, and expansion order for one search."""
    path, total_cost, expanded_count, expanded_order = result
    print(f"{name}:")
    print(f"  Path: {' -> '.join(path) if path is not None else 'No path found'}")
    print(f"  Total cost: {total_cost if total_cost is not None else 'N/A'}")
    print(f"  Nodes expanded: {expanded_count}")
    print(f"  Expansion order: {', '.join(expanded_order) or '(none)'}")


if __name__ == "__main__":
    sample_graph = {
        "S": [("A", 3), ("B", 1)],
        "A": [("C", 3), ("D", 4)],
        "B": [("A", 1), ("D", 5)],
        "C": [("G", 2)],
        "D": [("G", 1)],
        "G": [],
    }

    sample_heuristic = {
        "S": 7,
        "A": 5,
        "B": 6,
        "C": 2,
        "D": 1,
        "G": 0,
    }

    print_result("BFS", bfs(sample_graph, "S", "G"))
    print()
    print_result("DFS", dfs(sample_graph, "S", "G"))
    print()
    print_result("A*", astar(sample_graph, sample_heuristic, "S", "G"))
