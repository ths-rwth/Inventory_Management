"""Solver registry — keyed by category. Lazy-imported."""

from __future__ import annotations

import importlib

REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "bound": {
        "cp":            (".bound.cp",            "CPSolver"),
        "ilp_type":      (".bound.ilp_type",      "ILPType"),
        "ilp_type_nocard": (".bound.ilp_type_nocard", "ILPTypeNoCard"),
        "sat_type":      (".bound.sat_type",      "SatBitBlastingType"),
        "smt9":          (".bound.smt9",          "SMT9Solver"),
        "smt3":          (".bound.smt3",          "SMT3"),
        "smt3_max":      (".bound.smt3_max",      "SMT3maxSolver"),
        "smt4":          (".bound.smt4",          "SMT4"),
        "greedy":        (".bound.greedy",        "Greedy"),
        "genetic":       (".bound.genetic",       "GeneticSolver"),
    },
    "generator": {
        "genetic_endstate": (".generators.genetic_endstate", "GeneticEndStateGenerator"),
    },
    "unbound": {
        "smt6": (".unbound.smt6", "SMT6Router"),
        "greedy_reconstructor":   (".unbound.greedy_reconstructor",   "GreedyReconstructor"),
        "planning_reconstructor": (".unbound.planning_reconstructor", "PlanningReconstructor"),
        "smt5_swap":              (".unbound.smt5_swap",              "SMT5SwapRouter"),
    },
    "pipeline": {},  # empty — users compose at runtime.
}


def get_solver(category: str, name: str) -> type:
    if category not in REGISTRY:
        raise KeyError(f"unknown category {category!r}; known: {tuple(REGISTRY)}")
    bucket = REGISTRY[category]
    if name not in bucket:
        raise KeyError(
            f"unknown {category} solver {name!r}; known: {tuple(sorted(bucket))}"
        )
    module_name, class_name = bucket[name]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def list_solvers(category: str) -> tuple[str, ...]:
    if category not in REGISTRY:
        raise KeyError(f"unknown category {category!r}; known: {tuple(REGISTRY)}")
    return tuple(sorted(REGISTRY[category]))
