#!/usr/bin/env python3
"""
merge_tree_parts.py - Merge two or more partial domain-tree JSON files into
one complete tree, for cases where an LLM generation run hit its output
limit and had to continue in a follow-up response (e.g.
sklearn_v1.1.0_raw.json + sklearn_v1.1.0_raw#2.json).

Run this BEFORE normalize_tree.py, not after. A cell in part 1's `edges`
that points at a cell only defined in part 2 will look like a dangling
reference if you validate the parts separately — it isn't, once merged.

What this does:
  - Concatenates `cells` across all parts, de-duplicated by cell_id.
      - If the same cell_id appears in more than one part with IDENTICAL
        content, that's harmless (the LLM re-emitted it, e.g. included one
        cell from the end of cluster N again at the start of the
        continuation for context) — silently deduplicated, first
        occurrence kept.
      - If the same cell_id appears with DIFFERENT content across parts,
        that's flagged as a conflict — first occurrence is kept in the
        output, but you should look at the flagged ids by hand, since it
        usually means the continuation redefined something slightly
        differently than originally intended.
  - Merges `typestates.states`, de-duplicated by name, with the same
    identical-vs-conflicting distinction.
  - Merges `typestates.transitions`, de-duplicated by (from, to, via).
  - Merges `typestates.terminal_states`, de-duplicated, order-preserving.
  - Checks `domain`, `version`, and `initial_state` agree across parts and
    warns (does not fail) if they don't.

Usage:
    python merge_tree_parts.py part1.json part2.json [part3.json ...] \
        -o merged.json
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def state_name(s: Any) -> str:
    return s.get("name") if isinstance(s, dict) else s


def merge_trees(paths: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trees = [load(p) for p in paths]
    report: Dict[str, Any] = {
        "domain_mismatches": [],
        "version_mismatches": [],
        "initial_state_conflict": None,
        "duplicate_cell_ids_identical": [],
        "duplicate_cell_ids_conflicting": [],
        "duplicate_state_names_identical": [],
        "duplicate_state_names_conflicting": [],
        "total_cells_before_dedup": 0,
        "total_cells_after_dedup": 0,
        "total_transitions_before_dedup": 0,
        "total_transitions_after_dedup": 0,
    }

    base_domain = trees[0].get("domain")
    base_version = trees[0].get("version")
    for path, t in zip(paths[1:], trees[1:]):
        if t.get("domain") != base_domain:
            report["domain_mismatches"].append((path, base_domain, t.get("domain")))
        if t.get("version") != base_version:
            report["version_mismatches"].append((path, base_version, t.get("version")))

    # --- merge cells, de-duplicated by cell_id, first occurrence kept ---
    cells_by_id: "OrderedDict[str, dict]" = OrderedDict()
    for t in trees:
        for cell in t.get("cells", []):
            report["total_cells_before_dedup"] += 1
            cid = cell.get("cell_id")
            if cid in cells_by_id:
                if cells_by_id[cid] != cell:
                    report["duplicate_cell_ids_conflicting"].append(cid)
                else:
                    report["duplicate_cell_ids_identical"].append(cid)
                continue
            cells_by_id[cid] = cell
    report["total_cells_after_dedup"] = len(cells_by_id)

    # --- merge typestates.states, de-duplicated by name ---
    ts_list = [t.get("typestates", {}) for t in trees]
    states_by_name: "OrderedDict[str, Any]" = OrderedDict()
    for ts in ts_list:
        for s in ts.get("states", []):
            name = state_name(s)
            if name in states_by_name:
                if states_by_name[name] != s:
                    report["duplicate_state_names_conflicting"].append(name)
                else:
                    report["duplicate_state_names_identical"].append(name)
                continue
            states_by_name[name] = s

    # --- merge transitions, de-duplicated by (from, to, via) ---
    transitions: List[dict] = []
    seen_transitions = set()
    for ts in ts_list:
        for tr in ts.get("transitions", []):
            report["total_transitions_before_dedup"] += 1
            key = (tr.get("from"), tr.get("to"), tr.get("via"))
            if key in seen_transitions:
                continue
            seen_transitions.add(key)
            transitions.append(tr)
    report["total_transitions_after_dedup"] = len(transitions)

    # --- merge terminal_states, order-preserving de-dup ---
    terminal_states: List[str] = []
    seen_terminal = set()
    for ts in ts_list:
        for term in ts.get("terminal_states", []) or []:
            if term not in seen_terminal:
                seen_terminal.add(term)
                terminal_states.append(term)

    initial_states = {ts.get("initial_state") for ts in ts_list if ts.get("initial_state")}
    if len(initial_states) > 1:
        report["initial_state_conflict"] = sorted(initial_states)
        initial_state = sorted(initial_states)[0]
    else:
        initial_state = next(iter(initial_states)) if initial_states else None

    merged = {
        "domain": base_domain,
        "version": base_version,
        "typestates": {
            "domain": base_domain,
            "states": list(states_by_name.values()),
            "transitions": transitions,
            "initial_state": initial_state,
            "terminal_states": terminal_states,
        },
        "cells": list(cells_by_id.values()),
    }
    return merged, report


def print_report(report: Dict[str, Any]) -> None:
    print("=== Merge Report ===")
    if report["domain_mismatches"]:
        print(f"DOMAIN MISMATCHES: {report['domain_mismatches']}")
    if report["version_mismatches"]:
        print(f"Version differs across parts (informational): {report['version_mismatches']}")
    if report["initial_state_conflict"]:
        print(f"INITIAL STATE CONFLICT across parts: {report['initial_state_conflict']} "
              f"— kept the first alphabetically, check this by hand.")

    print(f"Cells: {report['total_cells_before_dedup']} before dedup -> "
          f"{report['total_cells_after_dedup']} after")
    print(f"  identical duplicates (harmless): {len(report['duplicate_cell_ids_identical'])}")
    print(f"  CONFLICTING duplicates (NEEDS A LOOK, first occurrence kept): "
          f"{len(report['duplicate_cell_ids_conflicting'])}")
    for cid in report["duplicate_cell_ids_conflicting"][:30]:
        print(f"    {cid}")

    print(f"Typestates: {len(report['duplicate_state_names_identical'])} identical duplicate "
          f"state names (harmless), {len(report['duplicate_state_names_conflicting'])} "
          f"CONFLICTING (NEEDS A LOOK, first occurrence kept)")
    for name in report["duplicate_state_names_conflicting"][:30]:
        print(f"    {name}")

    print(f"Transitions: {report['total_transitions_before_dedup']} before dedup -> "
          f"{report['total_transitions_after_dedup']} after")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="two or more partial tree JSON files, in generation order")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    if len(args.inputs) < 2:
        sys.exit("Provide at least two input files to merge (that's the point of this script).")

    merged, report = merge_trees(args.inputs)
    print_report(report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote merged tree to {args.output}")
    print("Now run normalize_tree.py on THIS merged file, not on the individual parts.")


if __name__ == "__main__":
    main()
