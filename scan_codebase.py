from pathlib import Path

print("=== 1. PYTHON MODULES IN SRC ===")
py_files = sorted(Path("src").rglob("*.py"))
for p in py_files:
    print(f"  {p}")

print("\n=== 2. TARGET ANCHORS & CONTEXT ===")
anchors = [
    "<UNRESOLVED_PORT>",
    "Code synthesis refused",
    "PRE-FLIGHT",
    "target_input",
    "Universal literal",
    "Estimator-shaped",
    "unbind",
    "PD_READ_CSV",
    "TRAIN_TEST_SPLIT",
]

for p in py_files:
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        continue

    matches = []
    for idx, line in enumerate(lines):
        for a in anchors:
            if a in line:
                matches.append((idx, a))
                break

    if matches:
        print(f"\n--- {p} ({len(matches)} match(es)) ---")
        printed_ranges = []
        for idx, anchor in matches:
            start = max(0, idx - 6)
            end = min(len(lines), idx + 7)
            if printed_ranges and start <= printed_ranges[-1][1]:
                printed_ranges[-1] = (printed_ranges[-1][0], max(end, printed_ranges[-1][1]))
            else:
                printed_ranges.append((start, end))

        for start, end in printed_ranges:
            print(f"  [Lines {start+1}-{end}]:")
            for i in range(start, end):
                marker = ">>>" if any(i == m[0] for m in matches) else "   "
                print(f"  {marker} {i+1:4d} | {lines[i]}")
