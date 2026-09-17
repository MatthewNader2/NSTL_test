# STATUS: partially wired / candidate for future removal. Do not expand until usage is confirmed.
import json
import os
from collections import defaultdict

def generate_report():
    if not os.path.exists('evaluation_results.json'):
        print("evaluation_results.json not found. Run eval_runner.py first.")
        return

    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)

    total_runs = len(results)
    if total_runs == 0:
        print("No results to process.")
        return

    total_passed = sum(1 for r in results if r['passed'])

    tier_counts = defaultdict(int)
    attrib_counts = defaultdict(int)
    profile_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    emb_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    llm_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    task_stats = defaultdict(lambda: {"passed": 0, "total": 0})

    for r in results:
        passed = r.get('passed', False)
        profile = r.get('profile', 'unknown')
        emb = r.get('embedder', 'unknown')
        llm = r.get('llm', 'unknown')
        task = r.get('task_id', 'unknown')
        tier = r.get('tier', 'semantically-validated' if passed else 'failed')
        attrib = r.get('pass_attribution', 'path' if passed else 'none')

        tier_counts[tier] += 1
        if passed or tier != 'failed':
            attrib_counts[attrib] += 1

        profile_stats[profile]['total'] += 1
        emb_stats[emb]['total'] += 1
        llm_stats[llm]['total'] += 1
        task_stats[task]['total'] += 1

        if passed:
            profile_stats[profile]['passed'] += 1
            emb_stats[emb]['passed'] += 1
            llm_stats[llm]['passed'] += 1
            task_stats[task]['passed'] += 1

    def format_rate(stats):
        return f"{stats['passed']}/{stats['total']} ({(stats['passed']/stats['total'])*100:.1f}%)"

    report_lines = []
    report_lines.append("# NSTL Evaluation System Report\n")
    report_lines.append(f"**Total Runs**: {total_runs}")
    report_lines.append(f"**Overall Success Rate**: {total_passed}/{total_runs} ({(total_passed/total_runs)*100:.1f}%)\n")

    report_lines.append("## Tiered Verdicts")
    report_lines.append("| Tier | Count | Percentage |")
    report_lines.append("|---|---|---|")
    for t_name in ("runs", "type-safe", "semantically-validated", "failed"):
        cnt = tier_counts.get(t_name, 0)
        report_lines.append(f"| {t_name} | {cnt} | {(cnt/total_runs)*100:.1f}% |")
    report_lines.append("\n")

    report_lines.append("## Pass Attribution (Structural)")
    report_lines.append("| Attribution | Count | Description |")
    report_lines.append("|---|---|---|")
    report_lines.append(f"| path | {attrib_counts.get('path', 0)} | Exact planner+binder emitted code passed |")
    report_lines.append(f"| repair | {attrib_counts.get('repair', 0)} | LLM self-repair repaired and passed code |")
    report_lines.append(f"| fallback | {attrib_counts.get('fallback', 0)} | Passed via fallback mechanism |")
    report_lines.append("\n")

    report_lines.append("## Success Rate by Profile")
    report_lines.append("| Profile | Success Rate |")
    report_lines.append("|---|---|")
    for k, v in sorted(profile_stats.items()):
        report_lines.append(f"| {k} | {format_rate(v)} |")
    report_lines.append("\n")

    report_lines.append("## Success Rate by Embedder")
    report_lines.append("| Embedder | Success Rate |")
    report_lines.append("|---|---|")
    for k, v in sorted(emb_stats.items()):
        report_lines.append(f"| {k} | {format_rate(v)} |")
    report_lines.append("\n")

    report_lines.append("## Success Rate by LLM (Profiles C/D)")
    report_lines.append("| LLM | Success Rate |")
    report_lines.append("|---|---|")
    for k, v in sorted(llm_stats.items()):
        if k != "auto":
            report_lines.append(f"| {k} | {format_rate(v)} |")
    report_lines.append("\n")
    
    report_lines.append("## Success Rate by Task")
    report_lines.append("| Task ID | Success Rate |")
    report_lines.append("|---|---|")
    for k, v in sorted(task_stats.items()):
        report_lines.append(f"| {k} | {format_rate(v)} |")
    report_lines.append("\n")
    
    report_lines.append("## Detailed Failures")
    failures = [r for r in results if not r['passed']]
    if not failures:
        report_lines.append("No failures! Great job.")
    else:
        for f in failures:
            report_lines.append(f"### {f['task_id']} (Profile {f['profile']}, Emb: {f['embedder']}, LLM: {f['llm']}, Tier: {f.get('tier', 'failed')})")
            report_lines.append(f"**Error**:\n```\n{f['error']}\n```")
            report_lines.append(f"**Generated Code**:\n```python\n{f['code']}\n```\n")
            if f.get('repaired_code'):
                report_lines.append(f"**Repaired Code**:\n```python\n{f['repaired_code']}\n```\n")

    report_content = "\n".join(report_lines)
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report_content)
        
    print("Metrics calculated. Report saved to evaluation_report.md")

if __name__ == "__main__":
    generate_report()
