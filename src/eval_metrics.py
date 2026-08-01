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

    profile_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    emb_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    llm_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    task_stats = defaultdict(lambda: {"passed": 0, "total": 0})

    for r in results:
        passed = r['passed']
        profile = r['profile']
        emb = r['embedder']
        llm = r['llm']
        task = r['task_id']

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
            report_lines.append(f"### {f['task_id']} (Profile {f['profile']}, Emb: {f['embedder']}, LLM: {f['llm']})")
            report_lines.append(f"**Error**:\n```\n{f['error']}\n```")
            report_lines.append(f"**Generated Code**:\n```python\n{f['code']}\n```\n")

    report_content = "\n".join(report_lines)
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report_content)
        
    print("Metrics calculated. Report saved to evaluation_report.md")

if __name__ == "__main__":
    generate_report()
