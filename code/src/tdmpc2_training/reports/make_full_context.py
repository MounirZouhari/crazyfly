"""Combine every document, result table and key source file into one PDF.

From the repository root:

    ~/.local/bin/uv run --no-project --with reportlab python code/src/tdmpc2_training/reports/make_full_context.py

Writes FULL_CONTEXT.md and FULL_CONTEXT.pdf next to this file. Meant for
bringing another assistant fully up to date. Each document becomes one part;
its headings move down one level so the parts stay the chapters. Built with
make_guide.py.
"""
import datetime
import re
import textwrap
from pathlib import Path

from make_guide import build

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TD = "code/src/tdmpc2_training"

DOCUMENTS = [
    (f"{TD}/reports/STATUS_2026-11-09.md", "Status on November 9 (newest; read first)"),
    (f"{TD}/reports/GUIDE.md", "The complete project guide"),
    (f"{TD}/reports/RESULTS_SUMMARY.md", "Results summary: one answer per question of Prof. Bellinger"),
    (f"{TD}/reports/HANDOFF.md", "Handoff after the Batch 2 overnight run"),
    (f"{TD}/reports/NEXT_STEPS.md", "Next research plan (as of November 2)"),
    (f"{TD}/reports/SUPERVISOR_UPDATE.md", "Supervisor update (November 2)"),
    (f"{TD}/reports/BATCH2_EXPLAINED.md", "Batch 2 in plain language"),
    (f"{TD}/EXPERIMENTS.md", "Experiment log: task, tests, launches"),
    (f"{TD}/README.md", "Code README: commands"),
    (f"{TD}/reports/DEPLOYMENT_READINESS.md", "Deployment readiness"),
    (f"{TD}/reports/LITERATURE.md", "Literature notes"),
    (f"{TD}/reports/RESEARCH_LOG.md", "Research log"),
    (f"{TD}/CONTEXT.md", "Batch 1 context (historical, October 13)"),
]

RESULT_TABLES = [
    "outputs/batch2/random_start_1M/analysis/summary.md",
    "outputs/batch2/fixed_start_1M/analysis/summary.md",
    "outputs/batch2/random_start_small_1M/analysis/summary.md",
    "outputs/batch2/randomized_small_1M/analysis/summary.md",
    "outputs/batch2/sharp_small_1M/analysis/summary.md",
    "outputs/batch2/random_start_2M/analysis/summary.md",
    "outputs/batch2/snapshot_summary/snapshot_evaluation.md",
    "outputs/batch2/snapshot_summary_policy_only/snapshot_evaluation_policy_only.md",
    "outputs/batch2/variance_decomposition/summary.md",
    "outputs/batch2/fluctuation_analysis/summary.md",
    "outputs/batch2/model_error_summary/summary.md",
    "outputs/batch2/model_error_summary_batch1/summary.md",
    "outputs/batch2/planner_variants_summary/summary.md",
    "outputs/batch2/planner_variants_summary_batch1/summary.md",
    "outputs/batch2/horizon_summary/summary.md",
    "outputs/batch2/smoothing_summary/summary.md",
    "outputs/batch2/precision_summary/summary.md",
    "outputs/batch2/flight_quality/summary.md",
    "results/tdmpc2_batch1/README.md",
    "results/tdmpc2_batch1/comparison_batch1/comparison.md",
    "results/tdmpc2_batch1/comparison_with_1M/comparison.md",
]

SOURCE_FILES = [
    ("code/src/gpu_training/constants.py", "Lab constants: hover thrust, reward weights, ranges"),
    ("code/src/gpu_training/crazyflie_env.py", "Lab MJX environment: reset, step, reward, observation"),
    (f"{TD}/crazyflie_hover.py", "TD-MPC2 adapter: one drone, random starts, metric target"),
    (f"{TD}/crazyflie_hover_dr.py", "H2: randomised mass, motors and wind"),
    (f"{TD}/crazyflie_hover_sharp.py", "Reward pilot: extra reward near the target"),
    (f"{TD}/train_hover.py", "Training entry point"),
    (f"{TD}/tdmpc2_jax/config_batch2_random_start_small.yaml", "Config D"),
    (f"{TD}/tdmpc2_jax/config_batch2_random_start.yaml", "Config R1"),
    (f"{TD}/tdmpc2_jax/config_batch2_fixed_start.yaml", "Config C1"),
    (f"{TD}/tdmpc2_jax/config_batch2_randomized_small.yaml", "Config H2"),
    (f"{TD}/tdmpc2_jax/config_batch2_sharp_small.yaml", "Config reward pilot"),
    (f"{TD}/tdmpc2_jax/config_hover.yaml", "Config Batch 1 small"),
    (f"{TD}/tdmpc2_jax/config_hover_standard.yaml", "Config Batch 1 published"),
    (f"{TD}/tdmpc2_jax/config.yaml", "Upstream default config"),
    (f"{TD}/hover_eval.py", "Test flights, test starts and metrics"),
    (f"{TD}/eval_hover.py", "Testing a saved agent"),
    (f"{TD}/eval_disturbances.py", "Disturbance tests (wind, gusts, mass, motors, delay)"),
    (f"{TD}/tdmpc2_jax/build.py", "Building the agent"),
    (f"{TD}/tdmpc2_jax/train.py", "Training loop"),
    (f"{TD}/tdmpc2_jax/tdmpc2.py", "TD-MPC2 agent: planning (MPPI) and updates"),
    (f"{TD}/tdmpc2_jax/world_model.py", "TD-MPC2 world model"),
    (f"{TD}/reports/experiment_registry.json", "Experiment registry"),
]

CODE_WIDTH = 98


def modified(path):
    return datetime.date.fromtimestamp((REPO / path).stat().st_mtime).isoformat()


def wrap_code(line):
    if len(line) <= CODE_WIDTH:
        return [line]
    indent = len(line) - len(line.lstrip())
    return textwrap.wrap(line, CODE_WIDTH, subsequent_indent=" " * (indent + 4),
                         drop_whitespace=False, break_on_hyphens=False) or [line]


def demote(text):
    """One document's Markdown, with headings one level down and only the syntax make_guide reads."""
    out, in_code = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            out.append("```" if in_code else line)
            continue
        if in_code:
            out += wrap_code(line.replace("\t", "    "))
            continue
        if line.startswith("% ") or line.startswith("[[TOC]]"):
            continue
        heading = re.match(r"(#+) (.*)", line)
        if heading:
            out.append("#" * min(len(heading.group(1)) + 1, 3) + " " + heading.group(2))
            continue
        line = re.sub(r"^(\s*)\* ", r"\1- ", line)
        line = re.sub(r"^(\s*)(\d+\.) ", r"\1- \2 ", line)
        out.append(line)
    if in_code:
        out.append("```")
    return "\n".join(out)


def code_block(text):
    lines = []
    for line in text.replace("\t", "    ").splitlines():
        lines += wrap_code(line)
    return "```\n" + "\n".join(lines) + "\n```"


def main():
    today = datetime.date.today().isoformat()
    parts = [
        "% Crazyflie + TD-MPC2: complete project context",
        "% Every document, result table and key source file, in one place",
        "% Mounir Zouhari, University of Ottawa (supervisor Prof. Colin Bellinger)",
        f"% Compiled {today}",
        "",
        "[[TOC]]",
    ]
    number = 0
    for path, title in DOCUMENTS:
        number += 1
        parts.append(f"\n# Part {number}. {title}\n")
        parts.append(f"Source: `{path}`, last modified {modified(path)}.\n")
        parts.append(demote((REPO / path).read_text(encoding="utf-8")))

    number += 1
    parts.append(f"\n# Part {number}. Raw result tables\n")
    parts.append("Every analysis writes a `summary.md`; these are copied unchanged, in the order "
                 "R1, C1, D, H2, reward pilot, R1 to 2M, then the follow-up tests and Batch 1. "
                 "Plot file names refer to images next to each summary, not included here.\n")
    for path in RESULT_TABLES:
        if not (REPO / path).exists():
            continue
        parts.append(f"\n## {path}\n")
        parts.append(f"Last modified {modified(path)}.\n")
        body = demote((REPO / path).read_text(encoding="utf-8"))
        parts.append(re.sub(r"^##? ", "### ", body, flags=re.M))

    number += 1
    parts.append(f"\n# Part {number}. Key source code\n")
    parts.append("Unchanged copies of the files that define the task, the experiments and the "
                 "algorithm. Long lines are wrapped for printing.\n")
    for path, title in SOURCE_FILES:
        parts.append(f"\n## {path}\n")
        parts.append(f"{title}. Last modified {modified(path)}.\n")
        parts.append(code_block((REPO / path).read_text(encoding="utf-8")))

    target = HERE / "FULL_CONTEXT.md"
    target.write_text("\n".join(parts) + "\n", encoding="utf-8")
    build(target, target.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
