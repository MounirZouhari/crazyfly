"""Build the Batch 1 PDF report (TD-MPC2 vs PPO, Crazyflie hover) from the result files."""
import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = Path.home() / "Projects/multi-agentCrazyflieLift/outputs"
DEST = OUT / "report_batch1"
DEST.mkdir(exist_ok=True)
PDF = DEST / "Crazyflie_TD-MPC2_vs_PPO_Batch1_Report.pdf"
TARGET = np.array([0.0, 0.0, 1.0])

# --------------------------------------------------------------------------- data

def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def band(row):
    if row["crashed"] == "True":
        return "crashed"
    e = float(row["late_error_m"])
    return "hover" if e < 0.10 else "close" if e < 0.30 else "drift" if e < 1.0 else "far"


BANDS = ["hover", "close", "drift", "far", "crashed"]
BAND_LABEL = {"hover": "Hover (< 10 cm)", "close": "Close (10–30 cm)", "drift": "Drifting (30–100 cm)",
              "far": "Far (> 1 m, still flying)", "crashed": "Crashed"}
BAND_COLOR = {"hover": "#1a7f4b", "close": "#7cc68d", "drift": "#f2c14e", "far": "#f08a4b", "crashed": "#c0392b"}


class Batch:
    def __init__(self, folder, label, single_dir=False):
        self.label = label
        dirs = [OUT / folder] if single_dir else sorted((OUT / folder).glob("seed_*"))
        self.seeds = []
        for d in dirs:
            summ = read(d / "evaluation/summary.csv")
            curve = read(d / "learning_curve.csv") if (d / "learning_curve.csv").exists() else []
            traj = read(d / "evaluation/trajectories.csv")
            self.seeds.append({"name": d.name, "summ": summ, "curve": curve, "traj": traj})
        self.starts = list(dict.fromkeys(r["start"] for r in self.seeds[0]["summ"]))

    def per_seed(self, start, col):
        return np.array([np.mean([float(r[col]) for r in s["summ"] if r["start"] == start]) for s in self.seeds])

    def all_rows(self, start=None):
        return [r for s in self.seeds for r in s["summ"] if start is None or r["start"] == start]

    def crashes(self, start=None):
        rows = self.all_rows(start)
        return sum(r["crashed"] == "True" for r in rows), len(rows)

    def bands(self):
        c = collections.Counter(band(r) for r in self.all_rows())
        return {b: c.get(b, 0) for b in BANDS}

    def train_minutes(self):
        return np.array([float(s["curve"][-1]["train_seconds"]) / 60 for s in self.seeds if s["curve"]])

    def crash_kinds(self):
        fell = out = 0
        for s in self.seeds:
            last = {}
            for r in s["traj"]:
                last[r["flight"]] = r
            for r in s["summ"]:
                if r["crashed"] == "True":
                    if float(last[r["flight"]]["z_m"]) < 0.06:
                        fell += 1
                    else:
                        out += 1
        return fell, out


def mse(values):
    v = np.asarray(values, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), se


def fm(values, d=2, unit=" m"):
    m, se = mse(values)
    return f"{m:.{d}f} ± {se:.{d}f}{unit}"


SEPT = Batch("crazyflie_hover_s0/2026-09-28_11-42-30", "September run (1 seed)", single_dir=True)
HOME = Batch("batch1_fixed_start", "TD-MPC2 small, home PC")
NARV = Batch("batch1_small_narval", "TD-MPC2 small, Narval")
STD = Batch("batch1_standard", "TD-MPC2 standard (big)")
LONG = Batch("batch1_standard_1M", "TD-MPC2 standard, 1M steps")
PPO = Batch("ppo_fixed_start", "PPO, same task")
LAB = Batch("ppo_existing_checkpoint", "Lab's existing PPO", single_dir=True)

# --------------------------------------------------------------------------- new figures

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig_task(path):
    grey, blue, red = "#9aa5b1", "#1f6f8b", "#d1495b"
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))

    # Side view (x horizontal, height vertical). plus_y / minus_y are hidden behind home here.
    a.axhline(0, color="#8d6e63", lw=3)
    a.text(0.55, 0.03, "floor", fontsize=8, color="#8d6e63")
    for x, z in [(0.5, 0.2), (-0.5, 0.2), (0, 1.5)]:
        a.scatter(x, z, s=70, color=grey, edgecolor="white", zorder=3)
    a.scatter(0, 0.2, s=130, color=blue, edgecolor="white", zorder=3)
    a.scatter(0, 1.0, s=280, marker="*", color=red, zorder=4)
    a.annotate("", xy=(0, 0.93), xytext=(0, 0.27), arrowprops=dict(arrowstyle="->", color=blue, lw=2))
    a.text(0.5, 0.29, "plus_x", ha="center", fontsize=8)
    a.text(-0.5, 0.29, "minus_x", ha="center", fontsize=8)
    a.text(0.07, 1.5, "above (0, 0, 1.5)", va="center", fontsize=8)
    a.text(0.08, 1.0, "TARGET (0, 0, 1)", va="center", fontsize=9, color=red, fontweight="bold")
    a.text(0.07, 0.6, "every training flight\ngoes this way", fontsize=8, color=blue, va="center")
    a.text(0, 0.06, "home (0, 0, 0.2)", ha="center", fontsize=8, color=blue)
    a.set_xlim(-0.9, 0.9); a.set_ylim(-0.05, 1.7)
    a.set_title("Side view"); a.set_xlabel("x (m)"); a.set_ylabel("height (m)")

    # Top view (x, y). home, above and the target are all above the same floor point.
    for x, y, name in [(0.5, 0, "plus_x"), (-0.5, 0, "minus_x"), (0, 0.5, "plus_y"), (0, -0.5, "minus_y")]:
        b.scatter(x, y, s=70, color=grey, edgecolor="white", zorder=3)
        b.text(x, y + 0.09, name, ha="center", fontsize=8)
    b.scatter(0, 0, s=280, marker="*", color=red, zorder=4)
    b.annotate("home, above and the TARGET\nare all over this point\n(heights 0.2, 1.5 and 1 m)",
               xy=(0.03, -0.03), xytext=(0.47, -0.72), ha="center", va="center", fontsize=7.5, color=red,
               arrowprops=dict(arrowstyle="->", color=red, lw=1))
    b.set_xlim(-0.9, 0.9); b.set_ylim(-0.9, 0.9)
    b.set_title("Top view (looking down)"); b.set_xlabel("x (m)"); b.set_ylabel("y (m)")
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_quality(path):
    batches = [SEPT, HOME, NARV, STD, LONG, PPO, LAB]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for k, bt in enumerate(batches[::-1]):
        counts = bt.bands()
        total = sum(counts.values())
        left = 0
        for bnd in BANDS:
            frac = counts[bnd] / total * 100
            ax.barh(k, frac, left=left, color=BAND_COLOR[bnd], edgecolor="white",
                    label=BAND_LABEL[bnd] if k == 0 else None)
            if frac >= 7:
                ax.text(left + frac / 2, k, f"{frac:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if bnd in ("hover", "crashed") else "black")
            left += frac
        ax.text(101, k, f"{total} flights", va="center", fontsize=8, color="#555")
    ax.set_yticks(range(len(batches)), [b.label for b in batches[::-1]])
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of all test flights (6 starts × 3 flights per seed)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)
    ax.set_title("How good was each test flight?")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_seed_dots(path):
    batches = [HOME, NARV, STD, LONG, PPO]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    rng = np.random.default_rng(1)
    for k, bt in enumerate(batches):
        v = bt.per_seed("home", "late_error_m")
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=40, color="#1f6f8b", alpha=0.8, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color="#d1495b", lw=3, zorder=4)
    sept = SEPT.per_seed("home", "late_error_m")[0]
    ax.scatter(-0.45, sept, marker="D", s=50, color="#555", zorder=4)
    ax.annotate("September\nrun", (-0.45, sept), xytext=(0, -26), textcoords="offset points",
                fontsize=8, color="#555", ha="center")
    ax.set_xlim(-0.75, len(batches) - 0.5)
    ax.axhline(0.10, color="black", ls=":", lw=1)
    ax.text(len(batches) - 0.55, 0.105, "10 cm", fontsize=8)
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3], ["1 cm", "3 cm", "10 cm", "30 cm", "1 m", "3 m"])
    ax.set_xticks(range(len(batches)), [b.label.replace(", ", "\n") for b in batches], fontsize=8.5)
    ax.set_ylabel("late-flight distance to target\n(training start, log scale)")
    ax.set_title("Every training run as one dot (red bar = median run)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


fig_task(DEST / "fig_task.png")
fig_quality(DEST / "fig_quality.png")
fig_seed_dots(DEST / "fig_seed_dots.png")

# --------------------------------------------------------------------------- styles

pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
# No italic DejaVu file on this system: italics fall back to the regular face.
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

INK = colors.HexColor("#1d2733")
ACCENT = colors.HexColor("#1f6f8b")
body = ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor("#555555"))
h1 = ParagraphStyle("h1", parent=body, fontName="DVB", fontSize=16, leading=20, spaceBefore=4, spaceAfter=8,
                    textColor=ACCENT)
h2 = ParagraphStyle("h2", parent=body, fontName="DVB", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=body, fontName="DVB", fontSize=24, leading=30, alignment=TA_CENTER,
                       textColor=ACCENT)
subtitle = ParagraphStyle("subtitle", parent=body, fontSize=12, leading=17, alignment=TA_CENTER)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=10.6, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DVB")

W = letter[0] - 1.5 * inch
story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def B(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 4))


def box(title_text, lines, tint="#eaf3f6", edge="#1f6f8b"):
    content = [Paragraph(f"<b>{title_text}</b>", body)] + [Paragraph(l, body) for l in lines]
    t = Table([[content]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(tint)),
                           ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(t)
    story.append(Spacer(1, 8))


def table(rows, widths, bold_last=False, align_right_from=1):
    data = [[Paragraph(str(c), cellb if i == 0 or (bold_last and i == len(rows) - 1) else cell) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8ee")),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if bold_last:
        style.append(("LINEABOVE", (0, -1), (-1, -1), 0.8, ACCENT))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))


def fig(path, caption, width=1.0):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = W * width
    img.drawHeight = W * width * ratio
    story.append(KeepTogether([img, Paragraph(caption, small)]))
    story.append(Spacer(1, 8))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(0.75 * inch, 0.5 * inch, "Crazyflie hover: TD-MPC2 vs PPO — Batch 1 report")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {doc.page}")
    canvas.restoreState()

# --------------------------------------------------------------------------- numbers used in the text

home_med = np.median(HOME.per_seed("home", "late_error_m"))
narv_med = np.median(NARV.per_seed("home", "late_error_m"))
pooled = np.concatenate([HOME.per_seed("home", "late_error_m"), NARV.per_seed("home", "late_error_m")])
ppo_home = PPO.per_seed("home", "late_error_m")
c_small = HOME.crashes()[0] + NARV.crashes()[0]
n_small = HOME.crashes()[1] + NARV.crashes()[1]
sept_home = SEPT.per_seed("home", "late_error_m")[0]

# ============================================================================ COVER
story.append(Spacer(1, 1.6 * inch))
P("Teaching a simulated drone to hover", title)
P("TD-MPC2 vs PPO on the Crazyflie hover task<br/>Batch 1 results: 11 seeds per method", subtitle)
story.append(Spacer(1, 0.3 * inch))
P("Mounir Zouhari · Supervisor: Prof. Colin Bellinger · University of Ottawa<br/>"
  "Report date: October 13, 2026 (updated with the 1-million-step results)",
  ParagraphStyle("c", parent=small, alignment=TA_CENTER, fontSize=9.5, leading=14))
story.append(Spacer(1, 0.5 * inch))
box("What this report is", [
    "A plain-language summary of every experiment run so far: the original September run, 11 training runs "
    "of TD-MPC2 on the home PC, 11 more on the Digital Research Alliance cluster (Narval), 11 runs of a "
    "bigger TD-MPC2, 3 runs of that bigger TD-MPC2 trained 20× longer, 11 runs of PPO on the same task, "
    "and a test of the lab's existing PPO model.",
    "No reinforcement-learning background is assumed. Section 2 explains every idea needed to read the "
    "results, and a glossary is at the end.",
])
story.append(PageBreak())

# ============================================================================ 1. SUMMARY
P("1. The short version", h1)
P("One drone, one job: take off from the floor and hover at a fixed point 1 metre up, for 10 seconds. "
  "We compared two learning methods, <b>TD-MPC2</b> and <b>PPO</b>, each trained 11 times "
  "(11 \"seeds\") so that luck can be told apart from real differences.")
long_home = LONG.per_seed("home", "late_error_m")
box("Main findings", [
    f"<b>1. TD-MPC2 learns to fly from very little practice.</b> After 50,000 steps (about 17 minutes of "
    f"simulated flying) it reliably takes off and holds the right height, but typically ends about "
    f"<b>{min(home_med, narv_med)*100:.0f}–{max(home_med, narv_med)*100:.0f} cm</b> to the side of the "
    f"target (median of each 11-seed batch), slowly drifting.",
    f"<b>2. With more practice the drift disappears.</b> The bigger TD-MPC2 trained for 1 million steps hovers "
    f"<b>{long_home.mean()*100:.1f} cm</b> from the target on average, within 10 cm for about 90% of every "
    f"flight, with no crashes from its training start (3 of 3 runs). All three reached this level after about "
    f"100,000–175,000 steps. PPO needed about <b>10–20 million</b> steps for similar precision: roughly "
    f"<b>100× more practice</b>.",
    f"<b>3. PPO is faster on the clock.</b> Because it simulates 2,048 drones at once, PPO reaches a "
    f"near-perfect hover in about 4 minutes on the same GPU; TD-MPC2 (one drone) needed about an hour to "
    f"reach a similar hover. PPO's typical (median) run "
    f"ends <b>{np.median(ppo_home)*100:.1f} cm</b> from the target, but 4 of 11 PPO runs failed or drifted.",
    f"<b>4. Trained long enough on one route, both methods memorise it.</b> From the 4 sideways starting "
    f"points they never practised, PPO and the long-trained TD-MPC2 crash in almost every flight. The "
    f"briefly trained TD-MPC2 was rougher but more robust there ({c_small} crashes in {n_small} flights), "
    f"probably because it had not specialised yet. The lab's PPO, trained on random starts, works from every "
    f"start: practising from many positions is what makes a controller robust (Batch 2).",
    "<b>5. Results vary a lot between seeds.</b> The same method, with the same settings, gave anything from "
    "a real 8 cm hover to a 2.3 m failure after 50,000 steps. A single run (like the September one) cannot "
    "show this.",
    "<b>6. The bigger, \"official\" TD-MPC2 settings are slow starters.</b> At 50,000 steps they were worse "
    "than the small settings (about 1 m off, mostly crashing); with more practice they became the best "
    "TD-MPC2 result.",
])
P("In one sentence: <i>TD-MPC2 needs about 100× less practice than PPO to hover precisely, PPO finishes "
  "sooner on the clock thanks to massive parallel simulation, and both only learn the route they practise.</i>")
story.append(PageBreak())

# ============================================================================ 2. BACKGROUND
P("2. Background in plain language", h1)
P("2.1 The task", h2)
P("A simulated <b>Crazyflie</b>, a 27-gram palm-sized drone about 9 cm across, starts 20 cm above the floor at "
  "position (0, 0, 0.2) metres. Its goal is the point (0, 0, 1), one metre up. Every 0.02 seconds "
  "(50 times per second) the controller chooses 4 numbers: <b>thrust</b> (how hard to push up) and three "
  "small twisting forces (<b>roll, pitch, yaw</b>) that tilt or turn the drone. A flight lasts at most 500 "
  "decisions, which is 10 seconds of simulated time. The physics is computed by MuJoCo/MJX, a standard "
  "robotics simulator.")
fig(DEST / "fig_task.png", "Figure 1. The task. During training every flight starts at the blue point and "
    "aims for the red star. After training, each drone is also tested from the 5 grey points it never "
    "practised from.")
P("A flight ends early (a <b>crash</b>) if the drone drops below 5 cm, or drifts more than 2 m further "
  "from the target than where it started (\"out of bounds\").")

P("2.2 What reinforcement learning (RL) is", h2)
P("RL is learning by trial and error, like learning to ride a bike. Nobody tells the drone which motor "
  "commands are correct. Instead it tries things, and after every 0.02 s step it receives a score called the "
  "<b>reward</b>. Over thousands of attempts it learns which behaviour collects the most reward. "
  "In this task the reward per step is:")
B(["<b>+0.001</b> for still flying (survival bonus),",
   "<b>up to +0.01</b> for being close to the target (largest when exactly on it),",
   "<b>+ how much closer</b> it got to the target during that step (negative if it moved away),",
   "<b>−1</b> once, if it crashes."])
P("The total over a flight is called the <b>return</b>. A good 10-second hover scores about +5 to +6.")

story.append(CondPageBreak(3 * inch))   # keep the heading with its table
P("2.3 The two learning methods", h2)
table([["", "TD-MPC2", "PPO"],
       ["Idea", "Learns a <b>model</b> of how the drone moves, then <b>imagines</b> many possible "
        "short action plans before every decision and picks the best one (\"model-based planning\").",
        "Learns a direct reflex: \"in this situation, do this\" (a <b>policy</b>), improved a little "
        "after every batch of experience. No imagination or planning."],
       ["Practice style", "1 drone, learns after every step, re-uses old experience many times.",
        "2,048 drones flying at once on the GPU; each experience is used for a few updates and then "
        "thrown away."],
       ["Strength", "Learns a lot from little experience (\"sample-efficient\").",
        "Each step is extremely cheap, so it can collect huge amounts of experience quickly."],
       ["Code used", "JAX version of TD-MPC2 by S. Flandermeyer (faithful to the 2024 paper by Hansen et al.).",
        "The lab's existing Brax PPO code and settings (gpu_training/ppo_training.py)."]],
      [0.16, 0.42, 0.42])

P("2.4 Seeds: why every experiment was run 11 times", h2)
P("Training involves a lot of randomness: the starting values of the neural networks, the random "
  "exploration at the beginning, sensor noise. The <b>seed</b> is a single number that fixes all of that "
  "randomness. Changing the seed gives a new, independent training run with exactly the same settings. "
  "Like training 11 students with the same course: if only one is tested, you cannot tell whether the course "
  "is good or the student was lucky. Professor Bellinger asked for 11 seeds per experiment, which is standard "
  "practice in RL research.")
P("2.5 \"Small\" and \"standard\" TD-MPC2", h2)
P("The same TD-MPC2 algorithm can be run with a small or a large brain and a short or long thinking time. "
  "The September run used reduced settings to train quickly; the \"standard\" settings are the ones published "
  "with TD-MPC2.")
table([["Setting", "Small (September, config_hover.yaml)", "Standard (published defaults)"],
       ["Brain size (internal summary / layer width)", "128 / 128 (about 0.4 M numbers)", "512 / 512 (about 5 M numbers)"],
       ["Plans imagined per decision", "128 plans × 3 refinement rounds", "512 plans × 6 rounds"],
       ["How far ahead it imagines", "3 steps (0.06 s)", "3 steps (0.06 s)"],
       ["Experience used per learning step", "64 samples", "256 samples"],
       ["Random flying before learning starts", "500 steps", "2,500 steps"]],
      [0.36, 0.32, 0.32])
story.append(PageBreak())

# ============================================================================ 3. HOW WE MEASURED
P("3. How the drones were tested", h1)
P("After training, each drone flies <b>test flights</b>. It no longer learns during these; we only watch. "
  "Every trained drone flew 3 test flights from each of the 6 starting points in Figure 1, always aiming "
  "for (0, 0, 1). That is 18 test flights per training run, or 198 per group of 11 seeds. All methods were "
  "tested in exactly the same way, with the same noise.")
table([["Measurement", "What it means", "Good value"],
       ["<b>Late-flight error</b>", "Average distance from the target during the last 2 seconds of the "
        "flight. The main score. 0.30 m = the drone ends up hovering 30 cm away from the target.", "below 0.10 m"],
       ["Horizontal / vertical error", "The same distance split into sideways and height parts.", "small"],
       ["Time within 10 cm", "Share of the flight (after the first 2 s of take-off) spent within 10 cm.", "near 100%"],
       ["Crash", "Fell below 5 cm, or flew out of bounds, before the 10 seconds were over.", "no crashes"],
       ["Return", "Total reward collected in a flight (see 2.2).", "about +5 to +6"],
       ["Learning curve", "Every 5,000 training steps (PPO: about every 4 M) the drone does 3 test "
        "flights, to show how it improves during training.", "going down (error) / up (return)"]],
      [0.22, 0.6, 0.18])
P("<b>How to read \"0.32 ± 0.09 m\".</b> The first number is the <b>average over the 11 seeds</b>. The "
  "second is the <b>standard error</b>: roughly how much that average could move if we repeated the whole "
  "11-seed experiment. Two results whose ± ranges overlap are not clearly different.")
P("<b>Median run.</b> Line up the 11 results from best to worst and take the 6th. Unlike the average, "
  "one disastrous run cannot drag it far, so it is the best single answer to \"what does a typical run "
  "achieve?\". In the plots, the median run is drawn as a thick red line and all others as thin grey lines.")
P("<b>Environment steps vs. wall-clock time.</b> An environment step is one 0.02 s decision of one drone. "
  "TD-MPC2 trained for 50,000 steps; PPO, with 2,048 drones at once, for about 79 million. Wall-clock time is "
  "the real minutes the computer needed. Comparing both is important: one method can need fewer steps "
  "but more minutes.")
story.append(PageBreak())

# ============================================================================ 4. EXPERIMENTS
P("4. All experiments at a glance", h1)
table([["#", "Experiment", "Where", "Runs", "Training", "Purpose"],
       ["A", "September run (TD-MPC2 small)", "home PC", "1", "50k steps", "The original result, re-tested"],
       ["B", "TD-MPC2 small", "home PC (RTX 3060)", "11", "50k steps", "How much do seeds vary?"],
       ["C", "TD-MPC2 small, repeated", "Narval cluster", "11", "50k steps", "Same as B, on the same hardware as D and E"],
       ["D", "TD-MPC2 standard (big)", "Narval cluster", "11", "50k steps", "Does a bigger brain help?"],
       ["E", "PPO on the same task", "Narval cluster", "11", "about 79M steps", "The fair PPO comparison"],
       ["F", "Lab's existing PPO model", "(already trained)", "1", "60M steps, random tasks", "Reference: how good can it get?"],
       ["G", "TD-MPC2 standard, long", "Narval cluster", "3", "1M steps", "Does more practice fix the drift?"]],
      [0.04, 0.26, 0.17, 0.08, 0.17, 0.28])
P("Narval runs used one 2/7 slice of an NVIDIA A100 GPU each. All methods were trained on exactly the same "
  "task: same start, target, reward, sensors, 10-second flights, and the same meaning of the 4 controls. "
  "The only exception is F, which the lab trained earlier on a harder version (random start and target "
  "every flight, slightly randomised drone physics, 30-second flights).")
story.append(PageBreak())

# ============================================================================ 5. RESULTS
P("5. Results", h1)
P("5.1 The September run, re-tested (experiment A)", h2)
sept_rows = [["Start", "Late error", "Crashes"]]
for st in SEPT.starts:
    v = SEPT.per_seed(st, "late_error_m")[0]
    c, n = SEPT.crashes(st)
    sept_rows.append([st, f"{v:.2f} m", f"{c}/{n}"])
other = [SEPT.per_seed(st, "late_error_m")[0] for st in SEPT.starts if st != "home"]
P(f"The saved September model was loaded in a fresh program and flown again. It reproduced its original "
  f"result ({sept_home:.2f} m from the training start), which confirms the saved model and the testing code "
  f"work. From the 5 new starts it never crashed, but stayed {min(other):.2f}–{max(other):.2f} m away from "
  f"the target.")
table(sept_rows, [0.4, 0.3, 0.3])

P("5.2 TD-MPC2 small, 11 seeds on the home PC (experiment B)", h2)
rows = [["Seed", "Late error", "Sideways error", "Height error", "Time within 10 cm", "Crashes (of 18 flights)"]]
for s, e, xy, z, t in zip(HOME.seeds, HOME.per_seed("home", "late_error_m"),
                          HOME.per_seed("home", "late_error_xy_m"), HOME.per_seed("home", "late_error_z_m"),
                          HOME.per_seed("home", "time_in_10cm")):
    c = sum(r["crashed"] == "True" for r in s["summ"])
    rows.append([s["name"][-2:].lstrip("0") or "0", f"{e:.2f} m", f"{xy:.2f} m", f"{z:.2f} m", f"{t:.0%}", f"{c}"])
rows.append(["All (mean ± SE)", fm(HOME.per_seed("home", "late_error_m")),
             fm(HOME.per_seed("home", "late_error_xy_m")), fm(HOME.per_seed("home", "late_error_z_m")),
             fm(HOME.per_seed("home", "time_in_10cm") * 100, 0, " %"), f"{HOME.crashes()[0]}/{HOME.crashes()[1]}"])
table(rows, [0.16, 0.16, 0.16, 0.14, 0.18, 0.2], bold_last=True)
P(f"<b>What it means.</b> The typical (median) run ended <b>{home_med:.2f} m</b> from the target, close "
  f"to the September run ({sept_home:.2f} m), so September was a typical result. But runs ranged from a real "
  f"hover (seed 6, 8 cm, within 10 cm for half the flight) to a failure (seed 0, 2.3 m). Height was learned "
  f"well; almost all of the error is sideways.")
box("Why did seed 0 fail?", [
    "Seed 0 learned to fly well: at 45,000 steps its test flights were 0.27 m from the target with no "
    "crashes, better than September. In the last 5,000 steps it suddenly got worse and its final test flights "
    "drifted out of bounds. This was checked with two independent testing methods (the new one and the "
    "original September one), which agree. This kind of late \"collapse\" is a known instability of RL "
    "training. It is a real result, not a bug, and it is kept in the averages.",
    "Seed 0 of September and seed 0 today ended differently even with the same seed number: GPU arithmetic "
    "is not bit-for-bit repeatable, and over 50,000 learning steps tiny differences snowball.",
])
fig(OUT / "batch1_fixed_start/analysis/learning_curves.png",
    "Figure 2. Experiment B during training. Left: average of 11 seeds with its standard-error band. Right: "
    "every seed (grey) and the median seed (red). Top: distance to target in test flights (lower is better). "
    "Bottom: return (higher is better). The return was still rising at 50,000 steps, so training was not "
    "finished.")
fig(OUT / "batch1_fixed_start/analysis/home_flights.png",
    "Figure 3. Experiment B: the final drones flying from the training start. All reach 1 m height within "
    "about 1 second (bottom). Then the distance to the target slowly grows (top): the sideways drift, about "
    "3 cm per second. Each line stops when one of its flights crashes.")

P("5.3 The same experiment repeated on the cluster (experiment C)", h2)
P(f"Experiment B was repeated on Narval with identical code and settings. Result: <b>{fm(NARV.per_seed('home', 'late_error_m'))}</b> "
  f"(median {narv_med:.2f} m, {NARV.crashes()[0]} crashes in {NARV.crashes()[1]} flights), compared with "
  f"{fm(HOME.per_seed('home', 'late_error_m'))} on the home PC. The ± ranges overlap, so the two batches are "
  f"consistent: the same method with different luck. Together they are <b>22 independent runs</b> with a "
  f"median of {np.median(pooled):.2f} m. Experiment C is used in the comparisons below because it ran on "
  f"the same hardware as D and E, which makes training times comparable.")

P("5.4 TD-MPC2 with standard (big) settings (experiment D)", h2)
P(f"The bigger version did clearly worse after 50,000 steps: <b>{fm(STD.per_seed('home', 'late_error_m'))}</b> "
  f"from the training start and {STD.crashes()[0]} crashes in {STD.crashes()[1]} flights. It was still "
  f"improving: its return rose steadily during training (Figure 4, bottom). The likely reason is that larger "
  f"networks need more data before they pay off. Note that the standard settings change about 10 things at "
  f"once (brain size, number of plans, batch size, and so on), so we cannot yet say which one matters.")
box("Careful: why the distance curve goes up while the drone improves", [
    "Early in training the drones crash right after take-off, close to the start, about 0.8 m from the "
    "target. Later they fly much longer before failing and drift far away first. Because the late-flight "
    "error is measured over the last 2 seconds before the flight ends, it gets larger even though the drone "
    "is improving. For runs that crash a lot, look at the return and the crash count instead.",
], tint="#fff6e6", edge="#edae49")
fig(OUT / "batch1_standard/analysis/learning_curves.png",
    "Figure 4. Experiment D during training. The return (bottom) keeps rising, but after 50,000 steps most "
    "flights still end in a crash.")

P("5.5 PPO on exactly the same task (experiment E)", h2)
P(f"PPO used the lab's own settings and code; only the task was changed to match TD-MPC2. From the "
  f"training start it was the most precise: median <b>{np.median(ppo_home)*100:.1f} cm</b>, and 7 of 11 seeds "
  f"stayed within 10 cm for the whole flight. But 4 of 11 seeds failed or drifted from the training start, "
  f"and from the new starting points PPO crashed in most flights (see 5.7).")
fig(OUT / "ppo_fixed_start/analysis/learning_curves.png",
    "Figure 5. Experiment E during training (x axis in thousands of steps, so the end is about 79 million). "
    "PPO reaches an almost perfect hover around 20–30 million steps, then gets somewhat worse again "
    "towards the end: its final model is not its best one.")

P("5.6 The lab's existing PPO model (experiment F)", h2)
lab_rows = [["Start", "Late error", "Crashes"]]
for st in LAB.starts:
    c, n = LAB.crashes(st)
    lab_rows.append([st, f"{LAB.per_seed(st, 'late_error_m')[0]*100:.1f} cm", f"{c}/{n}"])
P("This model was trained earlier by the lab on a harder version of the task (random start and target "
  "every flight), for 60 million steps according to the lab notebook's settings. Flown through our exact "
  "test, it was excellent from every start:")
table(lab_rows, [0.4, 0.3, 0.3])
P("This shows two things: the hover task is clearly learnable to a few centimetres, and practising from "
  "many different positions produces a controller that works from many positions. That is exactly what the "
  "next experiment (Batch 2, random starts and targets) will test for both methods.")
story.append(PageBreak())

# ============================================================================ 5.7 COMPARISON
P("5.7 Head-to-head comparison (experiments C, D, E)", h2)
cmp = [["", "TD-MPC2 small", "TD-MPC2 standard", "PPO"]]
cmp.append(["Training steps", "50,000", "50,000", "about 79 million"])
cmp.append(["Training time per seed (same GPU)", fm(NARV.train_minutes(), 1, " min"),
            fm(STD.train_minutes(), 1, " min"), fm(PPO.train_minutes(), 1, " min")])
for st, lab in [("home", "Late error, training start")]:
    cmp.append([lab] + [fm(b.per_seed(st, "late_error_m")) for b in (NARV, STD, PPO)])
cmp.append(["Median run, training start"] + [f"{np.median(b.per_seed('home', 'late_error_m')):.2f} m"
                                              for b in (NARV, STD, PPO)])
cmp.append(["Late error, average of all 6 starts"] +
           [fm(np.mean([b.per_seed(st, "late_error_m") for st in b.starts], axis=0)) for b in (NARV, STD, PPO)])
cmp.append(["Crashes, all 198 test flights"] + [f"{b.crashes()[0]} ({b.crashes()[0]/b.crashes()[1]:.0%})"
                                                 for b in (NARV, STD, PPO)])
table(cmp, [0.31, 0.23, 0.23, 0.23])
fig(OUT / "comparison_batch1/by_env_steps.png",
    "Figure 6. Learning against <b>practice</b> (environment steps, log scale: each grid line is 10× more). "
    "TD-MPC2 (blue) learns within 50,000 steps. PPO (orange) shows no progress until about 10 million steps. "
    "<b>TD-MPC2 is roughly 200× more sample-efficient.</b>")
fig(OUT / "comparison_batch1/by_train_time.png",
    "Figure 7. Learning against <b>real time</b>, all on the same GPU. PPO reaches its best hover within "
    "about 4 minutes; TD-MPC2 small is still about 30 cm off after 14 minutes. <b>On the clock, PPO wins</b>, "
    "because it simulates 2,048 drones at once while TD-MPC2 flies one.")
fig(OUT / "comparison_batch1/test_starts.png",
    "Figure 8. Final drones from each starting point (lower is better, dotted line = 10 cm). PPO is best from "
    "the training start (\"home\") but much worse from the 5 new starts. TD-MPC2 small is the most consistent.")
fig(DEST / "fig_quality.png",
    "Figure 9. Every test flight (all 6 starts) sorted into quality groups by its late-flight error. Green = "
    "precise, red = crashed. TD-MPC2 small after 50,000 steps rarely crashes but is rarely precise. PPO and "
    "the 1-million-step TD-MPC2 (section 5.8) are precise from the training start but crash from most new starts.")
fig(DEST / "fig_seed_dots.png",
    "Figure 10. Each dot is one complete training run, scored from the training start. The spread inside "
    "each column shows how much luck matters. Note the log scale: each labelled line is about 3× the one below.")

# ---------------------------------------------------------------------------- 5.8 LONG TRAINING
story.append(PageBreak())
P("5.8 Longer training: 1 million steps (experiment G)", h2)
P("Question: does the sideways drift come from too little practice? To test this, the standard (big) "
  "TD-MPC2 was trained 20× longer: 1,000,000 steps instead of 50,000, with 3 seeds. Only two settings "
  "changed besides the length: the replay memory was enlarged to 1 million steps (the published value, so "
  "nothing is forgotten) and the model was saved less often. Each run took about 6 hours on a GPU slice.")
rows = [["Seed", "Late error (training start)", "Height error", "Time within 10 cm", "Crashes, training start",
         "Crashes, all 18 flights"]]
for s, e, z, t in zip(LONG.seeds, long_home, LONG.per_seed("home", "late_error_z_m"),
                      LONG.per_seed("home", "time_in_10cm")):
    home_c = sum(r["crashed"] == "True" for r in s["summ"] if r["start"] == "home")
    all_c = sum(r["crashed"] == "True" for r in s["summ"])
    rows.append([s["name"][-1], f"{e*100:.1f} cm", f"{z*100:.1f} cm", f"{t:.0%}", f"{home_c}/3", f"{all_c}/18"])
rows.append(["All (mean ± SE)", fm(long_home * 100, 1, " cm"), fm(LONG.per_seed("home", "late_error_z_m") * 100, 1, " cm"),
             fm(LONG.per_seed("home", "time_in_10cm") * 100, 0, " %"),
             f"{LONG.crashes('home')[0]}/{LONG.crashes('home')[1]}", f"{LONG.crashes()[0]}/{LONG.crashes()[1]}"])
table(rows, [0.13, 0.2, 0.15, 0.17, 0.17, 0.18], bold_last=True)
fig(OUT / "batch1_standard_1M/analysis/learning_curves.png",
    "Figure 11. Experiment G during training (x axis: thousands of steps, up to 1 million). The error drops "
    "below the 10 cm line after about 100,000–175,000 steps and stays there, with the return close to its "
    "maximum (about 6). Occasional single test points jump to 20–60 cm, but the runs recover.")
box("So is the drift fixed?", [
    f"<b>From the training start: yes.</b> All 3 runs hover {long_home.min()*100:.0f}–{long_home.max()*100:.0f} cm "
    f"from the target, keep the height within about 1 cm, spend about 90% of the flight within 10 cm, and never "
    f"crash. This is more consistent than PPO (whose median run is more precise, but 4 of 11 runs failed). "
    f"So the drift seen after 50,000 steps was mainly a lack of practice.",
    "<b>From the sideways starts: no, it got worse.</b> From the 4 starts 0.5 m to the side, all 36 test "
    "flights crashed: sometimes the drone never takes off (it falls within half a second), otherwise it "
    "climbs, never gets closer than about 20–60 cm, and loses control after a few seconds. From \"above\" it "
    "still reaches about 23 cm without crashing.",
], tint="#eaf6ee", edge="#1a7f4b")
long_rows = [["Start", "Late error", "Crashes"]]
for st in LONG.starts:
    c, n = LONG.crashes(st)
    long_rows.append([st, fm(LONG.per_seed(st, "late_error_m"), 3), f"{c}/{n}"])
table(long_rows, [0.4, 0.3, 0.3])
P("<b>What this changes.</b> After 50,000 steps the small TD-MPC2 looked much more robust to new starts than "
  "PPO. This experiment suggests that was mostly because it had not specialised yet: once TD-MPC2 is trained "
  "long on a single route, it memorises that route just like PPO did. A fair test of robustness therefore "
  "needs training from many starting points, which is Batch 2.")
P("<b>A detail in the environment that may matter.</b> The drone's sensors report the direction to the "
  "target divided by the distance at the start of the flight (0.8 m from the training start, about 0.94 m "
  "from the sideways starts). So the same physical situation produces slightly different numbers depending "
  "on where the flight began, which makes a single-start controller even less likely to transfer. In "
  "Batch 2 this varies all the time during training, so the controller will learn to handle it.")
fig(OUT / "comparison_with_1M/by_env_steps.png",
    "Figure 12. All methods against practice (log scale). The 1-million-step TD-MPC2 (orange) reaches PPO's "
    "level of return (about 6) after roughly 200,000 steps; PPO (green) needs about 20 million.")
story.append(PageBreak())

# ============================================================================ 6. INTERPRETATION
P("6. What it all means", h1)
P("6.1 What a typical TD-MPC2 flight looks like after 50,000 steps", h2)
fell, outb = HOME.crash_kinds()
B(["Take-off to 1 m in about 0.8 seconds, a small overshoot to about 1.08 m, then the height is held within "
   "a few centimetres.",
   "Meanwhile the drone slides slowly sideways, about 3 cm per second, and after 10 seconds it is typically "
   "about 30 cm from the target.",
   "Each run drifts in its own direction, all around the compass. A problem in the simulator would push every "
   "run the same way, so this is a learning limitation, not a physics bug.",
   f"Crashes are uncommon and happen late (median 7 seconds into the flight), after the drift has built up: "
   f"in experiment B, {fell} by falling and {outb} by flying out of bounds."])
P("6.2 Why does it drift sideways?", h2)
P("A drone cannot push sideways directly. To move sideways it must tilt, accelerate, tilt back to brake, and "
  "level out exactly on the target: a slow chain that takes about a second. Nothing pulls it back on its own, "
  "so any small tilt makes the error grow over time. Height is much easier: more thrust immediately means "
  "up. Likely reasons TD-MPC2 has not mastered the sideways part yet:")
B(["<b>The reward barely cares about 30 cm.</b> Being exactly on target earns 0.0100 per step; being 30 cm "
   "away earns 0.0071. That difference is tiny compared with the −1 crash penalty: there is a strong "
   "reason not to crash, but only a weak reason to be precise.",
   "<b>Little practice at correcting.</b> Every training flight starts in the same place, so the drone rarely "
   "experiences being off to the side and coming back. 50,000 steps is the equivalent of only about 100 "
   "full 10-second flights.",
   "<b>Short look-ahead.</b> The planner imagines only 0.06 s ahead and relies on its learned estimate of "
   "the future beyond that, which is still inaccurate after 50,000 steps.",
   "The task itself is learnable: PPO reaches 1–3 cm from the training start, and the lab's PPO from all "
   "starts."])
P("<b>What experiment G showed.</b> With 20× more practice (1 million steps) the drift from the training start "
  "disappeared: about 5–6 cm, no crashes. So \"little practice at correcting\" was the main reason. The "
  "weak reward for precision did not prevent precise hovering once the drone had enough experience.")
story.append(CondPageBreak(4.2 * inch))   # keep the answers table on one page
P("6.3 Answers to the original questions", h2)
table([["Question", "Answer so far"],
       ["Does TD-MPC2 learn to hover?", "Yes. After 50,000 steps it takes off and holds height but typically "
        "drifts 15–35 cm sideways. After about 150,000 steps (bigger settings, 3 runs) it hovers within about "
        "5–6 cm with no crashes."],
       ["Was the September result reliable?", "It was a typical 50,000-step run. But seeds range from 8 cm to "
        "2.3 m, so one run alone hides most of the picture."],
       ["Does it work from new starting points?", "Only roughly, and only while training is short. After 50,000 "
        "steps it flies from new starts about as well as from its own; after 1 million steps it has "
        "specialised and crashes from the sideways starts. Training from many starts (Batch 2) is needed."],
       ["How does it compare with PPO?", "About 100× less practice needed for a precise hover, and more "
        "consistent between runs; but slower in real time on this GPU, because PPO simulates 2,048 drones "
        "at once. Both memorise the single route they are trained on."],
       ["Do the bigger, official settings help?", "Not at 50,000 steps (worse than the small settings), but "
        "with more practice they gave the best TD-MPC2 result. The small settings have not yet been trained "
        "longer, so a fair small-vs-big comparison at long training is still open."]],
      [0.32, 0.68])
story.append(PageBreak())

# ============================================================================ 7. CAVEATS
P("7. Limitations to keep in mind", h1)
B(["<b>Few flights per seed.</b> Each seed was tested with 3 flights per starting point. Averages over 11 "
   "seeds are reasonably stable; single-seed numbers less so.",
   "<b>Experiment G has only 3 seeds.</b> The 1-million-step result is very consistent across those 3, but "
   "it should be repeated with 11 seeds (and for the small settings too) before it is reported as final.",
   "<b>Final model only.</b> Scores use the model at the very end of training. Both TD-MPC2 (seed 0) and "
   "PPO (most seeds) were better a little earlier, so final scores include the effect of late instability.",
   "<b>Late-flight error and crashes.</b> For a crashed flight it measures where the drone was just before "
   "failing, which is not comparable with a flight that survived. Read it together with the crash count.",
   "<b>Not exactly repeatable.</b> GPU arithmetic is not bit-for-bit deterministic, so re-running a seed "
   "gives a similar, not identical, result. That is why groups of 11 seeds are compared.",
   "<b>Training budgets differ by design.</b> TD-MPC2 trained for 50,000 steps and PPO for about 79 million "
   "(the lab's setting of 60 million, rounded up by the PPO library). This is why both a steps comparison "
   "and a time comparison are shown.",
   "<b>Times.</b> Only experiments C, D and E ran on identical hardware. Experiment B ran two seeds at a "
   "time on the home GPU, so its 26 minutes per seed is not comparable.",
   "<b>Implementation details.</b> The JAX TD-MPC2 differs from the official PyTorch version in small ways "
   "(for example how the final action is picked and some optimiser details). These are the same in every "
   "TD-MPC2 run, so they do not affect the small-vs-standard comparison."])

# ============================================================================ 8. NEXT
P("8. Next steps", h1)
B(["<b>Confirm experiment G with 11 seeds.</b> Since the hover is reached after about 150,000 steps, about "
   "200,000 steps per run is enough (about 1.2 hours each on a GPU slice), for both the small and the "
   "standard settings.",
   "<b>Batch 2 (Professor Bellinger's next request).</b> Random start <i>and</i> target in every training "
   "flight, again 11 seeds per method. Both methods memorised their single training route, so this is the "
   "experiment that tests whether they can learn to fly from anywhere, as the lab's PPO does.",
   "<b>Training length in Batch 2.</b> Flying from anywhere is harder than one route, so Batch 2 will likely "
   "need more steps per run than 50,000; the learning curves will show when each method levels off.",
   "<b>The research question: scaling.</b> PPO wins on the clock because it simulates thousands of drones at "
   "once. What happens when TD-MPC2 is also given many parallel simulations? That is the next big step of "
   "the project."])

# ============================================================================ GLOSSARY
story.append(PageBreak())
P("Glossary", h1)
table([["Term", "Meaning"],
       ["Agent / brain", "The learning controller: the neural networks that decide the drone's actions."],
       ["Action", "The 4 numbers chosen every 0.02 s: thrust, roll, pitch, yaw."],
       ["Observation", "The 31 numbers the drone \"sees\": position, speed, tilt, direction to the target, "
        "last action (with a little sensor noise)."],
       ["Reward / return", "Score for one step / total score of a flight."],
       ["Environment step", "One decision of one drone (0.02 s of simulated time)."],
       ["Training run", "One complete training from scratch, ending with one trained model."],
       ["Seed", "The number that fixes all randomness of a training run."],
       ["Policy", "A rule that maps what the drone sees to what it does."],
       ["World model", "TD-MPC2's learned prediction of what will happen after an action."],
       ["Planning / MPC", "Imagining many action sequences with the world model and picking the best."],
       ["PPO", "Proximal Policy Optimization: a widely used RL method that improves a policy directly from "
        "large batches of experience."],
       ["Sample efficiency", "How much a method learns per step of experience."],
       ["Wall-clock time", "Real time the computer needed to train."],
       ["Mean ± standard error", "Average over seeds ± how precisely that average is known."],
       ["Median", "The middle value; not pulled around by one extreme run."],
       ["Checkpoint", "A saved copy of a trained model."],
       ["Overfitting / memorising", "Doing well only on exactly what was practised."],
       ["GPU slice", "A share (here 2/7) of one A100 graphics card on the Narval cluster."]],
      [0.28, 0.72])
P("Result files: outputs/batch1_fixed_start, batch1_small_narval, batch1_standard, ppo_fixed_start, "
  "ppo_existing_checkpoint and comparison_batch1 in the project folder. Each analysis folder contains a "
  "summary.md with the exact numbers.", small)

doc = SimpleDocTemplate(str(PDF), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                        title="Crazyflie hover: TD-MPC2 vs PPO (Batch 1)", author="Mounir Zouhari")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)
