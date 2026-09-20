"""六面板投稿主图；从已审计的 pilot 数据生成三列、两行的矢量图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_case import file_hash
from .report import load_results, paired, summarize

# 与 paper_v3_figures.py 的字体、主色、轴线及面板编号一致。
BLUE, GREY, TEAL = "#2369a5", "#737981", "#238473"
COLORS = {"native": GREY, "evspark": BLUE}
PATTERNS = {"medium": (768, 768), "short": (384, 1152)}
PATTERN_CODES = {"medium": "M", "short": "S"}


def workflow(ax):
    """用序列分支和回路表示搜索；条带仅作示意，不编码实验序列。"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    from matplotlib.path import Path as MplPath

    ax.set(xlim=(-.12, 1.02), ylim=(0, 1.02))
    ax.set_axis_off()

    def label(x, y, text, **kwargs):
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("fontsize", 8)
        ax.text(x, y, text, va="center", **kwargs)

    def sequence(x, y, width, *, extension=0, height=.025):
        # 灰色表示已有上下文；绿色表示本轮续写，不与后端配色混淆。
        ax.add_patch(Rectangle((x, y-height/2), width-extension, height,
                               facecolor="#c5cbd0", edgecolor="none", zorder=3))
        if extension:
            ax.add_patch(Rectangle((x+width-extension, y-height/2), extension, height,
                                   facecolor=TEAL, edgecolor="none", zorder=3))
        for xx in np.linspace(x+.02, x+width-.02, max(3, round(width*18))):
            ax.plot([xx, xx], [y-height/2, y+height/2], color="white", lw=.35, zorder=4)

    def arrow(start, end, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6,
                                    shrinkA=0, shrinkB=0, lw=.8, color=GREY, **kwargs))

    def model_node(y, title, detail):
        ax.add_patch(FancyBboxPatch((.10, y), .80, .108,
                                    boxstyle="round,pad=0.008,rounding_size=0.018",
                                    facecolor="#f0f6f4", edgecolor="#8fafa6", lw=.65))
        label(.50, y+.075, title, fontweight="bold", color="#215e52")
        label(.50, y+.030, detail, fontsize=7.5)

    # 候选横向展开，主路径居中；上下流程占满面板，回路独立放在外侧。
    label(.50, .977, "40,960 bp mouse prefix")
    sequence(.14, .935, .72)
    arrow((.50, .914), (.50, .899))
    label(.50, .872, "Generate per branch", fontweight="bold")
    label(.50, .813, "15 candidates × 128 bp", fontsize=7.8)

    # 三条示意分支代表候选集合，条带中的绿色末端表示本轮续写。
    for center in (.21, .50, .79):
        ax.plot([.50, center], [.779, .756], color="#a9b4ba", lw=.75)
        arrow((center, .756), (center, .733))
        sequence(center-.09, .713, .18, extension=.06, height=.031)
        ax.plot([center, .50], [.688, .652], color="#a9b4ba", lw=.75)
    arrow((.50, .652), (.50, .633))
    model_node(.514, "Score candidates", "Enformer + 4 Flashzoi")
    arrow((.50, .497), (.50, .476))
    label(.50, .448, "Retain the best 2", fontweight="bold")
    for center in (.32, .68):
        sequence(center-.12, .398, .24, extension=.07, height=.031)
        ax.plot([center, .50], [.374, .338], color="#a9b4ba", lw=.75)

    # 首轮只有一个父分支；保留分支返回生成步骤，共执行 24 轮。
    loop = MplPath([(.175, .398), (-.005, .398), (-.005, .866), (.17, .866)])
    ax.add_patch(FancyArrowPatch(path=loop, arrowstyle="-|>", mutation_scale=7,
                                color=GREY, lw=.8, fill=False))
    label(-.075, .637, "24 rounds", rotation=90, color=GREY, fontsize=7.5)
    arrow((.50, .338), (.50, .316))
    label(.50, .286, "Top design · 3,072 bp", fontweight="bold")
    sequence(.14, .238, .72, extension=.72, height=.031)
    arrow((.50, .212), (.50, .183))
    model_node(.055, "Final check", "Borzoi mouse replicate-0")


def draw(pairs, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8c9299", "axes.linewidth": .6, "axes.axisbelow": True,
        "grid.color": "#e6e9ed", "grid.linewidth": .55, "lines.linewidth": 1.5,
        "lines.markersize": 4, "savefig.dpi": 300, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.5, 5.55))
    grid = fig.add_gridspec(2, 3, left=.068, right=.981, bottom=.14, top=.94,
                           wspace=.45, hspace=.46, height_ratios=[1.26, 1])

    def panel(ax, letter, title, axis="y"):
        ax.set_title(f"{letter}  {title}", loc="left", pad=10, fontweight="bold")
        if axis:
            ax.grid(axis=axis)
        ax.tick_params(length=3, width=.6)

    ax = fig.add_subplot(grid[0, 0])
    # A 没有坐标轴标签，将相应空间用于展开流程，底部与 B/C 的标签对齐。
    bounds = ax.get_position()
    ax.set_position([bounds.x0, bounds.y0-.23*bounds.height,
                     bounds.width, 1.23*bounds.height])
    panel(ax, "a", "Guided sequence design", axis=None)
    workflow(ax)

    profile_panel = fig.add_subplot(grid[0, 1])
    profile_panel.set_axis_off()
    panel(profile_panel, "b", "Target and predictions", axis=None)
    # 与论文其他图相同，图例直接放在数据坐标轴内部，不单独占用面板区域。
    curve_handles = [
        Line2D([], [], color="#40454a", lw=1.15, label="Seed 1"),
        Line2D([], [], color="#40454a", lw=1.15, ls=(0, (3, 2)), label="Seed 2"),
        Patch(facecolor="#e7eaed", edgecolor="#aeb7be", linewidth=.35, label="Open"),
    ]
    for i, (name, (on, off)) in enumerate(PATTERNS.items()):
        profile_ax = profile_panel.inset_axes([0, .520 if i == 0 else -.015, 1, .415])
        profile_ax.tick_params(length=3, width=.6)
        for start in range(0, 3072, on+off):
            profile_ax.axvspan(start/1000, min(start+on, 3072)/1000,
                              color="#e7eaed", lw=0)
        selected = [row for pair in pairs for row in pair if row["pattern"] == name]
        seeds = sorted({row["seed"] for row in selected})
        for row in selected:
            values = np.asarray(row["checker"]["checker_profile"])
            profile_ax.plot((np.arange(len(values))+.5)*.032, values,
                            color=COLORS[row["backend"]], lw=1.05,
                            ls="-" if row["seed"] == seeds[0] else (0, (3, 2)))
        ymax = max(max(row["checker"]["checker_profile"]) for row in selected)
        profile_ax.set(xlim=(0, 3.072), ylim=(0, ymax*1.10),
                       ylabel="DNase", xticks=[0, 1, 2, 3])
        profile_ax.set_title(f"{PATTERN_CODES[name]} · {on} bp open / {off} bp closed",
                             loc="left", fontsize=7.4, pad=4, fontweight="normal")
        if i == 0:
            profile_ax.tick_params(labelbottom=False)
        else:
            # 短周期图的右侧无目标开放区，图例不会压住灰色背景或预测峰。
            profile_ax.legend(handles=curve_handles, loc="upper left",
                              bbox_to_anchor=(.675, .98), frameon=False,
                              fontsize=7, handlelength=1.8, handletextpad=.5,
                              labelspacing=.10, borderaxespad=0, borderpad=.1)
            profile_ax.set_xlabel("Generated position (kb)")

    time_ax = fig.add_subplot(grid[0, 2])
    quality_ax = fig.add_subplot(grid[1, 0])
    labels = []
    seed_order = sorted({a["seed"] for a, b in pairs})
    for i, pair in enumerate(pairs):
        labels.append(f'{PATTERN_CODES[pair[0]["pattern"]]} · s{seed_order.index(pair[0]["seed"])+1}')
        times = [row["total_s"]/60 for row in pair]
        scores = [row["checker"]["checker_auc"] for row in pair]
        time_ax.plot(times, [i, i], color="#bec5cb", lw=1, zorder=1)
        quality_ax.plot(scores, [i, i], color="#bec5cb", lw=1, zorder=1)
        for row, x, q in zip(pair, times, scores):
            # 空心 native 与实心 EvSpark 即使近乎重合也仍可辨别。
            native = row["backend"] == "native"
            style = dict(s=27 if native else 19, edgecolors=COLORS[row["backend"]],
                         facecolors="white" if native else COLORS[row["backend"]],
                         linewidths=1 if native else .5, zorder=3 if native else 4)
            time_ax.scatter(x, i, **style)
            quality_ax.scatter(q, i, **style)
        time_ax.text(np.mean(times), i-.18, f"{times[0]/times[1]:.2f}×",
                     ha="center", fontsize=8)
    time_ax.set(yticks=range(len(pairs)), yticklabels=labels, xlabel="Complete time (min)",
                xlim=(0, 15), xticks=[0, 5, 10, 15], ylim=(len(pairs)-.45, -.65))
    panel(time_ax, "c", "Complete design time", axis="x")
    quality_ax.axvline(.9, color=GREY, ls=":", lw=.9)
    quality_ax.set(yticks=range(len(pairs)), yticklabels=labels, xlabel="Final checker AUROC",
                   xlim=(.895, 1.005), xticks=[.90, .95, 1], ylim=(len(pairs)-.45, -.65))
    quality_ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    panel(quality_ax, "d", "Predicted quality", axis="x")

    budget_ax = fig.add_subplot(grid[1, 1])
    diversity_ax = fig.add_subplot(grid[1, 2])
    for i, backend in enumerate(COLORS):
        arm = summary["arms"][backend]
        curve = arm["budget_curve"]
        budget_ax.step(np.asarray(curve["seconds"])/60, curve["qualified_unique"],
                       where="post", color=COLORS[backend], lw=1.5)
        budget_ax.scatter(curve["seconds"][-1]/60, curve["qualified_unique"][-1],
                          s=15, color=COLORS[backend], zorder=3)
        diversity = arm["diversity"]
        values = diversity["aligned_identity"]
        diversity_ax.scatter(i+np.linspace(-.10, .10, len(values)), values, s=22,
                             edgecolors=COLORS[backend], linewidths=.75,
                             facecolors="white" if backend == "native" else COLORS[backend])
        diversity_ax.text(i, .91, f'{diversity["exact_unique"]}/{len(pairs)} unique',
                          ha="center", va="top", fontsize=8)
        diversity_ax.text(i, .79, f'$H_4$ = {diversity["fourmer_entropy_bits"]:.2f} bits',
                          ha="center", va="top", fontsize=7.5)
    panel(budget_ax, "e", "Fixed time budget")
    budget_ax.set(xlabel="Cumulative time (min)", ylabel="Qualified unique designs",
                  xlim=(0, 36), xticks=[0, 10, 20, 30], ylim=(0, len(pairs)+.75),
                  yticks=range(len(pairs)+1))
    budget_ax.text(.04, .96, f'Observed: {summary["common_budget_seconds"]/60:.2f} min',
                   transform=budget_ax.transAxes, va="top", fontsize=7.5)
    panel(diversity_ax, "f", "Sequence diversity")
    diversity_ax.set(xticks=[0, 1], xticklabels=["Native", "EvSpark"],
                     xlim=(-.5, 1.5), ylim=(0, 1), yticks=[0, .25, .5, .75, 1],
                     ylabel="Positionwise identity")

    # 后端颜色全图一致；种子线型放在 b 内，避免把两类编码混淆。
    handles = [
        Line2D([], [], color=GREY, marker="o", mfc="white", label="Native (B = 8)"),
        Line2D([], [], color=BLUE, marker="o", label="EvSpark (B = 4)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .02),
               ncol=2, frameon=False, columnspacing=2.5, handlelength=2)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(str(output)+"."+extension, bbox_inches="tight", pad_inches=.05)
    # 额外导出 A/B 局部，便于按实际内容审阅排版，主论文仍使用完整六面板图。
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    crop = Bbox.union([ax.get_tightbbox(renderer), profile_panel.get_tightbbox(renderer)])
    crop = crop.expanded(1.025, 1.025).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(str(output)+"_ab.png", bbox_inches=crop, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/phase5"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    audit = json.loads((root/"pilot_audit.json").read_text())
    assert audit["status"] == "passed" and audit["n_designs"] == 8
    for item in audit["sources"]:
        assert file_hash(item["source"]) == item["files"]["result.json"], "核验后的原始结果已改变"
    selection = json.loads((root/"selected_batches.json").read_text())
    pairs = paired(load_results(root, "pilot"), "pilot", selection)
    summary = summarize(pairs)
    assert summary == json.loads((root/"pilot_summary.json").read_text())
    output = args.output or root/"pilot_case_study"
    output.parent.mkdir(parents=True, exist_ok=True)
    draw(pairs, summary, output)
    print(str(output)+".pdf")


if __name__ == "__main__":
    main()
