from __future__ import annotations

import os
from pathlib import Path

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "results/.matplotlib")).resolve()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "results/.cache")).resolve()
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_selection_rate_by_time(fairness_metrics: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, part in fairness_metrics.groupby("scenario"):
        ax.plot(part["time_window"], part["selection_rate_A"], marker="o", label=f"{scenario} A")
        ax.plot(part["time_window"], part["selection_rate_B"], marker="s", linestyle="--", label=f"{scenario} B")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Selection rate")
    ax.set_title("Selection Rate by Time and Group")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_fairness_gap_by_time(fairness_metrics: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, part in fairness_metrics.groupby("scenario"):
        ax.plot(part["time_window"], part["selection_parity_gap"], marker="o", label=scenario)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Selection parity gap, B - A")
    ax.set_title("Fairness Gap by Time")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_p_values_by_time(hypothesis_tests: pd.DataFrame, output_path: str | Path) -> None:
    tests = hypothesis_tests[hypothesis_tests["test_name"] == "two_proportion_z_selection"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, part in tests.groupby("scenario"):
        ax.plot(part["time_window"], part["p_value"], marker="o", label=scenario)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.9, label="0.05")
    ax.set_yscale("log")
    ax.set_xlabel("Time window")
    ax.set_ylabel("p-value, log scale")
    ax.set_title("Selection-Rate Test p-values by Time")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bootstrap_ci(bootstrap_table: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, part in bootstrap_table.groupby("scenario"):
        ax.errorbar(
            part["time_window"],
            part["estimate"],
            yerr=[part["estimate"] - part["ci_lower"], part["ci_upper"] - part["estimate"]],
            marker="o",
            capsize=3,
            label=scenario,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Selection parity gap with bootstrap CI")
    ax.set_title("Bootstrap CI for Fairness Gap")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
