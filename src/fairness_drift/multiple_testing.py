from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests


METHODS = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "benjamini_hochberg": "fdr_bh",
}


def apply_multiple_testing_correction(
    results: pd.DataFrame,
    alpha: float = 0.05,
    p_col: str = "p_value",
) -> pd.DataFrame:
    corrected_frames = []
    clean = results.dropna(subset=[p_col]).copy()
    for method_name, statsmodels_method in METHODS.items():
        reject, p_adjusted, _, _ = multipletests(clean[p_col], alpha=alpha, method=statsmodels_method)
        frame = clean.copy()
        frame["correction_method"] = method_name
        frame["p_value_adjusted"] = p_adjusted
        frame["reject_null"] = reject
        frame["alpha"] = alpha
        corrected_frames.append(frame)
    return pd.concat(corrected_frames, ignore_index=True)
