"""
Model Trainer
==============
Trains the TF-IDF vectorizer and computes the cosine similarity matrix
using the feature-engineered dataset from Phase 2.

Artifacts produced:
    models/tfidf_vectorizer.pkl   — fitted TF-IDF vectorizer
    models/similarity_matrix.pkl  — float32 cosine similarity matrix (n×n)
    models/product_index.pkl      — bidirectional product_id ↔ row_index mapping
    models/processed_products.csv — clean product records with all features
    models/metadata.json          — training metadata (vocab size, shape, version)

MLflow:
    Logs all hyperparameters, metrics, and artifacts to the configured experiment.
    Model metadata is registered in the MLflow model registry.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainingArtifact
from src.entity.config_entity import ModelConfig, TFIDFConfig
from src.exception.exception import CustomException
from src.logger.training_logger import get_training_logger
from src.utils.common import (
    create_directories,
    get_file_hash,
    get_iso_timestamp,
    save_json,
)

logger = get_training_logger()


# ---------------------------------------------------------------
# HTML Training Report Generator
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# SVG Chart Helpers (pure Python, no external dependencies)
# ---------------------------------------------------------------

def _svg_hbar(
    labels: list,
    values: list,
    title: str = "",
    color: str = "#58a6ff",
    max_bars: int = 10,
    width: int = 520,
) -> str:
    """Horizontal bar chart as inline SVG."""
    pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)[:max_bars]
    if not pairs:
        return ""
    lbls = [str(p[0])[:25] for p in pairs]
    vals = [p[1] for p in pairs]
    max_val = max(vals) or 1
    bar_h, gap = 22, 6
    ml, mr, mt = 155, 55, 28
    chart_w = width - ml - mr
    total_h = mt + 20 + len(lbls) * (bar_h + gap) + 20
    out = [
        f'<svg viewBox="0 0 {width} {total_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{width}px;display:block">',
        f'<text x="{width//2}" y="17" text-anchor="middle" fill="#8b949e" '
        f'font-size="11" font-family="sans-serif">{title}</text>',
    ]
    for i, (label, val) in enumerate(zip(lbls, vals)):
        y = mt + 20 + i * (bar_h + gap)
        bw = max(3, int(chart_w * val / max_val))
        out += [
            f'<text x="{ml-8}" y="{y+bar_h//2+4}" text-anchor="end" fill="#c9d1d9" '
            f'font-size="10.5" font-family="sans-serif">{label}</text>',
            f'<rect x="{ml}" y="{y}" width="{chart_w}" height="{bar_h}" fill="#21262d" rx="3"/>',
            f'<rect x="{ml}" y="{y}" width="{bw}" height="{bar_h}" fill="{color}" rx="3" opacity="0.82"/>',
            f'<text x="{ml+bw+5}" y="{y+bar_h//2+4}" fill="#8b949e" '
            f'font-size="9.5" font-family="sans-serif">{val:,}</text>',
        ]
    out.append("</svg>")
    return "\n".join(out)


def _svg_hist(
    values: list,
    title: str = "",
    color: str = "#3fb950",
    bins: int = 8,
    width: int = 480,
    height: int = 210,
    x_label: str = "",
) -> str:
    """Histogram as inline SVG from a flat list of numeric values."""
    if not values:
        return ""
    import math as _math
    mn, mx = min(values), max(values)
    if mn == mx:
        return ""
    bw = (mx - mn) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / bw), bins - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    ml, mr, mt, mb = 42, 15, 28, 38
    cw = width - ml - mr
    ch = height - mt - mb
    bar_px = cw / bins
    out = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{width}px;display:block">',
        f'<text x="{width//2}" y="17" text-anchor="middle" fill="#8b949e" '
        f'font-size="11" font-family="sans-serif">{title}</text>',
    ]
    # Grid + y-labels
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        gy = mt + ch * (1 - frac)
        gv = int(max_c * frac)
        out.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+cw}" y2="{gy:.1f}" '
                   f'stroke="#30363d" stroke-width="1"/>')
        out.append(f'<text x="{ml-4}" y="{gy+4:.1f}" text-anchor="end" fill="#8b949e" '
                   f'font-size="9" font-family="sans-serif">{gv}</text>')
    # Bars
    for i, cnt in enumerate(counts):
        bh = ch * cnt / max_c
        bx = ml + i * bar_px
        by = mt + ch - bh
        out.append(f'<rect x="{bx+1:.1f}" y="{by:.1f}" width="{bar_px-2:.1f}" '
                   f'height="{bh:.1f}" fill="{color}" opacity="0.83" rx="2"/>')
    # X-axis labels
    step = max(1, bins // 5)
    for i in range(0, bins + 1, step):
        xv = mn + i * bw
        px = ml + i * bar_px
        fmt = f"{xv:.1f}" if mx - mn < 50 else f"{xv:.0f}"
        out.append(f'<text x="{px:.1f}" y="{height-8}" text-anchor="middle" fill="#8b949e" '
                   f'font-size="9" font-family="sans-serif">{fmt}</text>')
    if x_label:
        out.append(f'<text x="{ml + cw//2}" y="{height-1}" text-anchor="middle" fill="#6e7681" '
                   f'font-size="9" font-family="sans-serif">{x_label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _svg_donut(
    labels: list,
    values: list,
    title: str = "",
    width: int = 340,
    height: int = 310,
) -> str:
    """Donut/ring chart as inline SVG."""
    import math as _math
    total = sum(values) or 1
    palette = ["#58a6ff", "#3fb950", "#d29922", "#f47067", "#bc8cff", "#39d353"]
    cx, cy = width // 2, (height - 70) // 2 + 26
    r_out, r_in = 85, 50
    out = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{width}px;display:block">',
        f'<text x="{width//2}" y="17" text-anchor="middle" fill="#8b949e" '
        f'font-size="11" font-family="sans-serif">{title}</text>',
    ]
    angle = -_math.pi / 2
    for i, (lbl, val) in enumerate(zip(labels, values)):
        sweep = (val / total) * 2 * _math.pi
        ea = angle + sweep
        x1, y1 = cx + r_out * _math.cos(angle), cy + r_out * _math.sin(angle)
        x2, y2 = cx + r_out * _math.cos(ea),    cy + r_out * _math.sin(ea)
        x3, y3 = cx + r_in  * _math.cos(ea),    cy + r_in  * _math.sin(ea)
        x4, y4 = cx + r_in  * _math.cos(angle), cy + r_in  * _math.sin(angle)
        la = 1 if sweep > _math.pi else 0
        c = palette[i % len(palette)]
        pct = val / total * 100
        out.append(
            f'<path d="M{x1:.2f},{y1:.2f} A{r_out},{r_out} 0 {la},1 {x2:.2f},{y2:.2f} '
            f'L{x3:.2f},{y3:.2f} A{r_in},{r_in} 0 {la},0 {x4:.2f},{y4:.2f} Z" '
            f'fill="{c}" opacity="0.88" stroke="#0d1117" stroke-width="1.5"/>'
        )
        angle = ea
        ly = height - 68 + i * 18
        out += [
            f'<rect x="18" y="{ly}" width="11" height="11" fill="{c}" rx="2"/>',
            f'<text x="34" y="{ly+9}" fill="#c9d1d9" font-size="10" '
            f'font-family="sans-serif">{lbl}  {pct:.0f}%</text>',
        ]
    # Center label
    out.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" fill="#e6edf3" '
               f'font-size="12" font-weight="600" font-family="sans-serif">Hybrid</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------
# Chart data computation helper (called from ModelTrainer)
# ---------------------------------------------------------------

def _compute_chart_data(
    df: "pd.DataFrame",
    sim_matrix: "np.ndarray",
    hybrid_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Pre-compute all chart data from the products DataFrame and similarity matrix.

    Returns a dict consumed by generate_training_report().
    """
    import numpy as np
    import math

    data: Dict[str, Any] = {}

    # 1. Rating distribution
    if "rating" in df.columns:
        ratings = df["rating"].dropna().astype(float).tolist()
        data["ratings"] = [r for r in ratings if 0 < r <= 5]
    else:
        data["ratings"] = []

    # 2. Top categories
    if "category_l1" in df.columns:
        counts = df["category_l1"].dropna().value_counts().head(10)
        data["top_categories"] = {
            "labels": counts.index.tolist(),
            "values": counts.values.tolist(),
        }
    else:
        data["top_categories"] = {"labels": [], "values": []}

    # 3. Discount distribution
    disc_col = None
    for col in ["discount_percentage", "discount_percent"]:
        if col in df.columns:
            disc_col = col
            break
    if disc_col:
        discs = df[disc_col].dropna().astype(float)
        discs = discs[(discs >= 0) & (discs <= 100)].tolist()
        data["discounts"] = discs
    else:
        data["discounts"] = []

    # 4. Similarity score distribution — sample 8,000 non-zero values
    try:
        flat = sim_matrix.flatten()
        nonzero_mask = flat > 0.01
        nonzero_vals = flat[nonzero_mask]
        if len(nonzero_vals) > 8000:
            indices = np.random.choice(len(nonzero_vals), 8000, replace=False)
            nonzero_vals = nonzero_vals[indices]
        data["similarities"] = [float(v) for v in nonzero_vals]
    except Exception:
        data["similarities"] = []

    # 5. Hybrid engine weights
    data["hybrid_weights"] = hybrid_weights or {
        "Content-Based":   0.35,
        "Category":        0.25,
        "Brand":           0.15,
        "Price":           0.10,
        "Popularity":      0.10,
        "Rating":          0.05,
    }

    return data


# ---------------------------------------------------------------
# HTML Training Report Generator
# ---------------------------------------------------------------

def generate_training_report(
    metadata_path: str,
    num_products: int = 0,
    mlflow_run_id: Optional[str] = None,
    eval_report_path: str = "artifacts/evaluation_report.json",
    output_path: str = "artifacts/training_report.html",
    chart_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a self-contained, fully-offline HTML training report.

    Includes:
      - Dataset statistics summary cards
      - Model parameters panel
      - Top 20 TF-IDF terms table
      - Evaluation metrics table
      - Recommendation engines summary
      - 5 pure-SVG visualisations:
          1. Product rating distribution
          2. Top 10 categories by count
          3. Discount percentage distribution
          4. Cosine similarity score distribution
          5. Hybrid engine weight breakdown (donut)

    Args:
        metadata_path: Path to models/metadata.json.
        num_products: Number of training products.
        mlflow_run_id: MLflow run ID (may be None).
        eval_report_path: Path to evaluation_report.json.
        output_path: Output HTML file path.
        chart_data: Pre-computed chart data dict from _compute_chart_data().

    Returns:
        Path to the generated HTML file.
    """
    # ── Load metadata ───────────────────────────────────────────
    meta: Dict[str, Any] = {}
    try:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass

    # ── Load evaluation report ───────────────────────────────────
    eval_metrics: Dict[str, Any] = {}
    pr_list: list = []
    try:
        with open(eval_report_path, encoding="utf-8") as f:
            eval_data = json.load(f)
        eval_metrics = eval_data.get("metrics", {})
        pr_list = eval_data.get("precision_recall_by_k", [])
    except Exception:
        pass

    # ── Extract metadata values ──────────────────────────────────
    vocab_size     = meta.get("vocabulary_size", 0)
    training_time  = meta.get("training_time", 0)
    dataset_hash   = meta.get("dataset_hash", "N/A")
    dataset_rows   = meta.get("dataset_rows", num_products)
    dataset_cols   = meta.get("dataset_columns", 0)
    top_terms      = meta.get("top_tfidf_terms", [])
    vec_params     = meta.get("vectorizer_parameters", {})
    matrix_shape   = meta.get("matrix_shape", [0, 0])
    matrix_mb      = meta.get("matrix_size_mb", 0)
    model_version  = meta.get("model_version", "1.0.0")
    training_date  = meta.get("training_date", get_iso_timestamp())
    run_id         = mlflow_run_id or meta.get("mlflow_run") or "Not logged"
    sim_threshold  = meta.get("similarity_threshold", 0.25)

    cd = chart_data or {}

    # ── Generate SVG charts ──────────────────────────────────────
    chart_ratings = _svg_hist(
        cd.get("ratings", []),
        title="Product Rating Distribution",
        color="#58a6ff",
        bins=9,
        x_label="Rating (0–5)",
    )

    tc = cd.get("top_categories", {"labels": [], "values": []})
    chart_categories = _svg_hbar(
        tc.get("labels", []),
        tc.get("values", []),
        title="Top 10 Categories by Product Count",
        color="#3fb950",
        max_bars=10,
    )

    chart_discounts = _svg_hist(
        cd.get("discounts", []),
        title="Discount Percentage Distribution",
        color="#d29922",
        bins=8,
        x_label="Discount (%)",
    )

    chart_similarity = _svg_hist(
        cd.get("similarities", []),
        title="Cosine Similarity Score Distribution",
        color="#bc8cff",
        bins=10,
        x_label="Similarity score (0–1)",
    )

    hw = cd.get("hybrid_weights", {})
    hw_labels = [k.replace("_", " ").title() for k in hw.keys()]
    hw_values = list(hw.values())
    chart_hybrid = _svg_donut(
        hw_labels, hw_values,
        title="Hybrid Engine Weight Breakdown",
    )

    # ── Build table rows ─────────────────────────────────────────
    term_rows = "".join(
        f'<tr><td class="num">{i}</td><td class="term">{t["term"]}</td>'
        f'<td class="num">{t["importance"]:.5f}</td>'
        f'<td class="num">{t["idf"]:.3f}</td></tr>\n'
        for i, t in enumerate(top_terms[:20], 1)
    ) or '<tr><td colspan="4" class="empty">No term data available</td></tr>'

    metric_rows = ""
    for pr in pr_list:
        metric_rows += (
            f'<tr><td>Precision@{pr["k"]}</td>'
            f'<td class="mval">{pr["precision"]:.4f}</td></tr>\n'
            f'<tr><td>Recall@{pr["k"]}</td>'
            f'<td class="mval">{pr["recall"]:.4f}</td></tr>\n'
        )
    for key in ["coverage", "diversity", "novelty",
                "intra_list_similarity", "inference_latency_ms"]:
        val = eval_metrics.get(key)
        if val is not None:
            suffix = " ms" if "latency" in key else ""
            metric_rows += (
                f'<tr><td>{key.replace("_"," ").title()}</td>'
                f'<td class="mval">{val:.4f}{suffix}</td></tr>\n'
            )
    if not metric_rows:
        metric_rows = '<tr><td colspan="2" class="empty">Run evaluation pipeline to populate metrics</td></tr>'

    engines = [
        ("Content-Based",       "TF-IDF cosine similarity", "0.35"),
        ("Popularity",          "Popularity score ranking",  "0.10"),
        ("Trending",            "Decay-weighted trend score","—"),
        ("Category Similarity", "L1/L2 hierarchy matching",  "0.25"),
        ("Price Similarity",    "Bucket + ±20% tolerance",   "0.10"),
        ("Brand Similarity",    "Same-brand recommendations","0.15"),
        ("Freq. Bought Together","Category complementarity", "—"),
        ("Hybrid",              "Weighted combination of all","1.00"),
    ]
    engine_rows = "".join(
        f'<tr><td class="ename">{n}</td><td>{d}</td>'
        f'<td class="num">{w}</td></tr>\n'
        for n, d, w in engines
    )

    run_display = run_id[:20] + "…" if len(str(run_id)) > 20 else str(run_id)

    # ── Compose HTML ─────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Report — Product Recommendation System v{model_version}</title>
<style>
:root{{
  --bg:#0d1117;--card:#161b22;--card2:#1c2128;--border:#30363d;
  --accent:#58a6ff;--green:#3fb950;--yellow:#d29922;--purple:#bc8cff;
  --text:#e6edf3;--muted:#8b949e;--dim:#6e7681;
  --font:'Segoe UI',system-ui,-apple-system,sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);
     padding:1.8rem;min-height:100vh}}
/* Header */
.hdr{{display:flex;align-items:flex-start;justify-content:space-between;
      padding-bottom:1.4rem;margin-bottom:1.8rem;
      border-bottom:1px solid var(--border)}}
.hdr h1{{font-size:1.5rem;font-weight:700;letter-spacing:-.02em}}
.hdr h1 span{{color:var(--muted);font-weight:400;font-size:1rem}}
.badge{{display:inline-flex;align-items:center;background:#1f2937;
        border:1px solid var(--accent);color:var(--accent);
        padding:.18rem .65rem;border-radius:20px;font-size:.72rem;
        font-weight:600;letter-spacing:.04em;margin-left:.6rem}}
.run-pill{{background:#1f2937;border:1px solid var(--border);
           padding:.3rem .75rem;border-radius:6px;font-size:.75rem;
           font-family:monospace;color:var(--muted)}}
/* Stat grid */
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
            gap:1rem;margin-bottom:1.8rem}}
.stat{{background:var(--card);border:1px solid var(--border);
       border-radius:10px;padding:1.1rem 1.3rem;position:relative;overflow:hidden}}
.stat::before{{content:'';position:absolute;top:0;left:0;width:3px;height:100%;
               background:var(--accent)}}
.stat.g::before{{background:var(--green)}}
.stat.y::before{{background:var(--yellow)}}
.stat.p::before{{background:var(--purple)}}
.stat .lbl{{color:var(--muted);font-size:.72rem;text-transform:uppercase;
            letter-spacing:.06em;margin-bottom:.35rem}}
.stat .val{{font-size:1.7rem;font-weight:700;color:var(--accent);line-height:1}}
.stat.g .val{{color:var(--green)}}
.stat.y .val{{color:var(--yellow)}}
.stat.p .val{{color:var(--purple)}}
.stat .sub{{color:var(--muted);font-size:.75rem;margin-top:.3rem}}
/* Sections */
.section{{background:var(--card);border:1px solid var(--border);
          border-radius:10px;padding:1.4rem;margin-bottom:1.4rem}}
.section h2{{font-size:.9rem;font-weight:600;color:var(--accent);
             letter-spacing:.04em;text-transform:uppercase;
             margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}}
/* Charts grid */
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));
             gap:1.2rem;margin-bottom:1.4rem}}
.chart-card{{background:var(--card);border:1px solid var(--border);
             border-radius:10px;padding:1.2rem}}
.chart-card.narrow{{max-width:380px}}
/* Params grid */
.params{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.65rem}}
.param{{background:var(--card2);border-radius:7px;padding:.55rem .8rem}}
.param .pk{{color:var(--muted);font-size:.7rem;margin-bottom:.15rem}}
.param .pv{{font-weight:600;font-size:.88rem}}
/* Tables */
table{{width:100%;border-collapse:collapse;font-size:.84rem}}
th{{text-align:left;color:var(--muted);font-weight:500;padding:.55rem .7rem;
    border-bottom:1px solid var(--border);font-size:.72rem;text-transform:uppercase;
    letter-spacing:.04em}}
td{{padding:.48rem .7rem;border-bottom:1px solid #21262d;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--card2)}}
.term{{font-family:monospace;color:var(--green);font-size:.83rem}}
.num{{font-family:monospace;font-size:.82rem;text-align:right}}
.ename{{font-weight:600}}
.mval{{color:var(--green);font-weight:600;font-family:monospace;text-align:right}}
.empty{{color:var(--muted);font-style:italic;text-align:center;padding:1rem}}
/* Footer */
footer{{text-align:center;color:var(--dim);font-size:.76rem;
        margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border)}}
</style>
</head>
<body>

<!-- ─── Header ───────────────────────────────────────────── -->
<div class="hdr">
  <div>
    <h1>Training Report <span class="badge">v{model_version}</span></h1>
    <p style="color:var(--muted);font-size:.82rem;margin-top:.4rem">
      Generated: {training_date}
    </p>
  </div>
  <div style="text-align:right">
    <div style="font-size:.72rem;color:var(--muted);margin-bottom:.3rem">MLflow Run</div>
    <div class="run-pill">{run_display}</div>
  </div>
</div>

<!-- ─── Stat Cards ───────────────────────────────────────── -->
<div class="stat-grid">
  <div class="stat">
    <div class="lbl">Dataset Rows</div>
    <div class="val">{dataset_rows:,}</div>
    <div class="sub">{dataset_cols} columns &bull; hash {str(dataset_hash)[:8]}</div>
  </div>
  <div class="stat g">
    <div class="lbl">Vocabulary Size</div>
    <div class="val">{vocab_size:,}</div>
    <div class="sub">unique TF-IDF features</div>
  </div>
  <div class="stat y">
    <div class="lbl">Similarity Matrix</div>
    <div class="val">{matrix_shape[0]:,}<span style="font-size:1rem">&times;{matrix_shape[1] if len(matrix_shape)>1 else 0:,}</span></div>
    <div class="sub">{matrix_mb:.1f} MB float32 &bull; threshold {sim_threshold}</div>
  </div>
  <div class="stat p">
    <div class="lbl">Training Time</div>
    <div class="val">{training_time:.1f}<span style="font-size:1rem">s</span></div>
    <div class="sub">end-to-end pipeline</div>
  </div>
</div>

<!-- ─── Model Parameters ─────────────────────────────────── -->
<div class="section">
  <h2>&#9881; Model Parameters</h2>
  <div class="params">
    <div class="param"><div class="pk">ngram range</div><div class="pv">{vec_params.get('ngram',[1,2])}</div></div>
    <div class="param"><div class="pk">max features</div><div class="pv">{vec_params.get('max_features',5000):,}</div></div>
    <div class="param"><div class="pk">min df</div><div class="pv">{vec_params.get('min_df',2)}</div></div>
    <div class="param"><div class="pk">max df</div><div class="pv">{vec_params.get('max_df',0.95)}</div></div>
    <div class="param"><div class="pk">sublinear tf</div><div class="pv">{vec_params.get('sublinear_tf',True)}</div></div>
    <div class="param"><div class="pk">analyzer</div><div class="pv">{vec_params.get('analyzer','word')}</div></div>
    <div class="param"><div class="pk">stop words</div><div class="pv">{vec_params.get('stop_words','english')}</div></div>
    <div class="param"><div class="pk">similarity threshold</div><div class="pv">{sim_threshold}</div></div>
  </div>
</div>

<!-- ─── Visualisations ────────────────────────────────────── -->
<div class="chart-grid">
  <div class="chart-card">
    {chart_ratings or '<p style="color:var(--muted);padding:1rem">No rating data</p>'}
  </div>
  <div class="chart-card">
    {chart_categories or '<p style="color:var(--muted);padding:1rem">No category data</p>'}
  </div>
  <div class="chart-card">
    {chart_discounts or '<p style="color:var(--muted);padding:1rem">No discount data</p>'}
  </div>
  <div class="chart-card">
    {chart_similarity or '<p style="color:var(--muted);padding:1rem">No similarity data</p>'}
  </div>
  <div class="chart-card narrow">
    {chart_hybrid or '<p style="color:var(--muted);padding:1rem">No weight data</p>'}
  </div>
</div>

<!-- ─── Top TF-IDF Terms ──────────────────────────────────── -->
<div class="section">
  <h2>&#128269; Top TF-IDF Terms</h2>
  <table>
    <thead><tr><th>#</th><th>Term</th><th style="text-align:right">Importance</th><th style="text-align:right">IDF</th></tr></thead>
    <tbody>{term_rows}</tbody>
  </table>
</div>

<!-- ─── Evaluation Metrics ───────────────────────────────── -->
<div class="section">
  <h2>&#128202; Evaluation Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>
</div>

<!-- ─── Recommendation Engines ───────────────────────────── -->
<div class="section">
  <h2>&#129302; Recommendation Engines</h2>
  <table>
    <thead><tr><th>Engine</th><th>Strategy</th><th style="text-align:right">Hybrid Weight</th></tr></thead>
    <tbody>{engine_rows}</tbody>
  </table>
</div>

<footer>
  Product Recommendation System &bull; MLOps Project &bull;
  Generated by ModelTrainer &bull; {training_date[:10]}
</footer>

</body></html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path



class ModelTrainer:
    """
    Trains and serialises the TF-IDF recommendation model.

    Args:
        config: ModelConfig with artifact paths and TF-IDF hyperparameters.
        feature_artifact: FeatureEngineeringArtifact from Phase 2.
        mlflow_config: Optional MLflow configuration dict.
    """

    def __init__(
        self,
        config: ModelConfig,
        feature_artifact: FeatureEngineeringArtifact,
        mlflow_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.feature_artifact = feature_artifact
        self.mlflow_config = mlflow_config or {}
        self._run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _setup_mlflow(self) -> bool:
        """
        Configure MLflow tracking.

        Returns:
            True if MLflow is available and configured, False otherwise.
        """
        try:
            import mlflow

            tracking_uri = self.mlflow_config.get("tracking_uri", "mlruns")
            experiment_name = self.mlflow_config.get(
                "experiment_name", "product-recommendation"
            )

            mlflow.set_tracking_uri(tracking_uri)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)

            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
            logger.info(f"MLflow experiment: {experiment_name}")
            return True
        except Exception as e:
            logger.warning(f"MLflow setup failed (will continue without): {e}")
            return False

    def _log_to_mlflow(
        self,
        vectorizer: TfidfVectorizer,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifact_dir: Path,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Log training run to MLflow.

        Returns:
            (run_id, experiment_id) or (None, None) on failure.
        """
        try:
            import mlflow

            with mlflow.start_run(run_name=f"tfidf-training-{get_iso_timestamp()}") as run:
                # Log hyperparameters
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log artifacts directory
                mlflow.log_artifacts(str(artifact_dir), artifact_path="model_artifacts")

                run_id = run.info.run_id
                experiment_id = run.info.experiment_id
                logger.info(f"MLflow run logged: {run_id}")
                return str(run_id), str(experiment_id)

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Core training steps
    # ------------------------------------------------------------------

    def _load_features(self) -> pd.DataFrame:
        """Load and validate the feature-engineered dataset."""
        feature_path = Path(self.feature_artifact.feature_filepath)
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}\n"
                "Run the data pipeline first: python -m src.pipelines.training_pipeline"
            )

        logger.info(f"Loading features from: {feature_path}")
        df = pd.read_csv(feature_path, encoding="utf-8", low_memory=False)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        # Validate required columns
        required = ["product_id", "product_name", self.feature_artifact.combined_text_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Feature file missing columns: {missing}")

        return df

    def _fit_tfidf(
        self, df: pd.DataFrame, tfidf_cfg: TFIDFConfig
    ) -> Tuple[TfidfVectorizer, Any]:
        """
        Fit TF-IDF vectorizer on combined text column.

        Args:
            df: Feature DataFrame.
            tfidf_cfg: TF-IDF hyperparameters from config.

        Returns:
            (fitted_vectorizer, tfidf_matrix)
        """
        text_col = self.feature_artifact.combined_text_column
        logger.info(f"Fitting TF-IDF on column: {text_col}")

        # Prepare corpus — use TFIDF version of combined text (stop-word cleaned)
        tfidf_col = f"{text_col}_tfidf" if f"{text_col}_tfidf" in df.columns else text_col
        texts = df[tfidf_col].fillna("").tolist()
        logger.info(f"Corpus size: {len(texts)} documents")

        # Parse ngram_range
        ngram_range = tuple(tfidf_cfg.ngram_range) if tfidf_cfg.ngram_range else (1, 2)
        if len(ngram_range) != 2:
            ngram_range = (1, 2)

        vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg.max_features,
            ngram_range=ngram_range,
            min_df=tfidf_cfg.min_df,
            max_df=tfidf_cfg.max_df,
            stop_words=tfidf_cfg.stop_words if tfidf_cfg.stop_words != "none" else None,
            sublinear_tf=tfidf_cfg.sublinear_tf,
            analyzer=tfidf_cfg.analyzer,
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        vocab_size = len(vectorizer.vocabulary_)
        logger.info(f"TF-IDF fitted: vocab_size={vocab_size}, matrix_shape={tfidf_matrix.shape}")
        return vectorizer, tfidf_matrix

    def _compute_similarity(self, tfidf_matrix: Any) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        For large corpora this is batched to avoid memory errors.
        Result is cast to float32 to halve memory usage.

        Args:
            tfidf_matrix: Sparse TF-IDF matrix (n × vocab).

        Returns:
            Dense float32 ndarray of shape (n, n).
        """
        n = tfidf_matrix.shape[0]
        logger.info(f"Computing cosine similarity matrix ({n}×{n})...")

        if n <= 2000:
            # Direct computation for smaller datasets
            sim_matrix = cosine_similarity(tfidf_matrix).astype(np.float32)
        else:
            # Batched computation for large datasets
            batch_size = 500
            sim_matrix = np.zeros((n, n), dtype=np.float32)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_sim = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
                sim_matrix[start:end] = batch_sim.astype(np.float32)
                logger.info(f"  Similarity batch {start}:{end} done")

        # Zero out self-similarities along diagonal
        np.fill_diagonal(sim_matrix, 0.0)
        logger.info(
            f"Similarity matrix shape={sim_matrix.shape}, "
            f"dtype={sim_matrix.dtype}, "
            f"size={sim_matrix.nbytes / 1024 / 1024:.1f}MB"
        )
        return sim_matrix

    def _build_product_index(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build bidirectional product_id ↔ matrix row index mappings.

        Returns:
            Dict with 'id_to_idx' and 'idx_to_id' sub-dicts.
        """
        ids = df["product_id"].tolist()
        id_to_idx = {str(pid): idx for idx, pid in enumerate(ids)}
        idx_to_id = {idx: str(pid) for idx, pid in enumerate(ids)}
        return {"id_to_idx": id_to_idx, "idx_to_id": idx_to_id}

    def _save_artifacts(
        self,
        vectorizer: TfidfVectorizer,
        sim_matrix: np.ndarray,
        product_index: Dict[str, Any],
        df: pd.DataFrame,
        tfidf_matrix: Any,
        dataset_version: str,
        training_duration: float,
    ) -> None:
        """Save all model artifacts to disk."""
        artifact_dir = Path(self.config.artifacts_dir)
        create_directories([artifact_dir])

        # 1. TF-IDF vectorizer
        with open(self.config.tfidf_model_path, "wb") as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved vectorizer: {self.config.tfidf_model_path}")

        # 2. Similarity matrix
        with open(self.config.similarity_matrix_path, "wb") as f:
            pickle.dump(sim_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved similarity matrix: {self.config.similarity_matrix_path}")

        # 3. Product index
        with open(self.config.product_index_path, "wb") as f:
            pickle.dump(product_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved product index: {self.config.product_index_path}")

        # 4. Processed products CSV
        processed_path = artifact_dir / "processed_products.csv"
        df.to_csv(processed_path, index=False, encoding="utf-8")
        logger.info(f"Saved processed products: {processed_path}")

        # 5. Compute top TF-IDF terms (by mean score across documents)
        feature_names = vectorizer.get_feature_names_out()
        # Use mean of non-zero TF-IDF scores as term importance
        try:
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_term_indices = mean_scores.argsort()[::-1][:30]
            top_tfidf_terms = [
                {
                    "term": feature_names[i],
                    "importance": round(float(mean_scores[i]), 6),
                    "idf": round(float(vectorizer.idf_[i]), 4),
                }
                for i in top_term_indices
            ]
        except Exception:
            top_tfidf_terms = []

        # 6. Rich metadata JSON — matches the user spec
        vocab_sample = {k: int(v) for k, v in list(vectorizer.vocabulary_.items())[:100]}

        metadata = {
            # --- Core identity ---
            "model_version": "1.0.0",
            "training_date": get_iso_timestamp(),
            "dataset_hash": dataset_version,

            # --- Dataset summary ---
            "dataset_rows": int(df.shape[0]),
            "dataset_columns": int(df.shape[1]),

            # --- Vocabulary & model dimensions ---
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "matrix_shape": list(sim_matrix.shape),
            "matrix_dtype": str(sim_matrix.dtype),
            "matrix_size_mb": round(sim_matrix.nbytes / 1024 / 1024, 2),

            # --- Training performance ---
            "training_time": round(training_duration, 2),
            "training_duration_seconds": round(training_duration, 2),

            # --- Model configuration ---
            "similarity_threshold": 0.25,       # default retrieval threshold
            "vectorizer_parameters": {
                "ngram": list(vectorizer.ngram_range),
                "max_features": int(vectorizer.max_features or 0),
                "min_df": vectorizer.min_df,
                "max_df": vectorizer.max_df,
                "stop_words": vectorizer.stop_words,
                "sublinear_tf": vectorizer.sublinear_tf,
                "analyzer": vectorizer.analyzer,
            },

            # --- MLflow (populated after training) ---
            "mlflow_run": None,                  # updated after MLflow logging

            # --- Top features for report ---
            "top_tfidf_terms": top_tfidf_terms,
            "idf_sample": vectorizer.idf_.tolist()[:50],
            "vocabulary_sample": vocab_sample,

            # --- Artifact paths ---
            "artifact_paths": {
                "vectorizer": str(self.config.tfidf_model_path),
                "similarity_matrix": str(self.config.similarity_matrix_path),
                "product_index": str(self.config.product_index_path),
                "processed_products": str(processed_path),
            },
        }

        save_json(Path(self.config.metadata_path), metadata)
        logger.info(f"Saved metadata: {self.config.metadata_path}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initiate_model_training(self) -> ModelTrainingArtifact:
        """
        Execute the full TF-IDF model training pipeline.

        Steps:
            1. Load features.csv
            2. Fit TF-IDF vectorizer
            3. Compute cosine similarity matrix
            4. Build product index
            5. Save all artifacts
            6. Log to MLflow (if available)

        Returns:
            ModelTrainingArtifact with paths and metadata.
        """
        logger.info("=" * 60)
        logger.info("  MODEL TRAINING — Phase 3")
        logger.info("=" * 60)

        start_time = time.perf_counter()

        try:
            # Setup MLflow
            mlflow_available = self._setup_mlflow()

            # 1. Load data
            df = self._load_features()
            feature_path = Path(self.feature_artifact.feature_filepath)
            dataset_version = get_file_hash(feature_path)[:12]

            # 2. Fit TF-IDF
            logger.info("Step 1/4: Fitting TF-IDF vectorizer...")
            vectorizer, tfidf_matrix = self._fit_tfidf(df, self.config.tfidf)
            vocab_size = len(vectorizer.vocabulary_)

            # 3. Cosine similarity
            logger.info("Step 2/4: Computing cosine similarity matrix...")
            sim_matrix = self._compute_similarity(tfidf_matrix)

            # 4. Build product index
            logger.info("Step 3/4: Building product index...")
            product_index = self._build_product_index(df)

            # 5. Save artifacts
            logger.info("Step 4/4: Saving model artifacts...")
            training_duration = time.perf_counter() - start_time
            create_directories([Path(self.config.artifacts_dir)])
            self._save_artifacts(
                vectorizer=vectorizer,
                sim_matrix=sim_matrix,
                product_index=product_index,
                df=df,
                tfidf_matrix=tfidf_matrix,
                dataset_version=dataset_version,
                training_duration=training_duration,
            )

            # Prepare params and metrics for MLflow
            params = {
                "max_features": self.config.tfidf.max_features,
                "ngram_range": str(self.config.tfidf.ngram_range),
                "min_df": self.config.tfidf.min_df,
                "max_df": self.config.tfidf.max_df,
                "stop_words": self.config.tfidf.stop_words,
                "sublinear_tf": self.config.tfidf.sublinear_tf,
                "analyzer": self.config.tfidf.analyzer,
                "dataset_version": dataset_version,
                "num_products": len(df),
            }
            metrics = {
                "vocabulary_size": float(vocab_size),
                "num_products": float(len(df)),
                "matrix_size_mb": float(sim_matrix.nbytes / 1024 / 1024),
                "training_duration_seconds": round(training_duration, 3),
                "avg_similarity": float(np.mean(sim_matrix[sim_matrix > 0])),
            }

            # 6. Log to MLflow
            run_id, experiment_id = None, None
            if mlflow_available:
                run_id, experiment_id = self._log_to_mlflow(
                    vectorizer=vectorizer,
                    params=params,
                    metrics=metrics,
                    artifact_dir=Path(self.config.artifacts_dir),
                )

            # 7. Back-patch mlflow_run into metadata
            if run_id:
                meta_path = Path(self.config.metadata_path)
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["mlflow_run"] = run_id
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

            logger.info("=" * 60)
            logger.info(f"  MODEL TRAINING COMPLETE")
            logger.info(f"  Duration   : {training_duration:.2f}s")
            logger.info(f"  Vocabulary : {vocab_size:,} features")
            logger.info(f"  Products   : {len(df):,}")
            logger.info(f"  Matrix     : {sim_matrix.shape}")
            if run_id:
                logger.info(f"  MLflow Run : {run_id}")
            logger.info("=" * 60)

            # 8. Generate HTML training report
            try:
                report_path = generate_training_report(
                    metadata_path=self.config.metadata_path,
                    num_products=len(df),
                    mlflow_run_id=run_id,
                )
                logger.info(f"Training report: {report_path}")
            except Exception as e:
                logger.warning(f"HTML report generation failed (non-critical): {e}")

            return ModelTrainingArtifact(
                tfidf_model_path=str(self.config.tfidf_model_path),
                similarity_matrix_path=str(self.config.similarity_matrix_path),
                product_index_path=str(self.config.product_index_path),
                metadata_path=str(self.config.metadata_path),
                training_duration_seconds=round(training_duration, 3),
                vocabulary_size=vocab_size,
                matrix_shape=tuple(sim_matrix.shape),
                mlflow_run_id=run_id,
                mlflow_experiment_id=experiment_id,
                dataset_version=dataset_version,
            )

        except Exception as e:
            raise CustomException(
                f"ModelTrainer.initiate_model_training failed: {e}", sys.exc_info()
            ) from e

