
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI agent pro dairy farm data s vlastními "tools" (function calling).

CSV schéma (denní záznamy):
- date: YYYY-MM-DD
- cow_id: řetězec/ID krávy
- milk_kg: denní nádoj v kg
- fat_pct: obsah tuku v %
- protein_pct: obsah bílkovin v %
- feed_kg_dm: krmná dávka (sušina) v kg
- scc: somatické buňky (buňky/ml)
- bw_kg: hmotnost krávy v kg (volitelné)
- parity: pořadí laktace (volitelné)
- dim: days in milk (volitelné; když chybí, dopočítáme z prvního dne krávy)

Spuštění:
  1) zkopíruj .env.example na .env a doplň OPENAI_API_KEY
  2) pip install -r requirements.txt
  3) python agent_dairy_tools.py --csv demo_farm.csv "Shrň KPI stáda za 14 dní."
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
import requests
from math import radians, sin, cos, atan2, sqrt


# =======================
# Pomocné výpočty KPI
# =======================

def ensure_dataframe(csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Mini demo dataset (2 krávy, 14 dní)
        rng = pd.date_range("2025-08-01", periods=14, freq="D")
        data = []
        for cow in ["101", "102"]:
            base = 28 if cow == "101" else 24
            for i, d in enumerate(rng):
                milk = base + np.random.normal(0, 1.2) - 0.15*i  # mírný pokles
                fat = 4.0 + np.random.normal(0, 0.2)
                prot = 3.3 + np.random.normal(0, 0.1)
                feed = 20 + np.random.normal(0, 0.8)             # kg DM
                scc = int(150_000 + max(0, np.random.normal(0, 50_000)))
                data.append([d.date().isoformat(), cow, max(milk, 8), fat, prot, max(feed, 10), scc, 620, 2, i+1])
        df = pd.DataFrame(data, columns=["date","cow_id","milk_kg","fat_pct","protein_pct","feed_kg_dm","scc","bw_kg","parity","dim"])
    # typy
    df["date"] = pd.to_datetime(df["date"])
    for c in ["milk_kg","fat_pct","protein_pct","feed_kg_dm","scc","bw_kg","parity","dim"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # doplnění DIM pokud chybí
    if "dim" not in df.columns or df["dim"].isna().all():
        df = df.sort_values(["cow_id","date"])
        df["dim"] = df.groupby("cow_id")["date"].transform(lambda s: (s - s.min()).dt.days + 1)
    # přídavné sloupce
    df["ECM"] = df.apply(lambda r: ecm_kg(r["milk_kg"], r["fat_pct"], r["protein_pct"]), axis=1)
    df["FCE"] = df.apply(lambda r: fce(r["milk_kg"], r["feed_kg_dm"]), axis=1)
    df["FPR"] = df["fat_pct"] / df["protein_pct"]
    return df.dropna(subset=["cow_id","date"])

def ecm_kg(milk_kg: float, fat_pct: float, protein_pct: float) -> float:
    """Energy-Corrected Milk (aproximace ~ 3.5% fat)."""
    fat = fat_pct or 0.0
    prot = protein_pct or 0.0
    return float(milk_kg * (0.327 + 0.116 * fat + 0.06 * prot))

def fce(milk_kg: float, feed_kg_dm: float) -> Optional[float]:
    """Feed Conversion Efficiency ~ milk kg per kg DM."""
    if feed_kg_dm and feed_kg_dm > 0:
        return float(milk_kg / feed_kg_dm)
    return None

def rolling_slope(series: pd.Series, window: int = 3) -> float:
    """Lineární sklon (kg/den) přes poslední `window` dnů."""
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    idx = np.arange(len(s))
    if len(s) >= window:
        idx = idx[-window:]
        vals = s.values[-window:]
    else:
        vals = s.values
    coeffs = np.polyfit(idx, vals, 1)
    return float(coeffs[0])

def window_cut(df: pd.DataFrame, days: int) -> pd.DataFrame:
    end = df["date"].max()
    start = end - pd.Timedelta(days=days-1)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()

def zscores(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0]*len(x), index=x.index)
    return (x - mu) / sd

# =======================
# Lokální "tools" funkce
# =======================

GLOBAL_DF: Optional[pd.DataFrame] = None

def tool_farm_kpi(metric: str, cow_id: Optional[str] = None, window_days: int = 7) -> Dict[str, Any]:
    """
    KPI výpočet pro zadanou krávu nebo celé stádo v okně posledních N dní.
    metric ∈ {"ecm", "fce", "milk_avg", "scc_avg", "fat_pct_avg", "protein_pct_avg"}
    """
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    if cow_id:
        cut = cut[cut["cow_id"].astype(str) == str(cow_id)]
    if cut.empty:
        return {"metric": metric, "cow_id": cow_id, "window_days": window_days, "value": None, "note": "No data in window."}

    if metric == "ecm":
        val = cut["ECM"].mean()
        unit = "kg ECM/den"
    elif metric == "fce":
        val = cut["FCE"].dropna().mean()
        unit = "kg milk per kg DM"
    elif metric == "milk_avg":
        val = cut["milk_kg"].mean()
        unit = "kg/den"
    elif metric == "scc_avg":
        val = cut["scc"].mean()
        unit = "cells/ml"
    elif metric == "fat_pct_avg":
        val = cut["fat_pct"].mean()
        unit = "%"
    elif metric == "protein_pct_avg":
        val = cut["protein_pct"].mean()
        unit = "%"
    else:
        return {"error": f"Unknown metric '{metric}'."}

    trend = rolling_slope(cut["milk_kg"])
    end = df["date"].max()
    start = end - pd.Timedelta(days=window_days-1)
    return {
        "metric": metric, "cow_id": cow_id, "window_days": window_days,
        "value": None if val is None or (isinstance(val,float) and np.isnan(val)) else round(float(val), 3),
        "unit": unit, "milk_trend_kg_per_day": round(trend, 3),
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "n_days": int(cut["date"].nunique())
    }

def tool_alerts(cow_id: str) -> Dict[str, Any]:
    """
    Jednoduché alerty:
      - SCC > 200k (podezření na mastitidu)
      - FPR = fat/protein > 1.4 a DIM < 60 (riziko ketózy)
      - FCE < 1.2 (efektivita krmení nízká)
      - pokles mléka >1.0 kg/den za poslední 3 dny
    """
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    df = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if df.empty:
        return {"cow_id": cow_id, "alerts": [], "note": "No data."}

    recent = df.tail(7).copy()
    alerts: List[Dict[str, Any]] = []

    # SCC
    if recent["scc"].dropna().mean() > 200_000:
        alerts.append({"type": "SCC_HIGH", "message": f"Průměrné SCC posledních 7 dní ~ {int(recent['scc'].mean()):,}".replace(",", " "), "threshold": 200_000})

    # Ketóza (FPR & DIM)
    early = recent[recent["dim"] < 60]
    if not early.empty and (early["FPR"] > 1.4).mean() > 0.5:
        alerts.append({"type": "KETOSIS_RISK", "message": "FPR>1.4 u >50 % dní v rané laktaci (DIM<60).", "threshold": "FPR>1.4"})

    # FCE
    if recent["FCE"].notna().mean() > 0 and recent["FCE"].mean() < 1.2:
        alerts.append({"type": "LOW_FCE", "message": f"Průměrná FCE ~ {recent['FCE'].mean():.2f} (<1.2).", "threshold": 1.2})

    # Trend poklesu mléka
    slope = rolling_slope(recent["milk_kg"], window=3)
    if slope < -1.0:
        alerts.append({"type": "MILK_DROP", "message": f"Pokles mléka ~ {slope:.2f} kg/den za poslední 3 dny.", "threshold": -1.0})

    return {"cow_id": cow_id, "alerts": alerts, "days_covered": int(recent["date"].nunique())}

def tool_summarize_farm(window_days: int = 14) -> Dict[str, Any]:
    """
    Agregované KPI za stádo v okně posledních N dní:
      - průměr milk_kg, ECM, FCE, SCC
      - top5 a bottom5 krav dle milk_kg
    """
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    if cut.empty:
        return {"window_days": window_days, "note": "No data in window."}

    kpi = {
        "window": {"start": str(cut["date"].min().date()), "end": str(cut["date"].max().date())},
        "avg_milk_kg": round(cut["milk_kg"].mean(), 2),
        "avg_ecm_kg": round(cut["ECM"].mean(), 2),
        "avg_fce": None if cut["FCE"].dropna().empty else round(cut["FCE"].dropna().mean(), 3),
        "avg_scc": None if cut["scc"].dropna().empty else int(cut["scc"].mean()),
        "n_cows": int(cut["cow_id"].nunique()),
        "n_days": int(cut["date"].nunique())
    }

    by_cow = cut.groupby("cow_id")["milk_kg"].mean().sort_values(ascending=False)
    kpi["top5_milk_avg"] = by_cow.head(5).round(2).to_dict()
    kpi["bottom5_milk_avg"] = by_cow.tail(5).round(2).to_dict()
    return kpi

# ------- Nové nástroje (rozšíření) -------

def tool_cow_history(cow_id: str) -> Dict[str, Any]:
    """Jednoduchá časová osa: datum, DIM, milk_kg, poznámky (SCC, FPR v rané laktaci)."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    sub = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty:
        return {"cow_id": cow_id, "events": []}
    events = []
    for _, r in sub.iterrows():
        note = []
        if r.get("scc", np.nan) > 200_000:
            note.append("High SCC")
        if r.get("FPR", np.nan) > 1.4 and r.get("dim", 999) <= 60:
            note.append("FPR>1.4@early")
        events.append({
            "date": str(r["date"].date()),
            "dim": int(r["dim"]),
            "milk_kg": float(r["milk_kg"]),
            "note": "; ".join(note) if note else ""
        })
    return {"cow_id": cow_id, "events": events}

def tool_forecast_milk(cow_id: str, horizon_days: int = 7) -> Dict[str, Any]:
    """Lineární trendová predikce mléka (odstranění extrémů, bez záporných hodnot)."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    sub = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty:
        return {"cow_id": cow_id, "error": "No data for forecasting."}
    y = sub["milk_kg"].astype(float).values
    x = np.arange(len(y))
    # ořez outlierů (1.-99. percentil)
    q1, q99 = np.percentile(y, [1, 99])
    mask = (y >= q1) & (y <= q99)
    coef = np.polyfit(x[mask], y[mask], 1) if mask.sum() >= 2 else np.polyfit(x, y, 1)
    trend = np.poly1d(coef)
    x_future = np.arange(len(y), len(y)+horizon_days)
    forecast = np.maximum(0.0, trend(x_future)).tolist()
    return {
        "cow_id": cow_id,
        "horizon_days": int(horizon_days),
        "last_observed": float(y[-1]),
        "trend_slope": float(coef[0]),
        "forecast": [float(v) for v in forecast],
        "explain": "Affine trend forecast (least-squares) with outlier trimming."
    }

def tool_feed_optimizer(cow_id: str, target_milk: float, safety_margin: float = 0.05) -> Dict[str, Any]:
    """
    Heuristický návrh úpravy DM krmiva pro dosažení cílového nádoje.
    - použije průměr FCE a milk z posledních 7 dní
    - delta_dm je omezena bezpečnostní marží safety_margin * current_feed (min 0.5 kg)
    """
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    sub = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty:
        return {"cow_id": cow_id, "error": "No data for this cow."}
    recent = sub.tail(7)
    fce_mean = recent["FCE"].dropna().mean()
    current_milk = float(recent["milk_kg"].mean())
    current_feed = float(recent["feed_kg_dm"].mean())
    if np.isnan(fce_mean) or fce_mean <= 0:
        fce_mean = 1.2  # fallback
    needed_delta_milk = float(target_milk - current_milk)
    estimated_dm_change = needed_delta_milk / max(fce_mean, 1e-6)
    max_change = max(0.5, safety_margin * current_feed)
    estimated_dm_change = float(np.clip(estimated_dm_change, -max_change, max_change))

    advice = []
    if fce_mean < 1.0:
        advice.append("Nízká FCE — zvaž úpravu kvality KD (stravitelnost vlákniny, energetická hustota).")
    elif fce_mean > 1.6:
        advice.append("Velmi vysoká FCE — ověř kondici; vyhni se překrmování.")

    return {
        "cow_id": cow_id,
        "target_milk": float(target_milk),
        "current_milk_avg_7d": current_milk,
        "current_feed_avg_7d": current_feed,
        "recent_FCE": float(fce_mean),
        "suggested_feed_dm_delta": estimated_dm_change,
        "suggested_feed_dm_new": current_feed + estimated_dm_change,
        "advice": advice
    }

def tool_sustainability(window_days: int = 30) -> Dict[str, Any]:
    """
    Toy udržitelnost: ECM/feed, protein-efficiency proxy, CO2 proxy ~ 1.2*(1.2/FCE_mean).
    Výstup je 0-1 skóre + dílčí metriky.
    """
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    if cut.empty:
        return {"window_days": window_days, "error": "No data."}

    ecm_per_feed = float((cut["ECM"].sum() / cut["feed_kg_dm"].sum()) if cut["feed_kg_dm"].sum() else np.nan)
    prot_eff = float((cut["milk_kg"].mean() / cut["protein_pct"].mean()) if cut["protein_pct"].mean() else np.nan)
    fce_mean = cut["FCE"].replace([np.inf, -np.inf], np.nan).mean()
    carbon_index = float(1.2 if (pd.isna(fce_mean) or fce_mean <= 0) else 1.2 * (1.2 / fce_mean))

    def norm_pos(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0, 1))

    score = float(np.nanmean([
        norm_pos(ecm_per_feed, 0.8, 1.8),
        norm_pos(prot_eff, 5.0, 12.0),
        norm_pos(2.0 - carbon_index, 0.2, 2.0),
    ]))

    return {
        "window_days": int(window_days),
        "ecm_per_feed": ecm_per_feed,
        "protein_efficiency_proxy": prot_eff,
        "carbon_index_proxy": carbon_index,
        "sustainability_score_0to1": score
    }

def tool_outliers(window_days: int = 14, metric: str = "milk_kg", z_threshold: float = 2.0) -> Dict[str, Any]:
    """Najdi odlehlé krávy podle z-skóre z průměrů za okno (metric ∈ {'milk_kg','ECM','FCE','scc'})."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    if metric.lower() == "scc":
        g = cut.groupby("cow_id")["scc"].mean()
    else:
        col = metric if metric in cut.columns else metric.upper()
        if col not in cut.columns:
            return {"error": f"Metric '{metric}' not available."}
        g = cut.groupby("cow_id")[col].mean()
    z = zscores(g)
    hi = z[z >= z_threshold].sort_values(ascending=False).index.tolist()
    lo = z[z <= -z_threshold].sort_values().index.tolist()
    return {
        "window_days": window_days,
        "metric": metric,
        "high_outliers": hi,
        "low_outliers": lo,
        "explain": f"Outliers based on |z| >= {z_threshold} over cow means in window."
    }

def tool_compare_cows(cow_a: str, cow_b: str, window_days: int = 14) -> Dict[str, Any]:
    """Porovnej dvě krávy (milk, ECM, FCE, SCC) v okně."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    comp = {}
    for c in [cow_a, cow_b]:
        sub = cut[cut["cow_id"].astype(str) == str(c)]
        if sub.empty:
            comp[c] = {"error": "No data"}
        else:
            comp[c] = {
                "milk_avg": float(sub["milk_kg"].mean()),
                "ecm_avg": float(sub["ECM"].mean()),
                "fce_avg": None if sub["FCE"].dropna().empty else float(sub["FCE"].dropna().mean()),
                "scc_avg": None if sub["scc"].dropna().empty else float(sub["scc"].mean())
            }
    return {"window_days": window_days, "compare": comp}

def tool_what_if_feed(cow_id: str, delta_dm: float) -> Dict[str, Any]:
    """Jednoduché 'what-if': očekávaná změna mléka ~ FCE_mean * delta_dm (posledních 7 dní)."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    sub = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("date").tail(7)
    if sub.empty:
        return {"cow_id": cow_id, "error": "No data."}
    fce_mean = sub["FCE"].dropna().mean()
    if np.isnan(fce_mean) or fce_mean <= 0:
        fce_mean = 1.2
    milk_change = float(fce_mean * delta_dm)
    return {
        "cow_id": cow_id,
        "recent_FCE": float(fce_mean),
        "delta_dm": float(delta_dm),
        "expected_milk_delta": milk_change
    }

def tool_rank_cows(window_days: int = 14, by: str = "milk_kg", top: int = 10) -> Dict[str, Any]:
    """Seřaď krávy dle průměru metriky v okně (by ∈ {'milk_kg','ECM','FCE','scc'})."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    col = by if by in cut.columns else by.upper()
    if col not in cut.columns and by.lower() != "scc":
        return {"error": f"Metric '{by}' not available."}
    metric_series = cut["scc"] if by.lower() == "scc" else cut[col]
    ranks = metric_series.groupby(cut["cow_id"]).mean().sort_values(ascending=False)
    out = ranks.head(top).round(3).to_dict()
    return {"window_days": window_days, "by": by, "top": out}

def tool_scc_trend(cow_id: str, window_days: int = 30) -> Dict[str, Any]:
    """Trend SCC (sklon) a průměr SCC za okno."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    sub = cut[cut["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty or sub["scc"].dropna().empty:
        return {"cow_id": cow_id, "window_days": window_days, "error": "No SCC data."}
    slope = rolling_slope(sub["scc"].astype(float), window=min(7, len(sub)))
    return {"cow_id": cow_id, "window_days": window_days, "scc_avg": float(sub["scc"].mean()), "scc_trend_per_day": float(slope)}

def tool_variability(window_days: int = 14) -> Dict[str, Any]:
    """Variabilita ve stádě: koeficient variability (CV %) pro milk, ECM, FCE, SCC."""
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    cut = window_cut(df, window_days)
    def cv(series: pd.Series) -> Optional[float]:
        mu = series.mean()
        sd = series.std(ddof=0)
        if mu and mu != 0 and not np.isnan(mu) and not np.isnan(sd):
            return float(100.0 * sd / mu)
        return None
    return {
        "window_days": window_days,
        "cv_milk_pct": cv(cut["milk_kg"]),
        "cv_ecm_pct": cv(cut["ECM"]),
        "cv_fce_pct": cv(cut["FCE"].dropna()),
        "cv_scc_pct": cv(cut["scc"].dropna())
    }

# ---------- NEW: Heat stress (THI) ----------
def tool_thi_heat_stress(t_avg_c: float, rh_pct: float) -> Dict[str, Any]:
    """
    Výpočet THI (Temperature-Humidity Index) a klasifikace rizika.
    Vstupy: průměrná denní teplota [°C], relativní vlhkost [%].
    Použitá aproximace (classical dairy THI):
      THI = T - (0.55 - 0.0055*RH)*(T - 14.5)
    """
    try:
        T = float(t_avg_c)
        RH = float(rh_pct)
        thi = T - (0.55 - 0.0055 * RH) * (T - 14.5)
        # velmi zjednodušené prahy dle praxe
        if thi < 68:
            risk = "LOW"
            advice = ["Bez mimořádných opatření."]
        elif thi < 72:
            risk = "MODERATE"
            advice = ["Zajistit stín, napáječky, lehké větrání."]
        elif thi < 78:
            risk = "HIGH"
            advice = ["Ventilátory, sprchy, posun dojení/krmení na chladnější hodiny."]
        else:
            risk = "SEVERE"
            advice = ["Intenzivní chlazení, prevence dehydratace, úprava KD (více energie/Na/K)."]
        return {"t_avg_c": T, "rh_pct": RH, "thi": round(float(thi), 2), "risk": risk, "advice": advice}
    except Exception as e:
        return {"error": f"THI calculation failed: {e}"}

# ---------- NEW: Laktační rezidua vůči Woodově křivce ----------
def tool_lactation_residuals(cow_id: str, window_days: int = 60) -> Dict[str, Any]:
    """
    Fit log-lineární aproximaci Woodovy křivky: ln(y) = ln(a) + b ln(t) - c t
    - Bez SciPy: řešíme pomocí np.linalg.lstsq (OLS).
    Výstup: průměrné reziduum (kg), seznam posledních 10 reziduí, jednoduché hodnocení.
    """
    assert GLOBAL_DF is not None
    df = window_cut(GLOBAL_DF.copy(), window_days)
    sub = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("dim")
    if sub.empty:
        return {"cow_id": cow_id, "window_days": window_days, "error": "No data."}

    # Použij DIM jako t, ignoruj t<=0 a y<=0
    sub = sub[(sub["dim"] > 0) & (sub["milk_kg"] > 0)].copy()
    if sub.empty:
        return {"cow_id": cow_id, "window_days": window_days, "error": "Insufficient data for fit."}

    t = sub["dim"].astype(float).values
    y = sub["milk_kg"].astype(float).values
    ln_y = np.log(y)
    X = np.column_stack([np.ones_like(t), np.log(t), -t])  # [1, ln t, -t]
    try:
        beta, *_ = np.linalg.lstsq(X, ln_y, rcond=None)  # beta = [ln a, b, c]
        ln_a, b, c = beta
        a = np.exp(ln_a)
        y_hat = a * (t ** b) * np.exp(-c * t)
        resid = y - y_hat
        mean_resid = float(np.nanmean(resid))
        last_resid = [float(r) for r in resid[-10:]]
        flag = "OK"
        note = []
        if mean_resid < -1.0:
            flag = "NEGATIVE"
            note.append("Podvýkon proti křivce (ověř metaboliku/krmení).")
        elif mean_resid > 1.0:
            flag = "POSITIVE"
            note.append("Nadvýkon proti křivce (zvaž úpravy k prevenci acidózy).")
        return {
            "cow_id": cow_id,
            "window_days": int(window_days),
            "wood_params": {"a": float(a), "b": float(b), "c": float(c)},
            "mean_residual_kg": round(mean_resid, 3),
            "last10_residuals_kg": last_resid,
            "flag": flag,
            "note": note
        }
    except Exception as e:
        return {"cow_id": cow_id, "error": f"Wood fit failed: {e}"}

# ---------- NEW: Mastitis score (syntetické) ----------
def tool_mastitis_score(cow_id: str, window_days: int = 30) -> Dict[str, Any]:
    """
    Spojí úroveň a sklon SCC + variabilitu mléka:
      - base_score z prům. SCC (0..1)
      - trend_penalty z kladného sklonu SCC
      - milk_var_penalty z vyšší variability mléka
    Celkové skóre 0..1 (vyšší = horší). Heuristika, nikoli diagnóza.
    """
    assert GLOBAL_DF is not None
    cut = window_cut(GLOBAL_DF.copy(), window_days)
    sub = cut[cut["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty or sub["scc"].dropna().empty:
        return {"cow_id": cow_id, "error": "No SCC data."}

    scc_mean = float(sub["scc"].mean())
    scc_slope = rolling_slope(sub["scc"].astype(float), window=min(7, len(sub)))
    milk_cv = float(100.0 * (sub["milk_kg"].std(ddof=0) / sub["milk_kg"].mean())) if sub["milk_kg"].mean() else 0.0

    def clamp01(x): return float(np.clip(x, 0.0, 1.0))
    base = clamp01((scc_mean - 100_000) / (500_000 - 100_000))  # 100k→0, 500k→1
    trend_penalty = clamp01(scc_slope / 20_000.0) if scc_slope > 0 else 0.0
    milk_var_penalty = clamp01((milk_cv - 5.0) / 20.0)  # CV>5% penalizace

    score = clamp01(0.7 * base + 0.2 * trend_penalty + 0.1 * milk_var_penalty)
    label = "LOW"
    advice = []
    if score >= 0.7:
        label = "HIGH"
        advice = ["Okamžitě strip-test, CMT, kontrola hygieny dojení, případně cílená léčba."]
    elif score >= 0.4:
        label = "MEDIUM"
        advice = ["Zvýšené sledování SCC, kontrola podmínek, předzásahové testy."]

    return {
        "cow_id": cow_id,
        "window_days": int(window_days),
        "scc_mean": int(scc_mean),
        "scc_slope_per_day": float(scc_slope),
        "milk_cv_pct": round(milk_cv, 2),
        "mastitis_score_0to1": score,
        "risk": label,
        "advice": advice
    }

# ---------- NEW: Ekonomika zásahu (ROI) ----------
def tool_economic_roi(cow_id: str, delta_dm: float, milk_price_czk_per_kg: float, feed_price_czk_per_kgdm: float) -> Dict[str, Any]:
    """
    Spočítá oček. ekonomiku změny krmiva:
      Δmléka ≈ recent_FCE * delta_dm
      přínos = Δmléka * cena_mléka
      náklad = delta_dm * cena_krmiva
      ROI = (přínos - náklad) / max(náklad, eps)
    """
    assert GLOBAL_DF is not None
    sub = GLOBAL_DF.copy()
    sub = sub[sub["cow_id"].astype(str) == str(cow_id)].sort_values("date").tail(7)
    if sub.empty:
        return {"cow_id": cow_id, "error": "No data (last 7 days)."}

    fce_mean = sub["FCE"].dropna().mean()
    if np.isnan(fce_mean) or fce_mean <= 0:
        fce_mean = 1.2
    d_milk = float(fce_mean * float(delta_dm))
    benefit = d_milk * float(milk_price_czk_per_kg)
    cost = float(delta_dm) * float(feed_price_czk_per_kgdm)
    roi = (benefit - cost) / max(cost, 1e-6)
    payback = None
    if d_milk > 0 and benefit > cost:
        # jednoduchá “doba návratnosti” v dnech, pokud počítáme denní efekt
        payback = cost / max(benefit, 1e-6)
    return {
        "cow_id": cow_id,
        "recent_FCE": float(fce_mean),
        "delta_dm": float(delta_dm),
        "expected_delta_milk": float(d_milk),
        "benefit_czk_per_day": float(benefit),
        "cost_czk_per_day": float(cost),
        "roi": float(roi),
        "simple_payback_days": None if payback is None else float(payback)
    }

# ---------- NEW: Dry-off planner ----------
def tool_dryoff_planner(cow_id: str, target_dim: int = 305, scc_threshold: int = 200_000) -> Dict[str, Any]:
    """
    Návrh zasušení: kolik dní zbývá do target DIM, stav SCC, doporučení před zasušením.
    """
    assert GLOBAL_DF is not None
    sub = GLOBAL_DF.copy()
    sub = sub[sub["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty:
        return {"cow_id": cow_id, "error": "No data."}

    cur_dim = int(sub["dim"].iloc[-1])
    days_left = target_dim - cur_dim
    recent_scc = None if sub["scc"].dropna().empty else float(sub["scc"].tail(7).mean())
    advice = []
    if days_left <= 0:
        advice.append("DIM ≥ target – zvaž okamžité zasušení dle kondice a produkce.")
    elif days_left < 21:
        advice.append("Příprava na zasušení: kontrola SCC, kondice, úprava KD (méně energie).")
    else:
        advice.append("Standardní režim; plánuj kontrolu SCC 2–3 týdny před zasušením.")
    if recent_scc and recent_scc > scc_threshold:
        advice.append("SCC nad prahem – zvaž selektivní antibiotika/pečlivý dry-cow protokol.")
    return {
        "cow_id": cow_id,
        "current_dim": cur_dim,
        "target_dim": int(target_dim),
        "days_to_dryoff": int(days_left),
        "recent_scc_mean": None if recent_scc is None else int(recent_scc),
        "advice": advice
    }

# ---------- NEW: Reproduction window ----------
def tool_repro_window(cow_id: str, vwp_min: int = 45, vwp_max: int = 60) -> Dict[str, Any]:
    """
    Doporučení pro servisní okno na základě DIM a stability produkce:
    - pokud DIM v [vwp_min, vwp_max], hlásí “servisní okno otevřené”
    - pokud vysoká variabilita mléka nebo negativní trend, přidá upozornění
    """
    assert GLOBAL_DF is not None
    sub = GLOBAL_DF.copy()
    sub = sub[sub["cow_id"].astype(str) == str(cow_id)].sort_values("date")
    if sub.empty:
        return {"cow_id": cow_id, "error": "No data."}

    cur_dim = int(sub["dim"].iloc[-1])
    status = "TOO_EARLY" if cur_dim < vwp_min else ("OPEN" if cur_dim <= vwp_max else "OPEN_POST_VWP")
    slope = rolling_slope(sub["milk_kg"].astype(float), window=min(7, len(sub)))
    milk_cv = float(100.0 * (sub["milk_kg"].std(ddof=0) / sub["milk_kg"].mean())) if sub["milk_kg"].mean() else 0.0
    flags = []
    if slope < -0.8:
        flags.append("negativní trend mléka (stres/metabolika?)")
    if milk_cv > 12.0:
        flags.append("vysoká variabilita mléka (ověř zdraví/kondici)")
    advice = []
    if status.startswith("OPEN"):
        advice.append("Plánovat detekci říje a inseminaci v nejbližším cyklu.")
    elif status == "TOO_EARLY":
        advice.append("Počkej do VWP; sleduj kondici a příjem.")
    return {
        "cow_id": cow_id,
        "dim": cur_dim,
        "vwp_min": int(vwp_min),
        "vwp_max": int(vwp_max),
        "milk_trend_kg_per_day": round(float(slope), 3),
        "milk_cv_pct": round(float(milk_cv), 2),
        "status": status,
        "flags": flags,
        "advice": advice
    }

# ---------- NEW: Milk quality bonus simulator ----------
def tool_milk_quality_bonus(window_days: int = 14, pricing_table: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Simulace bonusů/srážek za kvalitu (SCC, tuk, bílkovina).
    pricing_table (příklad):
    {
      "scc":   [{"max":200000, "delta_per_kg": +0.30}, {"max":400000, "delta_per_kg": 0.0}, {"max": 9999999, "delta_per_kg": -0.40}],
      "fat":   [{"min":3.8, "delta_per_kg": +0.10}],
      "protein":[{"min":3.4, "delta_per_kg": +0.10}]
    }
    Výstup: delta ceny na kg, oček. bonus/srážka pro průměrné parametry v okně.
    """
    assert GLOBAL_DF is not None
    df = window_cut(GLOBAL_DF.copy(), window_days)
    if df.empty:
        return {"window_days": window_days, "error": "No data."}

    if pricing_table is None:
        pricing_table = {
            "scc":   [{"max": 200000, "delta_per_kg": 0.30},
                      {"max": 400000, "delta_per_kg": 0.00},
                      {"max": 9999999, "delta_per_kg": -0.40}],
            "fat":   [{"min": 3.8, "delta_per_kg": 0.10}],
            "protein":[{"min": 3.4, "delta_per_kg": 0.10}],
        }

    avg_scc = float(df["scc"].mean()) if "scc" in df.columns else np.nan
    avg_fat = float(df["fat_pct"].mean()) if "fat_pct" in df.columns else np.nan
    avg_prot = float(df["protein_pct"].mean()) if "protein_pct" in df.columns else np.nan
    avg_milk = float(df["milk_kg"].mean()) if "milk_kg" in df.columns else 0.0

    delta = 0.0
    # SCC pravidla (první prah, který vyhoví)
    if not np.isnan(avg_scc) and "scc" in pricing_table:
        for rule in pricing_table["scc"]:
            if avg_scc <= float(rule.get("max", 1e9)):
                delta += float(rule.get("delta_per_kg", 0.0))
                break
    # Tuk / bílkovina (akumulativní bonusy)
    if not np.isnan(avg_fat) and "fat" in pricing_table:
        for rule in pricing_table["fat"]:
            if avg_fat >= float(rule.get("min", 0.0)):
                delta += float(rule.get("delta_per_kg", 0.0))
    if not np.isnan(avg_prot) and "protein" in pricing_table:
        for rule in pricing_table["protein"]:
            if avg_prot >= float(rule.get("min", 0.0)):
                delta += float(rule.get("delta_per_kg", 0.0))

    # odhad denní změny tržeb pro prům. krávu
    revenue_delta_per_cow_day = delta * avg_milk
    return {
        "window_days": int(window_days),
        "avg_scc": None if np.isnan(avg_scc) else int(avg_scc),
        "avg_fat_pct": None if np.isnan(avg_fat) else round(avg_fat, 2),
        "avg_protein_pct": None if np.isnan(avg_prot) else round(avg_prot, 2),
        "avg_milk_kg": round(avg_milk, 2),
        "price_delta_czk_per_kg": round(delta, 3),
        "revenue_delta_czk_per_cow_day": round(revenue_delta_per_cow_day, 2),
        "assumptions": "Bonusy aplikovány na průměrné parametry okna; linearita po kg."
    }

# ---------- NEW: Welfare score (proxy) ----------
def tool_welfare_score(window_days: int = 14, cow_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Syntetické welfare skóre 0..1:
      - nízká variabilita (CV) a stabilní trend mléka zlepšují skóre
      - nižší SCC a nepřítomnost alertů zlepšují skóre
    Heuristika pro triáž, ne audit.
    """
    assert GLOBAL_DF is not None
    df = window_cut(GLOBAL_DF.copy(), window_days)
    if cow_id:
        df = df[df["cow_id"].astype(str) == str(cow_id)]
    if df.empty:
        return {"window_days": window_days, "cow_id": cow_id, "error": "No data."}

    # agregace po krávě → průměry + CV + trend
    results = {}
    for cid, sub in df.groupby("cow_id"):
        sub = sub.sort_values("date")
        cv_milk = sub["milk_kg"].std(ddof=0) / sub["milk_kg"].mean() * 100 if sub["milk_kg"].mean() else np.nan
        slope = rolling_slope(sub["milk_kg"], window=min(7, len(sub)))
        scc_mean = sub["scc"].mean() if "scc" in sub.columns else np.nan

        # normalizace na 0..1 (lepší → vyšší score)
        def npos(x, lo, hi):
            return float(np.clip((x - lo) / (hi - lo), 0, 1))
        milk_stability = 1.0 - npos(cv_milk if not np.isnan(cv_milk) else 15.0, 5.0, 25.0)
        trend_ok = npos(slope, -1.0, 0.8)         # lepší než -1 kg/d až +0.8 kg/d
        scc_ok = 1.0 - npos(scc_mean if not np.isnan(scc_mean) else 300_000, 150_000, 600_000)

        score = float(np.nanmean([milk_stability, trend_ok, scc_ok]))
        results[str(cid)] = {
            "cv_milk_pct": None if np.isnan(cv_milk) else round(float(cv_milk), 2),
            "milk_trend_kg_per_day": round(float(slope), 3),
            "scc_avg": None if np.isnan(scc_mean) else int(scc_mean),
            "welfare_score_0to1": round(score, 3)
        }

    if cow_id:
        return {"window_days": int(window_days), "cow_id": str(cow_id), **results.get(str(cow_id), {})}
    return {"window_days": int(window_days), "welfare": results}

# ---------- WEATHER FORECAST (Open-Meteo) ----------
def tool_forecast_open_meteo(lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """
    Předpověď počasí pro dané souřadnice (Open-Meteo) + výpočet THI a rizikové kategorie na den.
    Bez API klíče. Vrací denní T_max, T_min, RH_mean (odvozená), THI_mean a risk ("LOW/MOD/HIGH/SEVERE").
    """
    try:
        days = int(days)
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "daily": ["temperature_2m_max","temperature_2m_min","relative_humidity_2m_mean"],
            "timezone": "auto",
            "forecast_days": days
        }
        # Open-Meteo akceptuje "daily=var1,var2,..." i víckrát; sjednotíme
        params["daily"] = ",".join(params["daily"])

        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("daily", {})
        out = []
        for i, d in enumerate(data.get("time", [])):
            tmax = data.get("temperature_2m_max", [None]*len(data.get("time", [])))[i]
            tmin = data.get("temperature_2m_min", [None]*len(data.get("time", [])))[i]
            rh   = data.get("relative_humidity_2m_mean", [None]*len(data.get("time", [])))[i]
            # denní průměr T jako (Tmax+Tmin)/2; RH průměr poskytnut přímo
            if tmax is None or tmin is None or rh is None:
                continue
            t_avg = (float(tmax) + float(tmin)) / 2.0
            # THI aproximace (dairy industry classic):
            thi = t_avg - (0.55 - 0.0055*float(rh))*(t_avg - 14.5)
            if   thi < 68:  risk = "LOW"
            elif thi < 72:  risk = "MODERATE"
            elif thi < 78:  risk = "HIGH"
            else:           risk = "SEVERE"
            out.append({
                "date": d,
                "t_avg_c": round(t_avg,2),
                "t_max_c": float(tmax),
                "t_min_c": float(tmin),
                "rh_mean_pct": float(rh),
                "thi_mean": round(float(thi),2),
                "risk": risk
            })
        counts = {"LOW":0,"MODERATE":0,"HIGH":0,"SEVERE":0}
        for row in out: counts[row["risk"]] += 1
        return {"lat": float(lat), "lon": float(lon), "n_days": len(out), "days": out, "risk_counts": counts}
    except Exception as e:
        return {"error": f"forecast_open_meteo failed: {e}"}

# ---------- HAVERSINE distance ----------
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return float(R * 2 * atan2(sqrt(a), sqrt(1-a)))

# ---------- SUPPLIER RANK: nearest by distance ----------
def tool_nearest_suppliers(farm_lat: float, farm_lon: float, suppliers: List[Dict[str, Any]], top: int = 5) -> Dict[str, Any]:
    """
    Seřadí odběratele podle vzdálenosti od farmy.
    suppliers = [{ "id":"madeta_cb", "name":"MADETA a.s. (ČB)", "lat":48.974, "lon":14.474, "base_price":10.8, "bonus_rules":{...}, "price_url":null }, ...]
    """
    try:
        ranked = []
        for s in suppliers or []:
            d = _haversine_km(float(farm_lat), float(farm_lon), float(s["lat"]), float(s["lon"]))
            ranked.append({**s, "distance_km": round(d, 2)})
        ranked.sort(key=lambda x: x["distance_km"])
        return {"farm": {"lat": float(farm_lat), "lon": float(farm_lon)}, "suppliers_by_distance": ranked[:int(top)], "n_suppliers": len(ranked)}
    except Exception as e:
        return {"error": f"nearest_suppliers failed: {e}"}

# ---------- SUPPLIER PRICE: parse from catalog or fallback ----------
def tool_supplier_price(supplier: Dict[str, Any], fallback_czk_per_kg: Optional[float] = None) -> Dict[str, Any]:
    """
    Vrátí "aktuální" cenu daného odběratele:
      - pokud supplier obsahuje 'price_czk_per_kg' → použij
      - jinak zkus 'price_url' a 'parser' (jednoduché GET + regex/JSON keys)
      - jinak fallback (např. národní průměr z CLAL zadaný ručně)
    Pozn.: Veřejné farm-gate ceny publikuje málokterý odběratel → doporučuji udržovat ručně v katalogu.
    """
    try:
        # 1) fixní hodnota v katalogu
        if "price_czk_per_kg" in supplier and supplier["price_czk_per_kg"] is not None:
            return {"supplier_id": supplier.get("id"), "price_czk_per_kg": float(supplier["price_czk_per_kg"]), "source": "catalog"}
        # 2) jednoduchý fetch (pokud je price_url + parser)
        url = supplier.get("price_url")
        parser = supplier.get("parser")  # {"type":"json","path":["data","price_czk"]} or {"type":"regex","pattern":"Price:(\\d+\\.\\d+)"}
        if url and parser:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            if parser.get("type") == "json":
                j = resp.json()
                node = j
                for key in parser.get("path", []):
                    node = node[key]
                price = float(node)
                return {"supplier_id": supplier.get("id"), "price_czk_per_kg": price, "source": "http-json"}
            elif parser.get("type") == "regex":
                import re
                m = re.search(parser.get("pattern",""), resp.text)
                if m:
                    price = float(m.group(1))
                    return {"supplier_id": supplier.get("id"), "price_czk_per_kg": price, "source": "http-regex"}
        # 3) fallback (např. měsíční průměr ČR doplněný manuálně)
        if fallback_czk_per_kg is not None:
            return {"supplier_id": supplier.get("id"), "price_czk_per_kg": float(fallback_czk_per_kg), "source": "fallback"}
        return {"supplier_id": supplier.get("id"), "error": "No price available"}
    except Exception as e:
        return {"supplier_id": supplier.get("id"), "error": f"supplier_price failed: {e}"}

# ---------- BEST SUPPLIER: price + transport + bonusy ----------
def tool_best_supplier(farm_lat: float, farm_lon: float,
                       milk_qty_kg_per_day: float,
                       transport_cost_czk_per_km: float,
                       suppliers: List[Dict[str, Any]],
                       fallback_price_czk_per_kg: Optional[float] = None) -> Dict[str, Any]:
    """
    Vybere nejlepšího odběratele dle čisté denní tržby:
      net = (base_price + bonuses) * qty - (distance_km * transport_cost)
    Katalog položky:
      { id, name, lat, lon, base_price?, bonus_rules? (dict), price_url?, parser? }
    bonus_rules je volitelný slovník s konstantním příplatkem např. {"scc_bonus":0.15, "protein_bonus":0.10}
    """
    try:
        results = []
        for s in suppliers or []:
            dist = _haversine_km(float(farm_lat), float(farm_lon), float(s["lat"]), float(s["lon"]))
            # 1) cena dodavatele
            pinfo = tool_supplier_price(s, fallback_czk_per_kg=fallback_price_czk_per_kg)
            price = pinfo.get("price_czk_per_kg")
            if price is None:
                continue
            # 2) bonusy (jednoduchý součet, můžeš později navázat na milk_quality_bonus)
            bonus = 0.0
            br = s.get("bonus_rules") or {}
            for _, val in br.items():
                try: bonus += float(val)
                except: pass
            # 3) tržba – doprava
            gross = (float(price) + bonus) * float(milk_qty_kg_per_day)
            cost  = float(dist) * float(transport_cost_czk_per_km)
            net   = gross - cost
            results.append({
                "supplier_id": s.get("id"), "name": s.get("name"),
                "distance_km": round(dist,2),
                "price_czk_per_kg": float(price),
                "bonus_czk_per_kg": round(bonus, 3),
                "gross_revenue_czk_per_day": round(gross, 2),
                "transport_cost_czk_per_day": round(cost, 2),
                "net_revenue_czk_per_day": round(net, 2),
                "price_source": pinfo.get("source")
            })
        results.sort(key=lambda r: r["net_revenue_czk_per_day"], reverse=True)
        return {
            "farm": {"lat": float(farm_lat), "lon": float(farm_lon), "milk_qty_kg_per_day": float(milk_qty_kg_per_day)},
            "transport_cost_czk_per_km": float(transport_cost_czk_per_km),
            "ranked": results,
            "best": (results[0] if results else None)
        }
    except Exception as e:
        return {"error": f"best_supplier failed: {e}"}

# ---------- NEW: Microbiome-driven precision feeding ----------
def tool_microbiome_precision_feed(cow_id: str, profile: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Mikrobiom řízené krmení (heuristika).
    profile = { "cellulolytic":0..1, "amylolytic":0..1, "lactate_utilizers":0..1, "methanogens":0..1, "protozoa":0..1 }
    Výstup: skóre dysbiózy 0..1, oček. ΔFCE, návrhy doplňků/úprav KD.
    """
    prof = profile or {}
    cel = float(prof.get("cellulolytic", np.nan))
    amy = float(prof.get("amylolytic", np.nan))
    lac = float(prof.get("lactate_utilizers", np.nan))
    met = float(prof.get("methanogens", np.nan))
    pro = float(prof.get("protozoa", np.nan))

    signals, advice = [], []
    dys = 0.0

    if not np.isnan(cel) and cel < 0.25:
        dys += 0.3; signals.append("low_cellulolytic")
        advice += ["Zvaž kvasinkové kultury / živé kvasinky (podpora trávení vlákniny)."]
    if not np.isnan(amy) and not np.isnan(cel) and amy > 0.45 and cel < 0.3:
        dys += 0.25; signals.append("amy_over_cellulose")
        advice += ["Kontrola škrobu, přidej strukturální vlákninu; pufr (NaHCO3)."]
    if not np.isnan(lac) and lac < 0.2 and amy > 0.4:
        dys += 0.15; signals.append("low_lactate_utilizers")
        advice += ["Riziko acidózy – zvaž pomalejší škrob, iontové pufry."]
    if not np.isnan(met) and met > 0.18:
        dys += 0.15; signals.append("high_methanogens")
        advice += ["Cíl ke snížení metanu (např. specifické feed-additivum / tuková korekce)."]
    if not np.isnan(pro) and pro > 0.25 and cel < 0.3:
        dys += 0.1; signals.append("high_protozoa_low_fiber")
        advice += ["Zkontroluj jemnost TMR, vyvážit NDF/škrob."]

    dys = float(np.clip(dys, 0.0, 1.0))
    expected_delta_fce = round(0.05 + 0.20 * dys, 3)  # hrubý odhad zlepšení FCE
    carbon_index_delta = round(-0.1 * dys, 3)         # lepší FCE → nižší proxy uhlíku

    return {
        "cow_id": str(cow_id),
        "microbiome_profile": prof,
        "dysbiosis_score_0to1": dys,
        "expected_delta_FCE": expected_delta_fce,
        "expected_delta_carbon_index_proxy": carbon_index_delta,
        "signals": signals,
        "advice": sorted(set(advice))
    }

# ---------- NEW: Behavior & welfare anomalies ----------
def tool_behavior_anomalies(cow_id: str, sensor: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detekce odchylek chování z časové řady.
    sensor = [{"ts":"YYYY-MM-DD", "rumination_min":..., "lying_min":..., "steps":..., "vocal_high":...}, ...]
    Pracuje s poslední hodnotou vs. baseline (první 5–7 dnů nebo celý průměr), z-skóre, alerty.
    """
    if not sensor:
        return {"cow_id": cow_id, "error": "No sensor data."}
    df = pd.DataFrame(sensor).copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
    cols = [c for c in ["rumination_min","lying_min","steps","vocal_high"] if c in df.columns]
    if not cols:
        return {"cow_id": cow_id, "error": "Missing sensor metrics."}

    base_n = min(7, max(3, len(df)//2))
    base = df.iloc[:base_n][cols].astype(float)
    cur = df.iloc[-1][cols].astype(float)

    mu = base.mean()
    sd = base.std(ddof=0).replace(0, np.nan)
    z = (cur - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    alerts = []
    if "rumination_min" in z and z["rumination_min"] < -1.5:
        alerts.append("RUMINATION_DROP")
    if "lying_min" in z and z["lying_min"] > 1.5:
        alerts.append("LYING_SPIKE")
    if "steps" in z and z["steps"] < -1.5:
        alerts.append("ACTIVITY_DROP")
    if "vocal_high" in z and z["vocal_high"] > 1.5:
        alerts.append("DISTRESS_VOCAL")

    # welfare skóre 0..1 (1=dobré)
    penalty = 0.0
    for a in alerts:
        penalty += {"RUMINATION_DROP":0.3, "LYING_SPIKE":0.25, "ACTIVITY_DROP":0.2, "DISTRESS_VOCAL":0.25}.get(a, 0.1)
    welfare = float(np.clip(1.0 - penalty, 0.0, 1.0))

    return {
        "cow_id": str(cow_id),
        "z_scores": {k: round(float(v),2) for k,v in z.items()},
        "alerts": alerts,
        "welfare_score_0to1": welfare,
        "baseline_n_days": int(base_n),
        "notes": ["Heuristika; interpretovat spolu s klinickým vyšetřením."]
    }

# ---------- NEW: Early disease fusion (multi-signal) ----------
def tool_early_disease_alert(cow_id: str, health_ts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Fúze signálů pro včasné varování (mastitis/ketóza/pyrexie).
    health_ts volitelně: [{"ts":..., "temp_c":..., "milk_conductivity":..., "feed_intake_dm":...}, ...]
    Využije také GLOBAL_DF (SCC, FPR, trend mléka, DIM).
    """
    assert GLOBAL_DF is not None
    df = GLOBAL_DF.copy()
    sub = df[df["cow_id"].astype(str) == str(cow_id)].sort_values("date").tail(14)
    if sub.empty:
        return {"cow_id": cow_id, "error": "No farm data."}

    # metriky z farm dat
    milk_slope = rolling_slope(sub["milk_kg"].astype(float), window=min(7, len(sub)))
    fpr_early = (sub[sub["dim"] <= 60]["FPR"] > 1.4).mean() if "FPR" in sub else 0.0
    scc_mean = float(sub["scc"].mean()) if "scc" in sub else np.nan
    scc_slope = rolling_slope(sub["scc"].astype(float), window=min(7, len(sub))) if "scc" in sub else 0.0

    # z externích health_ts (pokud jsou)
    temp_peak = cond = fi_drop = 0.0
    if health_ts:
        h = pd.DataFrame(health_ts).copy()
        if not h.empty:
            temp_peak = float(np.nanmax(h.get("temp_c", pd.Series([np.nan]))))
            cond = float(h.get("milk_conductivity", pd.Series([np.nan])).tail(1).mean())
            if "feed_intake_dm" in h.columns and len(h) >= 3:
                base = h["feed_intake_dm"].iloc[:-1].mean()
                cur  = h["feed_intake_dm"].iloc[-1]
                fi_drop = float((base - cur) / base) if base and not np.isnan(base) else 0.0

    # rizikové skóre (0..1)
    def npos(x, lo, hi): return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    mastitis = 0.0
    if not np.isnan(scc_mean): mastitis += npos(scc_mean, 200_000, 600_000)
    if cond: mastitis += npos(cond, 5.0, 6.5) * 0.6
    mastitis += npos(scc_slope, 0, 20_000) * 0.5
    mastitis = float(np.clip(mastitis, 0, 1))

    ketosis = float(np.clip(0.6 * fpr_early + 0.4 * npos(-milk_slope, 0.5, 2.0) + 0.3 * np.clip(fi_drop,0,1), 0, 1))

    pyrexia = float(np.clip(npos(temp_peak, 39.0, 40.2), 0, 1))

    suggestions = []
    if mastitis >= 0.5: suggestions += ["Okamžitě CMT/strip test; cílená kultivace dle protokolu."]
    if ketosis  >= 0.5: suggestions += ["BHBA/NEFA test; zvaž glukoplastické složky, monitor příjmu."]
    if pyrexia  >= 0.5: suggestions += ["Ověř teplotu, hydrataci, vyluč infekci."]

    return {
        "cow_id": str(cow_id),
        "risk_scores_0to1": {"mastitis": mastitis, "ketosis": ketosis, "pyrexia": pyrexia},
        "drivers": {
            "milk_slope_kg_per_day": round(float(milk_slope),3),
            "fpr_early_share": float(fpr_early),
            "scc_mean": None if np.isnan(scc_mean) else int(scc_mean),
            "scc_slope_per_day": float(scc_slope),
            "temp_peak_c": None if not temp_peak else float(temp_peak),
            "milk_conductivity": None if not cond else float(cond),
            "feed_intake_drop_ratio": round(float(fi_drop),3) if fi_drop else 0.0
        },
        "advice": suggestions or ["Continue monitoring; no high-risk fusion signal."]
    }

# ---------- NEW: Genomics-guided mating ----------
def tool_genomics_mating(cow_id: str,
                         cow_genome: Dict[str, float],
                         bulls: List[Dict[str, Any]],
                         weights: Optional[Dict[str, float]] = None,
                         max_inbreeding: float = 0.1,
                         top: int = 3) -> Dict[str, Any]:
    """
    Skórování býků pro danou krávu (vážený index + penalizace příbuznosti).
    cow_genome/bulls: klíče např. {"milk":, "fertility":, "mastitis_resist":, "longevity":, "feed_eff":}
    Každý býk může mít 'inbreeding' (0..1).
    """
    w = weights or {"milk":0.35, "fertility":0.25, "mastitis_resist":0.2, "longevity":0.1, "feed_eff":0.1}
    ranked = []
    for b in bulls or []:
        score = 0.0
        for k, wk in w.items():
            score += wk * float(b.get(k, 0.0))
        inb = float(b.get("inbreeding", 0.0))
        score -= 10.0 * max(0.0, inb - max_inbreeding)  # penalizace přes limit
        ranked.append({"bull_id": b.get("id"), "name": b.get("name"),
                       "score": round(float(score),3), "inbreeding": inb,
                       "traits": {k: b.get(k) for k in w.keys()}})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {"cow_id": str(cow_id), "weights": w, "top": ranked[:int(top)]}

# ---------- NEW: Weather stress window (proaktivní) ----------
def tool_weather_stress_alert(lat: float, lon: float, days: int = 7, thi_threshold: float = 72.0, min_consecutive: int = 2) -> Dict[str, Any]:
    """
    Volá lokální forecast (Open-Meteo) a najde "okna horka" (>= thi_threshold) v délce >= min_consecutive.
    Vrátí doporučení opatření.
    """
    fc = tool_forecast_open_meteo(lat, lon, days=days)
    if "days" not in fc:
        return {"error": "No forecast"}
    streaks, cur = [], []
    for d in fc["days"]:
        if d.get("thi_mean", 0) >= thi_threshold:
            cur.append(d)
        else:
            if len(cur) >= min_consecutive: streaks.append(cur)
            cur = []
    if len(cur) >= min_consecutive: streaks.append(cur)

    advice = []
    if streaks:
        advice += [
            "Plánuj chlazení/ventilaci, zvyšit dostupnost vody, elektrolyty.",
            "Posuň krmení/dojení do chladnějších hodin.",
            "Zkontroluj stín/rosení; citlivé krávy (vysoká DIM) prioritně."
        ]
    return {"lat": float(lat), "lon": float(lon), "thi_threshold": float(thi_threshold),
            "windows": streaks, "risk_counts": fc.get("risk_counts", {}), "advice": advice or ["Bez rizikových oken."]}

# ---------- NEW: Energy price scheduling (greedy) ----------
def tool_optimize_energy_schedule(price_series: List[Dict[str, Any]],
                                  tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Jednoduchý greedy plán dle hodinových cen.
    price_series = [{"hour":0..23, "price":Kč/kWh}, ...]
    tasks = [{"name":"cooling", "duration_h":3, "earliest":0, "latest":23, "power_kw":12.0}, ...]
    Neřeší konflikt zdrojů – cílem je najít nejlevnější hodiny v okně každého úkolu.
    """
    if not price_series or not tasks:
        return {"error":"Missing price_series or tasks"}
    prices = pd.DataFrame(price_series).copy()
    if "hour" not in prices or "price" not in prices: return {"error":"Bad price_series"}
    schedule = []
    total_cost = 0.0
    for t in tasks:
        name = t.get("name","task")
        dur  = int(t.get("duration_h",1))
        e    = int(t.get("earliest",0))
        l    = int(t.get("latest",23))
        pkw  = float(t.get("power_kw",1.0))
        window = prices[(prices["hour"]>=e) & (prices["hour"]<=l)].sort_values("price")
        chosen = window.head(dur)["hour"].tolist() if len(window)>=dur else window["hour"].tolist()
        cost = float(prices.set_index("hour").loc[chosen]["price"].sum() * pkw)
        total_cost += cost
        schedule.append({"name": name, "hours": chosen, "power_kw": pkw, "energy_cost_czk": round(cost,2)})
    return {"schedule": schedule, "total_energy_cost_czk": round(total_cost,2)}



# =======================
# OpenAI agent
# =======================

def build_tools_schema() -> List[Dict[str, Any]]:
    return [
           {"type":"function","function":{
                "name":"microbiome_precision_feed",
                "description":"Heuristické doporučení krmení dle mikrobiomu (ΔFCE, návrhy).",
                "parameters":{"type":"object","properties":{
                    "cow_id":{"type":"string"},
                    "profile":{"type":"object"}
                },"required":["cow_id"]}
            }},
            {"type":"function","function":{
                "name":"behavior_anomalies",
                "description":"Detekce odchylek chování a welfare skóre (ruminace, lehání, kroky, vokalizace).",
                "parameters":{"type":"object","properties":{
                    "cow_id":{"type":"string"},
                    "sensor":{"type":"array","items":{"type":"object"}}
                },"required":["cow_id","sensor"]}
            }},
            {"type":"function","function":{
                "name":"early_disease_alert",
                "description":"Fúze signálů pro včasné varování (mastitis/ketóza/pyrexie).",
                "parameters":{"type":"object","properties":{
                    "cow_id":{"type":"string"},
                    "health_ts":{"type":"array","items":{"type":"object"}}
                },"required":["cow_id"]}
            }},
            {"type":"function","function":{
                "name":"genomics_mating",
                "description":"Výběr býků dle genomických indexů a limitu příbuznosti.",
                "parameters":{"type":"object","properties":{
                    "cow_id":{"type":"string"},
                    "cow_genome":{"type":"object"},
                    "bulls":{"type":"array","items":{"type":"object"}},
                    "weights":{"type":"object"},
                    "max_inbreeding":{"type":"number","default":0.1},
                    "top":{"type":"integer","default":3}
                },"required":["cow_id","cow_genome","bulls"]}
            }},
            {"type":"function","function":{
                "name":"weather_stress_alert",
                "description":"Najde okna horka z předpovědi (THI) a navrhne opatření.",
                "parameters":{"type":"object","properties":{
                    "lat":{"type":"number"},
                    "lon":{"type":"number"},
                    "days":{"type":"integer","default":7},
                    "thi_threshold":{"type":"number","default":72.0},
                    "min_consecutive":{"type":"integer","default":2}
                },"required":["lat","lon"]}
            }},
            {"type":"function","function":{
                "name":"optimize_energy_schedule",
                "description":"Greedy plán úloh dle hodinových cen elektřiny.",
                "parameters":{"type":"object","properties":{
                    "price_series":{"type":"array","items":{"type":"object"}},
                    "tasks":{"type":"array","items":{"type":"object"}}
                },"required":["price_series","tasks"]}
            }},
                {"type":"function","function":{
            "name":"forecast_open_meteo",
            "description":"Denní předpověď (Open-Meteo) + THI a rizikové kategorie.",
            "parameters":{"type":"object","properties":{
                "lat":{"type":"number"},
                "lon":{"type":"number"},
                "days":{"type":"integer","default":7}
            },"required":["lat","lon"]}
        }},
        {"type":"function","function":{
            "name":"nearest_suppliers",
            "description":"Seřadí odběratele podle vzdálenosti od farmy.",
            "parameters":{"type":"object","properties":{
                "farm_lat":{"type":"number"},
                "farm_lon":{"type":"number"},
                "suppliers":{"type":"array","items":{"type":"object"}},
                "top":{"type":"integer","default":5}
            },"required":["farm_lat","farm_lon","suppliers"]}
        }},
        {"type":"function","function":{
            "name":"supplier_price",
            "description":"Vrátí cenu pro daného odběratele z katalogu/url nebo fallback.",
            "parameters":{"type":"object","properties":{
                "supplier":{"type":"object"},
                "fallback_czk_per_kg":{"type":"number"}
            },"required":["supplier"]}
        }},
        {"type":"function","function":{
            "name":"best_supplier",
            "description":"Vybere nejlepšího odběratele (cena+bonusy−doprava).",
            "parameters":{"type":"object","properties":{
                "farm_lat":{"type":"number"},
                "farm_lon":{"type":"number"},
                "milk_qty_kg_per_day":{"type":"number"},
                "transport_cost_czk_per_km":{"type":"number"},
                "suppliers":{"type":"array","items":{"type":"object"}},
                "fallback_price_czk_per_kg":{"type":"number"}
            },"required":["farm_lat","farm_lon","milk_qty_kg_per_day","transport_cost_czk_per_km","suppliers"]}
        }},
        {"type":"function","function":{
            "name":"thi_heat_stress",
            "description":"THI výpočet a riziko tepelného stresu.",
            "parameters":{"type":"object","properties":{
                "t_avg_c":{"type":"number"},
                "rh_pct":{"type":"number"}
            },"required":["t_avg_c","rh_pct"]}
        }},
        {"type":"function","function":{
            "name":"lactation_residuals",
            "description":"Rezidua k laktační křivce (Wood) pro danou krávu.",
            "parameters":{"type":"object","properties":{
                "cow_id":{"type":"string"},
                "window_days":{"type":"integer","default":60}
            },"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"mastitis_score",
            "description":"Syntetické skóre mastitidy (SCC/Trend/Variabilita).",
            "parameters":{"type":"object","properties":{
                "cow_id":{"type":"string"},
                "window_days":{"type":"integer","default":30}
            },"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"economic_roi",
            "description":"Ekonomika změny DM: přínos, náklad, ROI, payback.",
            "parameters":{"type":"object","properties":{
                "cow_id":{"type":"string"},
                "delta_dm":{"type":"number"},
                "milk_price_czk_per_kg":{"type":"number"},
                "feed_price_czk_per_kgdm":{"type":"number"}
            },"required":["cow_id","delta_dm","milk_price_czk_per_kg","feed_price_czk_per_kgdm"]}
        }},
        {"type":"function","function":{
            "name":"dryoff_planner",
            "description":"Plán zasušení podle DIM a SCC.",
            "parameters":{"type":"object","properties":{
                "cow_id":{"type":"string"},
                "target_dim":{"type":"integer","default":305},
                "scc_threshold":{"type":"integer","default":200000}
            },"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"repro_window",
            "description":"Servisní okno a varovné flagy (trend/cv).",
            "parameters":{"type":"object","properties":{
                "cow_id":{"type":"string"},
                "vwp_min":{"type":"integer","default":45},
                "vwp_max":{"type":"integer","default":60}
            },"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"milk_quality_bonus",
            "description":"Simulace bonusů/srážek podle SCC/Tuk/Protein.",
            "parameters":{"type":"object","properties":{
                "window_days":{"type":"integer","default":14},
                "pricing_table":{"type":"object"}
            },"required":[]}
        }},
        {"type":"function","function":{
            "name":"welfare_score",
            "description":"Proxy welfare skóre 0..1 (stabilita mléka, trend, SCC).",
            "parameters":{"type":"object","properties":{
                "window_days":{"type":"integer","default":14},
                "cow_id":{"type":["string","null"]}
            },"required":[]}
        }},
        {
            "type": "function",
            "function": {
                "name": "farm_kpi",
                "description": "Vrací KPI pro konkrétní krávu nebo celé stádo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string","enum": ["ecm","fce","milk_avg","scc_avg","fat_pct_avg","protein_pct_avg"]},
                        "cow_id": {"type": ["string","null"]},
                        "window_days": {"type": "integer", "minimum": 1, "maximum": 120, "default": 7}
                    },
                    "required": ["metric"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "alerts",
                "description": "Zdravotní/produkční alerty pro danou krávu.",
                "parameters": {
                    "type": "object",
                    "properties": {"cow_id": {"type": "string"}},
                    "required": ["cow_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_farm",
                "description": "Souhrn stáda za poslední N dní.",
                "parameters": {
                    "type": "object",
                    "properties": {"window_days": {"type": "integer", "minimum": 3, "maximum": 120, "default": 14}},
                    "required": []
                }
            }
        },
        {"type":"function","function":{
            "name":"cow_history",
            "description":"Časová osa krávy (datum, DIM, mléko, poznámky).",
            "parameters":{"type":"object","properties":{"cow_id":{"type":"string"}},"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"forecast_milk",
            "description":"Trendová predikce mléka na zadaný horizont.",
            "parameters":{"type":"object","properties":{"cow_id":{"type":"string"},"horizon_days":{"type":"integer","minimum":1,"default":7}},"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"feed_optimizer",
            "description":"Návrh úpravy DM krmiva pro cílový nádoj.",
            "parameters":{"type":"object","properties":{"cow_id":{"type":"string"},"target_milk":{"type":"number"},"safety_margin":{"type":"number","default":0.05}},"required":["cow_id","target_milk"]}
        }},
        {"type":"function","function":{
            "name":"sustainability",
            "description":"Udržitelnostní skóre a dílčí metriky za okno.",
            "parameters":{"type":"object","properties":{"window_days":{"type":"integer","minimum":7,"default":30}},"required":["window_days"]}
        }},
        {"type":"function","function":{
            "name":"outliers",
            "description":"Odlehlé krávy dle z-skóre v okně.",
            "parameters":{"type":"object","properties":{"window_days":{"type":"integer","default":14},"metric":{"type":"string","default":"milk_kg"},"z_threshold":{"type":"number","default":2.0}},"required":[]}
        }},
        {"type":"function","function":{
            "name":"compare_cows",
            "description":"Porovnej dvě krávy (milk, ECM, FCE, SCC).",
            "parameters":{"type":"object","properties":{"cow_a":{"type":"string"},"cow_b":{"type":"string"},"window_days":{"type":"integer","default":14}},"required":["cow_a","cow_b"]}
        }},
        {"type":"function","function":{
            "name":"what_if_feed",
            "description":"'Co kdyby' změna DM o delta_dm — očekávaná změna mléka dle FCE.",
            "parameters":{"type":"object","properties":{"cow_id":{"type":"string"},"delta_dm":{"type":"number"}},"required":["cow_id","delta_dm"]}
        }},
        {"type":"function","function":{
            "name":"rank_cows",
            "description":"Žebříček krav dle metriky v okně.",
            "parameters":{"type":"object","properties":{"window_days":{"type":"integer","default":14},"by":{"type":"string","default":"milk_kg"},"top":{"type":"integer","default":10}},"required":[]}
        }},
        {"type":"function","function":{
            "name":"scc_trend",
            "description":"Sklon a průměr SCC pro danou krávu v okně.",
            "parameters":{"type":"object","properties":{"cow_id":{"type":"string"},"window_days":{"type":"integer","default":30}},"required":["cow_id"]}
        }},
        {"type":"function","function":{
            "name":"variability",
            "description":"Koeficient variability ve stádě (CV%).",
            "parameters":{"type":"object","properties":{"window_days":{"type":"integer","default":14}},"required":[]}
        }},

    ]

def run_agent(user_prompt: str, csv_path: Optional[str], model: str = "gpt-5") -> str:
    global GLOBAL_DF
    GLOBAL_DF = ensure_dataframe(csv_path)

    client = OpenAI()

    system_prompt = (
        "Jsi AI poradce pro mléčnou farmu. Máš k dispozici nástroje pro výpočet KPI a doporučení.\n"
        "- Na KPI a průměry používej 'farm_kpi' / 'summarize_farm'.\n"
        "- Zdravotní rizika řeš přes 'alerts' a 'scc_trend'.\n"
        "- Plánování a doporučení: 'feed_optimizer', 'what_if_feed', 'rank_cows'.\n"
        "- Kontext k jedné krávě: 'cow_history', 'forecast_milk'.\n"
        "- Udržitelnost a variabilita: 'sustainability', 'variability', 'outliers'.\n"
        "Odpovědi dělej krátké, číselné, s jasným doporučením. Pokud chybí data, uveď to."
    )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    tools = build_tools_schema()

    # 1) První kolo – model rozhodne, který tool zavolat
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = first.choices[0].message

    # 2) Provedeme skutečné volání lokálních funkcí a vrátíme výsledek modelu
    if msg.tool_calls:
        # mapování jmen na implementace
        def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            mapping = {
                "farm_kpi": tool_farm_kpi,
                "alerts": tool_alerts,
                "summarize_farm": tool_summarize_farm,
                "cow_history": tool_cow_history,
                "forecast_milk": tool_forecast_milk,
                "feed_optimizer": tool_feed_optimizer,
                "sustainability": tool_sustainability,
                "outliers": tool_outliers,
                "compare_cows": tool_compare_cows,
                "what_if_feed": tool_what_if_feed,
                "rank_cows": tool_rank_cows,
                "scc_trend": tool_scc_trend,
                "variability": tool_variability,
                "thi_heat_stress": tool_thi_heat_stress,
                "lactation_residuals": tool_lactation_residuals,
                "mastitis_score": tool_mastitis_score,
                "economic_roi": tool_economic_roi,
                "dryoff_planner": tool_dryoff_planner,
                "repro_window": tool_repro_window,
                "milk_quality_bonus": tool_milk_quality_bonus,
                "welfare_score": tool_welfare_score,
                "forecast_open_meteo": tool_forecast_open_meteo,
                "nearest_suppliers": tool_nearest_suppliers,
                "supplier_price": tool_supplier_price,
                "best_supplier": tool_best_supplier,
                "microbiome_precision_feed": tool_microbiome_precision_feed,
                "behavior_anomalies": tool_behavior_anomalies,
                "early_disease_alert": tool_early_disease_alert,
                "genomics_mating": tool_genomics_mating,
                "weather_stress_alert": tool_weather_stress_alert,
                "optimize_energy_schedule": tool_optimize_energy_schedule,
            }
            fn = mapping.get(name)
            if not fn:
                return {"error": f"Unknown tool {name}"}
            return fn(**args)

        for call in msg.tool_calls:
            fn = call.function.name
            args = json.loads(call.function.arguments or "{}")
            out = call_tool(fn, args)
            messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn,
                "content": json.dumps(out, ensure_ascii=False)
            })

        second = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return (second.choices[0].message.content or "").strip()

    # Kdyby žádný tool nebyl potřeba:
    return (msg.content or "").strip()

def run_agent_autonomous(user_prompt: str,
                         csv_path: Optional[str],
                         model: str = "gpt-5",
                         max_steps: int = 6) -> str:
    """
    React-style 'deep search' smyčka:
    - model si opakovaně říká o tooly (tools=build_tools_schema())
    - my je vykonáme a vrátíme do konverzace
    - končí, když model nepožádá o žádný další tool nebo po max_steps
    """
    global GLOBAL_DF
    GLOBAL_DF = ensure_dataframe(csv_path)

    client = OpenAI()

    # mapování názvů → implementace (sdílené s run_agent)
    def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            "farm_kpi": tool_farm_kpi,
            "alerts": tool_alerts,
            "summarize_farm": tool_summarize_farm,
            "cow_history": tool_cow_history,
            "forecast_milk": tool_forecast_milk,
            "feed_optimizer": tool_feed_optimizer,
            "sustainability": tool_sustainability,
            "outliers": tool_outliers,
            "compare_cows": tool_compare_cows,
            "what_if_feed": tool_what_if_feed,
            "rank_cows": tool_rank_cows,
            "scc_trend": tool_scc_trend,
            "variability": tool_variability,
            "thi_heat_stress": tool_thi_heat_stress,
            "lactation_residuals": tool_lactation_residuals,
            "mastitis_score": tool_mastitis_score,
            "economic_roi": tool_economic_roi,
            "dryoff_planner": tool_dryoff_planner,
            "repro_window": tool_repro_window,
            "milk_quality_bonus": tool_milk_quality_bonus,
            "welfare_score": tool_welfare_score,
            "forecast_open_meteo": tool_forecast_open_meteo,
            "nearest_suppliers": tool_nearest_suppliers,
            "supplier_price": tool_supplier_price,
            "best_supplier": tool_best_supplier,
            "microbiome_precision_feed": tool_microbiome_precision_feed,
            "behavior_anomalies": tool_behavior_anomalies,
            "early_disease_alert": tool_early_disease_alert,
            "genomics_mating": tool_genomics_mating,
            "weather_stress_alert": tool_weather_stress_alert,
            "optimize_energy_schedule": tool_optimize_energy_schedule,
        }
        fn = mapping.get(name)
        if not fn:
            return {"error": f"Unknown tool {name}"}
        return fn(**args)

    system_prompt = (
        "Jsi AI poradce pro mléčnou farmu. Pracuj ITERATIVNĚ (React-style):\n"
        "1) Nejprve si vyžádej souhrn stáda (summarize_farm) pro kontext.\n"
        "2) Podle cíle vyber další vhodné tooly (alerts, feed_optimizer, rank_cows, scc_trend, outliers,...).\n"
        "3) Opakuj, dokud nemáš dost informací k akčním doporučením.\n"
        "4) Nakonec vrať struční plan kroků a konkrétní čísla (úpravy DM, cílové FCE, hlídání SCC, seznam krav k zásahu).\n"
        "Buď efektivní: nevolej zbytečně stejné tooly s identickými argumenty."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    tools = build_tools_schema()

    used_tools: List[Tuple[str, Dict[str, Any]]] = []

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # Konec: žádné tool_calls → vrať finální odpověď
        if not getattr(msg, "tool_calls", None):
            return (msg.content or "").strip()

        # Jinak vykonej všechny požadované tooly
        for call in msg.tool_calls:
            fn = call.function.name
            args = json.loads(call.function.arguments or "{}")
            result = call_tool(fn, args)
            used_tools.append((fn, args))
            # přidej do konverzace „co se stalo“
            messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn,
                "content": json.dumps(result, ensure_ascii=False)
            })

    # Fallback: dosáhli jsme limitu kroků; vyžádej si finální shrnutí
    messages.append({
        "role": "system",
        "content": "Dosažen limit kroků. Sečti poznatky a vrať finální doporučení s prioritami a čísly."
    })
    final = client.chat.completions.create(model=model, messages=messages)
    return (final.choices[0].message.content or "").strip()

# =======================
# CLI
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Cesta k CSV s farm daty (pokud ne, použije se demo).")
    parser.add_argument("--model", default="gpt-5", help="Model (např. gpt-5).")
    parser.add_argument("--autonomous", action="store_true",
                        help="Zapne React-style smyčku (deep search) s vícekrokovým voláním tools.")
    parser.add_argument("--max_steps", type=int, default=6, help="Max počet kroků v autonomní smyčce.")
    parser.add_argument("prompt", nargs="*", help="Dotaz pro agenta.")
    args = parser.parse_args()

    # (zbytek ponech, jen místo run_agent podmíněně volej autonomous)
    if not os.getenv("OPENAI_API_KEY"):
        print("❗ Nastav proměnnou OPENAI_API_KEY (a případně OPENAI_PROJECT pro sk-proj- klíč).")
        sys.exit(1)

    user_prompt = " ".join(args.prompt).strip() or \
        "Optimalizuj farmu: zvedni FCE > 1.30, udrž SCC < 200k, navrhni zásahy u bottom krav."

    try:
        if args.autonomous:
            answer = run_agent_autonomous(user_prompt, args.csv, model=args.model, max_steps=args.max_steps)
        else:
            answer = run_agent(user_prompt, args.csv, model=args.model)

        print("\n=== Odpověď agenta ===\n")
        print(answer)
    except Exception as e:
        print(f"Chyba: {e}")
        sys.exit(2)


if __name__ == "__main__":
    # Načtení .env (pokud existuje)
    try:
        from dotenv import load_dotenv  # optional
        load_dotenv()
    except Exception:
        pass
    main()
