
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

# =======================
# OpenAI agent
# =======================

def build_tools_schema() -> List[Dict[str, Any]]:
    return [
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

# =======================
# CLI
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Cesta k CSV s farm daty (pokud ne, použije se demo).")
    parser.add_argument("--model", default="gpt-5", help="Model (např. gpt-5).")
    parser.add_argument("prompt", nargs="*", help="Dotaz pro agenta.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❗ Nastav proměnnou OPENAI_API_KEY (můžeš použít .env).")
        sys.exit(1)

    user_prompt = " ".join(args.prompt).strip() or \
        "Shrň KPI stáda za posledních 14 dní a doporuč úpravu krmení pro krávu 101."

    try:
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
