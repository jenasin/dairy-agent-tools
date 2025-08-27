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
from typing import Any, Dict, Optional

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

def rolling_drop(series: pd.Series, window: int = 3) -> float:
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
    end = df["date"].max()
    start = end - pd.Timedelta(days=window_days-1)
    cut = df[(df["date"] >= start) & (df["date"] <= end)]

    if cow_id:
        cut = cut[cut["cow_id"].astype(str) == str(cow_id)]

    if cut.empty:
        return {"metric": metric, "cow_id": cow_id, "window_days": window_days, "value": None, "note": "No data in window."}

    if metric == "ecm":
        val = cut.apply(lambda r: ecm_kg(r["milk_kg"], r["fat_pct"], r["protein_pct"]), axis=1).mean()
        unit = "kg ECM/den"
    elif metric == "fce":
        val = cut.apply(lambda r: fce(r["milk_kg"], r["feed_kg_dm"]), axis=1).dropna().mean()
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

    trend = rolling_drop(cut["milk_kg"])
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
    recent["ecm"] = recent.apply(lambda r: ecm_kg(r["milk_kg"], r["fat_pct"], r["protein_pct"]), axis=1)
    recent["fce"] = recent.apply(lambda r: fce(r["milk_kg"], r["feed_kg_dm"]), axis=1)
    recent["fpr"] = recent["fat_pct"] / recent["protein_pct"]

    alerts = []

    # SCC
    if "scc" in recent and recent["scc"].mean() > 200_000:
        alerts.append({"type": "SCC_HIGH", "message": f"Průměrné SCC posledních 7 dní ~ {int(recent['scc'].mean()):,}".replace(",", " "), "threshold": 200_000})

    # Ketóza (FPR & DIM)
    if "dim" in recent:
        early = recent[recent["dim"] < 60]
        if not early.empty and (early["fpr"] > 1.4).mean() > 0.5:
            alerts.append({"type": "KETOSIS_RISK", "message": "FPR>1.4 u >50 % dní v rané laktaci (DIM<60).", "threshold": "FPR>1.4"})

    # FCE
    if recent["fce"].notna().mean() > 0 and recent["fce"].mean() < 1.2:
        alerts.append({"type": "LOW_FCE", "message": f"Průměrná FCE ~ {recent['fce'].mean():.2f} (<1.2).", "threshold": 1.2})

    # Trend poklesu mléka
    slope = rolling_drop(recent["milk_kg"], window=3)
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
    end = df["date"].max()
    start = end - pd.Timedelta(days=window_days-1)
    cut = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if cut.empty:
        return {"window_days": window_days, "note": "No data in window."}

    cut["ecm"] = cut.apply(lambda r: ecm_kg(r["milk_kg"], r["fat_pct"], r["protein_pct"]), axis=1)
    cut["fce"] = cut.apply(lambda r: fce(r["milk_kg"], r["feed_kg_dm"]), axis=1)

    kpi = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "avg_milk_kg": round(cut["milk_kg"].mean(), 2),
        "avg_ecm_kg": round(cut["ecm"].mean(), 2),
        "avg_fce": None if cut["fce"].dropna().empty else round(cut["fce"].dropna().mean(), 3),
        "avg_scc": None if cut["scc"].dropna().empty else int(cut["scc"].mean()),
        "n_cows": int(cut["cow_id"].nunique()),
        "n_days": int(cut["date"].nunique())
    }

    by_cow = cut.groupby("cow_id")["milk_kg"].mean().sort_values(ascending=False)
    top5 = by_cow.head(5).round(2).to_dict()
    bot5 = by_cow.tail(5).round(2).to_dict()

    kpi["top5_milk_avg"] = top5
    kpi["bottom5_milk_avg"] = bot5
    return kpi

# =======================
# OpenAI agent
# =======================

def run_agent(user_prompt: str, csv_path: Optional[str], model: str = "gpt-5") -> str:
    global GLOBAL_DF
    GLOBAL_DF = ensure_dataframe(csv_path)

    client = OpenAI()

    system_prompt = (
        "Jsi AI poradce pro mléčnou farmu. Máš k dispozici nástroje pro výpočet KPI a alertů.\n"
        "- Vždy, když se ptají na ECM/FCE/SCC/milk averages, použij tool 'farm_kpi'.\n"
        "- Pokud se ptají na rizika či upozornění pro konkrétní krávu, použij tool 'alerts'.\n"
        "- Pro souhrn stáda použij 'summarize_farm'.\n"
        "Odpovídej stručně, s čísly a krátkým vysvětlením. Pokud chybí data, řekni to."
    )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "farm_kpi",
                "description": "Vrací KPI pro konkrétní krávu nebo celé stádo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": ["ecm","fce","milk_avg","scc_avg","fat_pct_avg","protein_pct_avg"],
                            "description": "Požadovaná metrika."
                        },
                        "cow_id": {"type": ["string","null"], "description": "ID krávy (nebo null pro stádo)."},
                        "window_days": {"type": "integer", "minimum": 1, "maximum": 90, "default": 7}
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
                "description": "Vrací zdravotní/produkční alerty pro danou krávu za poslední týden.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cow_id": {"type": "string", "description": "ID krávy."}
                    },
                    "required": ["cow_id"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_farm",
                "description": "Souhrnné KPI za celé stádo za poslední N dní.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "window_days": {"type": "integer", "minimum": 3, "maximum": 60, "default": 14}
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        }
    ]

    # 1) První kolo – model rozhodne, který tool zavolat
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )
    msg = first.choices[0].message

    # 2) Provedeme skutečné volání lokální funkce a vrátíme výsledek modelu
    if msg.tool_calls:
        for call in msg.tool_calls:
            fn = call.function.name
            args = json.loads(call.function.arguments or "{}")
            if fn == "farm_kpi":
                out = tool_farm_kpi(**args)
            elif fn == "alerts":
                out = tool_alerts(**args)
            elif fn == "summarize_farm":
                out = tool_summarize_farm(**args)
            else:
                out = {"error": f"Unknown tool {fn}"}

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
            temperature=0.2,
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
        "Zhodnoť FCE a ECM pro krávu 101 za posledních 7 dní a upozorni na rizika."

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
