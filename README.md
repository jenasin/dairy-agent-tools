# Dairy Tools Agent (OpenAI Function Calling)

Jednoduchý AI agent v Pythonu pro práci s **dairy farm** daty (CSV). Poskytuje tři nástroje:
- `farm_kpi(metric, cow_id, window_days)` – ECM, FCE, SCC, průměr mléka, tuk/bílkoviny
- `alerts(cow_id)` – jednoduché alerty (SCC, FPR>1.4 v rané laktaci, FCE, pokles mléka)
- `summarize_farm(window_days)` – agregované KPI za stádo + top/bottom 5 krav

## Struktura repa
```
.
├─ agent_dairy_tools.py
├─ demo_farm.csv
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ LICENSE
└─ test_local.py        # offline test bez OpenAI API
```

## Rychlý start (agent s OpenAI)
```bash
python -V  # 3.10+
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # doplň svůj OPENAI_API_KEY
python agent_dairy_tools.py --csv demo_farm.csv "Shrň KPI stáda za posledních 14 dní."
```

### Příklady dotazů
```bash
python agent_dairy_tools.py --csv demo_farm.csv "Zhodnoť FCE a ECM pro krávu 101 za posledních 7 dní a upozorni na rizika."
python agent_dairy_tools.py --csv demo_farm.csv "Spočti průměrné SCC stáda za 10 dní."
python agent_dairy_tools.py --csv demo_farm.csv "Shrň KPI stáda za 14 dní a ukaž top a bottom 5 krav dle mléka."
```

## Offline test (bez API klíče)
Chceš jen prověřit výpočty nad CSV bez volání OpenAI? Stačí:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_local.py
```
Vypíše ukázkové JSON výstupy z `summarize_farm`, `farm_kpi` a `alerts` nad `demo_farm.csv`.

## CSV schéma
Povinné sloupce: `date,cow_id,milk_kg,fat_pct,protein_pct,feed_kg_dm,scc`  
Volitelné: `bw_kg,parity,dim` (když `dim` chybí, dopočítá se z prvního dne dané krávy).

## Poznámky
- Výchozí model: `gpt-5` (změň parametrem `--model`).
- Pokud soubor CSV neuvedeš, agent si vygeneruje demo dataset (2 krávy × 14 dní).
- Alerty jsou heuristiky pro demo účely – prahy uprav dle farmy.

## Publikace na GitHub
```bash
git init
git add .
git commit -m "feat: dairy tools agent with KPI and alerts"
git branch -M main
git remote add origin https://github.com/jenasin/dairy-agent-tools.git  # změň podle svého účtu/názvu
git push -u origin main
```
