#!/usr/bin/env python3
# Rychlý offline test bez volání OpenAI API.
import json
import agent_dairy_tools as adt

# Nahraj demo CSV do globálního DF
df = adt.ensure_dataframe("demo_farm.csv")
adt.GLOBAL_DF = df

print("=== summarize_farm(14) ===")
print(json.dumps(adt.tool_summarize_farm(14), ensure_ascii=False, indent=2))

print("\n=== farm_kpi('ecm', cow_id='101', window_days=7) ===")
print(json.dumps(adt.tool_farm_kpi('ecm', cow_id='101', window_days=7), ensure_ascii=False, indent=2))

print("\n=== alerts('101') ===")
print(json.dumps(adt.tool_alerts('101'), ensure_ascii=False, indent=2))
