"""问题1.2/1.3：服务需求预测（理论需求 + 消费约束需求）"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import load_all
from src.q1_population import run_q1_1
from src.utils import (
    SERVICES, ELDER_TYPES, COMMUNITIES,
    DAYS_PER_MONTH, MONTHS_PER_YEAR, EMERGENCY_SERVICE,
    apply_consumption_cap, calc_effective_visits,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def calc_theory_demand(pop_yr5, demand_rate):
    """问题1.2：第5年末理论月需求次数

    理论总需求 = Σ_{老人类别} (该类老人数 × 月均需求次数)
    未考虑消费能力约束，取整输出
    """
    rows = []
    for comm in COMMUNITIES:
        for etype in ELDER_TYPES:
            count = pop_yr5.loc[comm, etype]
            for service in SERVICES:
                rate = demand_rate.loc[service, etype]
                rows.append({
                    '小区': comm, '老人类别': etype,
                    '服务': service, '老人数': count,
                    '人均需求(次/月)': rate,
                    '理论总需求(次/月)': round(count * rate),
                })
    df_detail = pd.DataFrame(rows)
    df_summary = df_detail.pivot_table(
        index='小区', columns='服务', values='理论总需求(次/月)', aggfunc='sum'
    )[SERVICES]
    df_summary = df_summary.round(0).astype(int)
    return df_detail, df_summary


def calc_effective_demand(pop_yr5, demand_rate, revenue_cost, consumption_cap, df_pop):
    """问题1.3：消费约束后的月均有效需求次数

    对每个小区×每类老人独立应用消费上限扣减 (约束#9)
    紧急救助价格=0, 不受消费约束影响 (约束#1)
    """
    base_prices = revenue_cost['营收(元/次)']
    base_costs = revenue_cost['直接支出(元/次)']

    results_effective = []
    results_fulfilled = []  # 需求满足率 (约束#25)

    for comm in COMMUNITIES:
        income = df_pop.loc[comm, '人均月收入']
        for etype in ELDER_TYPES:
            count = int(pop_yr5.loc[comm, etype])
            if count == 0:
                continue

            # 单人理论需求 (次/月)
            single_theory = pd.Series(dtype=float)
            for service in SERVICES:
                single_theory[service] = demand_rate.loc[service, etype]

            # 对单个老人做消费扣减 (约束#9: 个体独立)
            single_effective = apply_consumption_cap(
                single_theory, income, etype, consumption_cap,
                base_prices, base_costs
            )

            # 该类老人总需求 = 单人 × 人数
            for service in SERVICES:
                theory_val = count * single_theory.get(service, 0)
                eff_val = count * single_effective.get(service, 0)
                results_effective.append({
                    '小区': comm, '老人类别': etype,
                    '服务': service,
                    '理论需求(次/月)': theory_val,
                    '有效需求(次/月)': int(eff_val),
                })
                if single_theory.get(service, 0) > 0:
                    results_fulfilled.append({
                        '小区': comm, '老人类别': etype,
                        '服务': service,
                        '需求满足率': single_effective.get(service, 0) / single_theory.get(service, 0),
                        '理论需求': theory_val,
                        '有效需求': int(eff_val),
                    })

    df_effective = pd.DataFrame(results_effective)
    df_eff_summary = df_effective.pivot_table(
        index='小区', columns='服务', values='有效需求(次/月)', aggfunc='sum'
    )[SERVICES].round(0).astype(int)

    df_fulfilled = pd.DataFrame(results_fulfilled)

    return df_effective, df_eff_summary, df_fulfilled


def run_q1():
    """运行完整问题1，返回第5年末所有关键数据"""
    data = load_all()
    pop_results, pop_float = run_q1_1()
    yr5_pop = pop_results[5]
    yr5_float = pop_float[5]

    # 1.2 理论需求
    detail_theory, summary_theory = calc_theory_demand(yr5_pop, data['demand_rate'])
    summary_theory.to_csv(OUTPUT_DIR / 'q1_2_theory_demand.csv', encoding='utf-8-sig')

    # 1.3 有效需求 (消费约束)
    detail_eff, summary_eff, df_fulfilled = calc_effective_demand(
        yr5_pop, data['demand_rate'], data['revenue_cost'],
        data['consumption_cap'], data['population']
    )
    summary_eff.to_csv(OUTPUT_DIR / 'q1_3_effective_demand.csv', encoding='utf-8-sig')
    df_fulfilled.to_csv(OUTPUT_DIR / 'q1_3_fulfilled_ratio.csv', encoding='utf-8-sig')

    # 转为年化日均为后续Q2准备
    daily_eff = summary_eff / DAYS_PER_MONTH  # 日均有效需求人次

    return {
        'yr5_pop': yr5_pop,
        'yr5_float': yr5_float,
        'theory_demand': summary_theory,
        'effective_demand': summary_eff,
        'daily_eff_demand': daily_eff,
        'fulfilled_ratio': df_fulfilled,
        'data': data,
    }


if __name__ == '__main__':
    result = run_q1()
    print('=== 第5年末人口 ===')
    print(result['yr5_pop'].to_string())
    print('\n=== 理论需求(次/月) ===')
    print(result['theory_demand'].to_string())
    print('\n=== 有效需求(次/月) ===')
    print(result['effective_demand'].to_string())
    print('\n=== 日均有效需求 ===')
    print(result['daily_eff_demand'].round(2).to_string())
