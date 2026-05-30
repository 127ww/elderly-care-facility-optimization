"""问题4：灵敏度分析 — 4组参数变动后重新求解Q2+Q3"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import load_all
from src.q1_population import predict_population
from src.q2_location import run_q2, dfs_search, build_satisfaction_matrix, assign_greedy, COMMUNITIES
from src.q3_pricing import build_q3_data, solve_grid_probe
import src.utils as U
import src.q1_demand as q1d

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_PARAMS = {
    'new_elder_rate': 0.07,
    'p_sh': 0.045,  # 自理→半失能
    'p_hd': 0.10,   # 半失能→失能
    'cost_mult': 1.0,
    'budget': 120,
}

SCENARIOS = {
    'baseline': {},
    'S1_growth_8pct': {'new_elder_rate': 0.08},
    'S2_transfer_changed': {'p_sh': 0.055, 'p_hd': 0.095},
    'S3_cost_plus20pct': {'cost_mult': 1.20},
    'S4_budget_140': {'budget': 140},
}


def apply_scenario(scenario):
    """应用场景参数, 返回修改后的数据"""
    params = BASELINE_PARAMS.copy()
    params.update(scenario)

    data = load_all()
    df_pop = data['population']
    trans_prob = {('自理', '半失能'): params['p_sh'],
                  ('半失能', '失能'): params['p_hd']}

    old_new_elder = U.NEW_ELDER_RATE
    U.NEW_ELDER_RATE = params['new_elder_rate']

    pop_results, pop_float = predict_population(df_pop, trans_prob, years=5)

    U.NEW_ELDER_RATE = old_new_elder

    yr5_pop = pop_results[5]

    # 重新计算需求
    _, theory = q1d.calc_theory_demand(yr5_pop, data['demand_rate'])
    _, eff, _ = q1d.calc_effective_demand(
        yr5_pop, data['demand_rate'], data['revenue_cost'],
        data['consumption_cap'], df_pop
    )

    daily_eff = eff / U.DAYS_PER_MONTH
    daily_demand = {c: daily_eff.loc[c].sum() for c in COMMUNITIES}
    elder_pop = {c: int(yr5_pop.loc[c, '总老人']) for c in COMMUNITIES}

    return {
        'yr5_pop': yr5_pop, 'daily_eff_demand': daily_eff,
        'daily_demand': daily_demand, 'elder_pop': elder_pop,
        'theory_demand': theory, 'effective_demand': eff,
        'data': data, 'budget': params['budget'],
        'cost_mult': params['cost_mult'],
    }


def run_scenario(scenario_name, scenario_params):
    """运行单个场景的Q2+Q3"""
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'Scenario: {scenario_name}')
    print(f'Params: {scenario_params}')
    print(sep)

    sc = apply_scenario(scenario_params)

    # 修改全局参数用于Q2
    old_budget = U.BUDGET_CAPEX
    U.BUDGET_CAPEX = sc['budget']

    old_daily_cost = None  # store original if needed
    if sc['cost_mult'] != 1.0:
        # 修改日固定管理成本
        data = sc['data']
        for size in ['小型', '中型', '大型']:
            orig = data['station_cost'].loc[size, '日固定管理成本(元/日)']
            data['station_cost'].loc[size, '日固定管理成本(元/日)'] = orig * sc['cost_mult']

    # Q2: DFS搜索
    global _q1_sc, _S_sc, _daily_dem_sc, _elder_pop_sc
    import src.q2_location as q2l
    old_q1 = q2l._q1
    old_S = q2l._S
    old_daily = q2l._daily_dem
    old_elder = q2l._elder_pop

    q2l._q1 = sc
    q2l._S = build_satisfaction_matrix(sc['data']['distance'])
    q2l._daily_dem = sc['daily_demand']
    q2l._elder_pop = sc['elder_pop']

    # 寻找最大可行覆盖率 (传递budget和cost_mult参数)
    best_cov = 0.60
    best_sol, best_met = None, None
    for cov_req in np.arange(0.60, 0.95, 0.05):
        cov_req = round(cov_req, 2)
        sol, met = dfs_search(COMMUNITIES, cov_req, budget=sc['budget'])
        if sol:
            best_cov = cov_req
            best_sol = sol
            best_met = met
        else:
            break

    # 恢复Q2全局变量
    q2l._q1 = old_q1
    q2l._S = old_S
    q2l._daily_dem = old_daily
    q2l._elder_pop = old_elder
    U.BUDGET_CAPEX = old_budget

    if best_sol is None:
        print(f'  Q2: NO SOLUTION')
        return {'scenario': scenario_name, 'q2_feasible': False}

    n_stations = len(best_sol['stations'])
    budget_used = sum(
        {'小型': 18, '中型': 32, '大型': 45}[s['规模']]
        for s in best_sol['stations']
    )
    print(f'  Q2: {n_stations} stations coverage={best_met["coverage_rate"]:.2%} '
          f'sat={best_met["avg_satisfaction"]:.4f} budget={budget_used}wan')

    for s in best_sol['stations']:
        print(f'    {s["小区"]}: {s["规模"]}')

    # Q3: 完整Deb-PSO — 所有场景重跑（站点配置可能变化）
    from src.q3_pricing import solve_pso_deb
    from src.utils import calc_S1

    q3_tau = None; q3_pr = None
    try:
        sts_map = {s['小区']: s['规模'] for s in best_sol['stations']}
        st_comms_q3 = {}
        for j, i in best_sol['coverage'].items():
            st_comms_q3.setdefault(i, []).append(j)
        rev_cost = sc['data']['revenue_cost']
        q3d_simple = {
            'stations': sts_map, 'coverage': best_sol['coverage'],
            'st_comms': st_comms_q3,
            'base_prices': {s: rev_cost.loc[s, '营收(元/次)'] for s in rev_cost.index},
            'base_costs': {s: rev_cost.loc[s, '直接支出(元/次)'] for s in rev_cost.index},
            'S1': {(i,j): calc_S1(sc['data']['distance'].loc[i,j])
                   for i in sts_map for j in st_comms_q3.get(i,[])},
            'daily_eff': sc['daily_eff_demand'], 'yr5_pop': sc['yr5_pop'], 'data': sc['data'],
        }
        print(f'  Running Q3 PSO (500 particles, 200 gens)...')
        pso_q3 = solve_pso_deb(q3d_simple, n_particles=500, n_iter=200)
        if pso_q3 and pso_q3.get('feasible'):
            q3_tau = round(pso_q3['min_sat'], 4)
            q3_pr = {st: round(m['profit_rate'],4) for st,m in pso_q3['station_metrics'].items()}
            print(f'  Q3: tau={q3_tau} pr={q3_pr}')
        else:
            print(f'  Q3: INFEASIBLE')
    except Exception as e:
        print(f'  Q3 ERROR: {e}')

    return {
        'scenario': scenario_name,
        'q2_feasible': True,
        'n_stations': n_stations,
        'stations': [(s['小区'], s['规模']) for s in best_sol['stations']],
        'coverage': best_met['coverage_rate'],
        'satisfaction': best_met['avg_satisfaction'],
        'budget_used': budget_used,
        'total_elders': best_met['total_elders'],
        'covered_elders': best_met['covered_elders'],
        'q3_tau': q3_tau, 'q3_profit_rates': q3_pr,
    }


def run_q4():
    results = {}
    for name, params in SCENARIOS.items():
        results[name] = run_scenario(name, params)

    # 汇总对比表
    sep70 = '=' * 70
    print(f'\n{sep70}')
    print('=== Q4 Sensitivity Summary ===')
    print(sep70)
    hdr = f'{"Scenario":<25s} {"Stations":>8s} {"Budget":>7s} {"Coverage":>9s} {"Sat":>7s} {"Q3_tau":>8s}'
    print(hdr)
    print('-' * 65)

    baseline = results.get('baseline', {})
    for name, r in results.items():
        if r.get('q2_feasible'):
            sts = '+'.join(f'{s[0]}{s[1][0]}' for s in r['stations'])
            n_st = r['n_stations']
            bud = r['budget_used']
            cov = r['coverage']
            sat = r['satisfaction']
            q3t = r.get('q3_tau')
            q3t_str = f'{q3t:.4f}' if q3t else 'N/A'
            print(f'{name:<25s} {n_st:>2}站 {sts:<12s} {bud:>4}wan {cov:>8.2%} {sat:>7.4f} {q3t_str:>8s}')
        else:
            print(f'{name:<25s} {"INFEASIBLE":>30s}')

    rows = []
    for name, r in results.items():
        rows.append({
            '场景': name,
            '可解': r.get('q2_feasible', False),
            '站点数': r.get('n_stations', 0),
            '站点': '+'.join(f'{s[0]}{s[1][0]}' for s in r.get('stations', [])),
            '预算(万)': r.get('budget_used', 0),
            '覆盖率': r.get('coverage', 0),
            '满意度': r.get('satisfaction', 0),
            'Q3_tau': r.get('q3_tau'),
            '覆盖/总人数': f'{r.get("covered_elders",0)}/{r.get("total_elders",0)}',
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/'q4_sensitivity.csv', encoding='utf-8-sig', index=False)

    return results


if __name__ == '__main__':
    run_q4()
