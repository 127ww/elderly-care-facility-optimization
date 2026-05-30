"""问题2：服务站选址与规模优化 — DFS回溯 + 贪心分配"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import load_all, COMMUNITIES
from src.q1_demand import run_q1
from src.utils import (
    BUDGET_CAPEX, SERVICE_RADIUS, STATION_CAPACITY, SAT_LOWER,
    DAYS_PER_MONTH, DAYS_PER_YEAR, MONTHS_PER_YEAR,
    calc_S1, calc_coverage_rate, calc_profit_rate, calc_annual_subsidy,
    SERVICES, EMERGENCY_SERVICE,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# 站点状态编码: 0=不建, 1=小型, 2=中型, 3=大型
STATE_TO_SIZE = {1: '小型', 2: '中型', 3: '大型'}
STATE_TO_COST = {1: 18, 2: 32, 3: 45}  # 万元 (CapEx)
# Q2: S2=1.0(理想负载), S3=1.0(基准价) — 约束#27
Q2_S2, Q2_S3 = 1.0, 1.0

# 全局引用 (DFS中传递)
_q1 = None
_S = None
_daily_dem = None
_elder_pop = None


def build_satisfaction_matrix(df_dist):
    """预计算Q2满意度 S[i,j]=0.2*S1+0.3+0.5"""
    comms = df_dist.index.tolist()
    mat = {}
    for i in comms:
        for j in comms:
            s1 = calc_S1(df_dist.loc[i, j])
            mat[(i, j)] = 0.2 * s1 + 0.8 if s1 > 0 else 0.0
    return mat


def assign_greedy(stations, communities):
    """确定性贪心分配 + 平局按剩余容量裁决

    stations: {i: (size_str, capacity)}
    返回: (assignment, load, uncovered)
    """
    assignment = {}
    load = {i: 0.0 for i in stations}
    remaining = {i: cap for i, (_, cap) in stations.items()}

    for j in communities:
        # 收集可达站点的(满意度, 站点, 剩余容量)
        candidates = []
        for i, (_, cap) in stations.items():
            s = _S.get((i, j), 0)
            if s > 0 and remaining[i] > 0:
                candidates.append((s, remaining[i], i))
        if not candidates:
            continue
        # 按满意度降序, 平局按容量降序
        candidates.sort(key=lambda x: (-x[0], -x[1]))

        assigned = False
        for sat, _, i in candidates:
            eff = _daily_dem[j] * sat  # 约束#2
            if eff <= remaining[i]:
                assignment[j] = i
                load[i] += eff
                remaining[i] -= eff
                assigned = True
                break

    uncovered = set(communities) - set(assignment.keys())
    return assignment, load, uncovered


def calc_metrics(stations, assignment):
    """计算覆盖率/满意度/利润"""
    data = _q1['data']
    yr5_pop = _q1['yr5_pop']
    daily_eff = _q1['daily_eff_demand']
    df_cost = data['station_cost']
    rev_cost = data['revenue_cost']

    total_e = int(yr5_pop['总老人'].sum())
    covered_e = sum(int(yr5_pop.loc[j, '总老人']) for j in assignment)
    cov_rate = calc_coverage_rate(covered_e, total_e)

    st_comms = {}
    for j, i in assignment.items():
        st_comms.setdefault(i, []).append(j)

    st_m = {}
    wt_sat_sum = 0.0
    for i, covered in st_comms.items():
        size = stations[i]
        pop_i = sum(int(yr5_pop.loc[j, '总老人']) for j in covered)
        avg_sat = np.mean([_S[(i, j)] for j in covered]) if covered else 0.0
        wt_sat_sum += avg_sat * pop_i

        rev = dc = ec = non_em = 0.0
        for j in covered:
            for srv in SERVICES:
                mth = daily_eff.loc[j, srv] * DAYS_PER_MONTH
                ann = mth * MONTHS_PER_YEAR * avg_sat
                p = rev_cost.loc[srv, '营收(元/次)']
                c = rev_cost.loc[srv, '直接支出(元/次)']
                if srv == EMERGENCY_SERVICE:
                    ec += ann * c  # 约束#1: 成本照扣
                else:
                    rev += ann * p
                    dc += ann * c
                    non_em += ann

        sub = calc_annual_subsidy(non_em, size)
        fixed = df_cost.loc[size, '日固定管理成本(元/日)'] * DAYS_PER_YEAR
        profit = rev - dc - ec + sub - fixed
        pr = calc_profit_rate(rev, dc + ec, sub, fixed)

        st_m[i] = {
            '规模': size, '覆盖小区': covered, '覆盖老人数': pop_i,
            '日均有效人次': round(sum(daily_eff.loc[j].sum() * avg_sat for j in covered), 1),
            '年收入(万元)': round(rev/10000, 2),
            '年直接成本(万元)': round((dc+ec)/10000, 2),
            '年补贴(万元)': round(sub/10000, 2),
            '年固定成本(万元)': round(fixed/10000, 2),
            '年利润(万元)': round(profit/10000, 2),
            '利润率': round(pr, 4),
            '平均满意度': round(avg_sat, 4),
        }

    avg_sat = wt_sat_sum / max(1, covered_e)
    return {
        'coverage_rate': cov_rate, 'covered_elders': covered_e,
        'total_elders': total_e, 'avg_satisfaction': avg_sat,
        'station_metrics': st_m, 'station_comms': st_comms,
    }


def dfs_search(communities, min_coverage, budget=None):
    """DFS回溯搜索最优方案

    4^10=1,048,576状态空间; 预算剪枝后有效叶子≈O(10³)
    约束#11: 建设成本>budget→回溯; 约束#26: CapEx仅建设费
    """
    if budget is None:
        _budget = BUDGET_CAPEX
    else:
        _budget = budget

    n = len(communities)
    total_e = sum(_elder_pop.values())
    best_sol, best_met, best_score = None, None, -float('inf')
    nodes = [0]; leaves = [0]

    def dfs(idx, stations, cost):
        nodes[0] += 1
        if cost > _budget:  # 约束#11: 预算剪枝
            return

        if idx == n:
            if not stations:
                return
            leaves[0] += 1

            active = {i: (s, STATION_CAPACITY[s]) for i, s in stations.items()}
            asgn, _, _ = assign_greedy(active, communities)

            cov_e = sum(_elder_pop[j] for j in asgn)
            if calc_coverage_rate(cov_e, total_e) < min_coverage:
                return

            met = calc_metrics(stations, asgn)
            score = met['avg_satisfaction']

            nonlocal best_score, best_sol, best_met
            if score > best_score:
                best_score = score
                best_sol = {
                    'stations': [{'小区': i, '规模': s} for i, s in stations.items()],
                    'coverage': asgn,
                }
                best_met = met
            return

        comm = communities[idx]
        for st in [0, 1, 2, 3]:  # 4种状态
            if st == 0:
                dfs(idx + 1, stations.copy(), cost)
            else:
                dfs(idx + 1, {**stations, comm: STATE_TO_SIZE[st]},
                    cost + STATE_TO_COST[st])

    dfs(0, {}, 0)
    print(f'DFS: visited={nodes[0]} leaves={leaves[0]}')
    return best_sol, best_met


def run_q2(min_coverage=0.85):
    global _q1, _S, _daily_dem, _elder_pop
    _q1 = run_q1()
    data = _q1['data']
    _S = build_satisfaction_matrix(data['distance'])
    _daily_dem = {c: _q1['daily_eff_demand'].loc[c].sum() for c in COMMUNITIES}
    _elder_pop = {c: int(_q1['yr5_pop'].loc[c, '总老人']) for c in COMMUNITIES}

    sol, met = dfs_search(COMMUNITIES, min_coverage)

    if sol is None:
        print('WARNING: No feasible solution found.')
        return None, None

    pd.DataFrame(sol['stations']).to_csv(OUTPUT_DIR/'q2_stations.csv', encoding='utf-8-sig', index=False)
    pd.DataFrame(sol['coverage'].items(), columns=['小区','归属站点']).to_csv(
        OUTPUT_DIR/'q2_coverage.csv', encoding='utf-8-sig', index=False)
    if met:
        pd.DataFrame(met['station_metrics']).T.to_csv(OUTPUT_DIR/'q2_metrics.csv', encoding='utf-8-sig')
    return sol, met


if __name__ == '__main__':
    sol, met = run_q2()
    if sol:
        print('=== Q2 Optimal Solution ===')
        for s in sol['stations']:
            print(f"  {s['小区']}: {s['规模']}")
        print(f"\nCoverage: {met['coverage_rate']:.2%}")
        print(f"Avg Satisfaction: {met['avg_satisfaction']:.4f}")
        for i, m in met['station_metrics'].items():
            print(f"\n  Station {i}({m['规模']}): covers {m['覆盖小区']}")
            print(f"  Profit: {m['年利润(万元)']}wan | Margin: {m['利润率']:.2%} | Sat: {m['平均满意度']:.4f}")
