"""问题3：服务定价优化 — 网格探底 + PSO精细寻优 (无梯度)"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from src.data_loader import COMMUNITIES, SERVICES, NON_EMERGENCY_SERVICES
from src.q1_demand import run_q1
from src.q2_location import run_q2
from src.utils import (
    SAT_LOWER, SAT_UPPER, PRICE_FLOOR_FACTOR, PRICE_CEIL_FACTOR,
    DAYS_PER_MONTH, DAYS_PER_YEAR, MONTHS_PER_YEAR,
    STATION_CAPACITY, SUBSIDY_PER_VISIT, SUBSIDY_CAP,
    calc_S1, calc_S2, calc_S3, calc_annual_subsidy,
    INCOME_ADJUSTMENT, EMERGENCY_SERVICE, SEED,
)

np.random.seed(SEED)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def build_q3_data():
    q1_result = run_q1()
    q2_sol, q2_metrics = run_q2(min_coverage=0.80)
    data = q1_result['data']
    df_dist = data['distance']
    rev_cost = data['revenue_cost']
    daily_eff = q1_result['daily_eff_demand']
    yr5_pop = q1_result['yr5_pop']

    stations = {s['小区']: s['规模'] for s in q2_sol['stations']}
    coverage = q2_sol['coverage']
    st_comms = {}
    for j, i in coverage.items():
        st_comms.setdefault(i, []).append(j)

    base_prices = {s: rev_cost.loc[s, '营收(元/次)'] for s in SERVICES}
    base_costs = {s: rev_cost.loc[s, '直接支出(元/次)'] for s in SERVICES}

    S1 = {}
    for i in stations:
        for j in st_comms.get(i, []):
            S1[(i, j)] = calc_S1(df_dist.loc[i, j])

    return {
        'q1': q1_result, 'q2_sol': q2_sol,
        'stations': stations, 'coverage': coverage, 'st_comms': st_comms,
        'base_prices': base_prices, 'base_costs': base_costs,
        'S1': S1, 'daily_eff': daily_eff, 'yr5_pop': yr5_pop, 'data': data,
    }


def evaluate_pricing(price_factors, q3d):
    """评估定价方案 — 使用Q1.3真实需求数据

    返回: {community_sats, station_metrics, feasible, score, violations}
    score = tau × exp(-penalty)  (罚函数乘法形式, 避免量纲吞噬)
    """
    stations = q3d['stations']
    st_comms = q3d['st_comms']
    base_prices = q3d['base_prices']
    base_costs = q3d['base_costs']
    S1 = q3d['S1']
    daily_eff = q3d['daily_eff']  # Q1.3 真实有效需求 (人次/日)
    data = q3d['data']
    df_cost = data['station_cost']

    community_sats = {}
    station_metrics = {}
    feasible = True
    total_penalty = 0.0

    for st, size in stations.items():
        comms = st_comms.get(st, [])
        if not comms:
            continue

        factors = price_factors[st]
        actual_prices = {}
        for s in SERVICES:
            if s == EMERGENCY_SERVICE:
                actual_prices[s] = 0.0  # 约束#1
            else:
                actual_prices[s] = base_prices[s] * factors.get(s, 1.0)

        # Pass 1: S2=1.0预估满意度 → 计算负载 → 修正S2 (约束#27)
        sats_pass1 = {}
        for j in comms:
            inc_adj = INCOME_ADJUSTMENT[j]
            s1 = S1[(st, j)]
            weights = daily_eff.loc[j][NON_EMERGENCY_SERVICES]
            tw = weights.sum()
            avg_pr = sum((actual_prices[s] / base_prices[s]) * weights[s] / tw
                        for s in NON_EMERGENCY_SERVICES) if tw > 0 else 1.0
            s3 = calc_S3(avg_pr, inc_adj)
            sats_pass1[j] = np.clip(0.2 * s1 + 0.3 * 1.0 + 0.5 * s3, SAT_LOWER, SAT_UPPER)

        daily_eff_total = sum(daily_eff.loc[j].sum() * sats_pass1[j] for j in comms)
        daily_cap = STATION_CAPACITY[size]
        load_rate = daily_eff_total / daily_cap if daily_cap > 0 else 1.0
        actual_s2 = calc_S2(min(load_rate, 1.0))

        # 容量惩罚 (约束#10)
        if load_rate > 1.0:
            feasible = False
            total_penalty += (load_rate - 1.0) * 5.0

        # Pass 2: 实际S2最终计算
        annual_rev = 0.0
        annual_dc = 0.0
        annual_ec = 0.0
        annual_non_em = 0.0

        for j in comms:
            inc_adj = INCOME_ADJUSTMENT[j]
            s1 = S1[(st, j)]
            weights = daily_eff.loc[j][NON_EMERGENCY_SERVICES]
            tw = weights.sum()
            avg_pr = sum((actual_prices[s] / base_prices[s]) * weights[s] / tw
                        for s in NON_EMERGENCY_SERVICES) if tw > 0 else 1.0
            s3 = calc_S3(avg_pr, inc_adj)
            sat = np.clip(0.2 * s1 + 0.3 * actual_s2 + 0.5 * s3, SAT_LOWER, SAT_UPPER)
            community_sats[j] = sat

            for srv in SERVICES:
                mth = daily_eff.loc[j, srv] * DAYS_PER_MONTH
                ann = mth * MONTHS_PER_YEAR * sat  # 约束#2: 有效人次
                if srv == EMERGENCY_SERVICE:
                    annual_ec += ann * base_costs[srv]  # 约束#1: 成本照扣
                else:
                    annual_rev += ann * actual_prices[srv]
                    annual_dc += ann * base_costs[srv]
                    annual_non_em += ann  # 仅非紧急参与补贴

        annual_sub = calc_annual_subsidy(annual_non_em, size)  # 排除紧急救助
        annual_fixed = df_cost.loc[size, '日固定管理成本(元/日)'] * DAYS_PER_YEAR
        annual_profit = annual_rev - annual_dc - annual_ec + annual_sub - annual_fixed
        profit_rate = annual_profit / annual_fixed if annual_fixed > 0 else 0.0

        # 利润率惩罚 (约束#3, #20)
        if profit_rate < -0.001:
            feasible = False
            total_penalty += abs(profit_rate) * 3.0
        elif profit_rate > 0.081:
            feasible = False
            total_penalty += (profit_rate - 0.08) * 3.0

        daily_non_em_check = annual_non_em / DAYS_PER_YEAR
        cliff = daily_non_em_check > (SUBSIDY_CAP[size] / SUBSIDY_PER_VISIT)

        station_metrics[st] = {
            'size': size, 'comms': comms, 'profit_rate': profit_rate,
            'annual_profit': annual_profit, 'load_rate': load_rate,
            'subsidy_cliff': cliff, 'annual_revenue': annual_rev,
            'annual_subsidy': annual_sub, 'actual_s2': actual_s2,
            'daily_eff_total': daily_eff_total, 'daily_cap': daily_cap,
        }

    tau = min(community_sats.values()) if community_sats else 0.0
    # 归一化违规量 (Deb规则用: 容量超额+利润率越界)
    capacity_violation = 0.0; profit_violation = 0.0
    for st, m in station_metrics.items():
        if m['load_rate'] > 1.0: capacity_violation += m['load_rate'] - 1.0
        pr = m['profit_rate']
        if pr < -0.001: profit_violation += abs(pr)
        elif pr > 0.081: profit_violation += pr - 0.08
    normalized_violation = capacity_violation + profit_violation

    return {
        'community_sats': community_sats, 'min_sat': tau,
        'avg_sat': np.mean(list(community_sats.values())) if community_sats else 0.0,
        'station_metrics': station_metrics, 'feasible': feasible,
        'violation': normalized_violation,
    }


# ===== 阶段1: 网格探底 (每站统一价格因子, 3D) =====

def solve_grid_probe(q3d, n=11):
    """均匀价格网格搜索 — 为PSO提供先验引导"""
    stations = list(q3d['stations'].keys())
    levels = np.linspace(PRICE_FLOOR_FACTOR, PRICE_CEIL_FACTOR, n)
    best, best_score = None, -1.0

    for f_combo in product(levels, repeat=len(stations)):
        factors = {st: {s: f_combo[i] for s in NON_EMERGENCY_SERVICES}
                   for i, st in enumerate(stations)}
        res = evaluate_pricing(factors, q3d)
        if res['feasible'] and res['min_sat'] > best_score:
            best_score = res['min_sat']
            best = res
            best['price_factors'] = factors

    return best


# ===== 阶段2: PSO精细寻优 (15D差异化定价) =====

def solve_pso_deb(q3d, n_particles=500, n_iter=200):
    """PSO + Deb可行性规则 — 纯随机初始化, 无热启动

    Deb规则: 可行解按tau排序, 不可行解按违规量排序, 可行永远优于不可行
    """
    stations = list(q3d['stations'].keys())
    n_dims = len(stations) * len(NON_EMERGENCY_SERVICES)

    dim_map = []
    for st in stations:
        for srv in NON_EMERGENCY_SERVICES:
            dim_map.append((st, srv))

    lb = np.full(n_dims, PRICE_FLOOR_FACTOR)
    ub = np.full(n_dims, PRICE_CEIL_FACTOR)

    # 纯随机初始化 [0.5, 1.5] (宽于约束边界, 故意撒入不可行区)
    X = np.random.uniform(0.5, 1.5, (n_particles, n_dims))
    X = np.clip(X, lb, ub)
    V = np.random.uniform(-0.05, 0.05, (n_particles, n_dims))

    # 评估每个粒子的(feasible, tau, violation)
    def _eval(x):
        factors = {st: {} for st in stations}
        for k, (st, srv) in enumerate(dim_map):
            factors[st][srv] = float(x[k])
        return evaluate_pricing(factors, q3d)

    pop_eval = [_eval(X[i]) for i in range(n_particles)]
    pbest_X = X.copy()
    pbest_eval = list(pop_eval)
    gbest_X = X[0].copy()
    gbest_eval = pop_eval[0]

    def _better(a, b):
        """Deb规则: a better than b?"""
        if a['feasible'] and not b['feasible']: return True
        if not a['feasible'] and b['feasible']: return False
        if a['feasible'] and b['feasible']: return a['min_sat'] > b['min_sat']
        return a['violation'] < b['violation']

    for i in range(n_particles):
        if _better(pop_eval[i], gbest_eval):
            gbest_X = X[i].copy(); gbest_eval = pop_eval[i]

    history = []  # 每代gbest tau (仅可行时记录, 否则None)

    for it in range(n_iter):
        w = 0.9 - (0.9 - 0.4) * it / n_iter
        c1, c2 = 2.0, 2.0

        r1 = np.random.rand(n_particles, n_dims)
        r2 = np.random.rand(n_particles, n_dims)
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
        V = np.clip(V, -0.06, 0.06)
        X = np.clip(X + V, lb, ub)

        for i in range(n_particles):
            ev = _eval(X[i])
            if _better(ev, pbest_eval[i]): pbest_X[i] = X[i].copy(); pbest_eval[i] = ev
            if _better(ev, gbest_eval): gbest_X = X[i].copy(); gbest_eval = ev

        history.append(gbest_eval['min_sat'] if gbest_eval['feasible'] else None)

        if it % 50 == 0:
            st = 'OK' if gbest_eval['feasible'] else 'INFEAS'
            tv = gbest_eval['min_sat'] if gbest_eval['feasible'] else gbest_eval['violation']
            print(f'  iter={it:3d} {"tau" if gbest_eval["feasible"] else "viol"}={tv:.4f} [{st}]')

    factors = {st: {} for st in stations}
    for k, (st, srv) in enumerate(dim_map):
        factors[st][srv] = float(gbest_X[k])
    result = evaluate_pricing(factors, q3d)
    result['price_factors'] = factors
    result['pso_history'] = history
    return result


def _check_feasible(factors):
    """快速可行性检查"""
    q3d_cache = q3d_global
    return evaluate_pricing(factors, q3d_cache)


q3d_global = None


def run_q3():
    global q3d_global
    q3d = build_q3_data()
    q3d_global = q3d
    base_prices = q3d['base_prices']
    stations = list(q3d['stations'].keys())

    # 阶段1: 网格探底
    print('=== Stage 1: Grid Probe (3D uniform per station) ===')
    grid_result = solve_grid_probe(q3d, n=2)  # n=2粗探: 仅0.7/1.3两极, 定位可行域
    if grid_result:
        print(f'Grid best: tau={grid_result["min_sat"]:.4f} feasible={grid_result["feasible"]}')
    else:
        print('Grid: NO FEASIBLE SOLUTION')

    # 阶段2: PSO精细寻优
    print('\n=== Stage 2: PSO Refined (15D differentiated) ===')
    warm = grid_result['price_factors'] if grid_result else None
    pso_result = solve_pso_deb(q3d, n_particles=100, n_iter=200)

    # 选择最优可行解
    candidates = []
    if grid_result and grid_result.get('feasible'): candidates.append(('Grid', grid_result))
    if pso_result and pso_result.get('feasible'): candidates.append(('PSO', pso_result))

    if not candidates:
        print('NO FEASIBLE PRICING FOUND')
        return None
    method, result = max(candidates, key=lambda x: x[1]['min_sat'])

    print(f'\n=== Q3 Final Result (method={method}) ===')
    print(f'Min Satisfaction (tau): {result["min_sat"]:.4f}')
    print(f'Avg Satisfaction: {result["avg_sat"]:.4f}')

    # 详细输出
    for st in stations:
        m = result['station_metrics'][st]
        print(f'\nStation {st} ({m["size"]}):')
        print(f'  Load={m["daily_eff_total"]:.0f}/{m["daily_cap"]:.0f} ({m["load_rate"]:.1%})')
        print(f'  S2={m["actual_s2"]:.4f} SubsidyCliff={m["subsidy_cliff"]}')
        print(f'  Profit={m["annual_profit"]/10000:.2f}wan Rate={m["profit_rate"]:.4f}')
        print(f'  Revenue={m["annual_revenue"]/10000:.2f}wan Subsidy={m["annual_subsidy"]/10000:.2f}wan')
        factors = result['price_factors'][st]
        for srv in NON_EMERGENCY_SERVICES:
            f = factors[srv]
            print(f'  {srv}: {base_prices[srv]*f:.1f}yuan (f={f:.2f})')

    # 各社区满意度
    print('\nCommunity satisfaction:')
    for j in sorted(result['community_sats']):
        print(f'  {j}: S={result["community_sats"][j]:.4f}')

    # 保存CSV
    rows = [{'站点': st, '服务': srv,
             '价格因子': round(result['price_factors'][st][srv], 3),
             '最优价格(元)': round(base_prices[srv] * result['price_factors'][st][srv], 1),
             '基准价(元)': base_prices[srv]}
            for st in stations for srv in NON_EMERGENCY_SERVICES]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/'q3_pricing.csv', encoding='utf-8-sig', index=False)

    rows2 = [{'站点': st, '规模': result['station_metrics'][st]['size'],
              '利润率': round(result['station_metrics'][st]['profit_rate'], 4),
              '年利润(万元)': round(result['station_metrics'][st]['annual_profit']/10000, 2),
              '负载率': round(result['station_metrics'][st]['load_rate'], 4),
              'S2': round(result['station_metrics'][st]['actual_s2'], 4),
              '补贴触顶': result['station_metrics'][st]['subsidy_cliff']}
             for st in stations]
    pd.DataFrame(rows2).to_csv(OUTPUT_DIR/'q3_metrics.csv', encoding='utf-8-sig', index=False)

    rows3 = [{'小区': j, '满意度': round(result['community_sats'][j], 4),
              '归属站点': q3d['coverage'].get(j, '')}
             for j in sorted(result['community_sats'])]
    pd.DataFrame(rows3).to_csv(OUTPUT_DIR/'q3_satisfaction.csv', encoding='utf-8-sig', index=False)

    if 'pso_history' in result:
        pd.DataFrame({'iteration': range(len(result['pso_history'])),
                      'tau': result['pso_history']}).to_csv(
            OUTPUT_DIR/'q3_pso_history.csv', encoding='utf-8-sig', index=False)

    return result


if __name__ == '__main__':
    run_q3()
