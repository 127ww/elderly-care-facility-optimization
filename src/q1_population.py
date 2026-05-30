"""问题1.1：基于矩阵递推的老人数量预测模型"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import load_population
from src.utils import DEATH_RATE, NEW_ELDER_RATE, SEED, ELDER_TYPES, COMMUNITIES

np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# 新增老人的初始状态分配比例 (新增60岁老人中大部分为自理)
NEW_ELDER_DIST = [0.85, 0.12, 0.03]  # [自理, 半失能, 失能]


def build_transfer_survival_matrix(trans_prob):
    """构建单步复合矩阵 M = D × T

    时序: ①状态转移(T) → ②自然死亡(D: 各类×0.95)
    状态顺序: [自理(S), 半失能(H), 失能(D)]

    转移规则:
    - S → S: (1 - p_sh) * (1-d)    [未转移 × 存活]
    - S → H: p_sh * (1-d)           [转移为半失能 × 存活]
    - S → D: 0                      [自理不可直接变为失能]
    - H → S: 0                      [半失能不能恢复]
    - H → H: (1 - p_hd) * (1-d)     [未转移 × 存活]
    - H → D: p_hd * (1-d)           [转移为失能 × 存活]
    - D → S, D → H: 0               [失能不能恢复]
    - D → D: 1 * (1-d)              [存活]
    """
    p_sh = trans_prob[('自理', '半失能')]
    p_hd = trans_prob[('半失能', '失能')]
    surv = 1.0 - DEATH_RATE

    M = np.array([
        [(1 - p_sh) * surv, 0.0,               0.0],
        [p_sh * surv,       (1 - p_hd) * surv, 0.0],
        [0.0,                p_hd * surv,       surv],
    ])
    return M


def predict_population(df_pop, trans_prob, years=5):
    """预测未来years年末各小区各类老人数量 (延迟取整策略)

    递推公式: X_{t+1} = M·X_t + B_t
    其中 B_t = NEW_ELDER_RATE * |M·X_t| * new_elder_dist

    第1-4年保留浮点期望值，仅第5年末输出时四舍五入取整
    """
    M = build_transfer_survival_matrix(trans_prob)
    b = np.array(NEW_ELDER_DIST)

    communities = df_pop.index.tolist()
    current_X = {}
    for comm in communities:
        current_X[comm] = np.array([
            float(df_pop.loc[comm, '自理']),
            float(df_pop.loc[comm, '半失能']),
            float(df_pop.loc[comm, '失能']),
        ])

    results_float = {}

    for yr in range(1, years + 1):
        next_X = {}
        for comm in communities:
            vec = current_X[comm]
            after_transfer_death = M @ vec
            total_survived = after_transfer_death.sum()
            new_elders = NEW_ELDER_RATE * total_survived * b
            next_X[comm] = after_transfer_death + new_elders

        current_X = next_X
        results_float[yr] = {comm: next_X[comm].copy() for comm in communities}

    # 第5年末取整输出 (延迟取整)
    results = {}
    for yr in range(1, years + 1):
        rows = []
        for comm in communities:
            vec = results_float[yr][comm]
            rows.append({
                '小区': comm,
                '自理': int(round(vec[0])),
                '半失能': int(round(vec[1])),
                '失能': int(round(vec[2])),
                '总老人': int(round(vec.sum())),
            })
        results[yr] = pd.DataFrame(rows).set_index('小区')

    return results, results_float


def run_q1_1():
    """运行问题1.1"""
    df_pop, trans_prob = load_population()
    results, results_float = predict_population(df_pop, trans_prob, years=5)

    for yr, df in results.items():
        df.to_csv(OUTPUT_DIR / f'q1_1_year{yr}.csv', encoding='utf-8-sig')

    return results, results_float


if __name__ == '__main__':
    results, _ = run_q1_1()
    for yr in sorted(results):
        print(f'\n=== 第{yr}年末 ===')
        print(results[yr].to_string())
