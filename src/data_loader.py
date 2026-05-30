"""数据加载与预处理模块 — B题 嵌入式社区养老服务站"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob as _glob

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'B题'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

COMMUNITIES = list('ABCDEFGHIJ')
SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
NON_EMERGENCY_SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴']
ELDER_TYPES = ['自理', '半失能', '失能']
STATION_SIZES = ['小型', '中型', '大型']

def _find_file(pattern):
    """在B题目录下查找附件文件"""
    files = _glob.glob(str(DATA_DIR / pattern))
    if not files:
        raise FileNotFoundError(f'未找到匹配 {pattern} 的文件')
    return files[0]


def load_population():
    """加载附件1：小区基础数据

    Returns:
        df_pop: 人口与老人结构 DataFrame, index=小区
        trans_prob: 转移概率 dict
    """
    path = _find_file('附件1*')
    df_pop = pd.read_excel(path, sheet_name=0, header=1)
    df_pop = df_pop.dropna()
    df_pop.columns = ['小区', '总人口', '60+老人数', '自理', '半失能', '失能', '人均月收入']
    df_pop.set_index('小区', inplace=True)
    for col in df_pop.columns:
        df_pop[col] = pd.to_numeric(df_pop[col])

    trans_df = pd.read_excel(path, sheet_name=1, header=1)
    trans_df = trans_df.dropna()
    trans_prob = {}
    for _, row in trans_df.iterrows():
        text = str(row.iloc[0])
        val = float(row.iloc[1])
        # 解析 "自理 → 半失能" / "半失能 → 失能"
        if '自理' in text and '半失能' in text:
            trans_prob[('自理', '半失能')] = val
        elif '半失能' in text and '失能' in text:
            trans_prob[('半失能', '失能')] = val

    return df_pop, trans_prob


def load_service_demand():
    """加载附件2：服务需求数据

    Returns:
        demand_rate: 每位老人月均服务需求次数 DataFrame, index=服务项
        revenue_cost: 服务营收与支出 DataFrame, index=服务项
        consumption_cap: dict {'自理': 0.2, '半失能': 0.25, '失能': 0.3}
    """
    path = _find_file('附件2*')

    # Sheet 1: 每人月均服务需求次数
    demand_rate = pd.read_excel(path, sheet_name=0, header=1)
    demand_rate = demand_rate.dropna()
    demand_rate.columns = ['服务项目', '自理', '半失能', '失能']
    demand_rate.set_index('服务项目', inplace=True)
    for col in ELDER_TYPES:
        demand_rate[col] = pd.to_numeric(demand_rate[col])

    # Sheet 2: 营收及支出
    revenue_cost = pd.read_excel(path, sheet_name=1, header=1)
    revenue_cost = revenue_cost.dropna()
    revenue_cost.columns = ['服务项目', '营收(元/次)', '直接支出(元/次)']
    revenue_cost.set_index('服务项目', inplace=True)
    # 紧急救助: 营收="0（公益免费）" → 提取数字0
    for col in ['营收(元/次)', '直接支出(元/次)']:
        revenue_cost[col] = revenue_cost[col].apply(
            lambda x: float(str(x).split('（')[0].split('(')[0].strip()) if pd.notna(x) else 0.0
        )

    # Sheet 3: 月服务消费上限
    cap_df = pd.read_excel(path, sheet_name=2, header=None)
    consumption_cap = {}
    cap_map = {'自理': 1, '半失能': 2, '失能': 3}
    for etype, idx in cap_map.items():
        raw = str(cap_df.iloc[idx, 1])
        pct = float(raw.replace('≤', '').replace('%', '').strip()) / 100
        consumption_cap[etype] = pct

    return demand_rate, revenue_cost, consumption_cap


def load_station_cost():
    """加载附件3：服务站建设与运营成本

    Returns:
        df_cost: DataFrame, index=规模
            columns=[建设成本(万元), 日固定管理成本(元/日), 最大服务人次]
    """
    path = _find_file('附件3*')
    df = pd.read_excel(path, sheet_name=0, header=1).iloc[:3]
    df.columns = ['规模', '建设成本(万元)', '日固定管理成本(元/日)', '最大服务人次']
    df.set_index('规模', inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    return df


def load_distance_matrix():
    """加载附件4：小区间距离矩阵

    Returns:
        dist: (10,10) DataFrame, index/columns=ABCDEFGHIJ, values=距离(米)
    """
    path = _find_file('附件4*')
    df = pd.read_excel(path, sheet_name=0, header=1, index_col=0)
    df.columns = list('ABCDEFGHIJ')
    df.index = list('ABCDEFGHIJ')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    # 对角线(m)设为100 (本小区内部步行距离)
    for c in COMMUNITIES:
        df.loc[c, c] = 100
    return df


def load_satisfaction_rules():
    """加载附件5：满意度评分规则

    Returns dict with S1/S2/S3 piecewise thresholds
    """
    return {
        'weights': {'S1': 0.2, 'S2': 0.3, 'S3': 0.5},
        'S1_thresholds': [
            (0,   300, 1.00),
            (300, 500, 0.90),
            (500, 650, 0.75),
            (650, 1000, 0.60),
        ],  # (lower, upper, score)
        'S2_thresholds': [
            (0.00, 0.60, 1.00),
            (0.60, 0.75, 0.93),
            (0.75, 0.85, 0.85),
            (0.85, 0.95, 0.72),
            (0.95, 1.00, 0.60),
        ],  # load_rate interval -> S2
        'S3_thresholds': [
            (0.00, 1.00, 1.00),
            (1.00, 1.10, 0.90),
            (1.10, 1.20, 0.75),
            (1.20, float('inf'), 0.60),
        ],  # price_ratio interval -> S3
    }


def load_all():
    """加载全部数据"""
    df_pop, trans_prob = load_population()
    demand_rate, revenue_cost, consumption_cap = load_service_demand()
    df_cost = load_station_cost()
    df_dist = load_distance_matrix()
    sat_rules = load_satisfaction_rules()
    return {
        'population': df_pop,
        'trans_prob': trans_prob,
        'demand_rate': demand_rate,
        'revenue_cost': revenue_cost,
        'consumption_cap': consumption_cap,
        'station_cost': df_cost,
        'distance': df_dist,
        'satisfaction': sat_rules,
    }
