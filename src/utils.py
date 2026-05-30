"""公共工具函数与常数 — B题 嵌入式社区养老服务站"""
import numpy as np
import pandas as pd

# ===== 随机种子 =====
SEED = 42
np.random.seed(SEED)

# ===== 公共常数 =====
DEATH_RATE = 0.05
NEW_ELDER_RATE = 0.07
BUDGET_CAPEX = 120  # 总建设预算(万元), 仅限CapEx
SERVICE_RADIUS = 1000  # 有效服务半径(米)
PROFIT_RATE_CAP = 0.08  # 利润率上限
PROFIT_RATE_FLOOR = 0.00  # 利润率下界(保本)
PRICE_FLOOR_FACTOR = 0.70  # 定价下限(基准价70%)
PRICE_CEIL_FACTOR = 1.30  # 定价上限(基准价130%)
SUBSIDY_PER_VISIT = 2.0  # 补贴(元/人次), 紧急救助除外
SUBSIDY_CAP = {'小型': 1000, '中型': 1800, '大型': 2600}  # 日补贴上限(元/日)
STATION_CAPACITY = {'小型': 1000, '中型': 2000, '大型': 3000}  # 日服务能力(人次/日)
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365
MONTHS_PER_YEAR = 12
INTERNAL_DISTANCE = 100  # 本小区内部步行距离(米)
SAT_LOWER = 0.6
SAT_UPPER = 1.0
EMERGENCY_SERVICE = '紧急救助'

SUBSIDY_CRITICAL_VISITS = {  # 补贴临界人次
    '小型': SUBSIDY_CAP['小型'] / SUBSIDY_PER_VISIT,   # 500
    '中型': SUBSIDY_CAP['中型'] / SUBSIDY_PER_VISIT,   # 900
    '大型': SUBSIDY_CAP['大型'] / SUBSIDY_PER_VISIT,   # 1300
}

COMMUNITIES = list('ABCDEFGHIJ')
SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
ELDER_TYPES = ['自理', '半失能', '失能']
STATION_SIZES = ['小型', '中型', '大型']

# 收入调整因子: 以A区(3400元)为基准, 低收入小区价格敏感性更高
INCOME_ADJUSTMENT = {}
_base_income = 3400
_income_map = {
    'A': 3400, 'B': 3100, 'C': 3800, 'D': 2900, 'E': 3500,
    'F': 2700, 'G': 3600, 'H': 3000, 'I': 3300, 'J': 3200,
}
for _c, _inc in _income_map.items():
    INCOME_ADJUSTMENT[_c] = _base_income / _inc

# 需求扣减优先级: 从低优先级到高优先级(先被削减的是低优先级)
# 紧急救助(价格=0,公益免费)不参与扣减, 助餐为最高刚需
DEMAND_CUT_PRIORITY_LOW_TO_HIGH = ['助浴', '康复理疗', '上门护理', '日间照料', '助餐']

# ===== 时间量纲转换 =====

def monthly_to_annual(monthly_val):
    return monthly_val * MONTHS_PER_YEAR

def daily_to_annual(daily_val):
    return daily_val * DAYS_PER_YEAR

def annual_capacity(station_size):
    """站点年服务能力(人次/年)"""
    return STATION_CAPACITY[station_size] * DAYS_PER_YEAR

def annual_fixed_cost(station_size, df_cost):
    """年固定运营成本 = 日固定管理成本 × 365"""
    return df_cost.loc[station_size, '日固定管理成本(元/日)'] * DAYS_PER_YEAR

# ===== 满意度计算 =====

def calc_S1(distance_m):
    """距离满意度 S1"""
    if distance_m <= 300:
        return 1.00
    elif distance_m <= 500:
        return 0.90
    elif distance_m <= 650:
        return 0.75
    elif distance_m <= SERVICE_RADIUS:
        return 0.60
    else:
        return 0.0

def calc_S2(load_rate):
    """服务响应满意度 S2"""
    load_rate = np.clip(load_rate, 0, 1)
    if load_rate <= 0.60:
        return 1.00
    elif load_rate <= 0.75:
        return 0.93
    elif load_rate <= 0.85:
        return 0.85
    elif load_rate <= 0.95:
        return 0.72
    else:
        return 0.60

def calc_S3(price_ratio, income_adjust=1.0):
    """价格满意度 S3 (含收入调整因子)"""
    effective_ratio = price_ratio * income_adjust
    if effective_ratio <= 1.00:
        score = 1.00
    elif effective_ratio <= 1.10:
        score = 0.90
    elif effective_ratio <= 1.20:
        score = 0.75
    else:
        score = 0.60
    return np.clip(score, SAT_LOWER, SAT_UPPER)

def calc_satisfaction(distance_m, load_rate, price_ratio=1.0, income_adjust=1.0):
    """综合满意度 S = 0.2*S1 + 0.3*S2 + 0.5*S3"""
    s1 = calc_S1(distance_m)
    s2 = calc_S2(load_rate)
    s3 = calc_S3(price_ratio, income_adjust)
    s = 0.2 * s1 + 0.3 * s2 + 0.5 * s3
    return np.clip(s, SAT_LOWER, SAT_UPPER)

# ===== 覆盖率 =====

def calc_coverage_rate(covered_elder_count, total_elder_count):
    """覆盖率 = 至少享受1项服务的老年人数 / 总老年人口"""
    if total_elder_count == 0:
        return 0.0
    return covered_elder_count / total_elder_count

# ===== 利润率 (原题公式) =====

def calc_profit_rate(total_service_revenue, total_direct_cost, total_subsidy, annual_fixed):
    """利润率 = (服务总收入 - 直接支出 + 补贴 - 年固定运营成本) / 年固定运营成本"""
    if annual_fixed == 0:
        return float('inf')
    return (total_service_revenue - total_direct_cost + total_subsidy - annual_fixed) / annual_fixed

# ===== 距离/覆盖 =====

def build_coverage_mask(dist_matrix, radius=SERVICE_RADIUS):
    """预剪枝: 返回覆盖矩阵和有效(i,j)对"""
    communities = dist_matrix.index.tolist()
    coverage = {}
    valid_pairs = []
    for i in communities:
        for j in communities:
            d = dist_matrix.loc[i, j]
            coverage[(i, j)] = 1 if d <= radius else 0
            if d <= radius:
                valid_pairs.append((i, j))
    return coverage, valid_pairs

# ===== 有效服务人次 =====

def calc_effective_visits(theory_demand, satisfaction):
    """实际有效服务人次 = 理论需求人次 × 服务满意度"""
    return theory_demand * satisfaction

# ===== 补贴计算 (含临界点, 排除紧急救助) =====

def calc_daily_subsidy(daily_effective_visits_non_emergency, station_size):
    """计算日补贴额 (仅非紧急救助服务, 含上限截断)

    约束#1: 紧急救助不享受2元/人次补贴
    """
    raw_subsidy = daily_effective_visits_non_emergency * SUBSIDY_PER_VISIT
    return min(raw_subsidy, SUBSIDY_CAP[station_size])

def calc_annual_subsidy(annual_effective_visits_non_emergency, station_size):
    """年补贴额 = 365 × min(日均非紧急有效人次×2, 日补贴上限)"""
    daily_visits = annual_effective_visits_non_emergency / DAYS_PER_YEAR
    daily_subsidy = min(daily_visits * SUBSIDY_PER_VISIT, SUBSIDY_CAP[station_size])
    return daily_subsidy * DAYS_PER_YEAR

# ===== 消费约束需求扣减 (个体独立) =====

def apply_consumption_cap(demand_series, income, elder_type, consumption_cap,
                          base_prices, direct_costs):
    """按刚需优先顺序扣减需求至消费上限

    对某类老人个体: 先锁死助餐(刚需), 再从低优先级弹性服务(助浴/理疗/护理等)依次削减
    返回: 消费约束后的实际需求次数 Series
    """
    cap_amount = income * consumption_cap[elder_type]

    actual = demand_series.copy()
    total_cost = sum(actual[s] * base_prices[s] for s in actual.index if s in base_prices)

    if total_cost <= cap_amount:
        return np.floor(actual)

    # 按优先级从低到高削减弹性服务
    for service in DEMAND_CUT_PRIORITY_LOW_TO_HIGH:
        if total_cost <= cap_amount:
            break
        service_price = base_prices.get(service, 0)
        if service_price <= 0 or service not in actual.index:
            continue
        current_demand = actual[service]
        if current_demand <= 0:
            continue
        excess = total_cost - cap_amount
        max_cut = min(current_demand, np.ceil(excess / service_price))
        actual[service] = max(0, current_demand - max_cut)
        total_cost = sum(actual[s] * base_prices[s] for s in actual.index if s in base_prices)

    return np.floor(actual)

# ===== 预算剪枝: 枚举可行站点组合 =====

def enumerate_feasible_station_combos(budget=BUDGET_CAPEX, df_cost_in=None):
    """枚举预算约束下所有可行的站点数量组合"""
    costs = {'小型': 18, '中型': 32, '大型': 45}
    combos = []
    max_small = budget // costs['小型']
    max_medium = budget // costs['中型']
    max_large = budget // costs['大型']
    for n_large in range(max_large + 1):
        for n_medium in range(max_medium + 1):
            for n_small in range(max_small + 1):
                total = n_large * costs['大型'] + n_medium * costs['中型'] + n_small * costs['小型']
                if total <= budget:
                    combos.append({
                        '大型': n_large, '中型': n_medium, '小型': n_small,
                        '总站数': n_large + n_medium + n_small,
                        '总预算': total,
                    })
    return sorted(combos, key=lambda x: x['总站数'], reverse=True)
