# 嵌入式社区养老服务站选址与定价优化

26电工杯 B 题完整求解方案，涵盖人口预测、设施选址、差异化定价与灵敏度分析的全链条运筹决策框架。

## 问题概述

为 10 个小区规划嵌入式养老服务站——确定建站数量、位置、规模，并为每项服务制定差异化定价策略，在 120 万元建设预算和 8% 利润率上限约束下最大化老人满意度。

## 建模与算法

| 子问题 | 模型 | 算法 | 关键结果 |
|--------|------|------|----------|
| 人口预测 | 离散 Markov 状态转移 | 矩阵递推 + 刚需优先截断 | 五年末 7449 人，月有效需求 254,576 人次 |
| 选址规划 | SSCFLP 多目标 MILP | DFS + 强预算定界剪枝 | E大+F中+I中，109 万投资，覆盖率 80.06%，剪枝率 93% |
| 定价优化 | Max-Min 公平性 NLP | Deb-PSO (500 粒子, 15 维) | τ = 0.800，I 站利润率仅 0.18% |
| 灵敏度 | 四参数单变量扰动 | 完整重跑 Q2+Q3 管道 | 预算 140 万 → 覆盖率跃升至 100%，τ → 0.846 |

## 技术栈

Python 3 · NumPy · Pandas · Matplotlib · Seaborn · LaTeX (cumcmthesis)

## 文件结构

```
├── paper.pdf              # 竞赛论文 (29 页)
├── requirements.txt       # Python 依赖
├── src/                   # 核心代码
│   ├── data_loader.py     # 数据加载
│   ├── utils.py           # 满意度/补贴计算
│   ├── q1_population.py   # 人口预测
│   ├── q1_demand.py       # 需求计算
│   ├── q2_location.py     # DFS 选址搜索
│   ├── q3_pricing.py      # Deb-PSO 定价优化
│   ├── q4_sensitivity.py  # 灵敏度分析
│   └── visualization.py   # 图表生成
├── output/                # 模型输出 (CSV)
├── figures/               # 论文图表 (PDF)
└── CLAUDE.md              # 项目规范
```

## 复现

```bash
pip install -r requirements.txt
python src/q1_population.py
python src/q1_demand.py
python src/q2_location.py
python src/q3_pricing.py
python src/q4_sensitivity.py
python src/visualization.py
```

## 许可

MIT License
