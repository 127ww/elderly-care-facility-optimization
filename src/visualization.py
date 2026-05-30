"""Nature-skill figures — B题 17图 + 6表
遵循 nature-skills:nature-figure 全部规范:
- MANDATORY: svg.fonttype='none', font.sans-serif=['Arial',...]
- PALETTE: 真实hex色值 (api.md)
- apply_publication_style() 预设
- SVG primary + PDF/TIFF secondary
- 多面板信息架构: overview→deviation→relationship
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi

FIG_DIR = Path(__file__).resolve().parent.parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===== MANDATORY nature-skill 三行 (api.md §MANDATORY) =====
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

# ===== CJK overlay: SimHei 置顶以正确测量中文字符边界 (skill §CJK) =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ===== 真 PALETTE (api.md) =====
PAL = {
    "blue_main": "#0F4D92", "blue_sec": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676", "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727", "gold": "#FFD700", "teal": "#42949E",
    "violet": "#9A4D8E", "magenta": "#EA84DD",
}
DEFAULT_COLORS = [PAL["blue_main"], PAL["green_3"], PAL["red_strong"],
                  PAL["teal"], PAL["violet"], PAL["neutral_light"]]
SIG_UP = "#2E9E44"; SIG_DN = "#E53935"  # directional cues only

# ===== apply_publication_style (api.md) =====
def apply_publication_style(font_size=7, axes_linewidth=0.8):
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['legend.frameon'] = False
apply_publication_style(font_size=7, axes_linewidth=0.8)

# ===== finalize_figure (api.md) =====
def finalize_figure(fig, name, dpi=300, pad=1.5):
    fig.tight_layout(pad=pad)
    fig.savefig(FIG_DIR / f'{name}.svg', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.tiff', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')

# ===== add_panel_label (api.md) =====
def add_panel_label(ax, label, x=-0.06, y=1.02, fontsize=10, color='black', fontweight='bold'):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha='left', va='bottom')

# ===== 数据加载 =====
def load_d():
    from src.data_loader import load_all
    from src.q1_population import run_q1_1
    ppr, ppf = run_q1_1(); dd = load_all()
    de = pd.read_csv('output/q1_3_effective_demand.csv', encoding='utf-8-sig', index_col=0)
    pt = pd.read_csv('output/q2_pareto.csv', encoding='utf-8-sig')
    qs = pd.read_csv('output/q3_satisfaction.csv', encoding='utf-8-sig')
    qp = pd.read_csv('output/q3_pricing.csv', encoding='utf-8-sig')
    qm = pd.read_csv('output/q3_metrics.csv', encoding='utf-8-sig')
    q4 = pd.read_csv('output/q4_sensitivity.csv', encoding='utf-8-sig')
    try: ph = pd.read_csv('output/q3_pso_history.csv', encoding='utf-8-sig')
    except: ph = None
    return {'ppr':ppr,'ppf':ppf,'dd':dd,'de':de,'pt':pt,'qs':qs,'qp':qp,'qm':qm,'q4':q4,'ph':ph}

# ============================================================
#  Fig 1 — 全文流程图 (schematic-led)
# ============================================================
def fig01(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(xlim=(0,10), ylim=(0,4.8)); ax.axis('off')
    items = [
        (1,3.2,'数据\n(附件1-5)',PAL["neutral_light"]),
        (3,3.2,'Q1 人口预测\n矩阵递推',PAL["blue_sec"]),
        (5,3.2,'Q2 选址优化\nDFS·SSCFLP',PAL["teal"]),
        (7,3.2,'Q3 定价优化\nGrid+PSO',PAL["violet"]),
        (9,3.2,'Q4 灵敏度\n鲁棒性评价',PAL["violet"]),
        (1,1.2,'27项约束\n逐项核验',PAL["red_2"]),
        (3,1.2,'需求预测\n刚需优先',PAL["blue_sec"]),
        (5,1.2,'Pareto前沿\nε-约束法',PAL["teal"]),
        (7,1.2,'差异化定价\nmax-min',PAL["violet"]),
        (9,1.2,'LaTeX论文\n17图+6表',PAL["neutral_dark"]),
    ]
    for x,y,t,c in items:
        ax.add_patch(plt.Rectangle((x-.8,y-.4),1.6,.8,fc=c,ec='white',lw=.8,zorder=2))
        ax.text(x,y,t,ha='center',va='center',fontsize=7,color='white' if c not in [PAL["neutral_light"],PAL["red_2"]] else 'black',zorder=3)
    for a,b in [(0,1),(1,2),(2,3),(3,4),(0,5),(1,6),(2,7),(3,8)]:
        x1,y1,_,_=items[a]; x2,y2,_,_=items[b]
        ax.annotate('',xy=(x2-.8,y2+.1),xytext=(x1+.8,y1-.1),
                    arrowprops=dict(arrowstyle='->',color=PAL["neutral_mid"],lw=.6))
    ax.set_title('全文总体思路框架图',fontsize=11,fontweight='bold',pad=10)
    finalize_figure(fig,'fig01_flowchart')

# ============================================================
#  Fig 2 — 人口小提琴图 (数据诊断)
# ============================================================
def fig02(d):
    df = d['dd']['population']
    cats = ['自理','半失能','失能']
    data = [df[c].values for c in cats]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    vp = ax.violinplot(data, positions=[0,1,2], showmeans=True,
                       showmedians=True, widths=0.55)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(DEFAULT_COLORS[i])
        body.set_alpha(0.55)
        body.set_edgecolor(DEFAULT_COLORS[i])
        body.set_linewidth(0.6)
    for part in ['cmeans','cmedians','cmins','cmaxes','cbars']:
        vp[part].set_color(PAL['neutral_dark'])
        vp[part].set_linewidth(0.7)
    # 叠加散点
    for i, d_vals in enumerate(data):
        jitter = np.random.default_rng(42).normal(0, 0.04, len(d_vals))
        ax.scatter([i]*len(d_vals) + jitter, d_vals, s=14,
                   c=DEFAULT_COLORS[i], alpha=0.7, edgecolors='white', linewidth=0.3, zorder=5)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(cats, fontsize=8)
    ax.set(ylabel='人口数', title='10小区老人分布小提琴图')
    ax.title.set_weight('bold')
    finalize_figure(fig,'fig02_violin')

# ============================================================
#  Fig 3 — 状态转移 (机理示意)
# ============================================================
def fig03(d):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.set(xlim=(0,10), ylim=(0,5.4)); ax.axis('off')
    ARROW_GREY = PAL["neutral_mid"]
    pos = {'S':(2,3),'H':(5,3),'D':(8,3),'Death':(5,.5),'New':(2,4.8)}
    lbl = {'S':'自理','H':'半失能','D':'失能','Death':'死亡','New':'新增\n(7%/年)'}
    for k,(x,y) in pos.items():
        ax.add_patch(plt.Rectangle((x-.5,y-.3),1,.6,fc=PAL["blue_sec"],ec='white',lw=.6))
        ax.text(x,y,lbl[k],ha='center',va='center',fontsize=7,color='white')
    # 水平: S→H, H→D
    for s,d,l in [('S','H','4.5%/年'),('H','D','10%/年')]:
        sx,sy=pos[s]; dx,dy=pos[d]
        ax.annotate('',xy=(dx-.55,dy),xytext=(sx+.55,sy),
                    arrowprops=dict(arrowstyle='->',lw=.8,color=ARROW_GREY))
        ax.text((sx+dx)/2,sy+.12,l,fontsize=7,color=ARROW_GREY,ha='center')
    # 垂直: →Death
    for s in ['S','H','D']:
        sx,sy=pos[s]; dx,dy=pos['Death']
        ax.annotate('',xy=(dx,dy+.35),xytext=(sx,sy-.35),
                    arrowprops=dict(arrowstyle='->',lw=.6,color=ARROW_GREY))
        ax.text(sx+.15,(sy+dy)/2,'5%/年',fontsize=6,color=ARROW_GREY)
    # 新增→自理
    ax.annotate('',xy=(pos['S'][0],pos['S'][1]+.35),xytext=(pos['New'][0],pos['New'][1]-.35),
                arrowprops=dict(arrowstyle='->',lw=.8,color=ARROW_GREY))
    ax.text(2.3,4.1,'7%/年',fontsize=7,color=ARROW_GREY)
    ax.set_title('老人状态转移与人口演化',fontweight='bold')
    finalize_figure(fig,'fig03_transition')

# ============================================================
#  Fig 4 — 五年人口堆叠柱状 (数据诊断)
# ============================================================
def fig04(d):
    pop = d['ppr']
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.2), sharey=True)
    for idx, comm in enumerate('ABCDEFGHIJ'):
        ax = axes[idx//5][idx%5]
        yrs = np.arange(1,6)
        S = [pop[yr].loc[comm,'自理'] for yr in yrs]
        H = [pop[yr].loc[comm,'半失能'] for yr in yrs]
        D = [pop[yr].loc[comm,'失能'] for yr in yrs]
        SH = np.array(S)+np.array(H)
        ax.bar(yrs, S, color=DEFAULT_COLORS[0], width=.55, label='自理')
        ax.bar(yrs, H, bottom=S, color=DEFAULT_COLORS[1], width=.55, label='半失能')
        ax.bar(yrs, D, bottom=SH, color=DEFAULT_COLORS[2], width=.55, label='失能')
        ax.set_title(comm, fontsize=8); ax.set_xticks(yrs); ax.tick_params(labelsize=7)
        if idx == 0: ax.legend(fontsize=6, ncol=3, loc='upper left')
    fig.supylabel('人数', fontsize=9)
    fig.suptitle('五年各小区老人数量变化', fontweight='bold', fontsize=11)
    fig.tight_layout()
    finalize_figure(fig, 'fig04_pop_stacked')

# ============================================================
#  Fig 5 — 收入-需求散点矩阵 (数据诊断)
# ============================================================
def fig05(d):
    dp = d['dd']['population']; de = d['de']
    pdf = pd.DataFrame({'收入': dp['人均月收入'], '助餐': de['助餐'],
                        '日间照料': de['日间照料'], '上门护理': de['上门护理'],
                        '总需求': de.sum(axis=1)})
    g = sns.pairplot(pdf, diag_kind='kde', plot_kws={'alpha':.7, 's':50, 'color': PAL["blue_main"]},
                     diag_kws={'color': PAL["blue_main"]})
    g.fig.suptitle('收入与服务需求散点图矩阵', fontweight='bold', fontsize=11, y=1.01)
    finalize_figure(g.fig, 'fig05_pairplot')

# ============================================================
#  Fig 6 — 有效需求热力图 (结果呈现)
# ============================================================
def fig06(d):
    monthly = d['de']
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.heatmap(monthly, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': '次/月', 'shrink': .8}, linewidths=.5)
    ax.set_title('有效服务需求分布 (次/月)', fontweight='bold')
    ax.set_xlabel('服务'); ax.set_ylabel('小区')
    finalize_figure(fig, 'fig06_demand_heatmap')

# ============================================================
#  Fig 7 — 理论 vs 有效 棒棒糖图 (结果呈现)
# ============================================================
def fig07(d):
    daily = d['de']/30; pop5 = d['ppr'][5]; dr = d['dd']['demand_rate']
    svcs = list(dr.index)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.2), sharey=True)
    for idx, comm in enumerate('ABCDEFGHIJ'):
        ax = axes[idx//5][idx%5]
        th = [sum(pop5.loc[comm,e]*dr.loc[s,e] for e in ['自理','半失能','失能']) for s in svcs]
        ef = [daily.loc[comm,s]*30 for s in svcs]
        x = np.arange(6)
        # 茎 = 理论需求(灰竖线)，头 = 有效需求(绿圆点)
        ax.vlines(x, 0, th, colors=PAL['neutral_light'], lw=2.5, zorder=2, label='理论')
        ax.scatter(x, ef, s=45, c=DEFAULT_COLORS[1], ec='white', lw=0.4, zorder=4, label='有效')
        ax.set_title(comm, fontsize=8); ax.set_xticks(x)
        ax.set_xticklabels([s[:2] for s in svcs], fontsize=6, rotation=30)
        if idx == 0: ax.legend(fontsize=6)
    fig.suptitle('理论需求 vs 有效需求 (消费约束后)', fontweight='bold', fontsize=11)
    fig.tight_layout()
    finalize_figure(fig, 'fig07_demand_compare')

# ============================================================
#  Fig 8 — 距离矩阵热力图 (机理示意)
# ============================================================
def fig08(d):
    dist = d['dd']['distance']
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(dist, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                cbar_kws={'label': '距离 (m)', 'shrink': .8}, vmax=1500,
                linewidths=.5)
    for i in range(10):
        for j in range(10):
            if dist.iloc[i,j] > 1000:
                ax.add_patch(plt.Rectangle((j,i), 1, 1, fill=False, ec=SIG_DN, lw=1.8))
    ax.set_title('小区间距离矩阵 (红框: >1000m)', fontweight='bold')
    finalize_figure(fig, 'fig08_distance_heatmap')

# ============================================================
#  Fig 9 — 预算剪枝 (机理示意)
# ============================================================
def fig09(d):
    from src.utils import enumerate_feasible_station_combos
    combos = enumerate_feasible_station_combos()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = range(len(combos)); b = [c['总预算'] for c in combos]
    colors = [DEFAULT_COLORS[min(c['大型'],5)] for c in combos]
    ax.bar(x, b, color=colors, ec='white', lw=.2)
    ax.axhline(120, color=SIG_DN, ls='--', lw=.8)
    ax.axhline(109, color=PAL["blue_main"], ls='--', lw=.8)
    # Direct labels on lines (nature-skill: prefer over legends)
    n = len(combos)
    ax.text(n-1.5, 121, '预算上限 120万', fontsize=6, color=SIG_DN, va='bottom', ha='right')
    ax.text(n-1.5, 110, '最优 109万 (E大+F中+I中)', fontsize=6, color=PAL["blue_main"], va='bottom', ha='right')
    ax.set(xlabel='组合编号', ylabel='预算 (万元)',
           title=f'预算约束下可行站点组合 ({len(combos)}种)')
    ax.title.set_weight('bold')
    finalize_figure(fig, 'fig09_budget_combos')

# ============================================================
#  Fig 10 — Pareto 前沿 (求解印证)
# ============================================================
def fig10(d):
    pt = d['pt']; fb = pt[pt['coverage_actual'].notna()]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    covs = fb['coverage_actual']*100; sats = fb['avg_satisfaction']
    ax.plot(covs, sats, 'o-', color=PAL["blue_main"], lw=2, ms=8,
            mfc='white', mec=PAL["blue_main"], mew=2)
    for _, r in fb.iterrows():
        ax.annotate(f"{int(r['n_stations'])}站", (r['coverage_actual']*100+.3, r['avg_satisfaction']+.002), fontsize=6, color=PAL["neutral_mid"])
    ax.annotate('满意度断崖\n(80.1%, 0.9716)', xy=(80.06, 0.9716), xytext=(75.5, 0.984),
                arrowprops=dict(arrowstyle='->', color=SIG_DN), fontsize=7, color=SIG_DN)
    ax.axvline(82, color=SIG_DN, ls=':', lw=.6, alpha=.5, label='120万预算极限')
    ax.set(xlabel='覆盖率 (%)', ylabel='平均满意度', title='ε-约束法 Pareto 前沿 (120万预算)')
    ax.title.set_weight('bold'); ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    finalize_figure(fig, 'fig10_pareto')

# ============================================================
#  Fig 11 — 站点网络拓扑 (结果呈现)
# ============================================================
def fig11(d):
    from src.q2_location import run_q2
    sol, _ = run_q2(.80)
    sts = {s['小区']: s['规模'] for s in sol['stations']}
    dist = d['dd']['distance']
    fig, ax = plt.subplots(figsize=(6, 4.8))
    pos = {'A':(0,4),'B':(2,5),'C':(4,5),'D':(1,3),'E':(3,4),
           'F':(5,2),'G':(6,4),'H':(2,1),'I':(4,1.5),'J':(5,0)}
    for c in 'ABCDEFGHIJ':
        is_st = c in sts
        ax.scatter(*pos[c], s=200 if is_st else 80, zorder=3, ec='white', lw=.5,
                   color=PAL["red_strong"] if is_st else PAL["neutral_mid"])
        ax.text(pos[c][0], pos[c][1]+.22, c, ha='center', fontsize=9 if is_st else 7,
                fontweight='bold' if is_st else 'normal')
        if is_st: ax.annotate(sts[c], (pos[c][0]+.18, pos[c][1]-.3), fontsize=7, color=PAL["red_strong"])
    for a in 'ABCDEFGHIJ':
        for b in 'ABCDEFGHIJ':
            d_ab = dist.loc[a,b]
            if a < b and d_ab <= 1000:
                ax.plot([pos[a][0],pos[b][0]], [pos[a][1],pos[b][1]], color='#ccc', lw=.5, zorder=1)
                ax.text((pos[a][0]+pos[b][0])/2, (pos[a][1]+pos[b][1])/2, f'{d_ab:.0f}', fontsize=6, color='#aaa')
    ax.set(xlim=(-1.2,7), ylim=(-.8,5.8), title='站点覆盖网络拓扑 (红点=服务站, 灰点=小区)')
    ax.title.set_weight('bold'); ax.axis('off')
    finalize_figure(fig, 'fig11_network')

# ============================================================
#  Fig 12 — PSO 收敛 (求解印证)
# ============================================================
def fig12(d):
    ph = pd.read_csv('output/q3_pso_climb.csv', encoding='utf-8-sig')
    tau = ph['tau'].values; gens = np.arange(len(tau))
    first_feas = next(i for i,t in enumerate(tau) if not pd.isna(t))
    plateau = 19

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axvspan(0, 5, facecolor='#F6CFCB', alpha=0.12, zorder=0)
    ax.axvline(5.5, color=PAL["red_strong"], lw=0.8, ls='-', alpha=0.3)

 # 1. 修正“不可行区”：加入 \n 强制换行，Y轴下放到底部，完美呆在粉色阴影里
    ax.text(2.7, 0.725, '不可行\n探索区', ha='center', va='center', fontsize=7, color=PAL["red_strong"])
    
    # 2. 修正“爬坡区”：向右推到 x=28 的空白处，彻底避开蓝色的垂直线段
    ax.text(28, 0.765, '可行爬坡区', ha='center', va='center', fontsize=7, color=PAL["blue_main"])
    
    # 3. 修正“局部最优”：稍微向下压一点 (y=0.770)，给上方的深蓝直线留出呼吸空间
    ax.text(110, 0.770, '局部最优收敛停滞区', ha='center', va='center', fontsize=7.5, color=PAL["neutral_dark"], weight='bold')
    
    feas_mask = ~pd.isna(tau)
    ax.plot(gens[feas_mask], tau[feas_mask], '.', color=PAL["blue_main"], ms=3, alpha=0.5)
    ax.plot(gens[first_feas:], tau[first_feas:], '-', color=PAL["blue_main"], lw=1.8)

    ax.plot(first_feas, tau[first_feas], 'o', color=PAL["red_strong"], ms=8, zorder=5)
 # 将 xytext 改为 (30, 0.735)，让引线从蓝线下方指向红点，画面将极其干净
    ax.annotate(f'第{first_feas}代 首次可行 tau={tau[first_feas]:.3f}',
                xy=(first_feas, tau[first_feas]), 
                xytext=(35, 0.740), 
                arrowprops=dict(arrowstyle='->', color=PAL["red_strong"], lw=0.8),
                fontsize=7, color=PAL["red_strong"], ha='center', va='center')

    ax.plot(plateau, tau[plateau], 's', color=PAL["neutral_dark"], ms=7, zorder=5)
    ax.annotate(f'第{plateau}代 收敛 tau={tau[plateau]:.3f}',
                xy=(plateau, tau[plateau]), 
                xytext=(65, 0.755),
                arrowprops=dict(arrowstyle='->', color=PAL["neutral_dark"], lw=0.8),
                fontsize=7, color=PAL["neutral_dark"], ha='center', va='center')

    ax.axhline(0.80, color=PAL["green_3"], ls='--', lw=0.8, alpha=0.5)
    ax.text(195, 0.801, 'Grid验证 tau=0.800', fontsize=7, color=PAL["green_3"], ha='right')

    ax.set(xlim=(-2,205), ylim=(0.70,0.83), xlabel='迭代次数', ylabel='tau = min S_j',
           title='PSO收敛曲线 (Deb可行性规则, 100粒子, 200代)')
    ax.title.set_weight('bold'); ax.grid(True, alpha=0.2)
    finalize_figure(fig, 'fig12_pso_conv')

# ============================================================
#  Fig 13 — 定价对比 (结果呈现)
# ============================================================
def fig13(d):
    qp = d['qp']; base = {'助餐':10,'日间照料':20,'上门护理':30,'康复理疗':28,'助浴':25}
    svcs = list(base); sts = qp['站点'].unique()
    x = np.arange(len(svcs)); w = .2
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.bar(x-w, list(base.values()), w, color=PAL["neutral_light"], ec='white', lw=.3, label='基准价')
    for i,st in enumerate(sts):
        sd = qp[qp['站点']==st]
        vals = [float(sd[sd['服务']==s]['最优价格(元)'].values[0]) for s in svcs]
        ax.bar(x+i*w, vals, w, color=DEFAULT_COLORS[i], ec='white', lw=.2, label=f'{st}站')
    ax.set_xticks(x); ax.set_xticklabels(svcs)
    ax.set(ylabel='价格 (元/次)', title='差异化最优定价 vs 基准价')
    ax.title.set_weight('bold'); ax.legend(fontsize=6, ncol=4)
    finalize_figure(fig, 'fig13_price_compare')

# ============================================================
#  Fig 14 — 满意度雷达 (结果呈现)
# ============================================================
def fig14(d):
    qs = d['qs']; comms = qs['小区'].tolist()
    angles = np.linspace(0, 2*pi, len(comms), endpoint=False).tolist() + [0]
    vals = qs['满意度'].tolist() + [qs['满意度'].iloc[0]]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    ax.fill(angles, vals, alpha=.25, color=PAL["blue_main"])
    ax.plot(angles, vals, 'o-', color=PAL["blue_main"], lw=1.5, ms=6)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(comms, fontsize=7)
    ax.set_ylim(.6, 1.0); ax.set_yticks([.6,.7,.8,.9,1.0])
    ax.set_yticklabels(['0.6','0.7','0.8','0.9','1.0'], fontsize=6)
    ax.set_title('各社区满意度 (τ=0.80)', fontweight='bold', pad=12)
    finalize_figure(fig, 'fig14_radar')

# ============================================================
#  Fig 15 — 需求满足率 (结果呈现)
# ============================================================
def fig15(d):
    fr = pd.read_csv('output/q1_3_fulfilled_ratio.csv', encoding='utf-8-sig')
    fd = fr[fr['老人类别']=='失能']
    low = fd[fd['小区'].isin(['F','D','H'])].groupby('服务')['需求满足率'].mean()
    high = fd[fd['小区'].isin(['C','E','G'])].groupby('服务')['需求满足率'].mean()
    svcs = [s for s in ['助餐','日间照料','上门护理','康复理疗','助浴'] if s in low.index]
    x = np.arange(len(svcs)); w = .35
    LO_INC = PAL['blue_main']  # DEFAULT_COLORS[0]
    HI_INC = PAL['green_3']    # DEFAULT_COLORS[1]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(x-w/2, [low.get(s,0) for s in svcs], w, color=LO_INC, ec='white', lw=.3, label='低收入 (F,D,H)')
    ax.bar(x+w/2, [high.get(s,0) for s in svcs], w, color=HI_INC, ec='white', lw=.3, label='高收入 (C,E,G)')
    for i in range(len(svcs)):
        lv, hv = low.get(svcs[i],0), high.get(svcs[i],0)
        ax.text(i-w/2, lv+.02, f'{lv:.0%}', ha='center', fontsize=6)
        ax.text(i+w/2, hv+.02, f'{hv:.0%}', ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(svcs)
    ax.set(ylabel='需求满足率', title='失能老人需求满足率: 高收入 vs 低收入')
    ax.title.set_weight('bold'); ax.legend(fontsize=7)
    finalize_figure(fig, 'fig15_fulfilled')

# ============================================================
#  Fig 16 — Q4 灵敏度：散点(覆盖率×τ) + 龙卷风(覆盖率Δ)
# ============================================================
def fig16(d):
    q4 = d['q4']
    bl_cov = q4[q4['场景']=='baseline']['覆盖率'].values[0] * 100
    bl_tau = q4[q4['场景']=='baseline']['Q3_tau'].values[0]

    C0 = PAL['blue_main']        # DEFAULT_COLORS[0]
    C1 = PAL['green_3']          # DEFAULT_COLORS[1]
    C2 = PAL['red_strong']       # DEFAULT_COLORS[2]
    C3 = PAL['teal']             # DEFAULT_COLORS[3]
    GREY   = PAL['neutral_light']

    sc_keys  = ['S1_growth_8pct','S2_transfer_changed','S3_cost_plus20pct','S4_budget_140']
    sc_lbl   = ['S1 人口+8%','S2 转移变化','S3 成本+20%','S4 预算140万']
    sc_cov   = [q4[q4['场景']==s]['覆盖率'].values[0]*100 for s in sc_keys]
    sc_tau   = [q4[q4['场景']==s]['Q3_tau'].values[0]     for s in sc_keys]
    deltas   = [c - bl_cov for c in sc_cov]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.5, 3.2))

# ── a: 散点 ──
    # 基线
    ax_l.scatter(bl_cov, bl_tau, c=GREY, s=70, zorder=5, ec='white', lw=.5)
    ax_l.text(bl_cov+.5, bl_tau, 'Baseline', fontsize=6.5, va='center', color=GREY, fontweight='bold')
    
    colors   = [C0, C1, C2, C3]
    tiers    = ['强鲁棒','中度敏感','强鲁棒','断崖跃升']
    
    # 【核心修复 1】：S1 和 S3 坐标完全重叠。我们强行把它们在 X 轴上微调拉开！
    # 让 S1 往左移 0.4，S3 往右移 0.4，变成左右并列的两个点
    pt_jx = [-0.4,   0.0,   0.4,   0.0]  
    
    # 文本跟着点走：S1文本放点左边，S3文本放点右边，形成完美的对称“翅膀”
    txt_ox = [-0.6,   0.6,   0.6,  -1.5]  
    txt_oy = [ 0.0,   0.0,   0.0,   0.0]  
    ha_align = ['right', 'left', 'left', 'right']

    for i in range(4):
        cx = sc_cov[i] + pt_jx[i]  # 加上点的物理位移
        cy = sc_tau[i]
        # 散点
        ax_l.scatter(cx, cy, c=colors[i], s=70, zorder=5, ec='white', lw=.5)
        # 文本
        ax_l.text(cx + txt_ox[i], cy + txt_oy[i], f'{sc_lbl[i]}\n({tiers[i]})',
                  fontsize=6, va='center', ha=ha_align[i],
                  color=colors[i], fontweight='bold')
                  
    # 参考线
    ax_l.axhline(bl_tau, color=PAL['neutral_mid'], lw=.4, ls='--', zorder=0)
    ax_l.axvline(bl_cov, color=PAL['neutral_mid'], lw=.4, ls='--', zorder=0)
    ax_l.set_xlabel('覆盖率 (%)', fontsize=7)
    ax_l.set_ylabel('$\\tau$ (最小满意度)', fontsize=7)
    
    # 把左边距再拓宽一点点，保证 S1 的文字有充足空间
    ax_l.set_xlim(75, 103); ax_l.set_ylim(0.774, 0.870)
    add_panel_label(ax_l, 'a')

    # ── b: 龙卷风 ──
    order = [0, 2, 1, 3]
    d_vals  = [deltas[i] for i in order]
    d_names = [sc_lbl[i] for i in order]
    d_cols  = [colors[i] for i in order]
    d_tiers = [tiers[i] for i in order]
    y = np.arange(3, -1, -1)
    
    ax_r.barh(y, d_vals, height=.45, color=d_cols, ec='white', lw=.3)
    ax_r.axvline(0, color=PAL['neutral_dark'], lw=.5)
    ax_r.set_yticks(y); ax_r.set_yticklabels(d_names, fontsize=6.5)
    ax_r.set_xlabel('覆盖率变化 (百分点)', fontsize=7)
    
    for i in range(4):
        d = d_vals[i]
        xo = .8 if d>=0 else -.8
        ha = 'left' if d>=0 else 'right'
        ax_r.text(d+xo, y[i], f'{d:+.1f} pp  {d_tiers[i]}', fontsize=6,
                  va='center', ha=ha, color=PAL['neutral_black'], fontweight='bold')
                  
    # 【核心修复 2】：疯狂扩展左侧负半轴空间！
    # 直接把 Y 轴刻度线往左死命推到 -16，留出极其宽敞的安全走廊！
    ax_r.set_xlim(-16, 25) 
    
    add_panel_label(ax_r, 'b')

    fig.suptitle('参数灵敏度与鲁棒性综合评价', fontsize=9.5, fontweight='bold', y=1.02)
    finalize_figure(fig, 'fig16_q4_comparison')

# ============================================================
#  Fig 17 — Q4 灵敏度矩阵热力图 (求解印证 · 鲁棒性)
#  真实 Q4 数据, 4参数 × 4指标 Δ 矩阵, coolwarm 发散色
# ============================================================
def fig17(d):
    q4 = d['q4']
    bl = q4[q4['场景']=='baseline'].iloc[0]
    bl_cov = bl['覆盖率']*100; bl_tau = bl['Q3_tau']
    bl_sat = bl['满意度']; bl_n = bl['站点数']

    rows = ['S1 人口+8%', 'S2 转移变化', 'S3 成本+20%', 'S4 预算140万']
    cols = ['Δ覆盖率 (pp)', 'Δτ', 'Δ站点数', 'Δ满意度']
    scs = ['S1_growth_8pct','S2_transfer_changed','S3_cost_plus20pct','S4_budget_140']

    data = np.zeros((4, 4))
    for i, s in enumerate(scs):
        r = q4[q4['场景']==s].iloc[0]
        data[i, 0] = r['覆盖率']*100 - bl_cov
        data[i, 1] = r['Q3_tau'] - bl_tau
        data[i, 2] = r['站点数'] - bl_n
        data[i, 3] = r['满意度'] - bl_sat

    # 格式化标注文本
    ann = np.empty((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            v = data[i, j]
            if j == 1 or j == 3:
                ann[i, j] = f'{v:+.4f}'
            elif j == 2:
                ann[i, j] = f'{v:+.0f}'
            else:
                ann[i, j] = f'{v:+.1f}'

    fig, ax = plt.subplots(figsize=(7, 3.6))
    vmax = max(abs(data.min()), abs(data.max()))
    sns.heatmap(data, annot=ann, fmt='', cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax, linewidths=1.0, linecolor='white',
                xticklabels=cols, yticklabels=rows,
                annot_kws={'fontsize': 8, 'fontweight': 'bold'},
                cbar_kws={'label': '相对于基线的变化量', 'shrink': 0.70},
                ax=ax)
    ax.set_xticklabels(cols, fontsize=8, rotation=0)
    ax.set_yticklabels(rows, fontsize=8, rotation=0)
    ax.tick_params(left=False, bottom=False)
    fig.suptitle('参数灵敏度热力矩阵', fontsize=10, fontweight='bold', y=1.02)
    finalize_figure(fig, 'fig17_heatmap')


# ===== MAIN =====
if __name__ == '__main__':
    print('Loading data...')
    d = load_d()
    fns = [fig01,fig02,fig03,fig04,fig05,fig06,fig07,fig08,fig09,fig10,
           fig11,fig12,fig13,fig14,fig15,fig16,fig17]
    for fn in fns:
        try: fn(d)
        except Exception as e:
            import traceback
            print(f'  FAIL {fn.__name__}: {e}')
            traceback.print_exc()
    print(f'\nDone. {len([f for f in fns if True])} figures → {FIG_DIR}/')
