"""
金融衍生品定价:对数正态模拟 -> 蒙特卡洛定价 -> Bootstrap 误差估计
European Call Option Pricing
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# ============================================================
# 0. 市场参数 (Market Parameters)
# ============================================================
S0    = 100.0   # 当前股价 spot price
K     = 105.0   # 行权价 strike price
r     = 0.05    # 无风险利率 risk-free rate (年化)
sigma = 0.20    # 波动率 volatility (年化)
T     = 1.0     # 到期时间 maturity (年)

RNG = np.random.default_rng(42)  # 固定随机种子,保证结果可复现


# ============================================================
# 1. 对数正态分布模拟股价路径 (GBM path simulation)
#    S(t+dt) = S(t) * exp[(r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]
# ============================================================
def simulate_paths(S0, r, sigma, T, n_steps, n_paths, rng):
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    # 每一步的对数收益
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    # 累加得到对数价格路径,再取指数还原成价格
    log_paths = np.cumsum(log_increments, axis=1)
    S = S0 * np.exp(log_paths)
    # 在最前面拼上初始价格 S0
    S = np.hstack([np.full((n_paths, 1), S0), S])
    return S


# ============================================================
# 2. 蒙特卡洛计算期权收益期望 (Monte Carlo pricing)
#    Call = e^{-rT} * E[max(S_T - K, 0)]
# ============================================================
def mc_european_call(S0, K, r, sigma, T, n_paths, rng):
    # 欧式期权只需终值 S_T,可一步直接采样(对数正态)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs   # 贴现后的单条样本价值
    price = discounted.mean()               # 期望 = 期权价格
    return price, discounted, ST


# 解析解 Black-Scholes 作为"标准答案"对照
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ============================================================
# 3. Bootstrap 估计定价误差 (Bootstrap standard error & CI)
#    对 N 条贴现收益做有放回重采样,重复 B 次,看价格估计的分布
# ============================================================
def bootstrap_price(discounted_payoffs, n_boot, rng, chunk=500):
    n = len(discounted_payoffs)
    boot_prices = np.empty(n_boot)
    # 分块处理,避免一次性分配 n_boot * n 的巨大矩阵
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))     # 有放回抽样的下标
        boot_prices[start:start + b] = discounted_payoffs[idx].mean(axis=1)
    return boot_prices


# ============================================================
# 运行 (Run)
# ============================================================
N = 100_000  # 蒙特卡洛路径数

# --- Step 2: 定价 ---
mc_price, discounted, ST = mc_european_call(S0, K, r, sigma, T, N, RNG)
bs_price = black_scholes_call(S0, K, r, sigma, T)

# 经典蒙特卡洛标准误 SE = std / sqrt(N)
analytic_se = discounted.std(ddof=1) / np.sqrt(N)

# --- Step 3: Bootstrap ---
B = 20_000
boot_prices = bootstrap_price(discounted, B, RNG)
boot_se = boot_prices.std(ddof=1)
ci_low, ci_high = np.percentile(boot_prices, [2.5, 97.5])

print("=" * 55)
print(f"{'Black-Scholes 解析价格':<28}: {bs_price:.4f}")
print(f"{'蒙特卡洛估计价格':<32}: {mc_price:.4f}")
print(f"{'定价误差 (MC - BS)':<30}: {mc_price - bs_price:+.4f}")
print("-" * 55)
print(f"{'经典 MC 标准误 (std/sqrt N)':<27}: {analytic_se:.4f}")
print(f"{'Bootstrap 标准误':<32}: {boot_se:.4f}")
print(f"{'Bootstrap 95% 置信区间':<29}: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"{'真值是否落在区间内':<32}: {ci_low <= bs_price <= ci_high}")
print("=" * 55)

# 估值占比信息
itm_ratio = (ST > K).mean()
print(f"到期实值(S_T>K)比例: {itm_ratio:.1%}  |  到期均价 E[S_T]≈{ST.mean():.2f} (理论 {S0*np.exp(r*T):.2f})")


# ============================================================
# 可视化 (Visualization)
# ============================================================
plt.rcParams.update({'font.size': 10, 'figure.dpi': 110})
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) 样本股价路径
paths = simulate_paths(S0, r, sigma, T, n_steps=252, n_paths=60, rng=np.random.default_rng(7))
t_grid = np.linspace(0, T, 253)
for p in paths:
    axes[0, 0].plot(t_grid, p, lw=0.8, alpha=0.6)
axes[0, 0].axhline(K, color='red', ls='--', lw=1.5, label=f'Strike K={K:.0f}')
axes[0, 0].axhline(S0, color='black', ls=':', lw=1.0, label=f'S0={S0:.0f}')
axes[0, 0].set_title('(a) Simulated GBM Stock Price Paths', fontweight='bold')
axes[0, 0].set_xlabel('Time (years)'); axes[0, 0].set_ylabel('Stock Price')
axes[0, 0].legend(loc='upper left', fontsize=8)

# (b) 终值 S_T 的对数正态分布
axes[0, 1].hist(ST, bins=100, density=True, alpha=0.55, color='steelblue', label='MC samples')
x = np.linspace(ST.min(), np.percentile(ST, 99.7), 400)
# 解析对数正态密度
m = np.log(S0) + (r - 0.5 * sigma**2) * T          # ln S_T 的均值
s = sigma * np.sqrt(T)                               # ln S_T 的标准差
pdf = lognorm.pdf(x, s, scale=np.exp(m))
axes[0, 1].plot(x, pdf, 'r-', lw=2, label='Lognormal pdf (analytic)')
axes[0, 1].axvline(K, color='green', ls='--', lw=1.5, label=f'K={K:.0f}')
axes[0, 1].set_title('(b) Terminal Price $S_T$ ~ Lognormal', fontweight='bold')
axes[0, 1].set_xlabel('$S_T$'); axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(fontsize=8)

# (c) 贴现收益分布(大量为0 = 虚值期权)
axes[1, 0].hist(discounted, bins=100, color='darkorange', alpha=0.7)
axes[1, 0].set_title('(c) Discounted Payoffs $e^{-rT}\\max(S_T-K,0)$', fontweight='bold')
axes[1, 0].set_xlabel('Discounted Payoff'); axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_yscale('log')
zero_pct = (discounted == 0).mean()
axes[1, 0].text(0.55, 0.85, f'{zero_pct:.0%} of paths\nexpire worthless (=0)',
                transform=axes[1, 0].transAxes, fontsize=9,
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

# (d) Bootstrap 价格分布
axes[1, 1].hist(boot_prices, bins=80, density=True, color='mediumseagreen', alpha=0.7)
axes[1, 1].axvline(mc_price, color='blue', lw=2, label=f'MC price={mc_price:.3f}')
axes[1, 1].axvline(bs_price, color='red', ls='--', lw=2, label=f'BS true={bs_price:.3f}')
axes[1, 1].axvspan(ci_low, ci_high, color='green', alpha=0.15, label='95% CI')
axes[1, 1].set_title('(d) Bootstrap Distribution of Price Estimate', fontweight='bold')
axes[1, 1].set_xlabel('Estimated Price'); axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/pricing_overview.png', dpi=120, bbox_inches='tight')
print("\n[saved] pricing_overview.png")


# ============================================================
# 附加:收敛性分析 — MC 误差随路径数 N 的 1/sqrt(N) 衰减
# ============================================================
Ns = np.array([100, 300, 1000, 3000, 10_000, 30_000, 100_000, 300_000])
errors = []
for n in Ns:
    rng_c = np.random.default_rng(2024)
    reps = 50
    est = np.empty(reps)
    for j in range(reps):
        pr, _, _ = mc_european_call(S0, K, r, sigma, T, n, rng_c)
        est[j] = pr
    errors.append(est.std(ddof=1))   # 估计量的真实标准差
errors = np.array(errors)

fig2, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(Ns, errors, 'o-', color='crimson', lw=2, ms=7, label='Empirical MC std')
ref = errors[0] * np.sqrt(Ns[0] / Ns)   # 1/sqrt(N) 参考线
ax.loglog(Ns, ref, 'k--', lw=1.5, label=r'$\propto 1/\sqrt{N}$ reference')
ax.set_xlabel('Number of paths N'); ax.set_ylabel('Std. error of price estimate')
ax.set_title('Monte Carlo Convergence: error $\\sim 1/\\sqrt{N}$', fontweight='bold')
ax.grid(True, which='both', alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig('/home/claude/convergence.png', dpi=120, bbox_inches='tight')
print("[saved] convergence.png")
