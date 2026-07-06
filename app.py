import streamlit as st
import math
from scipy.stats import norm, f, ncf
from datetime import datetime

# ============================
# 基础工具函数
# ============================
def z_two_sided(alpha):
    return norm.ppf(1 - alpha / 2)

def z_one_sided(beta):
    return norm.ppf(1 - beta)

def fmt(v, digits=4):
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"

# ============================
# 12类样本量计算模块
# ============================
METHODS = []

# 1. 简单随机抽样（率）
def calc_srs_rate(vals):
    Z = z_two_sided(vals["alpha"])
    n_raw = (Z**2 * vals["P"] * (1 - vals["P"])) / (vals["d"]**2)
    n_adj = n_raw / (1 + (n_raw - 1) / vals["N"])
    return {"n": n_raw, "n_adj": n_adj}

METHODS.append({
    "id": "srs_rate",
    "category": "抽样调查",
    "name": "简单随机抽样（率）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "P": {"label": "预期总体率 P", "default": 0.5, "min": 0.001, "max": 0.999, "step": 0.01},
        "d": {"label": "容许误差 d", "default": 0.05, "min": 0.001, "step": 0.001},
        "N": {"label": "总体规模 N", "default": 10000, "min": 1, "step": 1}
    },
    "formula": r"n = \frac{Z_{\alpha/2}^{2} P(1-P)}{d^2},\quad n_{adj} = \frac{n}{1+\frac{n-1}{N}}",
    "results": [
        {"id": "n", "label": "原始样本量n", "desc": "不校正总体"},
        {"id": "n_adj", "label": "校正样本量nₐdj", "desc": "有限总体校正"}
    ],
    "calc": calc_srs_rate
})

# 2. 简单随机抽样（均数）
def calc_srs_mean(vals):
    Z = z_two_sided(vals["alpha"])
    n_raw = (Z**2 * vals["sigma"]**2) / (vals["d"]**2)
    n_adj = n_raw / (1 + (n_raw - 1) / vals["N"])
    return {"n": n_raw, "n_adj": n_adj}

METHODS.append({
    "id": "srs_mean",
    "category": "抽样调查",
    "name": "简单随机抽样（均数）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "sigma": {"label": "总体标准差 σ", "default": 10.0, "min": 0.001, "step": 0.1},
        "d": {"label": "容许误差 d", "default": 2.0, "min": 0.001, "step": 0.1},
        "N": {"label": "总体规模 N", "default": 10000, "min": 1, "step": 1}
    },
    "formula": r"n = \frac{Z_{\alpha/2}^2 \sigma^2}{d^2},\quad n_{adj} = \frac{n}{1+\frac{n-1}{N}}",
    "results": [
        {"id": "n", "label": "原始样本量n", "desc": "不校正总体"},
        {"id": "n_adj", "label": "校正样本量nₐdj", "desc": "有限总体校正"}
    ],
    "calc": calc_srs_mean
})

# 3. 整群抽样（率）
def calc_cluster_rate(vals):
    Z = z_two_sided(vals["alpha"])
    n_srs = (Z**2 * vals["P"] * (1 - vals["P"])) / (vals["d"]**2)
    Deff = 1 + (vals["m"] - 1) * vals["rho"]
    n_cluster = n_srs * Deff
    K = math.ceil(n_cluster / vals["m"])
    return {"n_srs": n_srs, "Deff": Deff, "n_cluster": n_cluster, "K": K}

METHODS.append({
    "id": "cluster_rate",
    "category": "抽样调查",
    "name": "整群随机抽样（率）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "P": {"label": "预期总体率 P", "default": 0.5, "min": 0.001, "max": 0.999, "step": 0.01},
        "d": {"label": "容许误差 d", "default": 0.05, "min": 0.001, "step": 0.001},
        "rho": {"label": "群内相关系数 ρ", "default": 0.05, "min": 0.0, "max": 0.99, "step": 0.001},
        "m": {"label": "平均每群人数 m", "default": 30, "min": 1, "step": 1}
    },
    "formula": r"n_{srs} = \frac{Z_{\alpha/2}^2 P(1-P)}{d^2},\; Deff=1+(m-1)\rho,\; n_{cluster}=n_{srs}\cdot Deff,\; K=\lceil n_{cluster}/m\rceil",
    "results": [
        {"id": "n_srs", "label": "简单随机样本量nₛᵣₛ", "desc": "单纯随机抽样所需样本"},
        {"id": "Deff", "label": "设计效应Deff", "desc": "整群抽样膨胀系数"},
        {"id": "n_cluster", "label": "总样本量n_cluster", "desc": "整群抽样总人数"},
        {"id": "K", "label": "需抽取群数K", "desc": "最少抽取群数量"}
    ],
    "calc": calc_cluster_rate
})

# 4. 整群抽样（均数）
def calc_cluster_mean(vals):
    Z = z_two_sided(vals["alpha"])
    n_srs = (Z**2 * vals["sigma"]**2) / (vals["d"]**2)
    Deff = 1 + (vals["m"] - 1) * vals["rho"]
    n_cluster = n_srs * Deff
    K = math.ceil(n_cluster / vals["m"])
    return {"n_srs": n_srs, "Deff": Deff, "n_cluster": n_cluster, "K": K}

METHODS.append({
    "id": "cluster_mean",
    "category": "抽样调查",
    "name": "整群随机抽样（均数）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "sigma": {"label": "总体标准差 σ", "default": 10.0, "min": 0.001, "step": 0.1},
        "d": {"label": "容许误差 d", "default": 2.0, "min": 0.001, "step": 0.1},
        "rho": {"label": "群内相关系数 ρ", "default": 0.05, "min": 0.0, "max": 0.99, "step": 0.001},
        "m": {"label": "平均每群人数 m", "default": 30, "min": 1, "step": 1}
    },
    "formula": r"n_{srs} = \frac{Z_{\alpha/2}^2 \sigma^2}{d^2},\; Deff=1+(m-1)\rho,\; n_{cluster}=n_{srs}\cdot Deff,\; K=\lceil n_{cluster}/m\rceil",
    "results": [
        {"id": "n_srs", "label": "简单随机样本量nₛᵣₛ", "desc": "单纯随机抽样所需样本"},
        {"id": "Deff", "label": "设计效应Deff", "desc": "整群抽样膨胀系数"},
        {"id": "n_cluster", "label": "总样本量n_cluster", "desc": "整群抽样总人数"},
        {"id": "K", "label": "需抽取群数K", "desc": "最少抽取群数量"}
    ],
    "calc": calc_cluster_mean
})

# 5. 单组率与总体率比较
def calc_one_prop(vals):
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])
    term1 = Za * math.sqrt(vals["P0"] * (1 - vals["P0"]))
    term2 = Zb * math.sqrt(vals["P1"] * (1 - vals["P1"]))
    numerator = (term1 + term2) ** 2
    denominator = (vals["P1"] - vals["P0"]) ** 2
    n = numerator / denominator
    return {"n": n}

METHODS.append({
    "id": "one_prop",
    "category": "率的比较",
    "name": "单组率与总体率比较",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.01},
        "P0": {"label": "总体率 P₀", "default": 0.30, "min": 0.001, "max": 0.999, "step": 0.01},
        "P1": {"label": "预期样本率 P₁", "default": 0.40, "min": 0.001, "max": 0.999, "step": 0.01}
    },
    "formula": r"n = \frac{\big(Z_{\alpha/2}\sqrt{p_0(1-p_0)} + Z_\beta\sqrt{p_1(1-p_1)}\big)^2}{(p_1-p_0)^2}",
    "results": [{"id": "n", "label": "所需样本量n", "desc": "单侧样本例数"}],
    "calc": calc_one_prop
})

# 6. 配对四格表率比较
def calc_paired_prop(vals):
    OR = vals["P10"] / vals["P01"]
    PD = vals["P01"] + vals["P10"]
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])
    inside_sqrt = (OR + 1)**2 - (OR - 1)**2 * PD
    numerator = (Za * (OR + 1) + Zb * math.sqrt(inside_sqrt)) ** 2
    denominator = ((OR - 1)**2) * PD
    N_pairs = numerator / denominator
    return {"N_pairs": N_pairs, "OR": OR, "PD": PD}

METHODS.append({
    "id": "paired_prop",
    "category": "率的比较",
    "name": "两组率比较（配对四格表）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.001},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.001},
        "P01": {"label": "P₀₁ (阴性转阳性)", "default": 0.10, "min": 0.001, "max": 0.99, "step": 0.001},
        "P10": {"label": "P₁₀ (阳性转阴性)", "default": 0.05, "min": 0.001, "max": 0.99, "step": 0.001}
    },
    "formula": r"N_{pairs} = \frac{\big[Z_{\alpha/2}(OR+1)+Z_\beta\sqrt{(OR+1)^2-(OR-1)^2 PD}\big]^2}{(OR-1)^2 PD}",
    "results": [
        {"id": "N_pairs", "label": "配对对子数N_pairs", "desc": "所需配对样本量"},
        {"id": "OR", "label": "比值比OR", "desc": "P10/P01"},
        {"id": "PD", "label": "不一致比例PD", "desc": "P01+P10"}
    ],
    "calc": calc_paired_prop
})

# 7. 两组独立率比较
def calc_two_prop(vals):
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])
    pA = vals["pA"]
    pB = vals["pB"]
    k = vals["k"]
    part = pA*(1-pA)/k + pB*(1-pB)
    coeff = ((Za + Zb) / (pA - pB)) ** 2
    nB = part * coeff
    nA = nB * k
    return {"nB": nB, "nA": nA}

METHODS.append({
    "id": "two_prop",
    "category": "率的比较",
    "name": "两组率比较（独立成组）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.01},
        "pA": {"label": "试验组率 pA", "default": 0.50, "min": 0.001, "max": 0.999, "step": 0.01},
        "pB": {"label": "对照组率 pB", "default": 0.30, "min": 0.001, "max": 0.999, "step": 0.01},
        "k": {"label": "样本分配比例 k=nA/nB", "default": 1.0, "min": 0.01, "step": 0.1}
    },
    "formula": r"n_B = \left(\frac{p_A(1-p_A)}{k}+p_B(1-p_B)\right)\left(\frac{Z_{\alpha/2}+Z_\beta}{p_A-p_B}\right)^2,\quad n_A = k\cdot n_B",
    "results": [{"id": "nB", "label": "对照组样本量nB", "desc": "对照组例数"},{"id": "nA", "label": "试验组样本量nA", "desc": "试验组例数"}],
    "calc": calc_two_prop
})

# 8. 三组率比较
def calc_three_prop(vals):
    ps = [vals["p1"], vals["p2"], vals["p3"]]
    pbar = sum(ps) / 3
    w2_sum = 0
    for p in ps:
        w2_sum += ((p - pbar)**2) / (pbar * (1 - pbar))
    w2 = w2_sum / 3
    lam = 12.65 if vals["beta"] <= 0.10 else 9.63
    N = lam / w2
    n_per = N / 3
    w = math.sqrt(w2)
    return {"w": w, "N": N, "n_per": n_per, "lambda": lam}

METHODS.append({
    "id": "three_prop",
    "category": "率的比较",
    "name": "三组率比较（卡方）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.01},
        "p1": {"label": "组1率 p₁", "default": 0.30, "min": 0.001, "max": 0.999, "step": 0.01},
        "p2": {"label": "组2率 p₂", "default": 0.40, "min": 0.001, "max": 0.999, "step": 0.01},
        "p3": {"label": "组3率 p₃", "default": 0.50, "min": 0.001, "max": 0.999, "step": 0.01}
    },
    "formula": r"w = \sqrt{\frac{1}{3}\sum \frac{(p_i-\bar{p})^2}{\bar{p}(1-\bar{p})}},\quad N = \lambda/w^2",
    "results": [
        {"id": "w", "label": "效应量w", "desc": "Cohen's w"},
        {"id": "N", "label": "总样本量N", "desc": "三组合计总例数"},
        {"id": "n_per", "label": "每组样本量", "desc": "每组平均例数"},
        {"id": "lambda", "label": "非中心参数λ", "desc": "查表常数"}
    ],
    "calc": calc_three_prop
})

# 9. 单样本均数t检验
def calc_one_mean(vals):
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])
    delta = abs(vals["mu1"] - vals["mu0"])
    sigma = vals["sigma"]
    n = ((Za + Zb) / (delta / sigma))**2 + 0.5 * Za**2
    return {"n": n, "delta": delta}

METHODS.append({
    "id": "one_mean",
    "category": "均数的比较",
    "name": "单组均数与总体均数比较",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.01},
        "mu0": {"label": "总体均数 μ₀", "default": 100.0, "step": 0.1},
        "mu1": {"label": "预期样本均数 μ₁", "default": 105.0, "step": 0.1},
        "sigma": {"label": "标准差 σ", "default": 15.0, "min": 0.001, "step": 0.1}
    },
    "formula": r"n = \left(\frac{Z_{\alpha/2}+Z_\beta}{\delta/\sigma}\right)^2 + \frac12 Z_{\alpha/2}^2,\quad \delta=|\mu_1-\mu_0|",
    "results": [{"id": "n", "label": "所需样本量n", "desc": "单组例数"},{"id": "delta", "label": "均数差值δ", "desc": "两组差值绝对值"}],
    "calc": calc_one_mean
})

# 10. 两组均数比较（方差齐）
def calc_two_mean_equal(vals):
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])
    diff = vals["mu1"] - vals["mu0"]
    sigma = vals["sigma"]
    k = vals["k"]
    term = (Za + Zb)**2 * sigma**2 * (1 + 1/k) / (diff**2)
    n0 = term + 0.25 * Za**2
    n1 = n0 * k
    return {"n0": n0, "n1": n1}

METHODS.append({
    "id": "two_mean_equal",
    "category": "均数的比较",
    "name": "两组均数比较（方差齐）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.1},
        "mu0": {"label": "对照组均数 μ₀", "default": 100.0, "step": 0.1},
        "mu1": {"label": "试验组均数 μ₁", "default": 108.0, "step": 0.1},
        "sigma": {"label": "合并标准差 σ", "default": 15.0, "min": 0.001, "step": 0.1},
        "k": {"label": "分配比例 k=n1/n0", "default": 1.0, "min": 0.01, "step": 0.1}
    },
    "formula": r"n_0 = \frac{(Z_{\alpha/2}+Z_\beta)^2 \sigma^2 (1+1/k)}{(\mu_1-\mu_0)^2} + \frac14 Z_{\alpha/2}^2,\quad n_1 = k\cdot n_0",
    "results": [{"id": "n0", "label": "对照组样本量n₀", "desc": "对照组例数"},{"id": "n1", "label": "试验组样本量n₁", "desc": "试验组例数"}],
    "calc": calc_two_mean_equal
})

# 11. 两组均数比较（方差不齐）
def calc_two_mean_unequal(vals):
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])
    diff = vals["mu1"] - vals["mu0"]
    s0 = vals["sigma0"]
    s1 = vals["sigma1"]
    k = vals["k"]
    term = (Za + Zb)**2 * (s0**2 + s1**2 / k) / (diff**2)
    n0 = term + 0.25 * Za**2
    n1 = n0 * k
    return {"n0": n0, "n1": n1}

METHODS.append({
    "id": "two_mean_unequal",
    "category": "均数的比较",
    "name": "两组均数比较（方差不齐）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.1},
        "mu0": {"label": "对照组均数 μ₀", "default": 100.0, "step": 0.1},
        "mu1": {"label": "试验组均数 μ₁", "default": 108.0, "step": 0.1},
        "sigma0": {"label": "对照组标准差 σ₀", "default": 15.0, "min": 0.001, "step": 0.1},
        "sigma1": {"label": "试验组标准差 σ₁", "default": 18.0, "min": 0.001, "step": 0.1},
        "k": {"label": "分配比例 k=n1/n0", "default": 1.0, "min": 0.01, "step": 0.1}
    },
    "formula": r"n_0 = \frac{(Z_{\alpha/2}+Z_\beta)^2 (\sigma_0^2+\sigma_1^2/k)}{(\mu_1-\mu_0)^2} + \frac14 Z_{\alpha/2}^2,\quad n_1 = k\cdot n_0",
    "results": [{"id": "n0", "label": "对照组样本量n₀", "desc": "对照组例数"},{"id": "n1", "label": "试验组样本量n₁", "desc": "试验组例数"}],
    "calc": calc_two_mean_unequal
})

# 12. 三组均数ANOVA
def anova_sample_size(alpha, beta, mus, sigma):
    k = len(mus)
    mu_bar = sum(mus) / k
    ss = sum((m - mu_bar)**2 for m in mus)
    f2 = ss / (k * sigma**2)
    if f2 <= 0:
        return None
    df1 = k - 1
    N = k + 1
    max_N = 10000
    target_power = 1 - beta
    while N <= max_N:
        df2 = N - k
        if df2 < 1:
            N += 1
            continue
        F_crit = f.ppf(1 - alpha, df1, df2)
        lam = N * f2
        power = 1 - ncf.cdf(F_crit, df1, df2, lam)
        if power >= target_power:
            break
        N += 1
    if N > max_N:
        N = float('inf')
    n_per = N / k
    final_lambda = N * f2 if N != float('inf') else float('inf')
    return {"f": math.sqrt(f2), "N": N, "n": n_per, "lambda": final_lambda}

METHODS.append({
    "id": "three_mean",
    "category": "均数的比较",
    "name": "三组均数比较（ANOVA）",
    "params": {
        "alpha": {"label": "置信水平 α", "default": 0.05, "min": 0.001, "max": 0.999, "step": 0.01},
        "beta": {"label": "β (1-把握度)", "default": 0.10, "min": 0.001, "max": 0.999, "step": 0.01},
        "mu1": {"label": "组1均数 μ₁", "default": 8.25, "step": 0.01},
        "mu2": {"label": "组2均数 μ₂", "default": 11.75, "step": 0.01},
        "mu3": {"label": "组3均数 μ₃", "default": 13.00, "step": 0.01},
        "sigma": {"label": "共同标准差 σ", "default": 3.5, "min": 0.001, "step": 0.1}
    },
    "formula": r"f = \sqrt{\frac{\sum(\mu_j-\bar{\mu})^2}{k\sigma^2}},\quad N=\lambda/f^2",
    "results": [
        {"id": "f", "label": "效应量f", "desc": "Cohen's f"},
        {"id": "lambda", "label": "非中心参数λ", "desc": "迭代计算值"},
        {"id": "N", "label": "总样本量N", "desc": "三组合计总例数"},
        {"id": "n", "label": "每组样本量", "desc": "每组平均例数"}
    ],
    "calc": lambda vals: anova_sample_size(vals["alpha"], vals["beta"], [vals["mu1"], vals["mu2"], vals["mu3"]], vals["sigma"])
})

# ============================
# 会话初始化
# ============================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_id" not in st.session_state:
    st.session_state.selected_id = "srs_rate"

st.set_page_config(page_title="公共卫生研究样本量计算软件 V1.0", layout="wide")

# ============================
# 首页【纯CSS + HTML按钮，永久居中】
# ============================
if st.session_state.page == "home":
    if st.query_params.get("goto") == "calc":
        st.session_state.page = "method"
        st.query_params.clear()
        st.rerun()

    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)), 
            url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1600&q=80");
            background-size: cover;
            background-position: center;
        }
        .block-container {
            padding-top: 8rem !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        .wrap-all {
            width: 100%;
            text-align: center;
        }
        .main-title {
            font-size: 3.6rem;
            color: #ffffff !important;
            font-weight: 700;
            text-shadow: 3px 3px 10px rgba(0,0,0,0.7);
            margin: 0 0 20px 0;
            letter-spacing: 1px;
        }
        .html-btn {
            display: inline-block;
            font-size: 1.6rem;
            font-weight: 700;
            padding: 0.9rem 3.2rem;
            border-radius: 50px;
            background: #2563eb;
            color: #ffffff !important;
            text-decoration: none;
            border: none;
            cursor: pointer;
            margin-bottom: 28px;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
            transition: all 0.2s ease;
            letter-spacing: 1px;
        }
        .html-btn:hover {
            background: #1d4ed8;
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
            color: #ffffff !important;
            text-decoration: none;
        }
        .desc-text {
            color: #ffffff !important;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
            font-size: 1.2rem;
            margin: 6px 0;
        }
        .copy-text {
            color: rgba(255,255,255,0.85) !important;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
            font-size: 0.95rem;
            margin: 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    current_year = datetime.now().year
    st.markdown(f'''
    <div class="wrap-all">
        <h1 class="main-title">📊 公共卫生研究样本量计算软件 V1.0</h1>
        <a class="html-btn" href="?goto=calc">🚀 点击开始</a>
        <p class="desc-text">基于经典统计公式，支持抽样调查、率比较、均数比较等 12 大类场景</p>
        <p class="copy-text">© {current_year} 长沙市疾病预防控制中心 版权所有 · 开发人员：谢赐福</p>
    </div>
    ''', unsafe_allow_html=True)
    st.stop()

# ============================
# 计算页面（深蓝色侧边栏 + 按钮高亮）
# ============================
st.markdown(
    """
    <style>
    /* ===== 侧边栏深蓝色 ===== */
    [data-testid="stSidebar"] {
        background-color: #1a3a5c !important;
        min-width:320px !important;
        max-width:320px !important;
    }
    /* 侧边栏所有文字白色 */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* 侧边栏子标题亮蓝色 */
    [data-testid="stSidebar"] .stSubheader {
        color: #8ab4f8 !important;
        font-weight: 600 !important;
    }
    /* 侧边栏标题 */
    [data-testid="stSidebar"] .stTitle {
        color: #ffffff !important;
    }
    /* 侧边栏按钮 - 默认透明 */
    [data-testid="stSidebar"] .stButton button {
        width:100%;
        text-align:left;
        padding: 8px 16px !important;
        white-space:nowrap;
        background-color: transparent !important;
        color: #c0d0e0 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 400 !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
    }
    /* 悬停效果 */
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255,255,255,0.12) !important;
        color: #ffffff !important;
    }
    /* 选中状态 - 蓝色高亮背景，白色文字 */
    [data-testid="stSidebar"] .stButton button[data-baseweb="button"][kind="primary"],
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4) !important;
    }
    [data-testid="stSidebar"] .stButton button[data-baseweb="button"][kind="primary"]:hover,
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: #3b82f6 !important;
    }
    /* 主界面背景 */
    .stApp {
        background:#f0f4f8 !important;
        background-image:none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 侧边栏导航
st.sidebar.title("📊 方法导航")
category_dict = {}
for item in METHODS:
    category_dict.setdefault(item["category"], []).append(item)

for cat_name, method_list in category_dict.items():
    st.sidebar.subheader(cat_name)
    for m in method_list:
        # 判断当前是否为选中状态
        is_active = (m["id"] == st.session_state.selected_id)
        # 使用 type="primary" 让选中的按钮显示蓝色背景
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(
            m["name"], 
            key=f"nav_{m['id']}", 
            use_container_width=True, 
            type=btn_type
        ):
            st.session_state.selected_id = m["id"]
            st.session_state.page = "method"
            st.rerun()

# 当前计算模块
current_method = next((m for m in METHODS if m["id"] == st.session_state.selected_id), METHODS[0])
st.title("📊 公共卫生研究样本量计算软件 V1.0")
st.caption("基于最新统计公式，支持抽样调查、率比较、均数比较等 12 大类场景")
st.header(current_method["name"])
st.latex(current_method["formula"])

# 参数输入区域
param_cols = st.columns(min(len(current_method["params"]), 4))
input_vals = {}
for idx, (key, param_info) in enumerate(current_method["params"].items()):
    col = param_cols[idx % len(param_cols)]
    input_vals[key] = col.number_input(
        label=param_info["label"],
        value=float(param_info["default"]),
        min_value=float(param_info.get("min", -99999)),
        max_value=float(param_info.get("max", 99999)),
        step=float(param_info.get("step", 0.01)),
        format="%.4f" if float(param_info.get("step",0.01)) < 0.01 else "%.2f"
    )

# 计算按钮
if st.button("🔢 计算样本量", type="primary"):
    try:
        res = current_method["calc"](input_vals)
        if res is None:
            st.error("参数无差异，无法计算样本量（各组指标完全一致）")
        else:
            st.divider()
            st.subheader("📋 计算结果")
            result_items = []
            for r_info in current_method["results"]:
                r_key = r_info["id"]
                if r_key in res:
                    val = res[r_key]
                    label_text = f"{r_info['label']}（{r_info['desc']}）"
                    result_items.append((label_text, val))
            if result_items:
                show_cols = st.columns(min(len(result_items), 4))
                for i, (lab, val) in enumerate(result_items):
                    display_val = fmt(val, 4) if isinstance(val, (int, float)) else str(val)
                    show_cols[i % len(show_cols)].metric(lab, display_val)
            st.caption("提示：理论样本量建议向上取整，实际研究适当增加样本量以应对失访")
    except Exception as e:
        st.error(f"计算失败：{str(e)}，请检查输入参数范围")

st.divider()
current_year = datetime.now().year
st.caption(f"© {current_year} 长沙市疾病预防控制中心 版权所有 · 开发人员：谢赐福 · Python+Streamlit")