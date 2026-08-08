import streamlit as st
import math
from scipy.stats import norm, f, ncf
from datetime import datetime
import streamlit.components.v1 as components

# ============================
# 1. 基础工具函数模块
# ============================

# 1.1 双侧Z值函数
def z_two_sided(alpha):
    """
    计算双侧检验对应的标准正态分布分位数。

    参数:
        alpha (float): 显著性水平，取值范围 (0, 1)。
                      通常取 0.05 或 0.01。

    返回:
        float: 标准正态分布的双侧分位数 Z_{alpha/2}。
              例如 alpha=0.05 时返回 1.959963984540054。

    数学公式: Z_{alpha/2} = Φ^{-1}(1 - alpha/2)
    其中 Φ^{-1} 为标准正态分布的逆累积分布函数。
    """
    return norm.ppf(1 - alpha / 2)

# 1.2 单侧Z值函数
def z_one_sided(beta):
    """
    计算单侧检验对应的标准正态分布分位数。

    参数:
        beta (float): 第二类错误概率，取值范围 (0, 1)。
                     通常取 0.20 或 0.10。

    返回:
        float: 标准正态分布的单侧分位数 Z_{beta}。
              例如 beta=0.10 时返回 1.2815515655446004。

    数学公式: Z_{beta} = Φ^{-1}(1 - beta)
    """
    return norm.ppf(1 - beta)

# 1.3 数值格式化函数
def fmt(v, digits=4):
    """
    格式化数值为指定小数位数的字符串。

    参数:
        v (float): 需要格式化的数值。
        digits (int): 保留的小数位数，默认为4。

    返回:
        str: 格式化后的字符串，如果数值无效则返回 "—"。

    说明: 此函数用于统一输出格式，便于阅读。
    """
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"

# 1.4 率参数校验函数
def check_rate_value(val, param_name):
    """
    校验率参数是否在 (0, 1) 范围内。

    参数:
        val (float): 待校验的率值。
        param_name (str): 参数名称，用于错误提示。

    返回:
        bool: 校验通过返回 True，否则返回 False。

    异常: 如果 val 不在 (0,1) 范围内，会通过 st.error 显示错误信息。
    """
    if val <= 0 or val >= 1:
        st.error(f"参数 {param_name} 必须在 0 到 1 之间，当前值为 {val}")
        return False
    return True

# 1.5 正数参数校验函数
def check_positive_value(val, param_name):
    """
    校验数值是否大于 0。

    参数:
        val (float): 待校验的数值。
        param_name (str): 参数名称，用于错误提示。

    返回:
        bool: 校验通过返回 True，否则返回 False。
    """
    if val <= 0:
        st.error(f"{param_name} 必须大于 0，当前值为 {val}")
        return False
    return True

# 1.6 大于等于1的校验函数
def check_ge_one(val, param_name):
    """
    校验数值是否大于等于 1。

    参数:
        val (float): 待校验的数值。
        param_name (str): 参数名称，用于错误提示。

    返回:
        bool: 校验通过返回 True，否则返回 False。
    """
    if val < 1:
        st.error(f"{param_name} 必须大于等于 1，当前值为 {val}")
        return False
    return True

# 1.7 综合参数校验函数
def validate_params(vals, method_name):
    """
    统一参数校验函数，根据不同的参数类型进行校验。

    参数:
        vals (dict): 参数字典。
        method_name (str): 方法名称，用于提示。

    返回:
        bool: 所有参数校验通过返回 True，否则返回 False。

    说明: 该函数会依次调用多个校验子函数，确保输入参数的有效性。
    """
    # 校验率参数
    rate_params = [
        "P",
        "pA",
        "pB",
        "P0",
        "P1",
        "p1",
        "p2",
        "p3",
        "rho"
    ]
    for key in rate_params:
        if key in vals:
            if not check_rate_value(vals[key], key):
                return False

    # 校验正数参数
    positive_params = {
        "alpha": "置信水平 α",
        "beta": "β (1-把握度)",
        "d": "容许误差 d",
        "sigma": "总体标准差 σ",
        "sigma0": "对照组标准差 σ₀",
        "sigma1": "试验组标准差 σ₁",
        "m": "平均每群人数 m",
        "k": "分配比例 k"
    }
    for key, label in positive_params.items():
        if key in vals:
            if not check_positive_value(vals[key], label):
                return False

    # 校验总体规模
    if "N" in vals and vals["N"] < 1:
        st.error(f"总体规模 N 必须大于等于 1，当前值为 {vals['N']}")
        return False

    return True


# ============================
# 2. 12类样本量计算模块
# ============================

# 定义 METHODS 列表，用于存储所有方法的信息。
METHODS = []

# ---------- 2.1 简单随机抽样（率） ----------
def calc_srs_rate(vals):
    """
    计算简单随机抽样（率）的样本量。

    参数:
        vals (dict): 包含 alpha, P, d, N 的字典。

    返回:
        dict: 包含原始样本量 n 和校正样本量 n_adj。

    公式:
        n = (Z_{alpha/2}^2 * P * (1-P)) / d^2
        n_adj = n / (1 + (n-1)/N)

    参考文献: Cochran (1977) Sampling Techniques, 第3版。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "简单随机抽样（率）"):
        return None

    # ----- 步骤 2: 获取 Z 值 -----
    Z = z_two_sided(vals["alpha"])

    # ----- 步骤 3: 计算初始样本量 -----
    numerator = Z ** 2 * vals["P"] * (1 - vals["P"])
    denominator = vals["d"] ** 2
    n_raw = numerator / denominator

    # ----- 步骤 4: 有限总体校正 -----
    n_adj = n_raw / (
        1 + (n_raw - 1) / vals["N"]
    )

    # ----- 步骤 5: 返回结果 -----
    return {
        "n": n_raw,
        "n_adj": n_adj
    }


METHODS.append(
    {
        "id": "srs_rate",
        "category": "抽样调查",
        "name": "简单随机抽样（率）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "P": {
                "label": "预期总体率 P",
                "default": 0.5,
                "step": 0.01
            },
            "d": {
                "label": "容许误差 d",
                "default": 0.05,
                "step": 0.001
            },
            "N": {
                "label": "总体规模 N",
                "default": 10000,
                "step": 1
            }
        },
        "formula": r"n = \frac{Z_{\alpha/2}^{2} P(1-P)}{d^2},\quad n_{adj} = \frac{n}{1+\frac{n-1}{N}}",
        "results": [
            {
                "id": "n",
                "label": "原始样本量 n",
                "desc": "（不校正总体）"
            },
            {
                "id": "n_adj",
                "label": "校正样本量 n_adj",
                "desc": "（有限总体校正）"
            }
        ],
        "calc": calc_srs_rate
    }
)


# ---------- 2.2 简单随机抽样（均数） ----------
def calc_srs_mean(vals):
    """
    计算简单随机抽样（均数）的样本量。

    参数:
        vals (dict): 包含 alpha, sigma, d, N 的字典。

    返回:
        dict: 包含原始样本量 n 和校正样本量 n_adj。

    公式:
        n = (Z_{alpha/2}^2 * sigma^2) / d^2
        n_adj = n / (1 + (n-1)/N)

    说明: 适用于连续变量均值的估计，需预先知道总体标准差。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "简单随机抽样（均数）"):
        return None

    # ----- 步骤 2: 获取 Z 值 -----
    Z = z_two_sided(vals["alpha"])

    # ----- 步骤 3: 计算初始样本量 -----
    numerator = Z ** 2 * vals["sigma"] ** 2
    denominator = vals["d"] ** 2
    n_raw = numerator / denominator

    # ----- 步骤 4: 有限总体校正 -----
    n_adj = n_raw / (
        1 + (n_raw - 1) / vals["N"]
    )

    # ----- 步骤 5: 返回结果 -----
    return {
        "n": n_raw,
        "n_adj": n_adj
    }


METHODS.append(
    {
        "id": "srs_mean",
        "category": "抽样调查",
        "name": "简单随机抽样（均数）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "sigma": {
                "label": "总体标准差 σ",
                "default": 10.0,
                "step": 0.1
            },
            "d": {
                "label": "容许误差 d",
                "default": 2.0,
                "step": 0.1
            },
            "N": {
                "label": "总体规模 N",
                "default": 10000,
                "step": 1
            }
        },
        "formula": r"n = \frac{Z_{\alpha/2}^2 \sigma^2}{d^2},\quad n_{adj} = \frac{n}{1+\frac{n-1}{N}}",
        "results": [
            {
                "id": "n",
                "label": "原始样本量 n",
                "desc": "（不校正总体）"
            },
            {
                "id": "n_adj",
                "label": "校正样本量 n_adj",
                "desc": "（有限总体校正）"
            }
        ],
        "calc": calc_srs_mean
    }
)


# ---------- 2.3 整群随机抽样（率） ----------
def calc_cluster_rate(vals):
    """
    计算整群随机抽样（率）的样本量和群数。

    参数:
        vals (dict): 包含 alpha, P, d, rho, m 的字典。

    返回:
        dict: 包含 n_srs, Deff, n_cluster, K。

    公式:
        n_srs = (Z^2 * P*(1-P)) / d^2
        Deff = 1 + (m-1)*rho
        n_cluster = n_srs * Deff
        K = ceil(n_cluster / m)

    说明: 引入设计效应 Deff 来调整样本量。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "整群随机抽样（率）"):
        return None

    # ----- 步骤 2: 获取 Z 值 -----
    Z = z_two_sided(vals["alpha"])

    # ----- 步骤 3: 计算简单随机样本量 -----
    n_srs = (
        Z ** 2
        * vals["P"]
        * (1 - vals["P"])
    ) / (vals["d"] ** 2)

    # ----- 步骤 4: 计算设计效应 -----
    Deff = 1 + (vals["m"] - 1) * vals["rho"]

    # ----- 步骤 5: 计算整群抽样样本量 -----
    n_cluster = n_srs * Deff

    # ----- 步骤 6: 计算所需群数 -----
    K = math.ceil(n_cluster / vals["m"])

    # ----- 步骤 7: 返回结果 -----
    return {
        "n_srs": n_srs,
        "Deff": Deff,
        "n_cluster": n_cluster,
        "K": K
    }


METHODS.append(
    {
        "id": "cluster_rate",
        "category": "抽样调查",
        "name": "整群随机抽样（率）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "P": {
                "label": "预期总体率 P",
                "default": 0.5,
                "step": 0.01
            },
            "d": {
                "label": "容许误差 d",
                "default": 0.05,
                "step": 0.001
            },
            "rho": {
                "label": "群内相关系数 ρ",
                "default": 0.05,
                "step": 0.001
            },
            "m": {
                "label": "平均每群人数 m",
                "default": 30,
                "step": 1
            }
        },
        "formula": r"n_{srs} = \frac{Z_{\alpha/2}^2 P(1-P)}{d^2},\; Deff=1+(m-1)\rho,\; n_{cluster}=n_{srs}\cdot Deff,\; K=\lceil n_{cluster}/m\rceil",
        "results": [
            {
                "id": "n_srs",
                "label": "单纯随机抽样所需样本 n_srs",
                "desc": ""
            },
            {
                "id": "Deff",
                "label": "设计效应 Deff",
                "desc": ""
            },
            {
                "id": "n_cluster",
                "label": "整群抽样所需样本 n_cluster",
                "desc": ""
            },
            {
                "id": "K",
                "label": "需抽取群数 K",
                "desc": ""
            }
        ],
        "calc": calc_cluster_rate
    }
)


# ---------- 2.4 整群随机抽样（均数） ----------
def calc_cluster_mean(vals):
    """
    计算整群随机抽样（均数）的样本量和群数。

    参数:
        vals (dict): 包含 alpha, sigma, d, rho, m 的字典。

    返回:
        dict: 包含 n_srs, Deff, n_cluster, K。

    公式:
        n_srs = (Z^2 * sigma^2) / d^2
        Deff = 1 + (m-1)*rho
        n_cluster = n_srs * Deff
        K = ceil(n_cluster / m)

    说明: 适用于整群抽样下连续变量均值的估计。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "整群随机抽样（均数）"):
        return None

    # ----- 步骤 2: 获取 Z 值 -----
    Z = z_two_sided(vals["alpha"])

    # ----- 步骤 3: 计算简单随机样本量 -----
    n_srs = (
        Z ** 2
        * vals["sigma"] ** 2
    ) / (vals["d"] ** 2)

    # ----- 步骤 4: 计算设计效应 -----
    Deff = 1 + (vals["m"] - 1) * vals["rho"]

    # ----- 步骤 5: 计算整群抽样样本量 -----
    n_cluster = n_srs * Deff

    # ----- 步骤 6: 计算所需群数 -----
    K = math.ceil(n_cluster / vals["m"])

    # ----- 步骤 7: 返回结果 -----
    return {
        "n_srs": n_srs,
        "Deff": Deff,
        "n_cluster": n_cluster,
        "K": K
    }


METHODS.append(
    {
        "id": "cluster_mean",
        "category": "抽样调查",
        "name": "整群随机抽样（均数）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "sigma": {
                "label": "总体标准差 σ",
                "default": 10.0,
                "step": 0.1
            },
            "d": {
                "label": "容许误差 d",
                "default": 2.0,
                "step": 0.1
            },
            "rho": {
                "label": "群内相关系数 ρ",
                "default": 0.05,
                "step": 0.001
            },
            "m": {
                "label": "平均每群人数 m",
                "default": 30,
                "step": 1
            }
        },
        "formula": r"n_{srs} = \frac{Z_{\alpha/2}^2 \sigma^2}{d^2},\; Deff=1+(m-1)\rho,\; n_{cluster}=n_{srs}\cdot Deff,\; K=\lceil n_{cluster}/m\rceil",
        "results": [
            {
                "id": "n_srs",
                "label": "单纯随机抽样所需样本 n_srs",
                "desc": ""
            },
            {
                "id": "Deff",
                "label": "设计效应 Deff",
                "desc": ""
            },
            {
                "id": "n_cluster",
                "label": "整群抽样所需样本 n_cluster",
                "desc": ""
            },
            {
                "id": "K",
                "label": "需抽取群数 K",
                "desc": ""
            }
        ],
        "calc": calc_cluster_mean
    }
)


# ---------- 2.5 单组率与总体率比较 ----------
def calc_one_prop(vals):
    """
    计算单组率与总体率比较的样本量。

    参数:
        vals (dict): 包含 alpha, beta, P0, P1 的字典。

    返回:
        dict: 包含样本量 n。

    公式:
        n = (Z_{alpha/2}*sqrt(p0*(1-p0)) + Z_beta*sqrt(p1*(1-p1)))^2 / (p1-p0)^2

    说明: 基于正态近似法，用于样本率与已知总体率的比较。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "单组率与总体率比较"):
        return None

    # ----- 步骤 2: 获取 Z 值 -----
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])

    # ----- 步骤 3: 计算平方根项 -----
    sqrt_p0 = math.sqrt(
        vals["P0"] * (1 - vals["P0"])
    )
    sqrt_p1 = math.sqrt(
        vals["P1"] * (1 - vals["P1"])
    )

    # ----- 步骤 4: 计算分子 -----
    term1 = Za * sqrt_p0
    term2 = Zb * sqrt_p1
    numerator = (term1 + term2) ** 2

    # ----- 步骤 5: 计算分母 -----
    denominator = (vals["P1"] - vals["P0"]) ** 2
    if denominator == 0:
        st.error("样本率与总体率不能相等")
        return None

    # ----- 步骤 6: 计算样本量 -----
    n = numerator / denominator

    # ----- 步骤 7: 返回结果 -----
    return {"n": n}


METHODS.append(
    {
        "id": "one_prop",
        "category": "率的比较",
        "name": "单组率与总体率比较",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.01
            },
            "P0": {
                "label": "总体率 P₀",
                "default": 0.30,
                "step": 0.01
            },
            "P1": {
                "label": "预期样本率 P₁",
                "default": 0.40,
                "step": 0.01
            }
        },
        "formula": r"n = \frac{\big(Z_{\alpha/2}\sqrt{p_0(1-p_0)} + Z_\beta\sqrt{p_1(1-p_1)}\big)^2}{(p_1-p_0)^2}",
        "results": [
            {
                "id": "n",
                "label": "所需样本量 n",
                "desc": ""
            }
        ],
        "calc": calc_one_prop
    }
)


# ---------- 2.6 配对四格表率比较 ----------
def calc_paired_prop(vals):
    """
    计算配对四格表率比较的样本量（对子数）。

    参数:
        vals (dict): 包含 alpha, beta, P01, P10 的字典。

    返回:
        dict: 包含 N_pairs, OR, PD。

    公式:
        OR = P10/P01
        PD = P01 + P10
        N_pairs = (Z_{alpha/2}*(OR+1) + Z_beta*sqrt((OR+1)^2 - (OR-1)^2*PD))^2 / ((OR-1)^2*PD)

    说明: 适用于配对设计的率比较，如诊断试验的一致性评价。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "两组率比较（配对四格表）"):
        return None

    # ----- 步骤 2: 单独校验 P01 和 P10 -----
    if vals["P01"] <= 0 or vals["P01"] >= 1:
        st.error("P₀₁ 必须在 0 到 1 之间")
        return None
    if vals["P10"] <= 0 or vals["P10"] >= 1:
        st.error("P₁₀ 必须在 0 到 1 之间")
        return None

    # ----- 步骤 3: 计算 OR 和 PD -----
    OR = vals["P10"] / vals["P01"]
    PD = vals["P01"] + vals["P10"]
    if PD >= 1:
        st.error("P₀₁ + P₁₀ 必须小于 1")
        return None

    # ----- 步骤 4: 获取 Z 值 -----
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])

    # ----- 步骤 5: 计算内部平方根 -----
    inside_sqrt = (
        (OR + 1) ** 2
        - (OR - 1) ** 2 * PD
    )
    if inside_sqrt < 0:
        st.error("计算出现负值，请检查参数")
        return None

    # ----- 步骤 6: 计算分子和分母 -----
    numerator = (
        Za * (OR + 1)
        + Zb * math.sqrt(inside_sqrt)
    ) ** 2
    denominator = ((OR - 1) ** 2) * PD
    N_pairs = numerator / denominator

    # ----- 步骤 7: 返回结果 -----
    return {
        "N_pairs": N_pairs,
        "OR": OR,
        "PD": PD
    }


METHODS.append(
    {
        "id": "paired_prop",
        "category": "率的比较",
        "name": "两组率比较（配对四格表）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.001
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.001
            },
            "P01": {
                "label": "P₀₁ (试验组阴性且对照组阳性)",
                "default": 0.10,
                "step": 0.001
            },
            "P10": {
                "label": "P₁₀ (试验组阳性且对照组阴性)",
                "default": 0.05,
                "step": 0.001
            }
        },
        "formula": r"N_{pairs} = \frac{\big[Z_{\alpha/2}(OR+1)+Z_\beta\sqrt{(OR+1)^2-(OR-1)^2 PD}\big]^2}{(OR-1)^2 PD}",
        "results": [
            {
                "id": "N_pairs",
                "label": "所需样本量 N_pairs",
                "desc": "（对子数）"
            },
            {
                "id": "OR",
                "label": "比值比 OR",
                "desc": "（P<sub>10</sub>/P<sub>01</sub>）"
            },
            {
                "id": "PD",
                "label": "不一致比例 PD",
                "desc": "（P<sub>01</sub>+P<sub>10</sub>）"
            }
        ],
        "calc": calc_paired_prop
    }
)


# ---------- 2.7 两组独立率比较 ----------
def calc_two_prop(vals):
    """
    计算两组独立率比较的样本量。

    参数:
        vals (dict): 包含 alpha, beta, pA, pB, k 的字典。

    返回:
        dict: 包含 nB (对照组) 和 nA (试验组)。

    公式:
        n_B = (p_A*(1-p_A)/k + p_B*(1-p_B)) * ((Z_{alpha/2}+Z_beta)/(p_A-p_B))^2
        n_A = k * n_B

    说明: 适用于两组独立样本率的比较，如随机对照试验。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "两组率比较（独立成组）"):
        return None

    # ----- 步骤 2: 检查率是否相等 -----
    if vals["pA"] == vals["pB"]:
        st.error("试验组率与对照组率不能相等")
        return None

    # ----- 步骤 3: 获取 Z 值 -----
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])

    # ----- 步骤 4: 计算中间量 -----
    part = (
        vals["pA"] * (1 - vals["pA"]) / vals["k"]
        + vals["pB"] * (1 - vals["pB"])
    )
    coeff = (
        (Za + Zb) / (vals["pA"] - vals["pB"])
    ) ** 2

    # ----- 步骤 5: 计算样本量 -----
    nB = part * coeff
    nA = nB * vals["k"]

    # ----- 步骤 6: 返回结果 -----
    return {
        "nB": nB,
        "nA": nA
    }


METHODS.append(
    {
        "id": "two_prop",
        "category": "率的比较",
        "name": "两组率比较（独立成组）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.01
            },
            "pA": {
                "label": "试验组率 pA",
                "default": 0.50,
                "step": 0.01
            },
            "pB": {
                "label": "对照组率 pB",
                "default": 0.30,
                "step": 0.01
            },
            "k": {
                "label": "样本分配比例 k",
                "default": 1.0,
                "step": 0.1
            }
        },
        "formula": r"n_B = \left(\frac{p_A(1-p_A)}{k}+p_B(1-p_B)\right)\left(\frac{Z_{\alpha/2}+Z_\beta}{p_A-p_B}\right)^2,\quad n_A = k\cdot n_B",
        "results": [
            {
                "id": "nB",
                "label": "对照组样本量 nB",
                "desc": "（对照组例数）"
            },
            {
                "id": "nA",
                "label": "试验组样本量 nA",
                "desc": "（试验组例数）"
            }
        ],
        "calc": calc_two_prop
    }
)


# ---------- 2.8 三组率比较（卡方） ----------
def calc_three_prop(vals):
    """
    计算三组率比较（卡方检验）的样本量。

    参数:
        vals (dict): 包含 alpha, beta, p1, p2, p3 的字典。

    返回:
        dict: 包含 w, N, n_per, lambda。

    公式:
        w = sqrt( (1/3) * sum((p_i - pbar)^2 / (pbar*(1-pbar))) )
        N = lambda / w^2
        n_per = N / 3

    说明: 基于 Cohen's w 效应量，适用于三组率的卡方检验。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "三组率比较（卡方）"):
        return None

    # ----- 步骤 2: 提取率值并计算平均率 -----
    ps = [vals["p1"], vals["p2"], vals["p3"]]
    pbar = sum(ps) / 3
    if pbar <= 0 or pbar >= 1:
        st.error("平均率必须在0到1之间")
        return None

    # ----- 步骤 3: 计算 w^2 -----
    w2_sum = 0
    for p in ps:
        w2_sum += (
            (p - pbar) ** 2
        ) / (pbar * (1 - pbar))
    w2 = w2_sum / 3

    if w2 <= 0:
        st.error("三组率不能完全相同")
        return None

    # ----- 步骤 4: 确定非中心参数 -----
    lam = 12.65 if vals["beta"] <= 0.10 else 9.63

    # ----- 步骤 5: 计算样本量 -----
    N = lam / w2
    n_per = N / 3
    w = math.sqrt(w2)

    # ----- 步骤 6: 返回结果 -----
    return {
        "w": w,
        "N": N,
        "n_per": n_per,
        "lambda": lam
    }


METHODS.append(
    {
        "id": "three_prop",
        "category": "率的比较",
        "name": "三组率比较（卡方）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.01
            },
            "p1": {
                "label": "组1率 p₁",
                "default": 0.30,
                "step": 0.01
            },
            "p2": {
                "label": "组2率 p₂",
                "default": 0.40,
                "step": 0.01
            },
            "p3": {
                "label": "组3率 p₃",
                "default": 0.50,
                "step": 0.01
            }
        },
        "formula": r"w = \sqrt{\frac{1}{3}\sum \frac{(p_i-\bar{p})^2}{\bar{p}(1-\bar{p})}},\quad N = \lambda/w^2",
        "results": [
            {
                "id": "w",
                "label": "效应量 w",
                "desc": "（Cohen's w）"
            },
            {
                "id": "lambda",
                "label": "非中心参数 λ",
                "desc": "（查表）"
            },
            {
                "id": "N",
                "label": "总样本量 N",
                "desc": "（三组合计）"
            },
            {
                "id": "n_per",
                "label": "每组样本量 n_per",
                "desc": ""
            }
        ],
        "calc": calc_three_prop
    }
)


# ---------- 2.9 单样本均数t检验 ----------
def calc_one_mean(vals):
    """
    计算单组均数与总体均数比较的样本量。

    参数:
        vals (dict): 包含 alpha, beta, mu0, mu1, sigma 的字典。

    返回:
        dict: 包含 n 和 delta。

    公式:
        n = ((Z_{alpha/2} + Z_beta) / (delta/sigma))^2 + 0.5 * Z_{alpha/2}^2
        delta = |mu1 - mu0|

    说明: 适用于单样本均数检验，如前后对照研究。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "单组均数与总体均数比较"):
        return None

    # ----- 步骤 2: 获取 Z 值 -----
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])

    # ----- 步骤 3: 计算差值绝对值 -----
    delta = abs(vals["mu1"] - vals["mu0"])
    if delta == 0:
        st.error("样本均数与总体均数不能相等")
        return None

    # ----- 步骤 4: 计算样本量 -----
    n = (
        (Za + Zb) / (delta / vals["sigma"])
    ) ** 2 + 0.5 * Za ** 2

    # ----- 步骤 5: 返回结果 -----
    return {
        "n": n,
        "delta": delta
    }


METHODS.append(
    {
        "id": "one_mean",
        "category": "均数的比较",
        "name": "单组均数与总体均数比较",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.01
            },
            "mu0": {
                "label": "总体均数 μ₀",
                "default": 100.0,
                "step": 0.1
            },
            "mu1": {
                "label": "预期样本均数 μ₁",
                "default": 105.0,
                "step": 0.1
            },
            "sigma": {
                "label": "标准差 σ",
                "default": 15.0,
                "step": 0.1
            }
        },
        "formula": r"n = \left(\frac{Z_{\alpha/2}+Z_\beta}{\delta/\sigma}\right)^2 + \frac12 Z_{\alpha/2}^2,\quad \delta=|\mu_1-\mu_0|",
        "results": [
            {
                "id": "n",
                "label": "所需样本量 n",
                "desc": ""
            },
            {
                "id": "delta",
                "label": "均数差值绝对值 δ",
                "desc": ""
            }
        ],
        "calc": calc_one_mean
    }
)


# ---------- 2.10 两组均数比较（方差齐） ----------
def calc_two_mean_equal(vals):
    """
    计算两组均数比较（方差齐）的样本量。

    参数:
        vals (dict): 包含 alpha, beta, mu0, mu1, sigma, k 的字典。

    返回:
        dict: 包含 n0 (对照组) 和 n1 (试验组)。

    公式:
        n0 = (Z_{alpha/2}+Z_beta)^2 * sigma^2 * (1+1/k) / (mu1-mu0)^2 + 0.25*Z_{alpha/2}^2
        n1 = k * n0

    说明: 假定两组方差相等，使用合并标准差，适用于两样本 t 检验。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "两组均数比较（方差齐）"):
        return None

    # ----- 步骤 2: 检查差值 -----
    diff = vals["mu1"] - vals["mu0"]
    if diff == 0:
        st.error("两组均数不能相等")
        return None

    # ----- 步骤 3: 获取 Z 值 -----
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])

    # ----- 步骤 4: 计算中间量 -----
    term = (
        (Za + Zb) ** 2
        * vals["sigma"] ** 2
        * (1 + 1 / vals["k"])
    ) / (diff ** 2)

    # ----- 步骤 5: 计算样本量 -----
    n0 = term + 0.25 * Za ** 2
    n1 = n0 * vals["k"]

    # ----- 步骤 6: 返回结果 -----
    return {
        "n0": n0,
        "n1": n1
    }


METHODS.append(
    {
        "id": "two_mean_equal",
        "category": "均数的比较",
        "name": "两组均数比较（方差齐）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.1
            },
            "mu0": {
                "label": "对照组均数 μ₀",
                "default": 100.0,
                "step": 0.1
            },
            "mu1": {
                "label": "试验组均数 μ₁",
                "default": 108.0,
                "step": 0.1
            },
            "sigma": {
                "label": "合并标准差 σ",
                "default": 15.0,
                "step": 0.1
            },
            "k": {
                "label": "分配比例 k",
                "default": 1.0,
                "step": 0.1
            }
        },
        "formula": r"n_0 = \frac{(Z_{\alpha/2}+Z_\beta)^2 \sigma^2 (1+1/k)}{(\mu_1-\mu_0)^2} + \frac14 Z_{\alpha/2}^2,\quad n_1 = k\cdot n_0",
        "results": [
            {
                "id": "n0",
                "label": "对照组样本量 n₀",
                "desc": ""
            },
            {
                "id": "n1",
                "label": "试验组样本量 n₁",
                "desc": ""
            }
        ],
        "calc": calc_two_mean_equal
    }
)


# ---------- 2.11 两组均数比较（方差不齐） ----------
def calc_two_mean_unequal(vals):
    """
    计算两组均数比较（方差不齐）的样本量。

    参数:
        vals (dict): 包含 alpha, beta, mu0, mu1, sigma0, sigma1, k 的字典。

    返回:
        dict: 包含 n0 (对照组) 和 n1 (试验组)。

    公式:
        n0 = (Z_{alpha/2}+Z_beta)^2 * (sigma0^2 + sigma1^2/k) / (mu1-mu0)^2 + 0.25*Z_{alpha/2}^2
        n1 = k * n0

    说明: 用于两组方差不相等的情况，使用 Welch 检验近似。
    """
    # ----- 步骤 1: 参数校验 -----
    if not validate_params(vals, "两组均数比较（方差不齐）"):
        return None

    # ----- 步骤 2: 检查差值 -----
    diff = vals["mu1"] - vals["mu0"]
    if diff == 0:
        st.error("两组均数不能相等")
        return None

    # ----- 步骤 3: 获取 Z 值 -----
    Za = z_two_sided(vals["alpha"])
    Zb = z_one_sided(vals["beta"])

    # ----- 步骤 4: 计算中间量 -----
    term = (
        (Za + Zb) ** 2
        * (vals["sigma0"] ** 2 + vals["sigma1"] ** 2 / vals["k"])
    ) / (diff ** 2)

    # ----- 步骤 5: 计算样本量 -----
    n0 = term + 0.25 * Za ** 2
    n1 = n0 * vals["k"]

    # ----- 步骤 6: 返回结果 -----
    return {
        "n0": n0,
        "n1": n1
    }


METHODS.append(
    {
        "id": "two_mean_unequal",
        "category": "均数的比较",
        "name": "两组均数比较（方差不齐）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.1
            },
            "mu0": {
                "label": "对照组均数 μ₀",
                "default": 100.0,
                "step": 0.1
            },
            "mu1": {
                "label": "试验组均数 μ₁",
                "default": 108.0,
                "step": 0.1
            },
            "sigma0": {
                "label": "对照组标准差 σ₀",
                "default": 15.0,
                "step": 0.1
            },
            "sigma1": {
                "label": "试验组标准差 σ₁",
                "default": 18.0,
                "step": 0.1
            },
            "k": {
                "label": "分配比例 k",
                "default": 1.0,
                "step": 0.1
            }
        },
        "formula": r"n_0 = \frac{(Z_{\alpha/2}+Z_\beta)^2 (\sigma_0^2+\sigma_1^2/k)}{(\mu_1-\mu_0)^2} + \frac14 Z_{\alpha/2}^2,\quad n_1 = k\cdot n_0",
        "results": [
            {
                "id": "n0",
                "label": "对照组样本量 n₀",
                "desc": ""
            },
            {
                "id": "n1",
                "label": "试验组样本量 n₁",
                "desc": ""
            }
        ],
        "calc": calc_two_mean_unequal
    }
)


# ---------- 2.12 三组均数ANOVA ----------
def anova_sample_size(alpha, beta, mus, sigma):
    """
    使用精确迭代法计算三组均数比较（ANOVA）的样本量。

    参数:
        alpha (float): 第一类错误概率。
        beta (float): 第二类错误概率。
        mus (list): 三组均数列表。
        sigma (float): 共同标准差。

    返回:
        dict: 包含 f, N, n, lambda。

    说明:
        基于非中心 F 分布的精确迭代，确保达到目标检验效能。
        最大迭代次数设为 100000，防止无限循环。
    """
    # ----- 步骤 1: 计算组数及组均值 -----
    k = len(mus)
    mu_bar = sum(mus) / k
    ss = sum((m - mu_bar) ** 2 for m in mus)
    f2 = ss / (k * sigma ** 2)

    if f2 <= 0:
        return None

    # ----- 步骤 2: 初始化迭代参数 -----
    df1 = k - 1
    N = k + 1
    max_N = 100000
    target_power = 1 - beta

    # ----- 步骤 3: 迭代计算 N -----
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

    return {
        "f": math.sqrt(f2),
        "N": N,
        "n": n_per,
        "lambda": final_lambda
    }


def calc_three_mean(vals):
    """
    计算三组均数比较（ANOVA）的样本量（封装函数）。

    参数:
        vals (dict): 包含 alpha, beta, mu1, mu2, mu3, sigma 的字典。

    返回:
        dict: 包含 f, N, n, lambda。
    """
    if not validate_params(vals, "三组均数比较（ANOVA）"):
        return None

    mus = [vals["mu1"], vals["mu2"], vals["mu3"]]
    if len(set(mus)) == 1:
        st.error("三组均数不能完全相同")
        return None

    return anova_sample_size(
        vals["alpha"],
        vals["beta"],
        mus,
        vals["sigma"]
    )


METHODS.append(
    {
        "id": "three_mean",
        "category": "均数的比较",
        "name": "三组均数比较（ANOVA）",
        "params": {
            "alpha": {
                "label": "置信水平 α",
                "default": 0.05,
                "step": 0.01
            },
            "beta": {
                "label": "β (1-把握度)",
                "default": 0.10,
                "step": 0.01
            },
            "mu1": {
                "label": "组1均数 μ₁",
                "default": 8.25,
                "step": 0.01
            },
            "mu2": {
                "label": "组2均数 μ₂",
                "default": 11.75,
                "step": 0.01
            },
            "mu3": {
                "label": "组3均数 μ₃",
                "default": 13.00,
                "step": 0.01
            },
            "sigma": {
                "label": "共同标准差 σ",
                "default": 3.5,
                "step": 0.1
            }
        },
        "formula": r"f = \sqrt{\frac{\sum(\mu_j-\bar{\mu})^2}{k\sigma^2}},\quad N=\lambda/f^2",
        "results": [
            {
                "id": "f",
                "label": "效应量 f",
                "desc": "（Cohen's f）"
            },
            {
                "id": "lambda",
                "label": "非中心参数 λ",
                "desc": "（迭代）"
            },
            {
                "id": "N",
                "label": "总样本量 N",
                "desc": "（三组合计）"
            },
            {
                "id": "n",
                "label": "每组样本量 n",
                "desc": ""
            }
        ],
        "calc": calc_three_mean
    }
)


# ============================
# 3. 会话初始化
# ============================

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_id" not in st.session_state:
    st.session_state.selected_id = "srs_rate"

st.set_page_config(
    page_title="公共卫生研究样本量计算软件 V1.0",
    layout="wide"
)

# ============================
# 4. 注入百度统计代码
# ============================

baidu_tongji_html = """
<script>
var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?38be9114db5a3298aa8ae53526815f3a";
  var s = document.getElementsByTagName("script")[0];
  s.parentNode.insertBefore(hm, s);
})();
</script>
"""
components.html(baidu_tongji_html, height=0)

# ============================
# 5. 首页
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
# 6. 计算页面（侧边栏样式与主界面）
# ============================

st.markdown(
    """
    <style>
    /* ===== 侧边栏样式（所有文字白色） ===== */
    [data-testid="stSidebar"] {
        background-color: #1a3a5c !important;
        min-width:320px !important;
        max-width:320px !important;
    }
    /* 强制侧边栏所有文字白色 */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* 侧边栏按钮样式 */
    [data-testid="stSidebar"] .stButton button {
        width:100%;
        text-align:left;
        padding: 8px 16px !important;
        white-space:nowrap;
        background-color: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 400 !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255,255,255,0.12) !important;
    }
    /* 选中状态按钮 */
    [data-testid="stSidebar"] .stButton button[data-baseweb="button"][kind="primary"],
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #2563eb !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4) !important;
    }
    [data-testid="stSidebar"] .stButton button[data-baseweb="button"][kind="primary"]:hover,
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: #3b82f6 !important;
    }
    /* 侧边栏子标题（如“抽样调查”）字体加粗 */
    [data-testid="stSidebar"] .stSubheader {
        font-weight: 600 !important;
    }

    /* ===== 主背景 ===== */
    .stApp {
        background:#f0f4f8 !important;
        background-image:none !important;
    }

    /* ===== 移动端适配 ===== */
    @media only screen and (max-width: 768px) {
        /* 强制主内容文字深色，解决荣耀浏览器白字 */
        section.main > div,
        section.main > div *,
        .stApp .main > div,
        .stApp .main > div * {
            color: #1a3a5c !important;
        }
        [data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 280px !important;
        }
        section.main > div {
            padding: 0.5rem 0.6rem !important;
        }
        .main-title {
            font-size: 2.2rem !important;
            color: #ffffff !important;
        }
        .stButton button {
            font-size: 1rem !important;
            padding: 0.5rem 1rem !important;
        }
        .katex-display {
            font-size: 0.9rem !important;
            overflow-x: auto !important;
            white-space: nowrap !important;
        }
        .row-widget.stColumns {
            flex-wrap: wrap !important;
        }
        .row-widget.stColumns > div {
            min-width: 45% !important;
            flex: 1 1 auto !important;
        }
        .element-container iframe {
            max-width: 100% !important;
        }
        .stApp {
            color-scheme: light !important;
        }
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
        is_active = (m["id"] == st.session_state.selected_id)
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
current_method = next(
    (m for m in METHODS if m["id"] == st.session_state.selected_id),
    METHODS[0]
)
st.title("📊 公共卫生研究样本量计算软件 V1.0")
st.caption("基于最新统计公式，支持抽样调查、率比较、均数比较等 12 大类场景")
st.header(current_method["name"])
st.latex(current_method["formula"])

# ============================
# 7. 参数输入区域
# ============================

param_cols = st.columns(min(len(current_method["params"]), 4))
input_vals = {}
for idx, (key, param_info) in enumerate(current_method["params"].items()):
    col = param_cols[idx % len(param_cols)]
    step_value = float(param_info.get("step", 0.01))
    input_vals[key] = col.number_input(
        label=param_info["label"],
        value=float(param_info["default"]),
        min_value=0.0,
        step=step_value,
        format="%.4f"
    )

# ============================
# 8. 计算按钮 + 结果展示
# ============================

if st.button("🔢 计算样本量", type="primary"):
    try:
        res = current_method["calc"](input_vals)
        if res is None:
            pass
        else:
            # 构建卡片 HTML
            cards_html = ""
            for r_info in current_method["results"]:
                r_key = r_info["id"]
                if r_key in res:
                    val = res[r_key]
                    label = r_info["label"] + r_info["desc"]
                    # 替换下标
                    label_html = label.replace("n_adj", "n<sub>adj</sub>")
                    label_html = label_html.replace("n_srs", "n<sub>srs</sub>")
                    label_html = label_html.replace("n_cluster", "n<sub>cluster</sub>")
                    label_html = label_html.replace("n_per", "n<sub>per</sub>")
                    label_html = label_html.replace("nA", "n<sub>A</sub>")
                    label_html = label_html.replace("nB", "n<sub>B</sub>")
                    label_html = label_html.replace("N_pairs", "N<sub>pairs</sub>")
                    # 数值格式化
                    if val is None or not math.isfinite(val):
                        display_val = "∞"
                    else:
                        display_val = fmt(val, 4) if isinstance(val, (int, float)) else str(val)
                    # 每个卡片
                    cards_html += f"""
                    <div style="flex: 1 1 200px; min-width: 150px; margin: 8px 4px;">
                        <div style="font-size: 1.1rem; color: #333; margin-bottom: 2px;">{label_html}</div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: #1a3a5c;">{display_val}</div>
                    </div>
                    """
            # 完整结果区块 HTML（浅红色）
            result_html = f"""
            <div style="background: #f8d7da; border: 2px solid #842029; border-radius: 12px; padding: 16px 20px; margin-top: 12px; width: 100%; box-sizing: border-box;">
                <hr style="margin: 0 0 8px 0; border-top: 1px solid #bbb;">
                <h3 style="margin: 0 0 8px 0; font-size: 1.5rem; color: #1a3a5c;">📋 计算结果</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 8px 16px; justify-content: flex-start; width: 100%;">
                    {cards_html}
                </div>
                <div style="font-size: 0.9rem; color: #842029; margin-top: 12px; border-top: 1px solid #e5b3b3; padding-top: 8px;">
                    💡 提示：理论样本量建议向上取整，实际研究适当增加样本量以应对失访
                </div>
            </div>
            """
            components.html(result_html, height=350, scrolling=True)
    except Exception as e:
        st.error(f"计算失败：{str(e)}，请检查输入参数范围")

# ============================
# 9. 参考文献
# ============================

ref_map = {
    "srs_rate": "[1] Cochran WG. Sampling Techniques[M]. 3rd ed. New York: John Wiley & Sons, 1977: 72-86.",
    "srs_mean": "[1] Cochran WG. Sampling Techniques[M]. 3rd ed. New York: John Wiley & Sons, 1977: 72-86.",
    "cluster_rate": "[1] Donner A, Klar N. Design and Analysis of Cluster Randomization Trials in Health Research[M]. London: Arnold, 2000: 25-45.",
    "cluster_mean": "[1] Donner A, Klar N. Design and Analysis of Cluster Randomization Trials in Health Research[M]. London: Arnold, 2000: 25-45.",
    "one_prop": "[1] Fleiss JL, Levin B, Paik MC. Statistical Methods for Rates and Proportions[M]. 3rd ed. New York: John Wiley & Sons, 2003. doi:10.1002/0471445428.",
    "paired_prop": "[1] Dupont WD, Plummer WD. Power and sample size calculations: a review and computer program[J]. Controlled Clinical Trials, 1990, 11(2): 116-128. doi:10.1016/0197-2456(90)90005-M.",
    "two_prop": "[1] Dupont WD, Plummer WD. Power and sample size calculations: a review and computer program[J]. Controlled Clinical Trials, 1990, 11(2): 116-128. doi:10.1016/0197-2456(90)90005-M.",
    "three_prop": "[1] Cohen J. Statistical Power Analysis for the Behavioral Sciences[M]. 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988: 215-271.",
    "one_mean": "[1] Cohen J. Statistical Power Analysis for the Behavioral Sciences[M]. 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988: 27-65.",
    "two_mean_equal": "[1] Satterthwaite FE. An approximate distribution of estimates of variance components[J]. Biometrics Bulletin, 1946, 2(6): 110-114. doi:10.2307/3002019.",
    "two_mean_unequal": "[1] Satterthwaite FE. An approximate distribution of estimates of variance components[J]. Biometrics Bulletin, 1946, 2(6): 110-114. doi:10.2307/3002019.",
    "three_mean": "[1] Cohen J. Statistical Power Analysis for the Behavioral Sciences[M]. 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988: 273-406."
}

ref_text = ref_map.get(current_method["id"], "")
if ref_text:
    st.markdown("### 📖 参考文献")
    st.markdown(ref_text)

# ============================
# 10. 参考示例
# ============================

example_map = {
    "srs_rate": """
**应用场景**：某市疾控中心欲估计辖区居民中高血压的患病率。采用简单随机抽样，要求估计的置信水平为95%，容许误差不超过5个百分点。根据文献报道，该地区高血压患病率约为50%。

**参数设定**：置信水平 α = 0.05，预期总体率 P = 0.50，容许误差 d = 0.05，总体规模 N = 10000。

**计算结果**：初始样本量 n ≈ 384.1459 人，经有限总体校正后 n_adj ≈ 369.9706 人。实际调查中建议向上取整，至少调查 370 人。

**结果解读**：在95%的置信水平下，若调查约370人，所得的患病率估计值与真实值的误差不超过5个百分点。
""",
    "srs_mean": """
**应用场景**：某社区卫生服务中心拟调查辖区老年人的平均收缩压水平。采用简单随机抽样，要求在95%的置信水平下，容许误差不超过2 mmHg。根据预调查，收缩压的标准差约为10 mmHg。

**参数设定**：置信水平 α = 0.05，总体标准差 σ = 10 mmHg，容许误差 d = 2 mmHg，总体规模 N = 10000。

**计算结果**：初始样本量 n ≈ 96.0365 人，经有限总体校正后 n_adj ≈ 95.1324 人。实际调查中建议至少调查 96 人。

**结果解读**：在95%的置信水平下，调查约96名老年人，其平均收缩压估计值与真实值的误差不超过2 mmHg。
""",
    "cluster_rate": """
**应用场景**：某市教育局拟调查全市中学生近视率。计划以班级为整群抽样单位，每个班级约30人。根据文献，近视率约为50%，群内相关系数估计为0.05。

**参数设定**：置信水平 α = 0.05，预期总体率 P = 0.50，容许误差 d = 0.05，群内相关系数 ρ = 0.05，平均每群人数 m = 30。

**计算结果**：简单随机抽样所需样本 n_srs ≈ 384.1459 人，设计效应 Deff = 2.45，需抽取群数 K ≈ 32 个班级。实际调查中，抽取32个班级共约 960 人（32 × 30）。

**结果解读**：由于班级内部学生近视情况具有一定相似性（ICC=0.05），整群抽样所需样本量约为简单随机抽样的2.45倍。需抽取约32个班级共约960名学生。
""",
    "cluster_mean": """
**应用场景**：某大学拟调查学生的肺活量平均水平。计划以班级为整群抽样单位，每个班级约30人。预调查肺活量标准差约为10 L，容许误差不超过2 L。

**参数设定**：置信水平 α = 0.05，总体标准差 σ = 10 L，容许误差 d = 2 L，群内相关系数 ρ = 0.05，平均每群人数 m = 30。

**计算结果**：简单随机抽样所需样本 n_srs ≈ 96.0365 人，设计效应 Deff = 2.45，需抽取群数 K ≈ 8 个班级。实际调查中，抽取8个班级共约 240 人（8 × 30）。

**结果解读**：需抽取约8个班级共约240名学生进行肺活量调查，所得平均肺活量估计值与真实值的误差不超过2 L。
""",
    "one_prop": """
**应用场景**：某研究者欲评估一种新的健康教育干预措施能否将辖区居民的健康知识知晓率从目前的30%提高到40%。拟采用单组前后比较设计，检验水准α=0.05，检验效能1-β=0.90。

**参数设定**：置信水平 α = 0.05，β = 0.10，总体率 P₀ = 0.30，预期样本率 P₁ = 0.40。

**计算结果**：所需样本量 n ≈ 232.8669 人，实际调查中建议至少调查 233 人。

**结果解读**：在95%的置信水平和90%的检验效能下，需要纳入约233名研究对象，才能检测出知晓率从30%提升到40%的差异。
""",
    "paired_prop": """
**应用场景**：某研究者欲比较两种方法对某种疾病的诊断一致性。采用配对设计，预期不一致对子中，方法A阳性且方法B阴性的比例为0.10，方法A阴性且方法B阳性的比例为0.05。

**参数设定**：置信水平 α = 0.05，β = 0.10，P₀₁ = 0.10，P₁₀ = 0.05。

**计算结果**：所需对子数 N_pairs ≈ 626.2807 对，比值比 OR = 0.5000，不一致比例 PD = 0.1500。实际调查中建议至少调查 627 对。

**结果解读**：在95%的置信水平和90%的检验效能下，需要约627对研究对象才能检验出两种方法诊断结果的一致性或差异性。
""",
    "two_prop": """
**应用场景**：某研究者欲开展一项随机对照试验，评价一种新药对某疾病的疗效。预期试验组有效率为50%，对照组有效率为30%，两组样本量相等。

**参数设定**：置信水平 α = 0.05，β = 0.10，试验组率 pA = 0.50，对照组率 pB = 0.30，分配比例 k = 1。

**计算结果**：对照组样本量 nB ≈ 120.8354 人，试验组样本量 nA ≈ 120.8354 人。实际调查中两组各至少调查 121 人。

**结果解读**：在95%的置信水平和90%的检验效能下，试验组和对照组各需约121人，共计242人，才能检测出两组有效率20个百分点的差异。
""",
    "three_prop": """
**应用场景**：某研究者欲比较三种健康教育方式对居民健康行为改变的效果。预期三组的有效率分别为30%、40%和50%，拟采用卡方检验进行比较。

**参数设定**：置信水平 α = 0.05，β = 0.10，组1率 p₁ = 0.30，组2率 p₂ = 0.40，组3率 p₃ = 0.50。

**计算结果**：效应量 w ≈ 0.1667，总样本量 N ≈ 455.4000 人，每组样本量 n_per ≈ 151.8000 人，非中心参数 λ = 12.65。实际调查中总样本至少 456 人，每组至少 152 人。

**结果解读**：在95%的置信水平和90%的检验效能下，三组共需约456人（每组约152人），才能检测出三组有效率之间的差异。
""",
    "one_mean": """
**应用场景**：某研究者欲评估一种新的教学方法能否将学生的考试成绩从目前的100分提高到105分。拟采用单组前后比较设计。

**参数设定**：置信水平 α = 0.05，β = 0.10，总体均数 μ₀ = 100 分，预期样本均数 μ₁ = 105 分，标准差 σ = 15 分。

**计算结果**：所需样本量 n ≈ 96.4875 人，均数差值 δ = 5 分。实际调查中建议至少调查 97 人。

**结果解读**：在95%的置信水平和90%的检验效能下，需要约97名学生才能检测出平均成绩提高5分的效应。
""",
    "two_mean_equal": """
**应用场景**：某研究者欲评价一种新型降压药的效果。拟开展随机对照试验，预期试验组平均收缩压可降低8 mmHg（从108 mmHg降至100 mmHg），两组标准差约为15 mmHg，样本量相等。

**参数设定**：置信水平 α = 0.05，β = 0.10，对照组均数 μ₀ = 100 mmHg，试验组均数 μ₁ = 108 mmHg，合并标准差 σ = 15 mmHg，分配比例 k = 1。

**计算结果**：对照组样本量 n₀ ≈ 74.8407 人，试验组样本量 n₁ ≈ 74.8407 人。实际调查中两组各至少调查 75 人。

**结果解读**：在95%的置信水平和90%的检验效能下，每组各需约75人，共计150人，才能检测出两组收缩压平均差值为8 mmHg的效应。
""",
    "two_mean_unequal": """
**应用场景**：某研究者欲比较两种不同的运动方案对老年人肺活量的改善效果。已知两组肺活量的变异程度不同，对照组标准差约为15，试验组标准差约为18。

**参数设定**：置信水平 α = 0.05，β = 0.10，对照组均数 μ₀ = 100，试验组均数 μ₁ = 108，对照组标准差 σ₀ = 15，试验组标准差 σ₁ = 18，分配比例 k = 1。

**计算结果**：对照组样本量 n₀ ≈ 81.1931 人，试验组样本量 n₁ ≈ 81.1931 人。实际调查中两组各至少调查 82 人。

**结果解读**：由于方差不齐，所需样本量略大于方差齐性假设下的结果。每组约82人，共计164人，方可检测出两组平均差值为8的效应。
""",
    "three_mean": """
**应用场景**：某研究者欲比较三种不同教学方法的教学效果，以学生考试成绩为结局指标。预期三组的平均成绩分别为8.25、11.75和13.00分，组内标准差约为3.5分。

**参数设定**：置信水平 α = 0.05，β = 0.10，组1均数 μ₁ = 8.25，组2均数 μ₂ = 11.75，组3均数 μ₃ = 13.00，共同标准差 σ = 3.5。

**计算结果**：效应量 f ≈ 0.5744，总样本量 N = 42 人，每组样本量 n = 14 人，非中心参数 λ ≈ 13.8571（迭代）。

**结果解读**：在95%的置信水平和90%的检验效能下，三组共需42人（每组14人），即可检测出三组平均成绩之间的差异。
"""
}

example_text = example_map.get(current_method["id"], "")
if example_text:
    st.markdown("### 📝 参考示例")
    st.markdown(example_text)

# ============================
# 11. 版权信息
# ============================

st.divider()
current_year = datetime.now().year
st.caption(f"© {current_year} 长沙市疾病预防控制中心 版权所有 · 开发人员：谢赐福 · Python+Streamlit")
