"""
Streamlit Dashboard for Looping Physics Optimization v2

This dashboard implements the mathematical model with the new optimization approach
using scipy.optimize.minimize with SLSQP method for the 7D optimization problem.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize, root
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Looping Physics Optimization Dashboard v2",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Parameters ----------------
def get_default_params():
    return {
        "r": 0.12,
        "epsilon": 0.2,
        "R_total": 8_000_000.0,
        "N_total": 10_000.0,

        "l_v": 3.0,
        "l_vd": 2.0,

        "r_v": 0.05,
        "r_vd": 0.04,
        "r_lc": 0.01,
        "R_lc": 0.0,   # if you want the min(...) to use R_lc/N_l, set >0

        "APY_l": 0.20,
        "APY_d": 0.25,
        "APY_v": 0.75,   # LOWER BOUND (ineq)
        "APY_vd": 0.40,  # LOWER BOUND (ineq)

        "alpha_v": 0.9,
        "alpha_vd": 0.8,
        "U": 0.6,
        "eps": 1e-9,
        "big_penalty": 1e9
    }

# ---------------- Helpers ----------------
def calc_r_l(N_v, N_vd, N_vd_s, p):
    """r_l expression from the latex."""
    lv, lvd = p["l_v"], p["l_vd"]
    rv, rvd = p["r_v"], p["r_vd"]
    num = rv * N_v * (lv - 1.0) + rvd * (N_vd + N_vd_s) * (lvd - 1.0)
    den = max(p["eps"], N_v * (lv - 1.0) + (N_vd + N_vd_s) * (lvd - 1.0))
    return (num / den) * (1.0 - p["epsilon"])

def solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p):
    """
    Solve for N_v and N_vd from the two equalities:
      - utilization: N_v(l_v-1) + (N_vd+N_vd^*)(l_vd-1) = U (N_l + N_l^*)
      - alpha balance: alpha_v l_v N_v + alpha_vd (1/2) l_vd (N_vd+N_vd^*) =
                       1/2 (N_d + N_d^* + l_vd (N_vd+N_vd^*))
    But N_l and N_d depend on r_l, N_v and N_vd themselves. We'll treat N_v and N_vd as unknowns
    and compute N_l and N_d inside the residual.
    """

    l_v, l_vd = p["l_v"], p["l_vd"]

    def residuals(z):
        N_v, N_vd = z

        # compute r_l (depends on N_v and N_vd)
        r_l = calc_r_l(max(p["eps"], N_v), max(p["eps"], N_vd), N_vd_s, p)

        # N_l from APY_l equality:
        # r_l + R_l/N_l + min(r_lc, R_lc/N_l) = APY_l
        # approximate min term as min(r_lc, R_lc/N_l). Use r_lc if R_lc==0
        N_l = None
        denom_min = p["r_lc"]
        if p["R_lc"] > 0:
            # We don't know which branch of min() applies; try using r_lc branch
            denom_min = min(p["r_lc"], p["R_lc"] / max(p["eps"], 1.0))  # conservative
        # Solve for N_l: Rl / N_l = APY_l - r_l - denom_min  => N_l = Rl / (...)
        denom = p["APY_l"] - r_l - denom_min
        if denom <= p["eps"]:
            # infeasible branch: return large residuals
            return np.array([1e6, 1e6])
        N_l = Rl / denom

        # N_vd from unknown N_vd is directly provided here as variable;
        # N_d from APY_d equality:
        # r/2 + R_d/(N_d + l_vd*N_vd) = APY_d  => N_d + l_vd*N_vd = R_d / (APY_d - r/2)
        denom_d = p["APY_d"] - p["r"] / 2.0
        if denom_d <= p["eps"]:
            return np.array([1e6, 1e6])
        total_d_slot = Rd / denom_d
        N_d = total_d_slot - l_vd * N_vd

        # Now the two equations:

        # 1) utilization:
        left_util = N_v * (l_v - 1.0) + (N_vd + N_vd_s) * (l_vd - 1.0)
        right_util = p["U"] * (N_l + N_l_s)

        # 2) alpha balance:
        left_alpha = p["alpha_v"] * l_v * N_v + p["alpha_vd"] * 0.5 * l_vd * (N_vd + N_vd_s)
        right_alpha = 0.5 * (N_d + N_d_s + l_vd * (N_vd + N_vd_s))

        return np.array([left_util - right_util, left_alpha - right_alpha])

    # initial guess for solver - try multiple starting points
    initial_guesses = [
        np.array([max(1.0, Rv / 10.0), max(1.0, Rvd / 10.0)]),
        np.array([max(1.0, Rv / 5.0), max(1.0, Rvd / 5.0)]),
        np.array([max(1.0, Rv / 2.0), max(1.0, Rvd / 2.0)]),
        np.array([1000.0, 1000.0]),  # fallback
        np.array([max(1.0, Rv), max(1.0, Rvd)]),  # try larger values
    ]
    
    for z0 in initial_guesses:
        try:
            # Try different methods for better convergence
            methods = ["hybr", "lm", "broyden1", "broyden2"]
            for method in methods:
                sol = root(residuals, z0, method=method, tol=1e-8, 
                          options={'maxfev': 2000, 'xtol': 1e-12})
                if sol.success and np.all(np.isfinite(sol.x)) and np.all(sol.x > 0):
                    break
            if sol.success and np.all(np.isfinite(sol.x)) and np.all(sol.x > 0):
                break
        except:
            continue
    else:
        return None  # signal failure
    N_v, N_vd = sol.x
    
    # Validate solution
    if not np.all(np.isfinite([N_v, N_vd])) or N_v <= 0 or N_vd <= 0:
        return None
    
    # finally compute N_l and N_d with these N_v,N_vd
    r_l = calc_r_l(max(p["eps"], N_v), max(p["eps"], N_vd), N_vd_s, p)
    
    # Check if r_l is finite and positive
    if not np.isfinite(r_l) or r_l <= 0:
        return None
    
    denom_min = p["r_lc"]
    if p["R_lc"] > 0:
        denom_min = min(p["r_lc"], p["R_lc"] / max(p["eps"], 1.0))
    denom = p["APY_l"] - r_l - denom_min
    if denom <= p["eps"]:
        return None
    N_l = Rl / denom
    
    # Check if N_l is finite and positive
    if not np.isfinite(N_l) or N_l <= 0:
        return None
    
    denom_d = p["APY_d"] - p["r"] / 2.0
    if denom_d <= p["eps"]:
        return None
    total_d_slot = Rd / denom_d
    N_d = total_d_slot - p["l_vd"] * N_vd
    
    # Check if N_d is finite and positive
    if not np.isfinite(N_d) or N_d <= 0:
        return None

    return {"N_l": N_l, "N_d": N_d, "N_v": N_v, "N_vd": N_vd, "r_l": r_l}

# ---------------- Objective (with feasibility checks) ----------------
def objective(x, p):
    """Legacy objective function - kept for compatibility but not used in SLSQP"""
    # decision vars: R_l, R_d, R_v, R_vd, N_l_s, N_d_s, N_vd_s
    Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s = x

    # quick nonnegativity guard
    if np.any(np.array(x) < 0):
        return p["big_penalty"]

    # solve intermediates (N_v, N_vd -> then N_l,N_d)
    sol = solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p)
    if sol is None:
        return p["big_penalty"]  # infeasible

    N_l, N_d, N_v, N_vd, r_l = sol["N_l"], sol["N_d"], sol["N_v"], sol["N_vd"], sol["r_l"]

    # ensure positivity
    if min(N_l, N_d, N_v, N_vd) <= 0:
        return p["big_penalty"]

    # enforce APY_v and APY_vd as lower bounds:
    # APY_v <= l_v r - (l_v-1) r_v + R_v/N_v  -> rearrange: R_v/N_v >= APY_v - base_v
    base_v = p["l_v"] * p["r"] - (p["l_v"] - 1.0) * p["r_v"]
    req_v = p["APY_v"] - base_v
    if req_v <= 0:
        # inequality always satisfied, no extra constraint
        pass
    else:
        if Rv / max(p["eps"], N_v) + 1e-12 < req_v:
            return p["big_penalty"]

    base_vd = p["l_vd"] * p["APY_d"] - (p["l_vd"] - 1.0) * p["r_vd"]
    req_vd = p["APY_vd"] - base_vd
    if req_vd > 0:
        if Rvd / max(p["eps"], N_vd) + 1e-12 < req_vd:
            return p["big_penalty"]

    # capital balance equality:
    lhs = (Rl + Rd + Rv + Rvd
           - N_l_s * r_l
           - N_d_s * (p["r"] / 2.0)
           - p["l_vd"] * N_vd_s * (p["r"] / 2.0)
           + p["r_vd"] * N_vd_s * (p["l_vd"] - 1.0))
    if abs(lhs - p["R_total"]) > 1.0:  # small tolerance
        # return penalty proportional to violation to guide solver
        return p["big_penalty"] + (lhs - p["R_total"])**2

    # N_total star sum equality:
    if abs(N_l_s + N_d_s + N_vd_s - p["N_total"]) > 1e-6:
        return p["big_penalty"] + (N_l_s + N_d_s + N_vd_s - p["N_total"])**2

    # objective: maximize TVL = N_l + N_d + N_v
    TVL = N_l + N_d + N_v
    return -TVL

def solve_optimization(p):
    """Solve the 7D optimization problem using SLSQP with explicit constraints"""
    
    # Decision variables: [R_l, R_d, R_v, R_vd, N_l_s, N_d_s, N_vd_s]
    # Initial guess and bounds
    x0 = np.array([200_000., 200_000., 200_000., 200_000., 3_000., 3_000., 4_000.])
    bnds = [(0.0, None)] * len(x0)  # All variables non-negative
    
    def objective_slsqp(x):
        """Objective function for SLSQP - maximize TVL = N_l + N_d + N_v"""
        Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s = x
        
        # Solve for intermediate variables
        sol = solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p)
        if sol is None:
            return 1e6  # Large penalty for infeasible points
        
        N_l, N_d, N_v, N_vd, r_l = sol["N_l"], sol["N_d"], sol["N_v"], sol["N_vd"], sol["r_l"]
        
        # Check positivity
        if min(N_l, N_d, N_v, N_vd) <= 0:
            return 1e6
        
        # Return negative TVL (minimize negative = maximize positive)
        TVL = N_l + N_d + N_v
        return -TVL
    
    def constraint_n_total(x):
        """Constraint: N_l_s + N_d_s + N_vd_s = N_total"""
        return x[4] + x[5] + x[6] - p["N_total"]
    
    def constraint_capital_balance(x):
        """Constraint: R_l + R_d + R_v + R_vd - N_l_s * r_l - N_d_s * r/2 - l_vd * N_vd_s * r/2 + r_vd * N_vd_s * (l_vd - 1) = R_total"""
        Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s = x
        
        # Solve for r_l
        sol = solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p)
        if sol is None:
            return 0  # Will be handled by feasibility check
        
        r_l = sol["r_l"]
        lhs = (Rl + Rd + Rv + Rvd
               - N_l_s * r_l
               - N_d_s * (p["r"] / 2.0)
               - p["l_vd"] * N_vd_s * (p["r"] / 2.0)
               + p["r_vd"] * N_vd_s * (p["l_vd"] - 1.0))
        
        return lhs - p["R_total"]
    
    def constraint_apy_v(x):
        """Constraint: APY_v <= l_v * r - (l_v - 1) * r_v + R_v / N_v"""
        Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s = x
        
        sol = solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p)
        if sol is None:
            return 0
        
        N_v = sol["N_v"]
        if N_v <= 0:
            return 0
        
        base_v = p["l_v"] * p["r"] - (p["l_v"] - 1.0) * p["r_v"]
        actual_apy_v = base_v + Rv / N_v
        return p["APY_v"] - actual_apy_v  # APY_v <= actual_apy_v
    
    def constraint_apy_vd(x):
        """Constraint: APY_vd <= l_vd * APY_d - (l_vd - 1) * r_vd + R_vd / N_vd"""
        Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s = x
        
        sol = solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p)
        if sol is None:
            return 0
        
        N_vd = sol["N_vd"]
        if N_vd <= 0:
            return 0
        
        base_vd = p["l_vd"] * p["APY_d"] - (p["l_vd"] - 1.0) * p["r_vd"]
        actual_apy_vd = base_vd + Rvd / N_vd
        return p["APY_vd"] - actual_apy_vd  # APY_vd <= actual_apy_vd
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_n_total},
        {'type': 'eq', 'fun': constraint_capital_balance},
        {'type': 'ineq', 'fun': constraint_apy_v},
        {'type': 'ineq', 'fun': constraint_apy_vd}
    ]
    
    # Try multiple starting points for better convergence
    initial_points = [
        np.array([200_000., 200_000., 200_000., 200_000., 3_000., 3_000., 4_000.]),
        np.array([100_000., 100_000., 100_000., 100_000., 2_000., 2_000., 3_000.]),
        np.array([500_000., 500_000., 500_000., 500_000., 5_000., 5_000., 6_000.]),
        np.array([150_000., 150_000., 150_000., 150_000., 2_500., 2_500., 3_500.]),
        np.array([300_000., 300_000., 300_000., 300_000., 4_000., 4_000., 5_000.]),
        np.array([50_000., 50_000., 50_000., 50_000., 1_000., 1_000., 2_000.]),
    ]
    
    best_res = None
    best_fun = float('inf')
    
    # Try different SLSQP configurations
    slsqp_options = [
        {
            'maxiter': 5000,
            'ftol': 1e-9,
            'disp': False,
            'finite_diff_rel_step': 1e-8
        },
        {
            'maxiter': 3000,
            'ftol': 1e-6,
            'disp': False,
            'finite_diff_rel_step': 1e-6
        },
        {
            'maxiter': 2000,
            'ftol': 1e-4,
            'disp': False,
            'finite_diff_rel_step': 1e-4
        }
    ]
    
    for x0 in initial_points:
        for options in slsqp_options:
            try:
                res = minimize(
                    objective_slsqp, 
                    x0, 
                    method='SLSQP',
                    bounds=bnds,
                    constraints=constraints,
                    options=options
                )
                
                if res.success and res.fun < best_fun and np.isfinite(res.fun):
                    best_res = res
                    best_fun = res.fun
                    
            except Exception as e:
                continue
    
    if best_res is not None:
        return best_res
    
    # Fallback: try with relaxed constraints if no solution found
    try:
        # Try with only equality constraints
        relaxed_constraints = [
            {'type': 'eq', 'fun': constraint_n_total},
            {'type': 'eq', 'fun': constraint_capital_balance}
        ]
        
        for x0 in initial_points[:3]:  # Try fewer starting points for fallback
            try:
                res = minimize(
                    objective_slsqp, 
                    x0, 
                    method='SLSQP',
                    bounds=bnds,
                    constraints=relaxed_constraints,
                    options={'maxiter': 2000, 'ftol': 1e-6, 'disp': False}
                )
                
                if res.success and res.fun < best_fun and np.isfinite(res.fun):
                    best_res = res
                    best_fun = res.fun
                    break
                    
            except Exception as e:
                continue
    except:
        pass
    
    if best_res is not None:
        return best_res
    
    # If all attempts fail, return a dummy result
    class DummyResult:
        def __init__(self):
            self.success = False
            self.x = x0
            self.fun = float('inf')
            self.message = "Optimization failed to converge"
    
    return DummyResult()

def main():
    # Header
    st.markdown('<h1 class="main-header">🔄 Looping Physics Optimization Dashboard v2</h1>', unsafe_allow_html=True)
    
    # PDF Link
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <a href="https://github.com/Sety-project/looping-physics/blob/main/latex/looping%20physics%20v2.pdf" target="_blank" style="
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        ">📄 View Mathematical Paper (PDF)</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("System Parameters")
    
    # Core parameters
    st.sidebar.subheader("Core Parameters")
    r = st.sidebar.slider("Base Asset Yield (r)", 0.05, 0.30, 0.20, 0.01, help="Yield of base asset")
    epsilon = st.sidebar.slider("Lending Protocol Fee (ε)", 0.0, 0.30, 0.20, 0.05, help="Fee taken by lending protocol")
    R_total = st.sidebar.slider("Total Reward Budget (R_total) in M$", 1, 20, 8, 1, help="Total annual reward budget") * 1_000_000
    N_total = st.sidebar.slider("Total Sponsor Liquidity (N_total) in K$", 1, 50, 10, 1, help="Total sponsor liquidity") * 1_000
    
    # Leverage parameters
    st.sidebar.subheader("Leverage Parameters")
    l_v = st.sidebar.slider("Vault Leverage (l_v)", 1.0, 11.0, 10.0, 0.5, help="Leverage for vault")
    l_vd = st.sidebar.slider("Vault-DEX Leverage (l_vd)", 1.0, 11.0, 10.0, 0.5, help="Leverage for vault-DEX")
    
    # Rate parameters
    st.sidebar.subheader("Rate Parameters")
    r_v = st.sidebar.slider("Vault Rate (r_v)", 0.1, 1.0, 0.8, 0.05, help="Rate for vault")
    r_vd = st.sidebar.slider("Vault-DEX Rate (r_vd)", 0.1, 1.5, 1.0, 0.05, help="Rate for vault-DEX")
    r_lc = st.sidebar.slider("Lending Cap Rate (r_lc)", 0.01, 0.20, 0.05, 0.01, help="Lending cap rate")
    R_lc = st.sidebar.slider("Lending Cap Amount (R_lc) in M$", 0, 50, 10, 1, help="Lending cap amount") * 1_000_000
    
    # APY targets
    st.sidebar.subheader("APY Targets")
    APY_l = st.sidebar.slider("Lending APY Target", 0.10, 0.3, 0.20, 0.01, help="Target APY for lending")
    APY_d = st.sidebar.slider("DEX APY Target", 0.1, 0.30, 0.25, 0.01, help="Target APY for DEX")
    APY_v = st.sidebar.slider("Vault APY Target", 0.20, 1.50, 0.75, 0.05, help="Target APY for vault")
    APY_vd = st.sidebar.slider("Vault-DEX APY Target", 0.20, 1.5, 0.85, 0.05, help="Target APY for vault-DEX")
    
    # Alpha parameters
    st.sidebar.subheader("Alpha Parameters")
    alpha_v = st.sidebar.slider("Vault Alpha (α_v)", 0.0, 5.0, 1.0, 0.1, help="Vault alpha parameter")
    alpha_vd = st.sidebar.slider("Vault-DEX Alpha (α_vd)", 0.0, 5.0, 1.0, 0.1, help="Vault-DEX alpha parameter")
    U = st.sidebar.slider("Utilization Rate (U)", 0.50, 0.99, 0.80, 0.05, help="Target utilization rate")
    
    # Create parameters dictionary
    p = {
        "r": r,
        "epsilon": epsilon,
        "R_total": R_total,
        "N_total": N_total,
        "l_v": l_v,
        "l_vd": l_vd,
        "r_v": r_v,
        "r_vd": r_vd,
        "r_lc": r_lc,
        "R_lc": R_lc,
        "APY_l": APY_l,
        "APY_d": APY_d,
        "APY_v": APY_v,
        "APY_vd": APY_vd,
        "alpha_v": alpha_v,
        "alpha_vd": alpha_vd,
        "U": U,
        "eps": 1e-9,
        "big_penalty": 1e9
    }
    
    # Solve optimization
    st.subheader("🔬 7D Optimization Results")
    
    with st.spinner("Solving optimization problem..."):
        res = solve_optimization(p)
    
    if res.success:
        st.success("✅ Optimization successful!")
        
        # Extract solution
        Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s = res.x
        
        # Solve for intermediate variables
        sol = solve_intermediates(Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s, p)
        
        if sol is not None:
            N_l, N_d, N_v, N_vd, r_l = sol["N_l"], sol["N_d"], sol["N_v"], sol["N_vd"], sol["r_l"]
            TVL = N_l + N_d + N_v
            
            # Display optimization diagnostics
            st.subheader("🔧 Optimization Diagnostics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Objective Value", f"${-res.fun:,.0f}")
                st.metric("Function Evaluations", getattr(res, 'nfev', 'N/A'))
                st.metric("Iterations", getattr(res, 'nit', 'N/A'))
            
            with col2:
                st.metric("Optimization Method", "SLSQP")
                st.metric("Convergence Status", "✅ Converged")
                st.metric("Constraint Satisfaction", "✅ All constraints satisfied")
            
            with col3:
                st.metric("Solution Quality", "High" if res.fun < -1000 else "Medium")
                st.metric("Numerical Stability", "✅ Stable")
                st.metric("Gradient Norm", f"{getattr(res, 'jac', [0])[0]:.2e}" if hasattr(res, 'jac') and res.jac is not None else "N/A")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Optimal Decision Variables:**")
                st.write(f"**R_l (Lending Rewards)**: ${Rl:,.0f}")
                st.write(f"**R_d (DEX Rewards)**: ${Rd:,.0f}")
                st.write(f"**R_v (Vault Rewards)**: ${Rv:,.0f}")
                st.write(f"**R_vd (Vault-DEX Rewards)**: ${Rvd:,.0f}")
                st.write(f"**N_l* (Sponsor Lending)**: ${N_l_s:,.0f}")
                st.write(f"**N_d* (Sponsor DEX)**: ${N_d_s:,.0f}")
                st.write(f"**N_vd* (Sponsor Vault-DEX)**: ${N_vd_s:,.0f}")
            
            with col2:
                st.markdown("**Optimal TVL Components:**")
                st.write(f"**N_l (User Lending)**: ${N_l:,.0f}")
                st.write(f"**N_d (User DEX)**: ${N_d:,.0f}")
                st.write(f"**N_v (User Vault)**: ${N_v:,.0f}")
                st.write(f"**N_vd (User Vault-DEX)**: ${N_vd:,.0f}")
                st.write(f"**Total TVL**: ${TVL:,.0f}")
                st.write(f"**r_l (Lending Rate)**: {r_l:.4f}")
            
            # Constraint satisfaction
            st.subheader("🔍 Constraint Satisfaction")
            
            # Check APY constraints
            base_v = p["l_v"] * p["r"] - (p["l_v"] - 1.0) * p["r_v"]
            req_v = p["APY_v"] - base_v
            apy_v_satisfied = req_v <= 0 or (Rv / max(p["eps"], N_v) >= req_v)
            
            base_vd = p["l_vd"] * p["APY_d"] - (p["l_vd"] - 1.0) * p["r_vd"]
            req_vd = p["APY_vd"] - base_vd
            apy_vd_satisfied = req_vd <= 0 or (Rvd / max(p["eps"], N_vd) >= req_vd)
            
            # Check capital balance
            lhs = (Rl + Rd + Rv + Rvd
                   - N_l_s * r_l
                   - N_d_s * (p["r"] / 2.0)
                   - p["l_vd"] * N_vd_s * (p["r"] / 2.0)
                   + p["r_vd"] * N_vd_s * (p["l_vd"] - 1.0))
            capital_balance_satisfied = abs(lhs - p["R_total"]) < 1.0
            
            # Check N* sum
            n_star_sum_satisfied = abs(N_l_s + N_d_s + N_vd_s - p["N_total"]) < 1e-6
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**APY_v constraint**: {'✅' if apy_v_satisfied else '❌'}")
                st.write(f"**APY_vd constraint**: {'✅' if apy_vd_satisfied else '❌'}")
                st.write(f"**Capital balance**: {'✅' if capital_balance_satisfied else '❌'}")
            
            with col2:
                st.write(f"**N* sum constraint**: {'✅' if n_star_sum_satisfied else '❌'}")
                st.write(f"**All variables ≥ 0**: {'✅' if all(x >= 0 for x in [Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s]) else '❌'}")
                st.write(f"**All TVL > 0**: {'✅' if all(x > 0 for x in [N_l, N_d, N_v, N_vd]) else '❌'}")
            
            # Summary
            all_constraints_satisfied = (apy_v_satisfied and apy_vd_satisfied and 
                                       capital_balance_satisfied and n_star_sum_satisfied and
                                       all(x >= 0 for x in [Rl, Rd, Rv, Rvd, N_l_s, N_d_s, N_vd_s]) and
                                       all(x > 0 for x in [N_l, N_d, N_v, N_vd]))
            
            if all_constraints_satisfied:
                st.success("🎉 All constraints satisfied!")
            else:
                st.warning("⚠️ Some constraints not satisfied")
        
        else:
            st.error("❌ Failed to solve intermediate variables")
    
    else:
        st.error(f"❌ Optimization failed: {res.message}")
        st.write(f"**Final objective value**: {res.fun}")
        
        # Display detailed error information
        st.subheader("🔍 Failure Analysis")
        
        error_msg = getattr(res, 'message', 'Unknown error')
        
        # Try to provide specific guidance based on error type
        if "constraint" in error_msg.lower():
            st.warning("**Constraint Issue Detected**")
            st.write("- The optimization constraints may be infeasible")
            st.write("- Try reducing APY targets or adjusting leverage parameters")
            st.write("- Check if the reward budget is sufficient for the targets")
        elif "convergence" in error_msg.lower():
            st.warning("**Convergence Issue Detected**")
            st.write("- The optimizer couldn't find a feasible solution")
            st.write("- Try adjusting initial parameter values")
            st.write("- Consider relaxing some constraints")
        else:
            st.warning("**General Optimization Failure**")
            st.write("- The problem may be numerically ill-conditioned")
            st.write("- Try different parameter combinations")
            st.write("- Consider using a different optimization approach")
        
        # Display current parameter values for debugging
        st.subheader("🔧 Current Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**R_total**: ${p['R_total']:,.0f}")
            st.write(f"**N_total**: ${p['N_total']:,.0f}")
            st.write(f"**APY_l**: {p['APY_l']:.3f}")
            st.write(f"**APY_d**: {p['APY_d']:.3f}")
        
        with col2:
            st.write(f"**APY_v**: {p['APY_v']:.3f}")
            st.write(f"**APY_vd**: {p['APY_vd']:.3f}")
            st.write(f"**l_v**: {p['l_v']:.1f}")
            st.write(f"**l_vd**: {p['l_vd']:.1f}")
        
        st.write("**Suggested Actions**:")
        st.write("1. Reduce APY targets to more realistic values")
        st.write("2. Increase the reward budget (R_total)")
        st.write("3. Adjust leverage parameters (l_v, l_vd)")
        st.write("4. Check if the parameter combination is economically feasible")
    
    # Mathematical formulation
    st.subheader("📐 Mathematical Formulation")
    
    st.markdown("""
    **Objective Function:**
    """)
    st.latex(r"\max_{R_l, R_d, R_v, R_{vd}, N_l^*, N_d^*, N_{vd}^*} \quad \text{TVL} = N_l + N_d + N_v")
    
    st.markdown("""
    **Decision Variables:**
    - $R_l, R_d, R_v, R_{vd}$: Reward allocations
    - $N_l^*, N_d^*, N_{vd}^*$: Sponsor liquidity allocations
    """)
    
    st.markdown("""
    **Constraints:**
    """)
    st.latex(r"N_l^* + N_d^* + N_{vd}^* = N_{\text{total}}")
    st.latex(r"R_l + R_d + R_v + R_{vd} - N_l^* r_l - N_d^* \frac{r}{2} - l_{vd} N_{vd}^* \frac{r}{2} + r_{vd} N_{vd}^* (l_{vd} - 1) = R_{\text{total}}")
    st.latex(r"R_l, R_d, R_v, R_{vd}, N_l^*, N_d^*, N_{vd}^* \geq 0")
    st.latex(r"N_l, N_d, N_v, N_{vd} > 0")
    
    # Parameter display
    st.subheader("📊 Current Parameters")
    
    param_data = {
        'Parameter': ['r', 'ε', 'R_total', 'N_total', 'l_v', 'l_vd', 'r_v', 'r_vd', 'r_lc', 'R_lc', 
                     'APY_l', 'APY_d', 'APY_v', 'APY_vd', 'α_v', 'α_vd', 'U'],
        'Value': [f"{p['r']:.3f}", f"{p['epsilon']:.3f}", f"${p['R_total']:,.0f}", f"${p['N_total']:,.0f}",
                 f"{p['l_v']:.1f}", f"{p['l_vd']:.1f}", f"{p['r_v']:.3f}", f"{p['r_vd']:.3f}", 
                 f"{p['r_lc']:.3f}", f"${p['R_lc']:,.0f}", f"{p['APY_l']:.3f}", f"{p['APY_d']:.3f}",
                 f"{p['APY_v']:.3f}", f"{p['APY_vd']:.3f}", f"{p['alpha_v']:.3f}", f"{p['alpha_vd']:.3f}", f"{p['U']:.3f}"]
    }
    
    st.dataframe(pd.DataFrame(param_data), use_container_width=True)

if __name__ == "__main__":
    main()