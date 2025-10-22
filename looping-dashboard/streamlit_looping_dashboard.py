"""
Streamlit Dashboard for Looping Physics Optimization

This dashboard implements the mathematical model from v1.tex with interactive sliders
and visualizations for the 2D linear programming problem.
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
from scipy.optimize import linprog
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Looping Physics Optimization Dashboard",
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

def calculate_user_tvl_direct(params):
    """Calculate user TVL directly from LaTeX equations"""
    
    R_l = params['R_l']
    R_d = params['R_d']
    R_v = params.get('R_v', 0)
    
    APY_l = params['APY_l']
    APY_d = params['APY_d']
    APY_v = params['APY_v']
    r_b = params['r_b']
    epsilon = params['epsilon']
    r = params['r']
    l = params['l']
    
    # Direct equations from LaTeX
    N_l = R_l / (APY_l - r_b * (1 - epsilon))
    N_d = R_d / (APY_d - r / 2)
    
    return {'N_l': N_l, 'N_d': N_d}

def calculate_sponsor_tvl_direct(params, user_tvl):
    """Calculate sponsor TVL using matrix solution for the system of equations"""
    
    import numpy as np
    
    l = params['l']
    U = params['U']
    alpha = params['alpha']
    APY_v = params['APY_v']
    r = params['r']
    r_b = params['r_b']
    N = params['N']
    
    N_l = user_tvl['N_l']
    N_d = user_tvl['N_d']
    
    # System of equations from LaTeX:
    # N_l* + N_d* = N
    # U(N_l+N_l*) = α(N_d+N_d*)
    
    # Rewrite as:
    # N_l* + N_d* = N
    # U*N_l* - α*N_d* = α*N_d - U*N_l
    
    N_l_star = (alpha * N_d + alpha * N - U * N_l) / (U + alpha)
    N_d_star = N - N_l_star
    
    # Calculate N_v from active constraint
    N_v = U * (N_l + N_l_star) / (l - 1)
    
    # Calculate R_v from new equation
    R_v = N_v * max(0, APY_v - l * r + (l - 1) * r_b)
    
    return {
        'N_l_star': N_l_star,
        'N_d_star': N_d_star,
        'N_v': N_v,
        'R_v': R_v,
        'constraint_1': U * (N_l + N_l_star),
        'constraint_2': alpha * (N_d + N_d_star),
        'total_tvl': N_l + N_d + N_v,
    }

def calculate_reward_budget_direct(params, sponsor_tvl):
    """Calculate R_v from budget constraint"""
    
    R_l = params['R_l']
    R_d = params['R_d']
    R = params['R']
    r_b = params['r_b']
    epsilon = params['epsilon']
    r = params['r']
    
    N_l_star = sponsor_tvl['N_l_star']
    N_d_star = sponsor_tvl['N_d_star']
    
    # Budget constraint: (R_l + R_d + R_v) - N_l* r_b(1-ε) - N_d* r/2 = R
    R_v = R + N_l_star * r_b * (1 - epsilon) + N_d_star * r / 2 - R_l - R_d
    
    return R_v

def solve_2d_simplex_optimization(params, user_tvl):
    """
    Solve the 2D LP in R_l and R_d using scipy.optimize.linprog.

    Maximize TVL = N_l + N_d + N_v
    Subject to:
        R_l >= 0
        R_d >= 0
        N_l^* >= 0
        N_d^* >= 0
        (R_l+R_d+R_v) - N_l^* r_b (1-eps) - N_d^* r/2 = R_total

    All intermediate quantities are linear in R_l and R_d for fixed parameters.
    """
    
    # Unpack params
    APY_l = params["APY_l"]
    APY_d = params["APY_d"]
    APY_v = params["APY_v"]
    r_b   = params["r_b"]
    r     = params["r"]
    eps   = params["epsilon"]
    l     = params["l"]
    U     = params["U"]
    alpha = params["alpha"]
    N     = params["N"]          # exogenous N
    R_total = params["R"]        # total required RHS of budget equality

    # --- Sanity checks on input parameter conditions ---
    try:
        assert (APY_v - l*r)/(l-1) <= r_b + 1e-12, "r_b is below lower bound (APY_v - l*r)/(l-1)"
        assert r_b <= APY_l/(1-eps) + 1e-12, "r_b is above upper bound APY_l/(1-eps)"
        assert r/2 <= APY_d + 1e-12, "APY_d must be >= r/2"
    except AssertionError as e:
        return {"success": False, "message": str(e), "res": None}

    # --- Precompute linear denominators (non-zero due to parameter constraints) ---
    denom_l = APY_l - r_b*(1 - eps)   # divisor for N_l: N_l = R_l / denom_l
    denom_d = APY_d - r/2             # divisor for N_d: N_d = R_d / denom_d

    if abs(denom_l) < 1e-12 or abs(denom_d) < 1e-12:
        return {"success": False, "message": "Denominator too small; check parameters (denom_l or denom_d ~ 0).", "res": None}

    # K factor for N_v:
    K = (1.0/(l - 1.0)) * (1.0 / (1.0/alpha + 1.0/U))  # scalar

    # s factor controlling R_v = s * N_v (s = max(0, APY_v - l*r + (l-1) r_b))
    s = max(0.0, APY_v - l*r + (l - 1.0)*r_b)

    # -------- Objective: maximize TVL = N_l + N_d + N_v
    # Express TVL = a_Rl * R_l + a_Rd * R_d + const
    # N_l = R_l / denom_l
    # N_d = R_d / denom_d
    # N_v = K * (N_l + N_d + N)
    # => TVL = (1 + K) * N_l + (1 + K) * N_d + K * N
    a_Rl = (1.0 + K) / denom_l
    a_Rd = (1.0 + K) / denom_d
    const_term = K * N

    # linprog minimizes c^T x. To maximize, minimize -TVL.
    c = np.array([-a_Rl, -a_Rd])  # decision vars: [R_l, R_d]

    # -------- Constraints
    # Variables: x = [R_l, R_d]

    # 1) Non-negativity done via bounds

    # 2) N_l^* >= 0  and  N_d^* >= 0
    # N_l^* = (alpha (N_d + N) - U N_l) / (U + alpha) >= 0
    # => alpha N_d + alpha N - U N_l >= 0
    # => -U*(R_l/denom_l) + alpha*(R_d/denom_d) >= -alpha * N
    # Multiply by -1 to put in A_ub x <= b_ub:
    #  U/denom_l * R_l - alpha/denom_d * R_d <= alpha * N

    # N_d^* = (U (N_l + N) - alpha N_d) / (U + alpha) >= 0
    # => U N_l + U N - alpha N_d >= 0
    # => U*(R_l/denom_l) - alpha*(R_d/denom_d) >= -U*N
    # Multiply by -1:
    # -U/denom_l * R_l + alpha/denom_d * R_d <= U * N

    A_ub = []
    b_ub = []

    # Constraint A:  U/denom_l * R_l - alpha/denom_d * R_d <= alpha * N
    A_ub.append([ U/denom_l, -alpha/denom_d ])
    b_ub.append(alpha * N)

    # Constraint B: -U/denom_l * R_l + alpha/denom_d * R_d <= U * N
    A_ub.append([ -U/denom_l, alpha/denom_d ])
    b_ub.append(U * N)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # 3) Budget equality: (R_l + R_d + R_v) - N_l^* r_b (1-eps) - N_d^* r/2 = R_total
    # 
    # Express everything as linear functions of R_l and R_d:
    # R_v = s * N_v = s * K * (N_l + N_d + N) = s * K * (R_l/denom_l + R_d/denom_d + N)
    # N_l^* = (alpha (N_d + N) - U N_l) / (U + alpha) = (alpha (R_d/denom_d + N) - U R_l/denom_l) / (U + alpha)
    # N_d^* = (U (N_l + N) - alpha N_d) / (U + alpha) = (U (R_l/denom_l + N) - alpha R_d/denom_d) / (U + alpha)
    #
    # The budget constraint becomes:
    # (R_l + R_d + s*K*(R_l/denom_l + R_d/denom_d + N)) - 
    #   [(alpha (R_d/denom_d + N) - U R_l/denom_l) / (U + alpha)] * r_b * (1-eps) - 
    #   [(U (R_l/denom_l + N) - alpha R_d/denom_d) / (U + alpha)] * r/2 = R_total
    #
    # Collecting coefficients of R_l and R_d:
    # R_l coefficient: 1 + s*K/denom_l + U*r_b*(1-eps)/(U+alpha)/denom_l - U*r/2/(U+alpha)/denom_l
    # R_d coefficient: 1 + s*K/denom_d - alpha*r_b*(1-eps)/(U+alpha)/denom_d + alpha*r/2/(U+alpha)/denom_d
    # Constant term: s*K*N - alpha*N*r_b*(1-eps)/(U+alpha) - U*N*r/2/(U+alpha)
    
    # Calculate the linear coefficients for the budget constraint
    coeff_Rl_budget = 1.0
    coeff_Rd_budget = 1.0
    const_total_LHS = 0.0
    
    # Add R_v contribution (if s > 0)
    if s > 0:
        coeff_Rl_budget += s * K / denom_l
        coeff_Rd_budget += s * K / denom_d
        const_total_LHS += s * K * N
    
    # Add N_l^* contribution: -N_l^* * r_b * (1-eps)
    # N_l^* = (alpha (R_d/denom_d + N) - U R_l/denom_l) / (U + alpha)
    # Coefficient of R_l: -(-U/denom_l) * r_b * (1-eps) / (U + alpha) = U*r_b*(1-eps)/(U+alpha)/denom_l
    # Coefficient of R_d: -(alpha/denom_d) * r_b * (1-eps) / (U + alpha) = -alpha*r_b*(1-eps)/(U+alpha)/denom_d
    # Constant: -(alpha*N) * r_b * (1-eps) / (U + alpha) = -alpha*N*r_b*(1-eps)/(U+alpha)
    coeff_Rl_budget += U * r_b * (1 - eps) / ((U + alpha) * denom_l)
    coeff_Rd_budget -= alpha * r_b * (1 - eps) / ((U + alpha) * denom_d)
    const_total_LHS -= alpha * N * r_b * (1 - eps) / (U + alpha)
    
    # Add N_d^* contribution: -N_d^* * r/2
    # N_d^* = (U (R_l/denom_l + N) - alpha R_d/denom_d) / (U + alpha)
    # Coefficient of R_l: -(U/denom_l) * r/2 / (U + alpha) = -U*r/2/(U+alpha)/denom_l
    # Coefficient of R_d: -(-alpha/denom_d) * r/2 / (U + alpha) = alpha*r/2/(U+alpha)/denom_d
    # Constant: -(U*N) * r/2 / (U + alpha) = -U*N*r/2/(U+alpha)
    coeff_Rl_budget -= U * r / 2 / ((U + alpha) * denom_l)
    coeff_Rd_budget += alpha * r / 2 / ((U + alpha) * denom_d)
    const_total_LHS -= U * N * r / 2 / (U + alpha)

    # Budget constraint as exact linear equality: coeffs * x = R_total - const_total_LHS
    A_eq = np.array([[ coeff_Rl_budget, coeff_Rd_budget ]])
    b_eq = np.array([ R_total - const_total_LHS ])
    

    # -------------------
    # Solve LP
    bounds = [(0, None), (0, None)]  # R_l >=0, R_d >=0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return {"success": False, "message": res.message, "res": res}

    R_l_opt, R_d_opt = res.x
    # compute derived quantities
    N_l = R_l_opt / denom_l
    N_d = R_d_opt / denom_d
    N_v = K * (N_l + N_d + N)
    TVL = N_l + N_d + N_v

    # compute N_l^*, N_d^*, R_v
    N_l_star = (alpha * (N_d + N) - U * N_l) / (U + alpha)
    N_d_star = (U * (N_l + N) - alpha * N_d) / (U + alpha)
    R_v_val = s * N_v
    
    # The linear programming solution should already satisfy the budget constraint exactly
    # since we implemented it as a linear equality constraint

    # Check all constraints for the solution
    constraints_satisfied = check_all_constraints(
        R_l_opt, R_d_opt, R_v_val, N_l_star, N_d_star, 
        params, N_l, N_d, N_v
    )

    return {
        "success": True,
        "R_l_opt": R_l_opt,
        "R_d_opt": R_d_opt,
        "R_v_opt": R_v_val,
        "N_l_opt": N_l,
        "N_d_opt": N_d,
        "N_v_opt": N_v,
        "N_l_star_opt": N_l_star,
        "N_d_star_opt": N_d_star,
        "optimal_tvl": TVL,
        "constraints_satisfied": constraints_satisfied,
        "result": res
    }

def check_all_constraints(R_l, R_d, R_v, N_l_star, N_d_star, params, N_l, N_d, N_v):
    """Check all 4 inequalities and 1 equality constraint from LaTeX"""
    
    # 4 Inequalities from LaTeX:
    # 1. R_l >= 0
    # 2. R_d >= 0  
    # 3. N_l* >= 0
    # 4. N_d* >= 0
    
    # 1 Equality constraint from LaTeX:
    # (R_l+R_d+R_v) - N_l* r_b(1-ε) - N_d* r/2 = R
    
    r_b = params['r_b']
    epsilon = params['epsilon']
    r = params['r']
    R = params['R']
    
    # Check inequalities
    constraint_1 = R_l >= 0
    constraint_2 = R_d >= 0
    constraint_3 = N_l_star >= 0
    constraint_4 = N_d_star >= 0
    
    # Check equality constraint
    lhs = (R_l + R_d + R_v) - N_l_star * r_b * (1 - epsilon) - N_d_star * r / 2
    constraint_5 = abs(lhs - R) < 1e-10  # Tight tolerance for exact equality
    
    return {
        'R_l >= 0': constraint_1,
        'R_d >= 0': constraint_2,
        'N_l* >= 0': constraint_3,
        'N_d* >= 0': constraint_4,
        'Budget equality': constraint_5,
        'budget_lhs': lhs,
        'budget_rhs': R,
        'all_satisfied': constraint_1 and constraint_2 and constraint_3 and constraint_4 and constraint_5
    }

def create_2d_simplex_visualization(params, user_tvl, sponsor_tvl, optimization_result):
    """Create 2D visualization of the simplex optimization with TVL colormap and constraints"""
    
    # Extract parameters
    l = params['l']
    U = params['U']
    alpha = params['alpha']
    APY_v = params['APY_v']
    r = params['r']
    r_b = params['r_b']
    N = params['N']
    R = params['R']
    epsilon = params['epsilon']
    APY_l = params['APY_l']
    APY_d = params['APY_d']
    
    # Create a grid for R_l and R_d
    R_l_range = np.linspace(0, R, 50)
    R_d_range = np.linspace(0, R, 50)
    R_l_grid, R_d_grid = np.meshgrid(R_l_range, R_d_range)
    
    # Calculate TVL for each point in the grid
    TVL_grid = np.zeros_like(R_l_grid)
    
    for i in range(R_l_grid.shape[0]):
        for j in range(R_l_grid.shape[1]):
            R_l_val = R_l_grid[i, j]
            R_d_val = R_d_grid[i, j]
            
            # Calculate N_l and N_d
            N_l_val = R_l_val / (APY_l - r_b * (1 - epsilon))
            N_d_val = R_d_val / (APY_d - r / 2)
            
            # Calculate N_l* and N_d* from constraints
            N_l_star_val = (alpha * N_d_val + alpha * N - U * N_l_val) / (U + alpha)
            N_d_star_val = N - N_l_star_val
            
            # Calculate N_v
            N_v_val = U * (N_l_val + N_l_star_val) / (l - 1)
            
            # Calculate TVL
            TVL_grid[i, j] = N_l_val + N_d_val + N_v_val
    
    # Create the plot
    fig = go.Figure()
    
    # Add TVL heatmap
    fig.add_trace(go.Heatmap(
        x=R_l_range,
        y=R_d_range,
        z=TVL_grid,
        colorscale='Viridis',
        name='TVL',
        opacity=0.7
    ))
    
    # Add constraint lines
    # Constraint 1: R_l + R_d <= R (Budget constraint)
    x_line = np.linspace(0, R, 100)
    y_line = R - x_line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='red', width=3),
        name='Budget: R_l + R_d ≤ R'
    ))
    
    # Constraint 2: N_l* ≥ 0
    # N_l* = (α*N_d + α*N - U*N_l) / (U + α) ≥ 0
    # => α*N_d + α*N - U*N_l ≥ 0
    # => α*R_d/(APY_d - r/2) + α*N - U*R_l/(APY_l - r_b(1-ε)) ≥ 0
    # => α*R_d/(APY_d - r/2) ≥ U*R_l/(APY_l - r_b(1-ε)) - α*N
    # => R_d ≥ (U*(APY_d - r/2)/(α*(APY_l - r_b(1-ε)))) * R_l - N*(APY_d - r/2)
    
    denom_l = APY_l - r_b * (1 - epsilon)
    denom_d = APY_d - r / 2
    
    if abs(denom_l) > 1e-12 and abs(denom_d) > 1e-12:
        slope_nl_star = (U * denom_d) / (alpha * denom_l)
        intercept_nl_star = -N * denom_d
        
        # Plot N_l* ≥ 0 constraint line
        x_nl_star = np.linspace(0, R, 100)
        y_nl_star = slope_nl_star * x_nl_star + intercept_nl_star
        # Only plot where y_nl_star >= 0 and within our range
        valid_mask = (y_nl_star >= 0) & (y_nl_star <= R)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=x_nl_star[valid_mask], y=y_nl_star[valid_mask],
                mode='lines',
                line=dict(color='blue', width=2, dash='dash'),
                name='N_l* ≥ 0'
            ))
    
    # Constraint 3: N_d* ≥ 0
    # N_d* = (U*N_l + U*N - α*N_d) / (U + α) ≥ 0
    # => U*N_l + U*N - α*N_d ≥ 0
    # => U*R_l/(APY_l - r_b(1-ε)) + U*N - α*R_d/(APY_d - r/2) ≥ 0
    # => U*R_l/(APY_l - r_b(1-ε)) ≥ α*R_d/(APY_d - r/2) - U*N
    # => R_l ≥ (α*(APY_l - r_b(1-ε))/(U*(APY_d - r/2))) * R_d - N*(APY_l - r_b(1-ε))
    
    if abs(denom_l) > 1e-12 and abs(denom_d) > 1e-12:
        slope_nd_star = (alpha * denom_l) / (U * denom_d)
        intercept_nd_star = -N * denom_l
        
        # Plot N_d* ≥ 0 constraint line
        y_nd_star = np.linspace(0, R, 100)
        x_nd_star = slope_nd_star * y_nd_star + intercept_nd_star
        # Only plot where x_nd_star >= 0 and within our range
        valid_mask = (x_nd_star >= 0) & (x_nd_star <= R)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=x_nd_star[valid_mask], y=y_nd_star[valid_mask],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='N_d* ≥ 0'
            ))
    
    # Add axes lines for R_l ≥ 0 and R_d ≥ 0
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, R],
        mode='lines',
        line=dict(color='purple', width=2),
        name='R_l ≥ 0'
    ))
    fig.add_trace(go.Scatter(
        x=[0, R], y=[0, 0],
        mode='lines',
        line=dict(color='purple', width=2),
        name='R_d ≥ 0'
    ))
    
    # Add current solution point
    fig.add_trace(go.Scatter(
        x=[params['R_l']], y=[params['R_d']],
        mode='markers',
        marker=dict(color='blue', size=15, symbol='circle'),
        name='Current Solution'
    ))
    
    # Add optimal solution point if available
    if optimization_result['success']:
        fig.add_trace(go.Scatter(
            x=[optimization_result['R_l_opt']], 
            y=[optimization_result['R_d_opt']],
            mode='markers',
            marker=dict(color='green', size=15, symbol='star'),
            name='Optimal Solution'
        ))
    
    # Update layout
    fig.update_layout(
        title='2D Simplex Optimization: TVL Colormap with Constraints',
        xaxis_title='R_l (Lending Rewards)',
        yaxis_title='R_d (DEX Rewards)',
        width=800,
        height=600
    )
    
    return fig

def check_liquidity_constraints(params, user_tvl, sponsor_tvl):
    """Check if N* satisfy liquidity constraints"""
    
    N_l = user_tvl['N_l']
    N_d = user_tvl['N_d']
    N_l_star = sponsor_tvl['N_l_star']
    N_d_star = sponsor_tvl['N_d_star']
    N = params['N']
    
    # Check N* > 0
    n_star_positive = N_l_star > -1e-6 and N_d_star > -1e-6
    
    # Check liquidity budget constraint (now equality: N_l* + N_d* = N)
    liquidity_budget_ok = abs((N_l_star + N_d_star) - N) < 1e-6
    
    # Check if N* satisfy the constraint relationships
    l = params['l']
    alpha = params['alpha']
    U = params['U']
    
    # From LaTeX equations:
    # U(N_l+N_l^*) = α(N_d+N_d^*)  (constraint 1)
    # N_v(l-1) = U(N_l+N_l^*)      (constraint 2)
    constraint_1_ok = abs(U * (N_l + N_l_star) - alpha * (N_d + N_d_star)) < 1e-6
    constraint_2_ok = abs(alpha * (N_d + N_d_star) - U * (N_l + N_l_star)) < 1e-6
    
    return {
        'n_star_positive': n_star_positive,
        'liquidity_budget_ok': liquidity_budget_ok,
        'constraint_1_ok': constraint_1_ok,
        'constraint_2_ok': constraint_2_ok,
        'total_sponsor_liquidity': N_l_star + N_d_star
    }

def calculate_reward_budget(params, user_tvl, sponsor_tvl):
    """Calculate R_v using reward budget constraint"""
    
    l = params['l']
    r_b = params['r_b']
    epsilon = params['epsilon']
    r = params['r']
    R_l = params['R_l']
    R_d = params['R_d']
    R = params['R']
    
    N_l_star = sponsor_tvl['N_l_star']
    N_d_star = sponsor_tvl['N_d_star']
    
    # Calculate R_v from budget constraint
    # (R_l + R_d + R_v) - N_l^* r_b(1-ε) - N_d^* r/2 = R
    R_v = R + N_l_star * r_b * (1 - epsilon) + N_d_star * r / 2 - R_l - R_d
    
    return R_v

def create_feasible_region_plot(params, coefficients):
    """Create visualization of feasible region with constraints"""
    
    # Extract parameters
    R_l_range = np.linspace(0, params['R'] * 2, 1000)
    R_d_range = np.linspace(0, params['R'] * 2, 1000)
    
    # Create meshgrid
    R_l_grid, R_d_grid = np.meshgrid(R_l_range, R_d_range)
    
    # Calculate constraint values
    gamma_constraint = coefficients['gamma_l'] * R_l_grid + coefficients['gamma_d'] * R_d_grid <= params['N'] + coefficients['gamma_0']
    delta_constraint = coefficients['delta_l'] * R_l_grid + coefficients['delta_d'] * R_d_grid <= coefficients['delta_0']
    epsilon_constraint = coefficients['epsilon_l'] * R_l_grid + coefficients['epsilon_d'] * R_d_grid <= coefficients['epsilon_0']
    positive_constraint = (R_l_grid >= 0) & (R_d_grid >= 0)
    
    # Combined feasible region
    feasible_region = gamma_constraint & delta_constraint & epsilon_constraint & positive_constraint
    
    # Calculate TVL for colormap
    TVL_grid = coefficients['alpha_l'] * R_l_grid + coefficients['alpha_d'] * R_d_grid + coefficients['alpha_0']
    TVL_grid = np.where(feasible_region, TVL_grid, np.nan)
    
    # Create plot
    fig = go.Figure()
    
    # Add TVL heatmap
    fig.add_trace(go.Heatmap(
        x=R_l_range,
        y=R_d_range,
        z=TVL_grid,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Total TVL"),
        opacity=0.7,
        name="TVL Heatmap"
    ))
    
    # Add constraint lines
    constraint_colors = ['red', 'blue', 'green', 'orange']
    constraint_names = ['γ constraint', 'δ constraint', 'ε constraint', 'R_l ≥ 0, R_d ≥ 0']
    
    # Plot constraint lines
    for i, (constraint, name, color) in enumerate(zip(
        [gamma_constraint, delta_constraint, epsilon_constraint, positive_constraint],
        constraint_names,
        constraint_colors
    )):
        # Find boundary of constraint
        boundary = np.zeros_like(constraint, dtype=bool)
        for j in range(1, constraint.shape[0]-1):
            for k in range(1, constraint.shape[1]-1):
                if constraint[j, k] and not all([constraint[j-1, k], constraint[j+1, k], constraint[j, k-1], constraint[j, k+1]]):
                    boundary[j, k] = True
        
        if np.any(boundary):
            y_coords, x_coords = np.where(boundary)
            fig.add_trace(go.Scatter(
                x=R_l_range[x_coords],
                y=R_d_range[y_coords],
                mode='markers',
                marker=dict(color=color, size=3),
                name=name,
                opacity=0.8
            ))
    
    fig.update_layout(
        title="Feasible Region with TVL Colormap",
        xaxis_title="R_l (Lending Rewards)",
        yaxis_title="R_d (DEX Rewards)",
        width=800,
        height=600
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔄 Looping Physics Optimization Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("System Parameters")
    
    # Core parameters
    st.sidebar.subheader("Core Parameters")
    l = st.sidebar.slider("Leverage (l)", 1., 11.0, 3.0, 1., help="Maximum leverage for loopers")
    r = st.sidebar.slider("Base Asset Yield (r)", 0.05, 0.30, 0.05, 0.01, help="Yield of base asset (e.g., mRe7)")
    r_b = st.sidebar.slider("Borrow Rate at 90% (r_b)", 0.0, 0.30, 0.30, 0.01, help="Borrow rate at 90% utilization")
    epsilon = st.sidebar.slider("Lending Protocol Fee (ε)", 0.0, 0.30, 0.20, 0.05, help="Fee taken by lending protocol")
    
    # System constraints
    st.sidebar.subheader("System Constraints")
    U = st.sidebar.slider("Utilization Rate (U)", 0.70, 0.95, 0.90, 0.05, help="Target utilization rate")
    alpha = st.sidebar.slider("DEX Leverage Ratio (α)", 0.3, 0.8, 0.5, 0.1, help="DEX liquidity to borrowing ratio")
    N = st.sidebar.slider("Sponsor Liquidity Budget (N) in M$", 1, 50, 10, 1, help="Total sponsor liquidity")*1000000
    R = st.sidebar.slider("Reward Budget (R) in M$", 1, 12, 8, 1, help="Total annual reward budget")*1000000
    
    # APY targets
    st.sidebar.subheader("APY Targets")
    APY_l = st.sidebar.slider("Lending APY Target", 0.10, 0.3, 0.30, 0.01, help="Target APY for lending")
    APY_d = st.sidebar.slider("DEX APY Target", 0.1, 0.30, 0.20, 0.01, help="Target APY for DEX")
    APY_v = st.sidebar.slider("Vault APY Target", 0.50, 1.50, 0.75, 0.05, help="Target APY for vault")
    
    
    # Set initial R_l and R_d to be solved by optimization
    R_l = R//3  # Initial guess, will be optimized
    R_d = R//3  # Initial guess, will be optimized
    
    # Create parameters dictionary
    params = {
        'l': l, 'r': r, 'r_b': r_b, 'epsilon': epsilon,
        'U': U, 'alpha': alpha, 'N': N, 'R': R,
        'APY_l': APY_l, 'APY_d': APY_d, 'APY_v': APY_v,
        'R_l': R_l, 'R_d': R_d
    }
    
    # Solve 2D simplex optimization first
    optimization_result = solve_2d_simplex_optimization(params, {'N_l': 0, 'N_d': 0})
    
    # Use optimized values if available, otherwise use initial guesses
    if optimization_result['success']:
        params['R_l'] = optimization_result['R_l_opt']
        params['R_d'] = optimization_result['R_d_opt']  
        R_v = optimization_result['R_v_opt']
        params['R_v'] = R_v
        
        # Calculate user TVL with optimized values
        user_tvl = calculate_user_tvl_direct(params)
        sponsor_tvl = calculate_sponsor_tvl_direct(params, user_tvl)
        user_tvl['N_v'] = sponsor_tvl['N_v']
        
        # Check constraints
        constraints = check_liquidity_constraints(params, user_tvl, sponsor_tvl)
    else:
        st.error(f"❌ Optimization failed: {optimization_result['message']}")
        # Set default values for failed optimization
        user_tvl = {'N_l': 0, 'N_d': 0, 'N_v': 0}
        sponsor_tvl = {'N_l_star': 0, 'N_d_star': 0, 'N_v': 0}
        constraints = {
            'n_star_positive': False,
            'liquidity_budget_ok': False,
            'constraint_1_ok': False,
            'constraint_2_ok': False,
            'total_sponsor_liquidity': 0
        }
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Status")
        
        # Display key metrics
        st.metric("Total TVL", f"${user_tvl['N_l'] + user_tvl['N_d'] + user_tvl['N_v']:,.0f}")
        st.metric("Sponsor Liquidity Used", f"${sponsor_tvl['N_l_star'] + sponsor_tvl['N_d_star']:,.0f}")
        st.metric("Remaining Budget", f"${N - (sponsor_tvl['N_l_star'] + sponsor_tvl['N_d_star']):,.0f}")
        st.metric("R_v (Vault Rewards)", f"${R_v:,.0f}")
        
        # Constraint status
        st.subheader("🔍 Constraint Status")
        
        if constraints['n_star_positive']:
            st.success("✅ N* values are positive")
        else:
            st.error("❌ N* values are negative")
        
        if constraints['liquidity_budget_ok']:
            st.success("✅ Liquidity budget constraint satisfied")
        else:
            st.error("❌ Liquidity budget exceeded")
        
        if constraints['constraint_1_ok'] and constraints['constraint_2_ok']:
            st.success("✅ Liquidity constraints satisfied")
        else:
            st.warning("⚠️ Liquidity constraints not satisfied")
    
    with col2:
        st.subheader("📈 User TVL Relationships")
        
        # Display user TVL
        tvl_data = {
            'Protocol': ['Lending', 'DEX', 'Vault'],
            'User TVL': [f"${user_tvl['N_l']:,.0f}", f"${user_tvl['N_d']:,.0f}", f"${user_tvl['N_v']:,.0f}"],
            'Sponsor TVL': [f"${sponsor_tvl['N_l_star']:,.0f}", f"${sponsor_tvl['N_d_star']:,.0f}", "$0"],
            'Rewards': [f"${params['R_l']:,.0f}", f"${params['R_d']:,.0f}", f"${R_v:,.0f}"]
        }
        
        st.dataframe(pd.DataFrame(tvl_data), use_container_width=True)
        
    
    # 2D Simplex Optimization
    st.subheader("🔬 2D Simplex Optimization")
    
    if optimization_result['success']:
        st.success("✅ Optimization successful!")
        
        # Display optimization results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Optimal Solution:**")
            st.write(f"**R_l (optimal)**: ${optimization_result['R_l_opt']:,.0f}")
            st.write(f"**R_d (optimal)**: ${optimization_result['R_d_opt']:,.0f}")
            st.write(f"**R_v (optimal)**: ${optimization_result['R_v_opt']:,.0f}")
            st.write(f"**Optimal TVL**: ${optimization_result['optimal_tvl']:,.0f}")
        
        with col2:
            st.markdown("**Optimal TVL Components:**")
            st.write(f"**N_l (optimal)**: ${optimization_result['N_l_opt']:,.0f}")
            st.write(f"**N_d (optimal)**: ${optimization_result['N_d_opt']:,.0f}")
            st.write(f"**N_v (optimal)**: ${optimization_result['N_v_opt']:,.0f}")
            st.write(f"**N_l* (optimal)**: ${optimization_result['N_l_star_opt']:,.0f}")
            st.write(f"**N_d* (optimal)**: ${optimization_result['N_d_star_opt']:,.0f}")
                
        # Display constraint satisfaction
        if 'constraints_satisfied' in optimization_result:
            opt_constraints = optimization_result['constraints_satisfied']
            st.markdown("**Constraint Satisfaction (4 Inequalities + 1 Equality):**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**R_l ≥ 0**: {'✅' if opt_constraints['R_l >= 0'] else '❌'} ({optimization_result['R_l_opt']:,.0f})")
                st.write(f"**R_d ≥ 0**: {'✅' if opt_constraints['R_d >= 0'] else '❌'} ({optimization_result['R_d_opt']:,.0f})")
                st.write(f"**N_l* ≥ 0**: {'✅' if opt_constraints['N_l* >= 0'] else '❌'} ({optimization_result['N_l_star_opt']:,.0f})")
            
            with col2:
                st.write(f"**N_d* ≥ 0**: {'✅' if opt_constraints['N_d* >= 0'] else '❌'} ({optimization_result['N_d_star_opt']:,.0f})")
                st.write(f"**Budget equality**: {'✅' if opt_constraints['Budget equality'] else '❌'}")
                st.write(f"**LHS**: {opt_constraints['budget_lhs']:,.0f} = **RHS**: {opt_constraints['budget_rhs']:,.0f}")
            
            # Debug information for budget constraint
            st.markdown("**Budget Constraint Debug:**")
            R_l_opt = optimization_result['R_l_opt']
            R_d_opt = optimization_result['R_d_opt']
            R_v_opt = optimization_result['R_v_opt']
            N_l_star_opt = optimization_result['N_l_star_opt']
            N_d_star_opt = optimization_result['N_d_star_opt']
            
            budget_lhs = (R_l_opt + R_d_opt + R_v_opt) - N_l_star_opt * params['r_b'] * (1 - params['epsilon']) - N_d_star_opt * params['r'] / 2
            budget_rhs = params['R']
            budget_error = abs(budget_lhs - budget_rhs)
            
            st.write(f"**Budget LHS**: (R_l + R_d + R_v) - N_l* × r_b × (1-ε) - N_d* × r/2")
            st.write(f"**= ({R_l_opt:,.0f} + {R_d_opt:,.0f} + {R_v_opt:,.0f}) - {N_l_star_opt:,.0f} × {params['r_b']:.3f} × {1-params['epsilon']:.3f} - {N_d_star_opt:,.0f} × {params['r']:.3f}/2**")

            
            if opt_constraints['all_satisfied']:
                st.success("🎉 All constraints satisfied!")
            else:
                st.warning("⚠️ Some constraints not satisfied")
        
    else:
        st.error(f"❌ Optimization failed: {optimization_result['message']}")
    
    # Create and display the 2D simplex visualization
    st.subheader("📈 2D Simplex Visualization")
    
    simplex_fig = create_2d_simplex_visualization(params, user_tvl, sponsor_tvl, optimization_result)
    st.plotly_chart(simplex_fig, use_container_width=True)
    
    # Feasible region visualization
    st.subheader("🗺️ Feasible Region Visualization")
    
    # Variable Formulas and Values
    st.subheader("📝 Variable Formulas and Values")
    
    # Create a comprehensive table of all variables
    variables_data = {
        'Variable': [
            'N_l', 'N_d', 'N_v', 'N_l*', 'N_d*', 'R_v',
            'N_l (optimal)', 'N_d (optimal)', 'N_v (optimal)', 
            'N_l* (optimal)', 'N_d* (optimal)', 'R_v (optimal)'
        ],
        'Formula': [
            'R_l / (APY_l - r_b(1-ε))',
            'R_d / (APY_d - r/2)',
            'U(N_l+N_l*) / (l-1)',
            '(α*N_d + α*N - U*N_l) / (U + α)',
            'N - N_l*',
            'N_v × max(0, APY_v - lr + (l-1)r_b)',
            'R_l_opt / (APY_l - r_b(1-ε))',
            'R_d_opt / (APY_d - r/2)',
            'U(N_l_opt+N_l*_opt) / (l-1)',
            '(α*N_d_opt + α*N - U*N_l_opt) / (U + α)',
            'N - N_l*_opt',
            'N_v_opt × max(0, APY_v - lr + (l-1)r_b)'
        ],
        'Current Value': [
            f"${user_tvl['N_l']:,.0f}",
            f"${user_tvl['N_d']:,.0f}",
            f"${user_tvl['N_v']:,.0f}",
            f"${sponsor_tvl['N_l_star']:,.0f}",
            f"${sponsor_tvl['N_d_star']:,.0f}",
            f"${R_v:,.0f}",
            f"${optimization_result.get('N_l_opt', 0):,.0f}" if optimization_result['success'] else "N/A",
            f"${optimization_result.get('N_d_opt', 0):,.0f}" if optimization_result['success'] else "N/A",
            f"${optimization_result.get('N_v_opt', 0):,.0f}" if optimization_result['success'] else "N/A",
            f"${optimization_result.get('N_l_star_opt', 0):,.0f}" if optimization_result['success'] else "N/A",
            f"${optimization_result.get('N_d_star_opt', 0):,.0f}" if optimization_result['success'] else "N/A",
            f"${optimization_result.get('R_v_opt', 0):,.0f}" if optimization_result['success'] else "N/A"
        ]
    }
    
    st.dataframe(pd.DataFrame(variables_data), use_container_width=True)
    
    
    # 2D Linear Programming Problem
    st.subheader("📐 2D Linear Programming Problem")
    
    st.markdown("""
    **Objective Function:**
    """)
    st.latex(f"\\max_{{R_l, R_d}} \\quad \\text{{TVL}} = N_l + N_d + N_v")
    
    st.markdown("""
    **Equations:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**User TVL relationships:**")
        st.latex(f"N_l = \\frac{{R_l}}{{APY_l - r_b(1-\\epsilon)}}")
        st.latex(f"N_d = \\frac{{R_d}}{{APY_d - \\frac{{r}}{{2}}}}")
        st.latex(f"N_v \\leq \\frac{{R_v}}{{APY_v - lr + (l-1)r_b}}")
        
        st.markdown("**Active constraints:**")
        st.latex(f"N_l^* + N_d^* = N")
        st.latex(f"U(N_l+N_l^*) = \\alpha(N_d+N_d^*)")
        st.latex(f"N_v(l-1) = U(N_l+N_l^*)")
    
    with col2:
        st.markdown("**R_v calculation:**")
        st.latex(f"R_v = N_v \\max(0, APY_v - lr + (l-1)r_b)")
        
        st.markdown("**Subject to constraints:**")
        st.latex(f"R_l \\geq 0, \\quad R_d \\geq 0, \\quad R_v \\geq 0")
        st.latex(f"N_l^* \\geq 0, \\quad N_d^* \\geq 0")
        st.latex(f"N_l^* + N_d^* \\leq N")
        st.latex(f"(R_l + R_d + R_v) - N_l^* r_b(1-\\epsilon) - N_d^* \\frac{{r}}{{2}} = R")
    
    # Parameter validation from LaTeX section
    st.subheader("✅ Parameter Validation")
    
    st.markdown("""
    **Input parameters must satisfy:**
    """)
    st.latex(f"\\frac{{APY_v - lr}}{{l-1}} \\leq r_b \\leq \\frac{{APY_l}}{{1-\\epsilon}}")
    st.latex(f"\\frac{{r}}{{2}} \\leq APY_d")
    
    # Calculate constraint bounds
    lower_rb_bound = (params['APY_v'] - params['l'] * params['r']) / (params['l'] - 1)
    upper_rb_bound = params['APY_l'] / (1 - params['epsilon'])
    min_apy_d = params['r'] / 2
    
    # Check constraints
    r_b_lower_ok = params['r_b'] >= lower_rb_bound
    r_b_upper_ok = params['r_b'] <= upper_rb_bound
    apy_d_ok = params['APY_d'] >= min_apy_d
    
    st.write(f"**r_b constraint**: {'✅' if (r_b_lower_ok and r_b_upper_ok) else '❌'}")
    st.write(f"  **{lower_rb_bound:.1%} ≤ r_b = {params['r_b']:.1%} ≤ {upper_rb_bound:.1%}**: {'✅' if (r_b_lower_ok and r_b_upper_ok) else '❌'}")
    
    st.write(f"**APY_d constraint**: {'✅' if apy_d_ok else '❌'}")
    st.write(f"  **{min_apy_d:.1%} ≤ APY_d = {params['APY_d']:.1%}**: {'✅' if apy_d_ok else '❌'}")
    
    # Check positive user TVL constraints
    user_vault_ok = params['APY_v'] - params['l'] * params['r'] + (params['l'] - 1) * params['r_b'] >= 0
    user_lending_ok = params['APY_l'] - params['r_b'] * (1 - params['epsilon']) >= 0
    user_dex_ok = params['APY_d'] - params['r'] / 2 >= 0
    
    st.markdown("""
    **User TVL positivity constraints:**
    """)
    st.write(f"**User Vault**: {'✅' if user_vault_ok else '❌'} ({params['APY_v'] - params['l'] * params['r'] + (params['l'] - 1) * params['r_b']:.1%})")
    st.write(f"**User Lending**: {'✅' if user_lending_ok else '❌'} ({params['APY_l'] - params['r_b'] * (1 - params['epsilon']):.1%})")
    st.write(f"**User DEX**: {'✅' if user_dex_ok else '❌'} ({params['APY_d'] - params['r'] / 2:.1%})")
    
    # Check if all constraints are satisfied
    all_parameter_constraints_ok = r_b_lower_ok and r_b_upper_ok and apy_d_ok
    all_user_constraints_ok = user_vault_ok and user_lending_ok and user_dex_ok
    all_system_constraints_ok = (
        constraints['n_star_positive'] and
        constraints['liquidity_budget_ok'] and
        constraints['constraint_1_ok'] and
        constraints['constraint_2_ok']
    )
    
    if all_parameter_constraints_ok and all_user_constraints_ok and all_system_constraints_ok:
        st.success("🎉 All parameter and system constraints are satisfied!")
    else:
        st.error("⚠️ Some constraints are not satisfied. Please adjust parameters.")
        
        if not all_parameter_constraints_ok:
            st.error("❌ Parameter validation failed")
        if not all_user_constraints_ok:
            st.error("❌ User TVL positivity constraints failed")
        if not all_system_constraints_ok:
            st.error("❌ System constraints failed")

if __name__ == "__main__":
    main()
