import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import itertools
import math

# Page configuration
st.set_page_config(
    page_title="Looping Physics Optimization",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
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

# Helper functions from v1.py
def safe_div(numer, denom, name):
    EPS = 1e-12
    if abs(denom) <= EPS:
        raise ZeroDivisionError(f"Denominator near-zero for '{name}': denom = {denom}")
    return numer / denom

def intersect(c1, c2):
    EPS = 1e-12
    a1, b1, c1rhs = c1
    a2, b2, c2rhs = c2
    det = a1 * b2 - a2 * b1
    if abs(det) <= EPS:
        return None
    x = (c1rhs * b2 - c2rhs * b1) / det
    y = (a1 * c2rhs - a2 * c1rhs) / det
    return np.array([x, y], dtype=float)

def validate_parameters(p):
    """Validate parameters according to LaTeX constraints"""
    errors = []
    warnings = []
    
    # Constraint 1: User vault APY constraint (equation 208)
    vault_constraint = p["APY_v"] - p["l"] * p["r"] + (p["l"] - 1) * p["r_b"]
    if vault_constraint < 0:
        errors.append(f"User vault constraint violated: APY_v - lr + (l-1)r_b = {vault_constraint:.6f} < 0")
    
    # Constraint 2: User lending APY constraint (equation 209)
    lending_constraint = p["APY_l"] - p["r_b"] * (1 - p["epsilon"])
    if lending_constraint < 0:
        errors.append(f"User lending constraint violated: APY_l - r_b(1-ε) = {lending_constraint:.6f} < 0")
    
    # Constraint 3: User DEX APY constraint (equation 210)
    dex_constraint = p["APY_d"] - p["r"] / 2
    if dex_constraint < 0:
        errors.append(f"User DEX constraint violated: APY_d - r/2 = {dex_constraint:.6f} < 0")
    
    # Constraint 4: All rewards must be positive (line 139)
    if p["R"] <= 0:
        errors.append(f"Reward budget must be positive: R = {p['R']} ≤ 0")
    
    # Constraint 5: All notional values must be positive (line 139)
    if p["N"] <= 0:
        errors.append(f"Liquidity budget must be positive: N = {p['N']} ≤ 0")
    
    # Constraint 6: Utilization must be between 0 and 1
    if not (0 < p["U"] <= 1):
        errors.append(f"Utilization must be in (0,1]: U = {p['U']}")
    
    # Constraint 7: Alpha must be positive
    if p["alpha"] <= 0:
        errors.append(f"Alpha must be positive: α = {p['alpha']} ≤ 0")
    
    # Constraint 8: Leverage must be greater than 1
    if p["l"] <= 1:
        errors.append(f"Leverage must be greater than 1: l = {p['l']} ≤ 1")
    
    # Constraint 9: Epsilon must be between 0 and 1
    if not (0 <= p["epsilon"] < 1):
        errors.append(f"Epsilon must be in [0,1): ε = {p['epsilon']}")
    
    # Additional warnings for edge cases
    if vault_constraint < 0.01:
        warnings.append(f"User vault constraint is very close to zero: {vault_constraint:.6f}")
    
    if lending_constraint < 0.01:
        warnings.append(f"User lending constraint is very close to zero: {lending_constraint:.6f}")
    
    if dex_constraint < 0.01:
        warnings.append(f"User DEX constraint is very close to zero: {dex_constraint:.6f}")
    
    return errors, warnings

def compute_optimization(p):
    """Compute the optimization problem from v1.py"""
    EPS = 1e-12
    
    # Compute intermediate denominators and check them
    den_A_l = p["APY_l"] - p["r_b"] * (1 - p["epsilon"])
    den_A_d = p["APY_d"] - p["r"] / 2
    den_A_v = p["APY_v"] - p["l"] * p["r"] + (p["l"] - 1) * p["r_b"]

    # checks
    for name, val in [("den_A_l", den_A_l), ("den_A_d", den_A_d), ("den_A_v", den_A_v)]:
        if abs(val) <= EPS:
            raise ZeroDivisionError(f"{name} is zero or too small: {val}")

    # compute intermediates
    A_l = safe_div(1.0, den_A_l, "A_l")
    A_d = safe_div(1.0, den_A_d, "A_d")
    A_v = safe_div(1.0, den_A_v, "A_v")

    B_l = 1.0 + (p["r_b"] * (1 - p["epsilon"])) / den_A_l
    B_d = 1.0 + (p["r"] / 2) / den_A_d

    C = (p["l"] - 1.0) * (1.0 / p["U"] + 1.0 / p["alpha"])

    # D as given
    D = 1.0 - ((p["l"] - 1.0) * p["r_b"] * (1 - p["epsilon"])) / (den_A_v * p["U"]) \
          - ((p["l"] - 1.0) * p["r"]) / (2.0 * p["alpha"] * den_A_v)

    if abs(D) <= EPS:
        raise ZeroDivisionError(f"D is zero or too small: D = {D}")

    # Coefficients
    alpha_l = A_l + A_v * B_l / D
    alpha_d = A_d + A_v * B_d / D
    alpha_0 = A_v * p["R"] / D

    gamma_l = -A_l + A_v * C * B_l / D
    gamma_d = -A_d + A_v * C * B_d / D
    gamma_0 = A_v * C * p["R"] / D

    delta_l = (p["U"] * den_A_v) / ((p["l"] - 1.0) * den_A_l) - B_l / D
    delta_d = -B_d / D
    delta_0 = p["R"] / D

    epsilon_l = -B_l / D
    epsilon_d = (p["alpha"] * den_A_v) / ((p["l"] - 1.0) * den_A_d) - B_d / D
    epsilon_0 = p["R"] / D

    # Find feasible vertices
    constraints = [
        (gamma_l, gamma_d, p["N"] + gamma_0),
        (delta_l, delta_d, delta_0),
        (epsilon_l, epsilon_d, epsilon_0)
    ]

    # x>=0, y>=0 lines
    lines = constraints + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    vertices = []
    for c1, c2 in itertools.combinations(lines, 2):
        pt = intersect(c1, c2)
        if pt is None:
            continue
        if np.any(pt < -1e-9):
            continue
        R_l, R_d = pt
        if all(a * R_l + b * R_d <= c + 1e-8 for a, b, c in constraints):
            vertices.append(pt)

    result = {
        'den_A_l': den_A_l,
        'den_A_d': den_A_d,
        'den_A_v': den_A_v,
        'D': D,
        'A_l': A_l,
        'A_d': A_d,
        'A_v': A_v,
        'B_l': B_l,
        'B_d': B_d,
        'C': C,
        'alpha_l': alpha_l,
        'alpha_d': alpha_d,
        'alpha_0': alpha_0,
        'gamma_l': gamma_l,
        'gamma_d': gamma_d,
        'gamma_0': gamma_0,
        'delta_l': delta_l,
        'delta_d': delta_d,
        'delta_0': delta_0,
        'epsilon_l': epsilon_l,
        'epsilon_d': epsilon_d,
        'epsilon_0': epsilon_0,
        'vertices': vertices,
        'constraints': constraints
    }

    if not vertices:
        # No feasible vertices found, but still return empty arrays for visualization
        result['all_vertices'] = np.array([]).reshape(0, 2)
        result['all_TVLs'] = []
        result['no_feasible_solution'] = True
    else:
        vertices = np.unique(np.round(np.array(vertices), 10), axis=0)
        def TVL(x, y): 
            return alpha_l * x + alpha_d * y + alpha_0
        vals = [TVL(x, y) for x, y in vertices]
        idx = int(np.argmax(vals))
        result['optimal_vertex'] = vertices[idx]
        result['optimal_TVL'] = vals[idx]
        result['all_vertices'] = vertices
        result['all_TVLs'] = vals
        result['no_feasible_solution'] = False

    return result

def compute_N_values(p, R_l, R_d, R_v):
    """Compute N, N*, and R values as defined in LaTeX"""
    # Beta coefficients from LaTeX equation (118-120)
    beta_l = 1.0 / (p["APY_l"] - p["r_b"] * (1 - p["epsilon"]))
    beta_d = 1.0 / (p["APY_d"] - p["r"] / 2)
    beta_v = 1.0 / (p["APY_v"] - p["l"] * p["r"] + (p["l"] - 1) * p["r_b"])
    
    # User TVL (N values) from equation (118-120)
    N_l = beta_l * R_l
    N_d = beta_d * R_d
    N_v = beta_v * R_v
    
    # Sponsor TVL (N* values) from equation (92-93)
    N_l_star = (beta_v * R_v * (p["l"] - 1)) / p["U"] - beta_l * R_l
    N_d_star = (beta_v * R_v * (p["l"] - 1)) / p["alpha"] - beta_d * R_d
    
    # Total TVL
    total_TVL = N_l + N_d + N_v + N_l_star + N_d_star
    
    return {
        'N_l': N_l,
        'N_d': N_d, 
        'N_v': N_v,
        'N_l_star': N_l_star,
        'N_d_star': N_d_star,
        'total_TVL': total_TVL,
        'beta_l': beta_l,
        'beta_d': beta_d,
        'beta_v': beta_v
    }

# Main app
def main():
    st.markdown('<h1 class="main-header">🔄 Looping Physics Optimization</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This app optimizes the bootstrapping of a looping ecosystem by solving a 2D linear programming problem.
    The optimization maximizes Total Value Locked (TVL) while respecting budget and liquidity constraints.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("📊 System Parameters")
    
    # Default parameters from v1.py
    default_params = {
        'epsilon': 0.2,
        'r': 0.2,
        'l': 10.0,
        'APY_l': 0.15,
        'APY_d': 0.25,
        'APY_v': 0.75,
        'r_b': 0.12,
        'R': 8_000_000.0,
        'r_star': 0.0,
        'N': 20_000_000.0,
        'U': 0.9,
        'alpha': 0.5
    }
    
    # Parameter inputs
    p = {}
    p['epsilon'] = st.sidebar.slider("ε (Protocol fee)", 0.0, 0.5, default_params['epsilon'], 0.01)
    p['r'] = st.sidebar.slider("r (Base asset yield)", 0.0, 1.0, default_params['r'], 0.01)
    p['l'] = st.sidebar.slider("l (Leverage)", 1.0, 20.0, default_params['l'], 0.1)
    p['APY_l'] = st.sidebar.slider("APY_l (Lending APY)", 0.0, 0.5, default_params['APY_l'], 0.01)
    p['APY_d'] = st.sidebar.slider("APY_d (DEX APY)", 0.0, 0.5, default_params['APY_d'], 0.01)
    p['APY_v'] = st.sidebar.slider("APY_v (Vault APY)", 0.0, 2.0, default_params['APY_v'], 0.01)
    p['r_b'] = st.sidebar.slider("r_b (Borrow rate)", 0.0, 0.5, default_params['r_b'], 0.01)
    p['R'] = st.sidebar.number_input("R (Reward budget)", 0, 100_000_000, int(default_params['R']), 100_000)
    p['N'] = st.sidebar.number_input("N (Liquidity budget)", 0, 100_000_000, int(default_params['N']), 100_000)
    p['U'] = st.sidebar.slider("U (Utilization)", 0.0, 1.0, default_params['U'], 0.01)
    p['alpha'] = st.sidebar.slider("α (DEX leverage factor)", 0.0, 1.0, default_params['alpha'], 0.01)
    p['r_star'] = default_params['r_star']  # Not used in optimization
    
    # Validate parameters first
    errors, warnings = validate_parameters(p)
    
    # Display validation results
    if errors:
        st.error("❌ Parameter Validation Failed")
        for error in errors:
            st.error(f"• {error}")
        st.stop()
    
    if warnings:
        st.warning("⚠️ Parameter Warnings")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    # Display constraint validation values
    st.subheader("🔍 Parameter Constraint Validation")
    
    col_val1, col_val2, col_val3 = st.columns(3)
    
    with col_val1:
        st.markdown("**User TVL Constraints**")
        st.markdown("""
        <div class="metric-container">
        """, unsafe_allow_html=True)
        
        vault_constraint = p["APY_v"] - p["l"] * p["r"] + (p["l"] - 1) * p["r_b"]
        lending_constraint = p["APY_l"] - p["r_b"] * (1 - p["epsilon"])
        dex_constraint = p["APY_d"] - p["r"] / 2
        
        st.metric("Vault: APY_v - lr + (l-1)r_b", f"{vault_constraint:.6f}", 
                 delta="✓ Valid" if vault_constraint >= 0 else "✗ Invalid")
        st.metric("Lending: APY_l - r_b(1-ε)", f"{lending_constraint:.6f}",
                 delta="✓ Valid" if lending_constraint >= 0 else "✗ Invalid")
        st.metric("DEX: APY_d - r/2", f"{dex_constraint:.6f}",
                 delta="✓ Valid" if dex_constraint >= 0 else "✗ Invalid")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_val2:
        st.markdown("**Budget Constraints**")
        st.markdown("""
        <div class="metric-container">
        """, unsafe_allow_html=True)
        
        st.metric("Reward Budget (R)", f"${p['R']:,.0f}",
                 delta="✓ Positive" if p['R'] > 0 else "✗ Invalid")
        st.metric("Liquidity Budget (N)", f"${p['N']:,.0f}",
                 delta="✓ Positive" if p['N'] > 0 else "✗ Invalid")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_val3:
        st.markdown("**System Parameters**")
        st.markdown("""
        <div class="metric-container">
        """, unsafe_allow_html=True)
        
        st.metric("Utilization (U)", f"{p['U']:.3f}",
                 delta="✓ Valid" if 0 < p['U'] <= 1 else "✗ Invalid")
        st.metric("Leverage (l)", f"{p['l']:.1f}",
                 delta="✓ Valid" if p['l'] > 1 else "✗ Invalid")
        st.metric("Alpha (α)", f"{p['alpha']:.3f}",
                 delta="✓ Positive" if p['alpha'] > 0 else "✗ Invalid")
        st.metric("Epsilon (ε)", f"{p['epsilon']:.3f}",
                 delta="✓ Valid" if 0 <= p['epsilon'] < 1 else "✗ Invalid")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Run optimization
    try:
        result = compute_optimization(p)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if result.get('no_feasible_solution', False):
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ No Feasible Solution:</strong> The constraint region is empty. Try adjusting parameters.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <strong>✅ Optimization Successful!</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎯 Optimal Solution")
                st.markdown("""
                <div class="metric-container">
                """, unsafe_allow_html=True)
                
                R_l_opt, R_d_opt = result['optimal_vertex']
                R_v_opt = p['R'] - R_l_opt - R_d_opt  # From budget constraint
                
                st.metric("R_l (Lending rewards)", f"${R_l_opt:,.0f}")
                st.metric("R_d (DEX rewards)", f"${R_d_opt:,.0f}")
                st.metric("R_v (Vault rewards)", f"${R_v_opt:,.0f}")
                st.metric("Total TVL", f"${result['optimal_TVL']:,.0f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Display N, N*, and R values as defined in LaTeX
            if 'optimal_vertex' in result:
                R_l_opt, R_d_opt = result['optimal_vertex']
                R_v_opt = p['R'] - R_l_opt - R_d_opt
                
                N_values = compute_N_values(p, R_l_opt, R_d_opt, R_v_opt)
                
                st.subheader("📊 N, N*, and R Values")
                
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    st.markdown("**User TVL (N values)**")
                    st.markdown("""
                    <div class="metric-container">
                    """, unsafe_allow_html=True)
                    st.metric("N_l (Lending)", f"${N_values['N_l']:,.0f}")
                    st.metric("N_d (DEX)", f"${N_values['N_d']:,.0f}")
                    st.metric("N_v (Vault)", f"${N_values['N_v']:,.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2b:
                    st.markdown("**Sponsor TVL (N* values)**")
                    st.markdown("""
                    <div class="metric-container">
                    """, unsafe_allow_html=True)
                    st.metric("N_l* (Lending)", f"${N_values['N_l_star']:,.0f}")
                    st.metric("N_d* (DEX)", f"${N_values['N_d_star']:,.0f}")
                    st.metric("N_v* (Vault)", "0 (users only)")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2c:
                    st.markdown("**Reward Budget (R values)**")
                    st.markdown("""
                    <div class="metric-container">
                    """, unsafe_allow_html=True)
                    st.metric("R_l (Lending)", f"${R_l_opt:,.0f}")
                    st.metric("R_d (DEX)", f"${R_d_opt:,.0f}")
                    st.metric("R_v (Vault)", f"${R_v_opt:,.0f}")
                    st.metric("Total R", f"${p['R']:,.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Beta coefficients
                st.subheader("🔢 Beta Coefficients")
                col_beta1, col_beta2, col_beta3 = st.columns(3)
                with col_beta1:
                    st.metric("β_l", f"{N_values['beta_l']:.6f}")
                with col_beta2:
                    st.metric("β_d", f"{N_values['beta_d']:.6f}")
                with col_beta3:
                    st.metric("β_v", f"{N_values['beta_v']:.6f}")
                
                # Explanation about negative N* values
                if N_values['N_l_star'] < 0 or N_values['N_d_star'] < 0:
                    st.info("""
                    **ℹ️ About Negative N* Values:**
                    
                    Negative sponsor TVL (N*) values mean the sponsor should **withdraw** liquidity rather than provide it. 
                    This occurs when reward payments (R_l, R_d) are high enough that the sponsor's required liquidity 
                    contribution becomes negative according to the LaTeX equations (92-93):
                    
                    - N_l* = (β_v R_v(l-1))/U - β_l R_l
                    - N_d* = (β_v R_v(l-1))/α - β_d R_d
                    
                    This is mathematically valid and indicates the sponsor can extract value from the system.
                    """)
        
        
        # Visualization - always show feasible region
        if 'all_vertices' in result:
            st.subheader("📈 Constraint Visualization")
            
            # Create enhanced feasible region plot with constraints
            fig = go.Figure()
            
            # Get vertices and TVLs
            vertices = result['all_vertices']
            TVLs = result['all_TVLs']
            constraints = result['constraints']
            
            # Find the range for plotting - ensure all constraints are visible
            if len(vertices) > 0:
                max_x = max(vertices[:, 0]) * 1.2
                max_y = max(vertices[:, 1]) * 1.2
                min_x = max(0, min(vertices[:, 0]) * 0.8)
                min_y = max(0, min(vertices[:, 1]) * 0.8)
            else:
                # No vertices found, use a reasonable default range
                max_x = p['R'] * 0.5  # Half of reward budget
                max_y = p['R'] * 0.5
                min_x = 0
                min_y = 0
            
            # Extend range to ensure all constraint lines are visible
            # Calculate constraint intersection points to determine proper range
            constraint_intersections = []
            for i, (a1, b1, c1) in enumerate(constraints):
                for j, (a2, b2, c2) in enumerate(constraints[i+1:], i+1):
                    det = a1 * b2 - a2 * b1
                    if abs(det) > 1e-10:
                        x_int = (c1 * b2 - c2 * b1) / det
                        y_int = (a1 * c2 - a2 * c1) / det
                        if x_int >= 0 and y_int >= 0:
                            constraint_intersections.append([x_int, y_int])
            
            # Also check intersections with axes
            for a, b, c in constraints:
                if abs(a) > 1e-10:  # Intersection with y=0 axis
                    x_axis_int = c / a
                    if x_axis_int >= 0:
                        constraint_intersections.append([x_axis_int, 0])
                if abs(b) > 1e-10:  # Intersection with x=0 axis
                    y_axis_int = c / b
                    if y_axis_int >= 0:
                        constraint_intersections.append([0, y_axis_int])
            
            if constraint_intersections:
                constraint_intersections = np.array(constraint_intersections)
                max_x = max(max_x, np.max(constraint_intersections[:, 0]) * 1.2)
                max_y = max(max_y, np.max(constraint_intersections[:, 1]) * 1.2)
                min_x = min(min_x, np.min(constraint_intersections[:, 0]) * 0.8)
                min_y = min(min_y, np.min(constraint_intersections[:, 1]) * 0.8)
            
            # Create grid for constraint lines and TVL heatmap
            x_range = np.linspace(min_x, max_x, 100)
            y_range = np.linspace(min_y, max_y, 100)
            
            # Create TVL heatmap data
            X, Y = np.meshgrid(x_range, y_range)
            TVL_grid = np.zeros_like(X)
            
            # Calculate TVL for each point in the grid
            alpha_l = result['alpha_l']
            alpha_d = result['alpha_d'] 
            alpha_0 = result['alpha_0']
            
            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    x_val = x_range[i]
                    y_val = y_range[j]
                    # Check if point satisfies all constraints
                    satisfies_constraints = True
                    for a, b, c in constraints:
                        if a * x_val + b * y_val > c + 1e-8:
                            satisfies_constraints = False
                            break
                    if satisfies_constraints and x_val >= 0 and y_val >= 0:
                        TVL_grid[j, i] = alpha_l * x_val + alpha_d * y_val + alpha_0
                    else:
                        TVL_grid[j, i] = np.nan  # Not feasible
            
            # Add TVL heatmap as background - FIX RANGE ISSUE
            feasible_points = np.sum(~np.isnan(TVL_grid))
            st.write(f"**TVL Heatmap Debug:** {feasible_points} feasible points out of {TVL_grid.size} total points")
            
            if feasible_points > 0:
                min_tvl = np.nanmin(TVL_grid)
                max_tvl = np.nanmax(TVL_grid)
                st.write(f"**TVL Range:** ${min_tvl:,.0f} to ${max_tvl:,.0f}")
                
                # Only show heatmap if we have variation in TVL values
                if max_tvl - min_tvl > 1000:  # Only show if there's meaningful variation
                    fig.add_trace(go.Heatmap(
                        x=x_range,
                        y=y_range,
                        z=TVL_grid,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Total TVL", x=1.02),
                        opacity=0.8,
                        name="TVL Heatmap",
                        hovertemplate="R_l: %{x:,.0f}<br>R_d: %{y:,.0f}<br>TVL: $%{z:,.0f}<extra></extra>",
                        visible=True
                    ))
                    st.write("✓ TVL Heatmap added with color variation")
                else:
                    st.write("⚠️ TVL values too constant, skipping heatmap")
            else:
                st.write("⚠️ No feasible points for TVL heatmap")
            
            # Plot constraint lines - SIMPLIFIED APPROACH
            constraint_names = ['γ constraint', 'δ constraint', 'ε constraint']
            constraint_colors = ['blue', 'green', 'orange']
            
            # Debug: Print constraint information
            st.write("**Constraint Debug Info:**")
            st.write(f"Number of constraints: {len(constraints)}")
            
            # Print detailed constraint info
            for i, (a, b, c) in enumerate(constraints):
                st.write(f"Constraint {i}: a={a:.6f}, b={b:.6f}, c={c:.0f}")
                if abs(b) > 1e-15:
                    slope = -a/b
                    y_intercept = c/b
                    st.write(f"  Slope: {slope:.6f}, Y-intercept: {y_intercept:.0f}")
                else:
                    x_intercept = c/a
                    st.write(f"  Vertical line at x={x_intercept:.0f}")
            
            # Create a fixed wide range for all lines
            x_plot = np.linspace(0, p['R'], 500)
            y_plot = np.linspace(0, p['R'], 500)
            
            # Plot each constraint line individually - FORCE ALL TO SHOW
            for i, (a, b, c) in enumerate(constraints):
                st.write(f"{constraint_names[i]}: {a:.6f}*R_l + {b:.6f}*R_d ≤ {c:.0f}")
                
                # Create a much wider range to catch all lines
                x_wide = np.linspace(0, p['R'] * 2, 1000)
                
                if abs(b) > 1e-15:  # Sloped line - more lenient threshold
                    # Calculate y values for the constraint line
                    y_values = (c - a * x_wide) / b
                    
                    # Find valid range where y >= 0 and within reasonable bounds
                    valid_indices = (y_values >= 0) & (y_values <= p['R'] * 2)
                    
                    if np.any(valid_indices):
                        fig.add_trace(go.Scatter(
                            x=x_wide[valid_indices],
                            y=y_values[valid_indices],
                            mode='lines',
                            line=dict(color=constraint_colors[i], width=8),
                            name=constraint_names[i],
                            hovertemplate=f"{constraint_names[i]}: {a:.4f}*R_l + {b:.4f}*R_d ≤ {c:.0f}<extra></extra>"
                        ))
                        st.write(f"  ✓ Added {constraint_names[i]} (sloped, {np.sum(valid_indices)} points)")
                    else:
                        # Try with even wider range
                        x_very_wide = np.linspace(-p['R'], p['R'] * 3, 1000)
                        y_very_wide = (c - a * x_very_wide) / b
                        valid_very_wide = (y_very_wide >= 0) & (y_very_wide <= p['R'] * 3)
                        if np.any(valid_very_wide):
                            fig.add_trace(go.Scatter(
                                x=x_very_wide[valid_very_wide],
                                y=y_very_wide[valid_very_wide],
                                mode='lines',
                                line=dict(color=constraint_colors[i], width=8),
                                name=constraint_names[i],
                                hovertemplate=f"{constraint_names[i]}: {a:.4f}*R_l + {b:.4f}*R_d ≤ {c:.0f}<extra></extra>"
                            ))
                            st.write(f"  ✓ Added {constraint_names[i]} (sloped, very wide range, {np.sum(valid_very_wide)} points)")
                        else:
                            st.write(f"  ✗ Skipped {constraint_names[i]} (no valid points even with wide range)")
                else:  # Vertical line
                    x_val = c / a if abs(a) > 1e-15 else 0
                    fig.add_trace(go.Scatter(
                        x=[x_val, x_val],
                        y=[0, p['R'] * 2],
                        mode='lines',
                        line=dict(color=constraint_colors[i], width=8),
                        name=constraint_names[i],
                        hovertemplate=f"{constraint_names[i]}: R_l ≤ {x_val:.0f}<extra></extra>"
                    ))
                    st.write(f"  ✓ Added {constraint_names[i]} (vertical at x={x_val:.0f})")
            
            # FORCE ALL 3 CONSTRAINTS TO SHOW - SIMPLE APPROACH
            st.write("**FORCING ALL 3 CONSTRAINTS TO SHOW:**")
            
            # Create a very wide range
            x_force = np.linspace(0, p['R'] * 3, 200)
            
            for i, (a, b, c) in enumerate(constraints):
                st.write(f"FORCING {constraint_names[i]}: a={a:.6f}, b={b:.6f}, c={c:.0f}")
                
                # Calculate y values
                if abs(b) > 1e-30:  # Extremely lenient
                    y_force = (c - a * x_force) / b
                    
                    # Plot the entire line, including negative values
                    fig.add_trace(go.Scatter(
                        x=x_force,
                        y=y_force,
                        mode='lines',
                        line=dict(color=constraint_colors[i], width=6, dash='solid'),
                        name=f"{constraint_names[i]} (FORCED)",
                        hovertemplate=f"{constraint_names[i]}: {a:.4f}*R_l + {b:.4f}*R_d ≤ {c:.0f}<extra></extra>",
                        showlegend=True
                    ))
                    st.write(f"  ✓ FORCED {constraint_names[i]} - plotted {len(x_force)} points")
                    
                    # Also plot just the positive part
                    positive_mask = y_force >= 0
                    if np.any(positive_mask):
                        fig.add_trace(go.Scatter(
                            x=x_force[positive_mask],
                            y=y_force[positive_mask],
                            mode='lines',
                            line=dict(color=constraint_colors[i], width=8, dash='dash'),
                            name=f"{constraint_names[i]} (positive)",
                            hovertemplate=f"{constraint_names[i]} (positive): {a:.4f}*R_l + {b:.4f}*R_d ≤ {c:.0f}<extra></extra>",
                            showlegend=True
                        ))
                        st.write(f"  ✓ FORCED {constraint_names[i]} (positive part) - {np.sum(positive_mask)} points")
                else:
                    # Vertical line
                    x_val = c / a if abs(a) > 1e-30 else 0
                    fig.add_trace(go.Scatter(
                        x=[x_val, x_val],
                        y=[0, p['R'] * 3],
                        mode='lines',
                        line=dict(color=constraint_colors[i], width=8, dash='solid'),
                        name=f"{constraint_names[i]} (FORCED)",
                        hovertemplate=f"{constraint_names[i]}: R_l ≤ {x_val:.0f}<extra></extra>",
                        showlegend=True
                    ))
                    st.write(f"  ✓ FORCED {constraint_names[i]} (vertical at x={x_val:.0f})")
            
            # Add R_l >= 0 and R_d >= 0 axes
            fig.add_trace(go.Scatter(
                x=[0, max_x],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=3),
                name='R_d ≥ 0',
                hovertemplate="R_d ≥ 0<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[0, max_y],
                mode='lines',
                line=dict(color='red', width=3),
                name='R_l ≥ 0',
                hovertemplate="R_l ≥ 0<extra></extra>"
            ))
            
            # Plot feasible vertices (only if they exist)
            if len(vertices) > 0:
                fig.add_trace(go.Scatter(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='white',
                        line=dict(width=4, color='black'),
                        symbol='circle'
                    ),
                    text=[f"TVL: ${tv:,.0f}<br>R_l: {v[0]:,.0f}<br>R_d: {v[1]:,.0f}" for v, tv in zip(vertices, TVLs)],
                    hovertemplate="%{text}<extra></extra>",
                    name="Feasible Vertices"
                ))
            else:
                # Add a text annotation when no feasible vertices exist
                fig.add_annotation(
                    x=max_x/2,
                    y=max_y/2,
                    text="No Feasible Region<br>Constraints do not intersect<br>in the positive quadrant",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(size=16, color="red"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=2
                )
            
            # Highlight optimal point
            if 'optimal_vertex' in result:
                opt_vertex = result['optimal_vertex']
                fig.add_trace(go.Scatter(
                    x=[opt_vertex[0]],
                    y=[opt_vertex[1]],
                    mode='markers',
                    marker=dict(
                        size=25,
                        color='red',
                        symbol='star',
                        line=dict(width=4, color='white')
                    ),
                    name="Optimal Point",
                    hovertemplate=f"OPTIMAL<br>R_l: {opt_vertex[0]:,.0f}<br>R_d: {opt_vertex[1]:,.0f}<br>TVL: ${result['optimal_TVL']:,.0f}<extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                title="2D Linear Programming: Constraints, Feasible Region, and Optimal Solution",
                xaxis_title="R_l (Lending Rewards)",
                yaxis_title="R_d (DEX Rewards)",
                showlegend=True,
                width=900,
                height=700,
                xaxis=dict(range=[min_x, max_x]),
                yaxis=dict(range=[min_y, max_y]),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add constraint information
            st.subheader("📋 Constraint Details")
            constraint_info = []
            for i, (a, b, c) in enumerate(constraints):
                constraint_info.append({
                    'Constraint': constraint_names[i],
                    'Equation': f"{a:.4f}·R_l + {b:.4f}·R_d ≤ {c:,.0f}",
                    'Type': 'Linear inequality'
                })
            
            constraint_info.append({
                'Constraint': 'R_l ≥ 0',
                'Equation': 'R_l ≥ 0',
                'Type': 'Non-negativity'
            })
            constraint_info.append({
                'Constraint': 'R_d ≥ 0', 
                'Equation': 'R_d ≥ 0',
                'Type': 'Non-negativity'
            })
            
            df_constraints = pd.DataFrame(constraint_info)
            st.dataframe(df_constraints, use_container_width=True)
        
        # Summary table
        if 'all_vertices' in result and len(result['all_vertices']) > 0:
            st.subheader("📋 All Feasible Solutions")
            
            df_data = []
            for i, (vertex, tvl) in enumerate(zip(result['all_vertices'], result['all_TVLs'])):
                R_l, R_d = vertex
                R_v = p['R'] - R_l - R_d
                df_data.append({
                    'Solution': f"#{i+1}",
                    'R_l': f"${R_l:,.0f}",
                    'R_d': f"${R_d:,.0f}",
                    'R_v': f"${R_v:,.0f}",
                    'TVL': f"${tvl:,.0f}",
                    'Optimal': "⭐" if i == np.argmax(TVLs) else ""
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in optimization: {str(e)}")
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Parameter Error:</strong> Please check that all parameters satisfy the constraints:
        <ul>
        <li>APY_v - lr + (l-1)r_b ≥ 0</li>
        <li>APY_l - r_b(1-ε) ≥ 0</li>
        <li>APY_d - r/2 ≥ 0</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
