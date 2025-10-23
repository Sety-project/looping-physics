# Looping Physics Dashboard v2

A Streamlit dashboard for optimizing looping strategies in DeFi protocols, implementing the advanced 7D optimization model with scipy.optimize.minimize.

## Features

- **Interactive Parameter Controls**: Adjust key parameters with sliders
- **7D Nonlinear Optimization**: Advanced optimization using SLSQP method
- **Multi-Protocol Support**: Lending, DEX, Vault, and Vault-DEX protocols
- **Constraint Satisfaction**: Real-time validation of all optimization constraints
- **Mathematical Formulation**: Display of the complete 7D optimization model
- **Performance Metrics**: TVL calculations and constraint satisfaction

## Quick Start

### Option 1: Using the launcher script (Recommended)
```bash
python3 run_dashboard.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run streamlit_looping_dashboard.py
```

### Option 3: Using the virtual environment
```bash
# Activate virtual environment
source venv/bin/activate

# Run the dashboard
streamlit run streamlit_looping_dashboard.py
```

## Installation

If you need to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Adjust Parameters**: Use the sidebar sliders to modify:
   - APY targets (lending, DEX, vault)
   - Interest rates and leverage
   - Budget constraints
   - System parameters

2. **View Optimization**: The dashboard will automatically:
   - Solve the 2D linear programming problem
   - Display optimal reward allocations
   - Show constraint satisfaction status
   - Visualize the optimization space

3. **Analyze Results**: Review:
   - Total Value Locked (TVL) calculations
   - User and sponsor TVL breakdowns
   - Constraint satisfaction metrics
   - Mathematical formulation details

## Mathematical Model

The dashboard implements the 7D Nonlinear Optimization Problem:

- **Objective**: Maximize TVL = N_l + N_d + N_v
- **Variables**: R_l, R_d, R_v, R_vd (rewards), N_l*, N_d*, N_vd* (sponsor liquidity)
- **Constraints**: Budget equality, N* sum equality, APY bounds, and positivity constraints
- **Solution**: Real-time optimization using scipy.optimize.minimize with SLSQP method

## Files

- `streamlit_looping_dashboard.py`: Main dashboard application
- `run_dashboard.py`: Simple launcher script
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Requirements

- Python 3.8+
- Streamlit
- NumPy, Pandas, Plotly, Matplotlib, SciPy

## Troubleshooting

If you encounter issues:

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Port conflicts**: The dashboard runs on port 8501 by default
3. **Parameter constraints**: Some parameter combinations may not have valid solutions
4. **Browser issues**: Try refreshing the page or clearing browser cache

## Mathematical Background

This dashboard implements the advanced 7D optimization model for looping strategies in DeFi. The model optimizes reward allocation across multiple protocols (lending, DEX, vault, vault-DEX) and sponsor liquidity allocation to maximize total value locked while satisfying complex nonlinear constraints including APY bounds, budget equality, and utilization constraints.
