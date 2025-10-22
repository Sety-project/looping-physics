# Looping Physics Dashboard

A Streamlit dashboard for optimizing looping strategies in DeFi protocols, implementing the mathematical model from the LaTeX paper.

## Features

- **Interactive Parameter Controls**: Adjust key parameters with sliders
- **2D Linear Programming Optimization**: Real-time optimization of reward allocation
- **Constraint Visualization**: Visual representation of optimization constraints
- **Mathematical Formulation**: Display of the complete mathematical model
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

The dashboard implements the 2D Linear Programming Problem from the LaTeX paper:

- **Objective**: Maximize TVL = N_l + N_d + N_v
- **Variables**: R_l (lending rewards), R_d (DEX rewards)
- **Constraints**: Budget, liquidity, and positivity constraints
- **Solution**: Real-time optimization using scipy.optimize.linprog

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

This dashboard implements the bootstrapping optimization model for looping strategies in DeFi, as described in the accompanying LaTeX paper. The model optimizes reward allocation across lending protocols, DEX liquidity, and looping vaults to maximize total value locked while satisfying various constraints.
