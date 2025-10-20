# Looping Physics Optimization

A comprehensive 2D linear programming optimization tool for looping strategies in DeFi, featuring an interactive Streamlit web application with real-time visualization.

## 🚀 Features

- **Interactive Web App**: Built with Streamlit for easy parameter adjustment and real-time optimization
- **2D Linear Programming**: Optimizes TVL (Total Value Locked) subject to multiple constraints
- **Constraint Visualization**: Shows all constraint lines (γ, δ, ε) with feasible region and optimal solution
- **TVL Heatmap**: Color-coded visualization of TVL across the feasible region
- **Parameter Validation**: Comprehensive validation according to LaTeX-defined constraints
- **Real-time Results**: Displays N, N*, and R values as defined in the mathematical model

## 📊 Mathematical Model

The optimization problem maximizes TVL subject to constraints:

- **Objective**: Maximize TVL = α_l × R_l + α_d × R_d + α_0
- **Constraints**:
  - γ constraint: γ_l × R_l + γ_d × R_d ≤ N + γ_0
  - δ constraint: δ_l × R_l + δ_d × R_d ≤ δ_0
  - ε constraint: ε_l × R_l + ε_d × R_d ≤ ε_0
  - Non-negativity: R_l ≥ 0, R_d ≥ 0

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sety-project/looping-physics.git
   cd looping-physics
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Run the Core Optimization

```bash
python v1.py
```

## 📁 Project Structure

```
looping-physics/
├── streamlit_app.py          # Main Streamlit application
├── v1.py                     # Core optimization logic
├── requirements.txt          # Python dependencies
├── run_app.sh               # Quick start script
├── latex/                   # LaTeX documentation
│   └── looping physics v1.tex
└── README.md                # This file
```

## 🔧 Parameters

The application accepts the following parameters:

- **N**: Total supply
- **U**: Utilization rate
- **l**: Leverage factor
- **alpha**: Alpha parameter
- **epsilon**: Epsilon parameter
- **R**: Reserve parameter

## 📈 Visualization Features

- **Constraint Lines**: All three constraint lines (γ, δ, ε) are displayed
- **Feasible Region**: Shaded area where all constraints are satisfied
- **Optimal Point**: Marked optimal solution
- **TVL Heatmap**: Color gradient showing TVL distribution
- **Vertices**: All feasible vertices are highlighted

## 🧮 Mathematical Background

The optimization is based on the mathematical model defined in `latex/looping physics v1.tex`, which includes:

- Intermediate calculations for denominators and coefficients
- Beta coefficients (β_l, β_d, β_v)
- N* calculations for optimal liquidity provision
- Comprehensive constraint definitions

## 🐛 Troubleshooting

### Constraint Lines Not Showing
The app includes aggressive debugging to ensure all constraint lines are visible. If lines still don't appear, check the debug output in the app.

### TVL Heatmap Issues
The heatmap automatically handles NaN values and provides debug information about TVL range and feasible points.

## 📝 License

This project is part of the Sety project research into DeFi optimization strategies.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📞 Support

For questions or support, please open an issue in the GitHub repository.
