# Looping Physics Projects

This directory contains two main projects:

## 🔄 Looping Physics Optimization Dashboard
**Location**: `looping-dashboard/`

A comprehensive Streamlit dashboard implementing the mathematical model from `v1.tex` for optimizing looping strategies in DeFi ecosystems.

**Features:**
- Interactive parameter sliders for all system parameters
- Real-time calculation of N, N*, and R values
- Constraint validation and feasible region visualization
- TVL colormap across the feasible region
- Complete implementation of the 2D linear programming problem

**Quick Start:**
```bash
cd looping-dashboard
./run_dashboard.sh
```

## 📈 Curve Pool Price Impact Simulator
**Location**: `curvesim-fork/`

A Streamlit application for simulating price impact and slippage in Curve Finance pools, forked from the original curvesim library.

**Features:**
- Multi-chain support (Mainnet, Arbitrum, Optimism, etc.)
- Real-time price impact calculation
- Interactive visualizations with Plotly
- Pool information display and analysis

**Quick Start:**
```bash
cd curvesim-fork
./run_streamlit.sh
```

## 📁 Project Structure

```
looping physics/
├── looping-dashboard/          # Looping Physics Optimization Dashboard
│   ├── streamlit_looping_dashboard.py
│   ├── test_simple.py
│   ├── requirements.txt
│   ├── run_dashboard.sh
│   └── README.md
├── curvesim-fork/             # Curve Pool Price Impact Simulator
│   ├── streamlit_price_impact_simple.py
│   ├── streamlit_price_impact.py
│   ├── requirements_streamlit.txt
│   ├── run_streamlit.sh
│   └── README.md
└── latex/                     # Mathematical model documentation
    └── looping physics v1.tex
```

## 🚀 Getting Started

### For Looping Physics Optimization:
1. Navigate to `looping-dashboard/`
2. Run `./run_dashboard.sh`
3. Open http://localhost:8504

### For Curve Price Impact Simulation:
1. Navigate to `curvesim-fork/`
2. Run `./run_streamlit.sh`
3. Open http://localhost:8503

## 📚 Documentation

Each project has its own comprehensive README with:
- Detailed installation instructions
- Usage examples
- Mathematical model explanations
- Troubleshooting guides

## 🤝 Contributing

Both projects are open for contributions. Please refer to the individual README files for specific contribution guidelines.

## 📄 License

Both projects maintain the same MIT license as their respective base libraries.