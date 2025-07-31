# Monte Carlo Derivative Pricing Tool

## Project Overview

This project implements a comprehensive Monte Carlo derivative pricing tool for equity options using three advanced mathematical models:

- **Black-Scholes Model**: Classical option pricing with constant volatility
- **Heston Model**: Stochastic volatility model with mean-reverting variance
- **Dupire Model**: Local volatility model with volatility surface

## Features

### Core Functionality
- **Real-time Market Data**: Integration with Yahoo Finance for historical price data
- **Monte Carlo Simulation**: High-performance simulation engine with configurable parameters
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, and Rho for all models
- **Model Calibration**: Parameter calibration to market data
- **Sensitivity Analysis**: Comprehensive parameter sensitivity studies

### User Interface
- **Professional GUI**: Modern Tkinter-based interface with multiple tabs
- **Interactive Charts**: Real-time visualization of results and analysis
- **Parameter Forms**: User-friendly forms for model setup and configuration
- **Results Export**: Export to Excel and JSON formats

### Advanced Analytics
- **Model Comparison**: Side-by-side comparison of pricing results
- **Convergence Analysis**: Monte Carlo convergence visualization
- **Volatility Surface**: 3D visualization of local volatility surfaces
- **Path Visualization**: Monte Carlo simulation path plotting

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- numpy >= 1.24.3
- pandas >= 2.0.3
- scipy >= 1.11.1
- matplotlib >= 3.7.1
- seaborn >= 0.12.2
- yfinance >= 0.2.18
- openpyxl >= 3.1.2
- tkinter (included with Python)

### Running the Application
```bash
python main.py
```

## Project Structure

```
monte_carlo_pricing/
├── main.py                     # Application entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── docs/                       # Documentation
│   └── project_report.pdf      # Comprehensive project report
├── src/                        # Source code
│   ├── data/                   # Data management
│   │   └── data_manager.py     # Market data fetching and caching
│   ├── models/                 # Pricing models
│   │   ├── black_scholes.py    # Black-Scholes implementation
│   │   ├── heston.py           # Heston stochastic volatility
│   │   └── dupire.py           # Dupire local volatility
│   ├── pricing/                # Pricing engine
│   │   └── pricing_engine.py   # Main pricing orchestration
│   └── gui/                    # User interface
│       ├── main_gui.py         # Main application window
│       ├── parameter_forms.py  # Parameter input forms
│       ├── results_display.py  # Results visualization
│       └── charts.py           # Chart management
├── data/                       # Data storage
│   └── cache/                  # Cached market data
└── results/                    # Output files
    ├── exports/                # Excel exports
    └── charts/                 # Saved charts
```

## Models Implementation

### Black-Scholes Model
The Black-Scholes model assumes constant volatility and provides analytical solutions for European options.

**Key Features:**
- Analytical pricing formula
- Complete Greeks calculation
- Monte Carlo verification
- Implied volatility calculation

**Parameters:**
- S₀: Current stock price
- K: Strike price
- T: Time to maturity
- r: Risk-free rate
- σ: Volatility

### Heston Model
The Heston model incorporates stochastic volatility with mean reversion, providing more realistic volatility dynamics.

**Key Features:**
- Stochastic volatility process
- Semi-analytical pricing (when possible)
- Monte Carlo simulation with Euler scheme
- Feller condition validation

**Parameters:**
- S₀: Current stock price
- K: Strike price
- T: Time to maturity
- r: Risk-free rate
- v₀: Initial variance
- κ: Mean reversion speed
- θ: Long-term variance
- σᵥ: Volatility of volatility
- ρ: Correlation between price and volatility

### Dupire Model
The Dupire model uses local volatility surfaces constructed from market option prices.

**Key Features:**
- Local volatility surface construction
- Market-calibrated volatility
- Forward PDE solution
- Monte Carlo simulation with time-dependent volatility

**Parameters:**
- S₀: Current stock price
- r: Risk-free rate
- q: Dividend yield
- Local volatility surface σ(S,t)

## Usage Guide

### 1. Loading Market Data
1. Launch the application using `python main.py`
2. Navigate to the **Market Data** tab
3. Enter a stock symbol (e.g., "AAPL", "GOOGL", "MSFT")
4. Select the time period for historical data
5. Click "Load Data" to fetch market information

### 2. Setting Up Models

#### Black-Scholes Setup
1. Go to **Model Parameters** → **Black-Scholes** tab
2. The form will auto-populate with market data
3. Adjust parameters as needed:
   - Strike price relative to current price
   - Time to maturity (in years)
   - Volatility (decimal format, e.g., 0.2 = 20%)
4. Select option type (Call/Put)
5. Click "Setup Model" and then "Price Option"

#### Heston Setup
1. Navigate to **Model Parameters** → **Heston** tab
2. Configure stochastic volatility parameters:
   - Initial variance (v₀): Typically current volatility squared
   - Mean reversion speed (κ): Controls how fast volatility reverts
   - Long-term variance (θ): Long-run average variance
   - Vol of vol (σᵥ): Volatility of the variance process
   - Correlation (ρ): Usually negative for equity options
3. Verify Feller condition: 2κθ ≥ σᵥ²
4. Click "Setup Model" and "Price Option"

#### Dupire Setup
1. Go to **Model Parameters** → **Dupire** tab
2. Set basic parameters (spot, risk-free rate, dividend yield)
3. Click "Construct Surface" to build local volatility surface
4. Once surface is ready, click "Setup Model" and "Price Option"

### 3. Analyzing Results

#### Pricing Results
- Navigate to **Pricing Results** tab
- View side-by-side price comparison
- Analyze confidence intervals and standard errors
- Review pricing statistics and model agreement

#### Greeks Analysis
- Go to **Greeks Analysis** tab
- Click "Calculate All Greeks" for sensitivity measures
- View Delta, Gamma, Theta, Vega, and Rho values
- Compare Greeks across different models

#### Charts and Visualization
- Access **Charts & Analysis** tab
- Select chart type:
  - **Sensitivity Analysis**: Parameter sensitivity curves
  - **Monte Carlo Paths**: Simulation path visualization
  - **Greeks Analysis**: Greeks vs. spot price relationships
  - **Volatility Surface**: 3D volatility surface (Dupire)
  - **Price Convergence**: Monte Carlo convergence analysis
  - **Model Comparison**: Comparative price analysis

### 4. Advanced Features

#### Model Calibration
1. Use **Models** → **Calibrate Models** menu
2. The system will attempt to calibrate Heston and Dupire models
3. Review calibration results and parameter estimates

#### Sensitivity Analysis
1. Access **Analysis** → **Sensitivity Analysis**
2. Select parameter for analysis (S₀, K, T, r, σ)
3. Choose sensitivity range (±5% to ±50%)
4. View results in Charts tab

#### Export Results
- **File** → **Save Results**: Export to JSON format
- **File** → **Export to Excel**: Create comprehensive Excel report
- Charts can be saved individually using chart controls

## Mathematical Background

### Monte Carlo Simulation
The Monte Carlo method simulates random paths of the underlying asset price using stochastic differential equations:

**Black-Scholes SDE:**
```
dS = rS dt + σS dW
```

**Heston SDE System:**
```
dS = rS dt + √v S dW₁
dv = κ(θ - v) dt + σᵥ√v dW₂
dW₁ dW₂ = ρ dt
```

**Dupire Local Volatility:**
```
dS = rS dt + σ(S,t)S dW
```

### Greeks Calculation
Greeks measure price sensitivities:

- **Delta (Δ)**: ∂V/∂S - Price sensitivity to spot
- **Gamma (Γ)**: ∂²V/∂S² - Delta sensitivity to spot  
- **Theta (Θ)**: ∂V/∂t - Time decay
- **Vega (ν)**: ∂V/∂σ - Volatility sensitivity
- **Rho (ρ)**: ∂V/∂r - Interest rate sensitivity

### Model Calibration
Parameter calibration minimizes the difference between model and market prices:

```
min Σᵢ (V_model(Kᵢ, Tᵢ) - V_market(Kᵢ, Tᵢ))²
```

## Performance Considerations

### Computational Efficiency
- **Black-Scholes**: Instantaneous analytical calculation
- **Heston**: Semi-analytical when possible, otherwise Monte Carlo
- **Dupire**: Monte Carlo with optimized local volatility lookup

### Memory Management
- Efficient array operations using NumPy
- Smart caching of market data
- Configurable simulation parameters

### Accuracy vs. Speed Trade-offs
- Default: 100,000 Monte Carlo simulations
- High precision: 1,000,000+ simulations
- Quick estimates: 10,000 simulations

## Error Handling and Validation

### Input Validation
- Parameter bounds checking
- Feller condition verification for Heston
- Market data availability confirmation

### Numerical Stability
- Variance positivity enforcement in Heston
- Gradient checks for calibration
- Convergence monitoring

### Error Recovery
- Graceful degradation when models fail
- Alternative calculation methods
- Comprehensive error reporting

## Testing and Validation

### Model Verification
- Black-Scholes analytical vs. Monte Carlo comparison
- Put-call parity validation
- Greeks finite difference verification

### Benchmark Tests
- Performance profiling with different simulation sizes
- Memory usage monitoring
- Accuracy testing against known solutions

## Known Limitations

1. **Market Data**: Limited to publicly available equity data
2. **Option Types**: Currently supports European options only
3. **Dividends**: Simplified continuous dividend yield model
4. **Calibration**: Requires sufficient market option data for Dupire

## Future Enhancements

### Planned Features
- American option pricing with early exercise
- Exotic option types (Asian, Barrier, etc.)
- Multi-asset correlation models
- Real-time options data integration

### Technical Improvements
- GPU acceleration for Monte Carlo
- Advanced calibration algorithms
- Machine learning volatility forecasting
- Web-based interface option

## Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies
3. Run test suite: `python -m pytest tests/`
4. Follow PEP 8 coding standards

### Code Structure Guidelines
- Modular design with clear separation of concerns
- Comprehensive docstrings and type hints
- Unit tests for all major functions
- Performance benchmarks for critical paths

## Troubleshooting

### Common Issues

**Market Data Loading Fails:**
- Check internet connection
- Verify stock symbol exists
- Try different time periods

**Model Setup Errors:**
- Ensure all parameters are positive where required
- Check Feller condition for Heston model
- Verify time to maturity > 0

**Performance Issues:**
- Reduce number of Monte Carlo simulations
- Close other applications to free memory
- Consider using analytical methods when available

**Calibration Failures:**
- Ensure sufficient market data points
- Check parameter bounds and constraints
- Try different initial parameter guesses

## License and Disclaimer

This software is provided for educational and research purposes. Users should validate results independently before making any financial decisions. The authors assume no responsibility for trading losses or other financial consequences.

## Contact and Support

For technical support, bug reports, or feature requests, please refer to the project documentation or contact the development team.

## Acknowledgments

This project implements well-established mathematical models and numerical methods from quantitative finance literature. Special thanks to the open-source Python community for providing the foundational libraries that make this work possible.