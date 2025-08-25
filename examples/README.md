# PSplines Examples

This directory contains comprehensive examples demonstrating the capabilities of the PSplines library.

## Running the Examples

Each example is a standalone Python script that can be run independently:

```bash
cd examples
python 01_basic_usage.py
python 02_parameter_selection.py
python 03_uncertainty_methods.py
python 04_real_world_application.py
```

Or using uv from the project root:

```bash
uv run python examples/01_basic_usage.py
```

## Example Descriptions

### 1. Basic Usage (`01_basic_usage.py`)
**Demonstrates:** Core PSpline functionality
- Creating and fitting P-splines
- Making predictions with uncertainty
- Computing derivatives
- Visualizing fits and residuals

**Topics Covered:**
- Data generation and fitting
- Prediction with analytical standard errors
- First and second derivative computation
- Model diagnostics and visualization

### 2. Parameter Selection (`02_parameter_selection.py`)
**Demonstrates:** Automatic smoothing parameter selection
- Cross-validation optimization
- AIC-based selection
- L-curve and V-curve methods
- Bias-variance tradeoff analysis

**Topics Covered:**
- Multiple parameter selection methods
- Comparing different smoothing levels
- Method performance comparison
- Visual analysis of optimal parameters

### 3. Uncertainty Methods (`03_uncertainty_methods.py`)
**Demonstrates:** Different uncertainty quantification approaches
- Analytical standard errors (delta method)
- Parametric bootstrap confidence intervals
- Bayesian credible intervals
- Coverage analysis

**Topics Covered:**
- Multiple uncertainty quantification methods
- Method comparison and validation
- Coverage analysis
- Computational considerations

### 4. Real-World Application (`04_real_world_application.py`)
**Demonstrates:** Complete time series analysis workflow
- Multi-level trend decomposition
- Anomaly detection using residuals
- Rate of change analysis via derivatives
- Forecasting with uncertainty

**Topics Covered:**
- Time series smoothing at different scales
- Trend extraction and seasonal decomposition
- Derivative-based change point analysis
- Residual-based anomaly detection
- Forecasting and extrapolation

## Example Data

All examples use synthetically generated data to ensure reproducibility and to highlight specific features:

- **Example 1**: Noisy sine wave with Gaussian noise
- **Example 2**: Damped oscillation with moderate noise  
- **Example 3**: Sine wave with heteroscedastic noise
- **Example 4**: Complex time series with trend, seasonality, and anomalies

## Output Files

Each example generates visualization plots saved as PNG files:
- `basic_usage_output.png`
- `parameter_selection_output.png` 
- `uncertainty_methods_output.png`
- `timeseries_analysis_output.png`

## Requirements

All examples require the same dependencies as the main PSplines package. Example 4 additionally uses:
- `matplotlib.dates` for time series plotting
- `datetime` for date handling

## Notes

- Examples are designed to be educational and may include simplified approaches for clarity
- Computational times may vary; Bayesian examples (Example 3) can take 30-60 seconds
- All examples include comprehensive comments explaining the methodology
- Error handling demonstrates input validation features

## Customization

Each example includes parameters at the top that can be modified:
- Data generation seeds for reproducibility
- Sample sizes and noise levels  
- Model parameters (nseg, degree, lambda)
- Visualization options

Feel free to experiment with these parameters to explore different scenarios!