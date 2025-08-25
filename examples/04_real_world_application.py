#!/usr/bin/env python3
"""
Example 4: Real-World Application - Time Series Analysis
========================================================

This example demonstrates applying P-splines to a realistic time series analysis
scenario, including:
- Trend estimation and seasonal decomposition
- Derivative analysis for rate of change
- Forecasting with uncertainty
- Anomaly detection using residuals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from psplines import PSpline
from psplines.optimize import cross_validation

def generate_realistic_timeseries(n_points=365, seed=42):
    """Generate realistic time series data with trend, seasonality, and noise."""
    np.random.seed(seed)
    
    # Create time axis (daily data for one year)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    t = np.arange(n_points)  # Time as numeric values
    
    # Components
    # 1. Long-term trend (declining)
    trend = 100 - 15 * t / n_points + 0.02 * (t / n_points)**2 * 50
    
    # 2. Seasonal components
    yearly_season = 10 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle
    weekly_season = 3 * np.sin(2 * np.pi * t / 7)        # Weekly cycle
    
    # 3. Random walk component (smooth random variations)
    rw = np.cumsum(0.5 * np.random.randn(n_points))
    rw = rw - np.linspace(rw[0], rw[-1], n_points)  # Detrend
    
    # 4. Noise
    noise = 2 * np.random.randn(n_points)
    
    # 5. A few anomalies
    anomaly_indices = np.random.choice(n_points, size=5, replace=False)
    anomalies = np.zeros(n_points)
    anomalies[anomaly_indices] = 15 * np.random.randn(5)
    
    # Combine components
    y = trend + yearly_season + weekly_season + rw + noise + anomalies
    
    return dates, t, y, {
        'trend': trend,
        'yearly_season': yearly_season, 
        'weekly_season': weekly_season,
        'random_walk': rw,
        'noise': noise,
        'anomalies': anomalies,
        'anomaly_indices': anomaly_indices
    }

def detect_anomalies(residuals, threshold=2.5):
    """Simple anomaly detection using residual analysis."""
    std_resid = np.std(residuals)
    anomalies = np.abs(residuals) > threshold * std_resid
    return anomalies

def forecast_with_uncertainty(spline, t_obs, n_forecast=30, method='linear_trend'):
    """Generate forecasts with uncertainty estimates."""
    # Extend time axis
    t_forecast = np.arange(t_obs[-1] + 1, t_obs[-1] + 1 + n_forecast)
    t_extended = np.concatenate([t_obs, t_forecast])
    
    # Simple linear trend extrapolation for demonstration
    if method == 'linear_trend':
        # Estimate trend from last part of the data
        last_n = min(30, len(t_obs) // 4)
        trend_slope = (spline.fitted_values[-1] - spline.fitted_values[-last_n]) / last_n
        
        # Extrapolate
        forecast_values = spline.fitted_values[-1] + trend_slope * np.arange(1, n_forecast + 1)
        
        # Estimate forecast uncertainty (increases with time)
        base_se = np.sqrt(spline.sigma2)
        forecast_se = base_se * np.sqrt(1 + 0.1 * np.arange(1, n_forecast + 1))
        
    return t_forecast, forecast_values, forecast_se

def main():
    print("PSplines Real-World Application: Time Series Analysis")
    print("=" * 60)
    
    # Generate realistic time series data
    print("1. Generating realistic time series data...")
    dates, t, y, components = generate_realistic_timeseries(n_points=300)
    print(f"   Generated {len(t)} daily observations")
    print(f"   Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Exploratory analysis
    print(f"\n2. Data summary:")
    print(f"   Mean: {np.mean(y):.2f}")
    print(f"   Std: {np.std(y):.2f}")
    print(f"   Range: [{np.min(y):.2f}, {np.max(y):.2f}]")
    
    # Fit P-splines with different smoothing levels
    print(f"\n3. Fitting P-splines for different analyses...")
    
    # Long-term trend (heavy smoothing)
    print("   - Extracting long-term trend...")
    spline_trend = PSpline(t, y, nseg=20, lambda_=1000)  # Heavy smoothing
    spline_trend.fit()
    trend_fitted = spline_trend.fitted_values
    
    # Medium-term variations (moderate smoothing)
    print("   - Fitting medium-term variations...")
    spline_medium = PSpline(t, y, nseg=40)
    # Use cross-validation for optimal smoothing
    best_lambda, _ = cross_validation(spline_medium)
    spline_medium.lambda_ = best_lambda
    spline_medium.fit()
    print(f"     Optimal λ = {best_lambda:.6f}")
    
    # Fine variations (light smoothing)
    print("   - Capturing fine variations...")
    spline_fine = PSpline(t, y, nseg=60, lambda_=0.1)  # Light smoothing
    spline_fine.fit()
    
    # Derivative analysis (rate of change)
    print("   - Computing rate of change...")
    rate_of_change = spline_medium.derivative(t, deriv_order=1)
    
    # Anomaly detection
    print("   - Detecting anomalies...")
    residuals = y - spline_medium.fitted_values
    anomalies_detected = detect_anomalies(residuals)
    n_anomalies = np.sum(anomalies_detected)
    print(f"     Detected {n_anomalies} potential anomalies")
    
    # Forecasting
    print("   - Generating forecasts...")
    t_forecast, forecast_values, forecast_se = forecast_with_uncertainty(
        spline_medium, t, n_forecast=30
    )
    
    # Create comprehensive analysis plots
    print(f"\n4. Creating analysis plots...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Raw data and different smoothing levels
    ax1 = plt.subplot(3, 3, (1, 2))
    plt.plot(dates, y, 'o-', alpha=0.5, markersize=2, color='gray', label='Observed data')
    plt.plot(dates, trend_fitted, 'r-', linewidth=3, label='Long-term trend (λ=1000)')
    plt.plot(dates, spline_medium.fitted_values, 'b-', linewidth=2, 
            label=f'Medium-term fit (λ={best_lambda:.3f})')
    plt.plot(dates, spline_fine.fitted_values, 'g-', linewidth=1, alpha=0.8,
            label='Fine variations (λ=0.1)')
    
    # Highlight detected anomalies
    if n_anomalies > 0:
        anomaly_dates = [dates[i] for i in range(len(dates)) if anomalies_detected[i]]
        anomaly_values = y[anomalies_detected]
        plt.scatter(anomaly_dates, anomaly_values, color='red', s=80, 
                   marker='x', linewidth=3, label=f'Detected anomalies ({n_anomalies})')
    
    plt.title('Time Series Analysis: Multiple Smoothing Levels', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Plot 2: Rate of change analysis
    ax2 = plt.subplot(3, 3, 3)
    plt.plot(dates, rate_of_change, 'purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Rate of Change (1st Derivative)')
    plt.xlabel('Date')
    plt.ylabel('Rate of Change')
    plt.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # Plot 3: Residual analysis
    ax3 = plt.subplot(3, 3, 4)
    plt.plot(dates, residuals, 'o-', alpha=0.6, markersize=3, color='orange')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add threshold lines for anomaly detection
    threshold = 2.5 * np.std(residuals)
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'±{2.5}σ threshold')
    plt.axhline(y=-threshold, color='red', linestyle='--', alpha=0.7)
    
    plt.title('Residuals and Anomaly Detection')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # Plot 4: Forecasting with uncertainty
    ax4 = plt.subplot(3, 3, (5, 6))
    
    # Plot historical data and fit
    n_recent = 60  # Show last 60 days plus forecast
    recent_idx = slice(-n_recent, None)
    plt.plot(dates[recent_idx], y[recent_idx], 'o-', alpha=0.6, markersize=3, 
            color='gray', label='Historical data')
    plt.plot(dates[recent_idx], spline_medium.fitted_values[recent_idx], 'b-', 
            linewidth=2, label='Fitted values')
    
    # Plot forecast
    forecast_dates = [dates[-1] + timedelta(days=i) for i in range(1, len(forecast_values) + 1)]
    plt.plot(forecast_dates, forecast_values, 'r-', linewidth=2, label='Forecast')
    plt.fill_between(forecast_dates, 
                    forecast_values - 1.96 * forecast_se,
                    forecast_values + 1.96 * forecast_se,
                    alpha=0.3, color='red', label='95% Forecast CI')
    
    # Add vertical line at forecast start
    plt.axvline(x=dates[-1], color='black', linestyle=':', alpha=0.7, label='Forecast start')
    
    plt.title('Forecasting with Uncertainty')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # Plot 5: Seasonal decomposition approximation
    ax5 = plt.subplot(3, 3, 7)
    
    # Extract seasonal component (residuals from trend)
    seasonal_approx = spline_medium.fitted_values - trend_fitted
    plt.plot(dates, seasonal_approx, 'green', linewidth=1.5, alpha=0.8)
    plt.title('Seasonal Component (Medium - Trend)')
    plt.xlabel('Date')
    plt.ylabel('Seasonal Effect')
    plt.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # Plot 6: Model diagnostics
    ax6 = plt.subplot(3, 3, 8)
    
    # QQ plot for residual normality
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = np.linspace(-2.5, 2.5, n)
    
    plt.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, s=20)
    plt.plot([-3, 3], [-3, 3], 'r--', alpha=0.8, label='Perfect normal')
    plt.title('Q-Q Plot: Residual Normality')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Smoothing parameter effect
    ax7 = plt.subplot(3, 3, 9)
    
    # Show effect of different lambda values
    lambda_demo = [0.01, 1.0, 100.0]
    colors_demo = ['blue', 'green', 'red']
    
    recent_t = t[recent_idx]
    recent_y = y[recent_idx]
    recent_dates = dates[recent_idx]
    
    plt.plot(recent_dates, recent_y, 'o', alpha=0.4, markersize=3, color='gray', label='Data')
    
    for lam, color in zip(lambda_demo, colors_demo):
        demo_spline = PSpline(recent_t, recent_y, nseg=20, lambda_=lam)
        demo_spline.fit()
        plt.plot(recent_dates, demo_spline.fitted_values, color=color, 
                linewidth=2, alpha=0.8, label=f'λ = {lam}')
    
    plt.title('Effect of Smoothing Parameter')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('examples/timeseries_analysis_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print analysis summary
    print(f"\n5. Analysis Summary:")
    print(f"   Model Performance:")
    print(f"     - RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"     - R²: {1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2):.4f}")
    print(f"     - Effective DoF: {spline_medium.ED:.2f}")
    
    print(f"\n   Trend Analysis:")
    overall_trend = trend_fitted[-1] - trend_fitted[0]
    print(f"     - Overall trend: {overall_trend:.2f} units over {len(t)} days")
    print(f"     - Average daily change: {overall_trend/len(t):.4f}")
    
    print(f"\n   Rate of Change Statistics:")
    print(f"     - Mean rate: {np.mean(rate_of_change):.4f}")
    print(f"     - Max rate: {np.max(rate_of_change):.4f}")
    print(f"     - Min rate: {np.min(rate_of_change):.4f}")
    
    print(f"\n   Anomaly Detection:")
    print(f"     - Detected anomalies: {n_anomalies}")
    print(f"     - Anomaly rate: {n_anomalies/len(t)*100:.1f}%")
    if n_anomalies > 0:
        anomaly_dates_str = [dates[i].strftime('%Y-%m-%d') for i in range(len(dates)) if anomalies_detected[i]]
        print(f"     - Dates: {', '.join(anomaly_dates_str[:5])}{'...' if n_anomalies > 5 else ''}")
    
    print(f"\n   Forecast Summary:")
    print(f"     - Forecast horizon: {len(forecast_values)} days")
    print(f"     - Final forecast: {forecast_values[-1]:.2f} ± {1.96*forecast_se[-1]:.2f}")
    print(f"     - Trend direction: {'Increasing' if forecast_values[-1] > forecast_values[0] else 'Decreasing'}")
    
    print("\n✓ Time series analysis example completed!")
    print("  Plot saved as 'examples/timeseries_analysis_output.png'")

if __name__ == "__main__":
    main()