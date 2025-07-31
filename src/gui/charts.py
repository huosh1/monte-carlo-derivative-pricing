"""
Charts and Visualization Module
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ChartManager:
    """
    Manages all charts and visualizations
    """
    
    def __init__(self, parent):
        """
        Initialize chart manager
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        self.current_figure = None
        self.current_canvas = None
        self.setup_chart_interface()
    
    def setup_chart_interface(self):
        """Setup chart interface"""
        # Control frame
        control_frame = ttk.LabelFrame(self.parent, text="Chart Controls", padding="10")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Chart type selection
        ttk.Label(control_frame, text="Chart Type:").pack(side='left', padx=5)
        
        self.chart_type_var = tk.StringVar(value="sensitivity")
        chart_types = [
            ("Sensitivity Analysis", "sensitivity"),
            ("Monte Carlo Paths", "paths"),
            ("Greeks Analysis", "greeks"),
            ("Volatility Surface", "vol_surface"),
            ("Price Convergence", "convergence"),
            ("Model Comparison", "comparison")
        ]
        
        for text, value in chart_types:
            ttk.Radiobutton(control_frame, text=text, variable=self.chart_type_var, 
                          value=value).pack(side='left', padx=5)
        
        # Chart display frame
        self.chart_frame = ttk.LabelFrame(self.parent, text="Charts", padding="10")
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create initial empty chart
        self.create_empty_chart()
    
    def create_empty_chart(self):
        """Create empty chart placeholder"""
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        self.current_figure = Figure(figsize=(12, 8))
        ax = self.current_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Select a chart type and data to display', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_sensitivity_analysis(self, sensitivity_results):
        """
        Plot sensitivity analysis results
        
        Args:
            sensitivity_results (dict): Sensitivity analysis results
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        # Create new figure with subplots
        self.current_figure = Figure(figsize=(15, 10))
        
        # Number of models
        models = list(sensitivity_results.keys())
        valid_models = [m for m in models if 'error' not in sensitivity_results[m]]
        
        if not valid_models:
            self.create_error_chart("No valid sensitivity data available")
            return
        
        # Create subplots
        n_models = len(valid_models)
        rows = (n_models + 1) // 2
        cols = 2 if n_models > 1 else 1
        
        for i, model_name in enumerate(valid_models):
            result = sensitivity_results[model_name]
            
            if 'values' not in result or 'prices' not in result:
                continue
            
            ax = self.current_figure.add_subplot(rows, cols, i + 1)
            
            values = result['values']
            prices = result['prices']
            parameter = result.get('parameter', 'Parameter')
            original_value = result.get('original_value', 0)
            
            # Plot sensitivity curve
            ax.plot(values, prices, 'b-', linewidth=2, label=f'{model_name.title()} Price')
            ax.axvline(x=original_value, color='r', linestyle='--', alpha=0.7, 
                      label='Current Value')
            
            ax.set_xlabel(f'{parameter}')
            ax.set_ylabel('Option Price ($)')
            ax.set_title(f'{model_name.title()} - {parameter} Sensitivity')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics text
            price_range = max(prices) - min(prices)
            param_range = max(values) - min(values)
            sensitivity = price_range / param_range if param_range != 0 else 0
            
            stats_text = f'Sensitivity: ${sensitivity:.3f} per unit\nPrice Range: ${price_range:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.current_figure.suptitle(f'Sensitivity Analysis - {parameter}', fontsize=16)
        self.current_figure.tight_layout()
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def plot_monte_carlo_paths(self, pricing_engine, model_name='black_scholes', num_paths=100):
        """
        Plot Monte Carlo simulation paths
        
        Args:
            pricing_engine: PricingEngine instance
            model_name (str): Model name
            num_paths (int): Number of paths to display
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        try:
            # Generate paths
            path_data = pricing_engine.generate_monte_carlo_paths(model_name, num_paths)
            
            if 'error' in path_data:
                self.create_error_chart(f"Error generating paths: {path_data['error']}")
                return
            
            self.current_figure = Figure(figsize=(14, 10))
            
            if model_name == 'heston' and 'stock_paths' in path_data:
                # For Heston, plot both stock and volatility paths
                ax1 = self.current_figure.add_subplot(2, 2, 1)
                ax2 = self.current_figure.add_subplot(2, 2, 2)
                ax3 = self.current_figure.add_subplot(2, 1, 2)
                
                stock_paths = path_data['stock_paths']
                variance_paths = path_data['variance_paths']
                
                # Plot stock price paths
                time_steps = np.linspace(0, 1, stock_paths.shape[1])
                for i in range(min(50, stock_paths.shape[0])):
                    ax1.plot(time_steps, stock_paths[i], alpha=0.3, linewidth=0.5)
                
                ax1.set_title('Stock Price Paths')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Stock Price')
                ax1.grid(True, alpha=0.3)
                
                # Plot variance paths
                for i in range(min(50, variance_paths.shape[0])):
                    ax2.plot(time_steps, np.sqrt(variance_paths[i]), alpha=0.3, linewidth=0.5)
                
                ax2.set_title('Volatility Paths')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Volatility')
                ax2.grid(True, alpha=0.3)
                
                # Plot final price distribution
                final_prices = path_data['final_prices']
                ax3.hist(final_prices, bins=50, alpha=0.7, density=True, edgecolor='black')
                ax3.axvline(np.mean(final_prices), color='red', linestyle='--', 
                           label=f'Mean: ${np.mean(final_prices):.2f}')
                ax3.set_title('Final Stock Price Distribution')
                ax3.set_xlabel('Final Stock Price')
                ax3.set_ylabel('Density')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            else:
                # For other models, plot stock paths and distribution
                ax1 = self.current_figure.add_subplot(2, 1, 1)
                ax2 = self.current_figure.add_subplot(2, 1, 2)
                
                if 'paths' in path_data:
                    paths = path_data['paths']
                elif 'stock_paths' in path_data:
                    paths = path_data['stock_paths']
                else:
                    self.create_error_chart("No path data available")
                    return
                
                # Plot sample paths
                time_steps = np.linspace(0, 1, paths.shape[1])
                for i in range(min(100, paths.shape[0])):
                    ax1.plot(time_steps, paths[i], alpha=0.2, linewidth=0.5)
                
                # Highlight mean path
                mean_path = np.mean(paths, axis=0)
                ax1.plot(time_steps, mean_path, 'r-', linewidth=2, label='Mean Path')
                
                ax1.set_title(f'{model_name.title()} - Monte Carlo Paths')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Stock Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot final price distribution
                final_prices = path_data['final_prices']
                ax2.hist(final_prices, bins=50, alpha=0.7, density=True, edgecolor='black')
                ax2.axvline(np.mean(final_prices), color='red', linestyle='--', 
                           label=f'Mean: ${np.mean(final_prices):.2f}')
                ax2.axvline(np.median(final_prices), color='green', linestyle='--', 
                           label=f'Median: ${np.median(final_prices):.2f}')
                
                ax2.set_title('Final Stock Price Distribution')
                ax2.set_xlabel('Final Stock Price')
                ax2.set_ylabel('Density')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            self.current_figure.suptitle(f'{model_name.title()} Monte Carlo Simulation', fontsize=16)
            self.current_figure.tight_layout()
            
        except Exception as e:
            self.create_error_chart(f"Error plotting paths: {e}")
            return
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def plot_greeks_sensitivity(self, pricing_engine):
        """
        Plot Greeks sensitivity analysis
        
        Args:
            pricing_engine: PricingEngine instance
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        try:
            # Calculate Greeks for different spot prices
            if 'black_scholes' not in pricing_engine.models:
                self.create_error_chart("Black-Scholes model required for Greeks sensitivity")
                return
            
            bs_model = pricing_engine.models['black_scholes']
            original_S0 = bs_model.S0
            
            # Create range of spot prices
            spot_range = np.linspace(original_S0 * 0.7, original_S0 * 1.3, 50)
            
            deltas = []
            gammas = []
            thetas = []
            vegas = []
            prices = []
            
            for S in spot_range:
                bs_model.S0 = S
                deltas.append(bs_model.delta())
                gammas.append(bs_model.gamma())
                thetas.append(bs_model.theta())
                vegas.append(bs_model.vega())
                prices.append(bs_model.price())
            
            # Restore original spot price
            bs_model.S0 = original_S0
            
            # Create plots
            self.current_figure = Figure(figsize=(15, 12))
            
            # Price and Delta
            ax1 = self.current_figure.add_subplot(2, 3, 1)
            ax1.plot(spot_range, prices, 'b-', linewidth=2, label='Option Price')
            ax1.axvline(original_S0, color='r', linestyle='--', alpha=0.7, label='Current Spot')
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Option Price')
            ax1.set_title('Option Price vs Spot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = self.current_figure.add_subplot(2, 3, 2)
            ax2.plot(spot_range, deltas, 'g-', linewidth=2, label='Delta')
            ax2.axvline(original_S0, color='r', linestyle='--', alpha=0.7, label='Current Spot')
            ax2.set_xlabel('Spot Price')
            ax2.set_ylabel('Delta')
            ax2.set_title('Delta vs Spot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gamma
            ax3 = self.current_figure.add_subplot(2, 3, 3)
            ax3.plot(spot_range, gammas, 'orange', linewidth=2, label='Gamma')
            ax3.axvline(original_S0, color='r', linestyle='--', alpha=0.7, label='Current Spot')
            ax3.set_xlabel('Spot Price')
            ax3.set_ylabel('Gamma')
            ax3.set_title('Gamma vs Spot')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Theta
            ax4 = self.current_figure.add_subplot(2, 3, 4)
            ax4.plot(spot_range, thetas, 'purple', linewidth=2, label='Theta')
            ax4.axvline(original_S0, color='r', linestyle='--', alpha=0.7, label='Current Spot')
            ax4.set_xlabel('Spot Price')
            ax4.set_ylabel('Theta')
            ax4.set_title('Theta vs Spot')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Vega
            ax5 = self.current_figure.add_subplot(2, 3, 5)
            ax5.plot(spot_range, vegas, 'brown', linewidth=2, label='Vega')
            ax5.axvline(original_S0, color='r', linestyle='--', alpha=0.7, label='Current Spot')
            ax5.set_xlabel('Spot Price')
            ax5.set_ylabel('Vega')
            ax5.set_title('Vega vs Spot')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Greeks summary
            ax6 = self.current_figure.add_subplot(2, 3, 6)
            current_idx = np.argmin(np.abs(spot_range - original_S0))
            greek_names = ['Delta', 'Gamma', 'Theta', 'Vega']
            greek_values = [deltas[current_idx], gammas[current_idx], 
                           thetas[current_idx], vegas[current_idx]]
            
            bars = ax6.bar(greek_names, greek_values, color=['green', 'orange', 'purple', 'brown'])
            ax6.set_title('Current Greeks Values')
            ax6.set_ylabel('Greek Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, greek_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            ax6.grid(True, alpha=0.3)
            
            self.current_figure.suptitle('Greeks Sensitivity Analysis', fontsize=16)
            self.current_figure.tight_layout()
            
        except Exception as e:
            self.create_error_chart(f"Error plotting Greeks: {e}")
            return
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def plot_volatility_surface(self, pricing_engine):
        """
        Plot volatility surface (for Dupire model)
        
        Args:
            pricing_engine: PricingEngine instance
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        try:
            if 'dupire' not in pricing_engine.models:
                self.create_error_chart("Dupire model required for volatility surface")
                return
            
            dupire_model = pricing_engine.models['dupire']
            
            if dupire_model.local_vol_surface is None:
                self.create_error_chart("Local volatility surface not constructed")
                return
            
            # Create mesh for surface plot
            strikes = np.linspace(dupire_model.strikes.min(), dupire_model.strikes.max(), 50)
            maturities = np.linspace(dupire_model.maturities.min(), dupire_model.maturities.max(), 30)
            
            K_mesh, T_mesh = np.meshgrid(strikes, maturities)
            vol_surface = np.zeros_like(K_mesh)
            
            for i in range(len(maturities)):
                for j in range(len(strikes)):
                    vol_surface[i, j] = dupire_model.get_local_volatility(strikes[j], maturities[i])
            
            self.current_figure = Figure(figsize=(15, 10))
            
            # 3D surface plot
            ax1 = self.current_figure.add_subplot(2, 2, 1, projection='3d')
            surf = ax1.plot_surface(K_mesh, T_mesh, vol_surface, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('Strike')
            ax1.set_ylabel('Time to Maturity')
            ax1.set_zlabel('Local Volatility')
            ax1.set_title('Local Volatility Surface')
            
            # Contour plot
            ax2 = self.current_figure.add_subplot(2, 2, 2)
            contour = ax2.contour(K_mesh, T_mesh, vol_surface, levels=15)
            ax2.clabel(contour, inline=True, fontsize=8)
            ax2.set_xlabel('Strike')
            ax2.set_ylabel('Time to Maturity')
            ax2.set_title('Volatility Contours')
            ax2.grid(True, alpha=0.3)
            
            # Volatility smile at different maturities
            ax3 = self.current_figure.add_subplot(2, 2, 3)
            for i, T in enumerate([0.25, 0.5, 1.0]):
                if T <= dupire_model.maturities.max():
                    vols = [dupire_model.get_local_volatility(K, T) for K in strikes]
                    ax3.plot(strikes, vols, label=f'T={T:.2f}', linewidth=2)
            
            ax3.set_xlabel('Strike')
            ax3.set_ylabel('Local Volatility')
            ax3.set_title('Volatility Smile')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Term structure at different strikes
            ax4 = self.current_figure.add_subplot(2, 2, 4)
            S0 = dupire_model.S0
            for strike_mult in [0.9, 1.0, 1.1]:
                K = S0 * strike_mult
                if dupire_model.strikes.min() <= K <= dupire_model.strikes.max():
                    vols = [dupire_model.get_local_volatility(K, T) for T in maturities]
                    ax4.plot(maturities, vols, label=f'K/S₀={strike_mult:.1f}', linewidth=2)
            
            ax4.set_xlabel('Time to Maturity')
            ax4.set_ylabel('Local Volatility')
            ax4.set_title('Volatility Term Structure')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            self.current_figure.suptitle('Local Volatility Surface Analysis', fontsize=16)
            self.current_figure.tight_layout()
            
        except Exception as e:
            self.create_error_chart(f"Error plotting volatility surface: {e}")
            return
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def plot_price_convergence(self, pricing_engine, model_name='black_scholes'):
        """
        Plot Monte Carlo price convergence
        
        Args:
            pricing_engine: PricingEngine instance
            model_name (str): Model name
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        try:
            if model_name not in pricing_engine.models:
                self.create_error_chart(f"{model_name} model not found")
                return
            
            model = pricing_engine.models[model_name]
            
            # Different numbers of simulations
            sim_counts = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
            prices = []
            std_errors = []
            
            for n_sim in sim_counts:
                if hasattr(model, 'monte_carlo_price'):
                    result = model.monte_carlo_price(num_simulations=n_sim)
                    prices.append(result['price'])
                    std_errors.append(result.get('std_error', 0))
                else:
                    # For models without MC method, use analytical price
                    prices.append(model.price() if hasattr(model, 'price') else 0)
                    std_errors.append(0)
            
            self.current_figure = Figure(figsize=(12, 8))
            
            # Price convergence
            ax1 = self.current_figure.add_subplot(2, 1, 1)
            ax1.plot(sim_counts, prices, 'bo-', linewidth=2, markersize=6)
            
            # Add error bars if available
            if any(std_errors):
                ax1.errorbar(sim_counts, prices, yerr=[1.96*se for se in std_errors], 
                           fmt='bo-', capsize=5, alpha=0.7)
            
            # Reference line (analytical price if available)
            if hasattr(model, 'price') and model_name == 'black_scholes':
                analytical_price = model.price()
                ax1.axhline(y=analytical_price, color='r', linestyle='--', 
                           label=f'Analytical: ${analytical_price:.4f}')
                ax1.legend()
            
            ax1.set_xscale('log')
            ax1.set_xlabel('Number of Simulations')
            ax1.set_ylabel('Option Price ($)')
            ax1.set_title(f'{model_name.title()} - Price Convergence')
            ax1.grid(True, alpha=0.3)
            
            # Standard error convergence
            ax2 = self.current_figure.add_subplot(2, 1, 2)
            if any(std_errors):
                ax2.loglog(sim_counts, std_errors, 'ro-', linewidth=2, markersize=6, label='Std Error')
                
                # Theoretical convergence rate (1/sqrt(n))
                theoretical = std_errors[0] * np.sqrt(sim_counts[0] / np.array(sim_counts))
                ax2.loglog(sim_counts, theoretical, 'g--', alpha=0.7, label='Theoretical (1/√n)')
                ax2.legend()
            
            ax2.set_xlabel('Number of Simulations')
            ax2.set_ylabel('Standard Error')
            ax2.set_title('Standard Error Convergence')
            ax2.grid(True, alpha=0.3)
            
            self.current_figure.suptitle(f'{model_name.title()} Monte Carlo Convergence Analysis', fontsize=14)
            self.current_figure.tight_layout()
            
        except Exception as e:
            self.create_error_chart(f"Error plotting convergence: {e}")
            return
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def plot_model_comparison(self, comparison_results):
        """
        Plot model comparison results
        
        Args:
            comparison_results (dict): Model comparison results
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        try:
            if 'prices' not in comparison_results or not comparison_results['prices']:
                self.create_error_chart("No pricing data available for comparison")
                return
            
            prices = comparison_results['prices']
            models = list(prices.keys())
            price_values = list(prices.values())
            
            self.current_figure = Figure(figsize=(14, 10))
            
            # Price comparison bar chart
            ax1 = self.current_figure.add_subplot(2, 2, 1)
            bars = ax1.bar(models, price_values, color=['skyblue', 'lightgreen', 'salmon'][:len(models)])
            ax1.set_ylabel('Option Price ($)')
            ax1.set_title('Model Price Comparison')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, price_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'${value:.4f}', ha='center', va='bottom')
            
            # Price differences from mean
            ax2 = self.current_figure.add_subplot(2, 2, 2)
            mean_price = np.mean(price_values)
            differences = [p - mean_price for p in price_values]
            colors = ['red' if d < 0 else 'green' for d in differences]
            
            bars2 = ax2.bar(models, differences, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax2.set_ylabel('Difference from Mean ($)')
            ax2.set_title('Price Deviations from Mean')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars2, differences):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${value:.4f}', ha='center', 
                        va='bottom' if height >= 0 else 'top')
            
            # Statistics summary
            ax3 = self.current_figure.add_subplot(2, 2, 3)
            if 'statistics' in comparison_results:
                stats = comparison_results['statistics']
                stat_names = ['Mean', 'Std Dev', 'Min', 'Max', 'Range']
                stat_values = [
                    stats.get('mean', 0),
                    stats.get('std', 0),
                    stats.get('min', 0),
                    stats.get('max', 0),
                    stats.get('range', 0)
                ]
                
                bars3 = ax3.bar(stat_names, stat_values, color='lightcoral')
                ax3.set_ylabel('Value ($)')
                ax3.set_title('Price Statistics')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars3, stat_values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'${value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Model agreement visualization
            ax4 = self.current_figure.add_subplot(2, 2, 4)
            
            # Create pie chart showing relative price contributions
            wedges, texts, autotexts = ax4.pie(price_values, labels=models, autopct='%1.1f%%',
                                              colors=['skyblue', 'lightgreen', 'salmon'][:len(models)])
            ax4.set_title('Relative Price Distribution')
            
            # Enhance pie chart text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            self.current_figure.suptitle('Model Comparison Analysis', fontsize=16)
            self.current_figure.tight_layout()
            
        except Exception as e:
            self.create_error_chart(f"Error plotting model comparison: {e}")
            return
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def create_error_chart(self, error_message):
        """Create error message chart"""
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        self.current_figure = Figure(figsize=(10, 6))
        ax = self.current_figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error: {error_message}', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, color='red', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Chart Error', fontsize=14)
        
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.chart_frame)
        self.current_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_canvas.draw()
    
    def save_current_chart(self, filename=None):
        """Save current chart to file"""
        if not self.current_figure:
            return False
        
        if filename is None:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                title="Save Chart",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
        
        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
                return True
            except Exception as e:
                print(f"Error saving chart: {e}")
                return False
        
        return False
    
    def clear_chart(self):
        """Clear current chart"""
        self.create_empty_chart()