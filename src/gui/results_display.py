"""
Results Display Module
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd

class ResultsDisplay:
    """
    Display pricing results and model comparisons
    """
    
    def __init__(self, parent):
        """
        Initialize results display
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        self.setup_display()
    
    def setup_display(self):
        """Setup results display interface"""
        # Create main notebook for different result types
        self.results_notebook = ttk.Notebook(self.parent)
        self.results_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pricing Results Tab
        self.pricing_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.pricing_frame, text="Pricing Results")
        self.setup_pricing_display()
        
        # Model Comparison Tab
        self.comparison_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.comparison_frame, text="Model Comparison")
        self.setup_comparison_display()
        
        # Statistics Tab
        self.stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.stats_frame, text="Statistics")
        self.setup_statistics_display()
    
    def setup_pricing_display(self):
        """Setup pricing results display"""
        # Results table frame
        table_frame = ttk.LabelFrame(self.pricing_frame, text="Option Prices by Model", padding="10")
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for pricing results
        columns = ('Model', 'Price', 'Std Error', 'Confidence Interval', 'Method')
        self.pricing_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.pricing_tree.heading('Model', text='Model')
        self.pricing_tree.heading('Price', text='Price ($)')
        self.pricing_tree.heading('Std Error', text='Std Error')
        self.pricing_tree.heading('Confidence Interval', text='95% CI')
        self.pricing_tree.heading('Method', text='Method')
        
        self.pricing_tree.column('Model', width=120)
        self.pricing_tree.column('Price', width=100)
        self.pricing_tree.column('Std Error', width=100)
        self.pricing_tree.column('Confidence Interval', width=200)
        self.pricing_tree.column('Method', width=120)
        
        # Scrollbar
        pricing_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.pricing_tree.yview)
        self.pricing_tree.configure(yscrollcommand=pricing_scrollbar.set)
        
        self.pricing_tree.pack(side='left', fill='both', expand=True)
        pricing_scrollbar.pack(side='right', fill='y')
        
        # Summary frame
        summary_frame = ttk.LabelFrame(self.pricing_frame, text="Pricing Summary", padding="10")
        summary_frame.pack(fill='x', padx=10, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=6, wrap='word')
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient='vertical', command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side='left', fill='both', expand=True)
        summary_scrollbar.pack(side='right', fill='y')
    
    def setup_comparison_display(self):
        """Setup model comparison display"""
        # Comparison table
        comp_table_frame = ttk.LabelFrame(self.comparison_frame, text="Model Comparison", padding="10")
        comp_table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        columns = ('Metric', 'Value', 'Description')
        self.comparison_tree = ttk.Treeview(comp_table_frame, columns=columns, show='headings', height=10)
        
        self.comparison_tree.heading('Metric', text='Metric')
        self.comparison_tree.heading('Value', text='Value')
        self.comparison_tree.heading('Description', text='Description')
        
        self.comparison_tree.column('Metric', width=150)
        self.comparison_tree.column('Value', width=100)
        self.comparison_tree.column('Description', width=300)
        
        comp_scrollbar = ttk.Scrollbar(comp_table_frame, orient='vertical', command=self.comparison_tree.yview)
        self.comparison_tree.configure(yscrollcommand=comp_scrollbar.set)
        
        self.comparison_tree.pack(side='left', fill='both', expand=True)
        comp_scrollbar.pack(side='right', fill='y')
        
        # Price differences frame
        diff_frame = ttk.LabelFrame(self.comparison_frame, text="Price Differences", padding="10")
        diff_frame.pack(fill='x', padx=10, pady=5)
        
        self.differences_text = tk.Text(diff_frame, height=6, wrap='word')
        diff_scrollbar = ttk.Scrollbar(diff_frame, orient='vertical', command=self.differences_text.yview)
        self.differences_text.configure(yscrollcommand=diff_scrollbar.set)
        
        self.differences_text.pack(side='left', fill='both', expand=True)
        diff_scrollbar.pack(side='right', fill='y')
    
    def setup_statistics_display(self):
        """Setup statistics display"""
        # Statistics table
        stats_frame = ttk.LabelFrame(self.stats_frame, text="Pricing Statistics", padding="10")
        stats_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        columns = ('Statistic', 'Value', 'Interpretation')
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show='headings', height=8)
        
        self.stats_tree.heading('Statistic', text='Statistic')
        self.stats_tree.heading('Value', text='Value')
        self.stats_tree.heading('Interpretation', text='Interpretation')
        
        self.stats_tree.column('Statistic', width=150)
        self.stats_tree.column('Value', width=120)
        self.stats_tree.column('Interpretation', width=300)
        
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_tree.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
        
        # Model performance frame
        perf_frame = ttk.LabelFrame(self.stats_frame, text="Model Performance Analysis", padding="10")
        perf_frame.pack(fill='x', padx=10, pady=5)
        
        self.performance_text = tk.Text(perf_frame, height=6, wrap='word')
        perf_scrollbar = ttk.Scrollbar(perf_frame, orient='vertical', command=self.performance_text.yview)
        self.performance_text.configure(yscrollcommand=perf_scrollbar.set)
        
        self.performance_text.pack(side='left', fill='both', expand=True)
        perf_scrollbar.pack(side='right', fill='y')
    
    def display_pricing_results(self, results):
        """
        Display pricing results
        
        Args:
            results (dict): Pricing results from all models
        """
        # Clear existing results
        for item in self.pricing_tree.get_children():
            self.pricing_tree.delete(item)
        
        # Add results to table
        for model_name, result in results.items():
            if isinstance(result, dict):
                price = result.get('price', 'N/A')
                std_error = result.get('std_error', 'N/A')
                ci = result.get('confidence_interval', 'N/A')
                
                # Format values
                if isinstance(price, (int, float)) and not np.isnan(price):
                    price_str = f"${price:.4f}"
                else:
                    price_str = str(price)
                
                if isinstance(std_error, (int, float)) and not np.isnan(std_error):
                    std_error_str = f"${std_error:.4f}"
                else:
                    std_error_str = "N/A"
                
                if isinstance(ci, tuple) and len(ci) == 2:
                    ci_str = f"[${ci[0]:.4f}, ${ci[1]:.4f}]"
                else:
                    ci_str = "N/A"
                
                # Determine method
                if 'confidence_interval' in result:
                    method = "Monte Carlo"
                else:
                    method = "Analytical"
                
                self.pricing_tree.insert('', 'end', values=(
                    model_name.replace('_', ' ').title(),
                    price_str,
                    std_error_str,
                    ci_str,
                    method
                ))
        
        # Update summary
        self.update_pricing_summary(results)
    
    def update_pricing_summary(self, results):
        """Update pricing summary text"""
        self.summary_text.delete('1.0', 'end')
        
        prices = []
        valid_results = {}
        
        for model_name, result in results.items():
            if isinstance(result, dict) and 'price' in result:
                price = result['price']
                if isinstance(price, (int, float)) and not np.isnan(price):
                    prices.append(price)
                    valid_results[model_name] = result
        
        if not prices:
            self.summary_text.insert('end', "No valid pricing results available.")
            return
        
        # Calculate summary statistics
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        min_price = np.min(prices)
        max_price = np.max(prices)
        price_range = max_price - min_price
        
        summary = f"""PRICING SUMMARY
{'=' * 50}

Number of Models: {len(valid_results)}
Price Statistics:
  • Average Price: ${mean_price:.4f}
  • Standard Deviation: ${std_price:.4f}
  • Minimum Price: ${min_price:.4f}
  • Maximum Price: ${max_price:.4f}
  • Price Range: ${price_range:.4f}
  • Coefficient of Variation: {(std_price/mean_price)*100:.2f}%

Model Agreement: {'High' if std_price/mean_price < 0.05 else 'Moderate' if std_price/mean_price < 0.1 else 'Low'}
"""
        
        self.summary_text.insert('end', summary)
    
    def display_model_comparison(self, comparison):
        """
        Display model comparison results
        
        Args:
            comparison (dict): Comparison results
        """
        # Clear existing comparison results
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        
        # Display price comparison
        if 'prices' in comparison:
            prices = comparison['prices']
            for model_name, price in prices.items():
                self.comparison_tree.insert('', 'end', values=(
                    f"{model_name.title()} Price",
                    f"${price:.4f}",
                    f"Option price from {model_name} model"
                ))
        
        # Display statistics
        if 'statistics' in comparison:
            stats = comparison['statistics']
            
            stat_items = [
                ('Mean Price', stats.get('mean', 0), 'Average price across all models'),
                ('Price Std Dev', stats.get('std', 0), 'Standard deviation of prices'),
                ('Min Price', stats.get('min', 0), 'Minimum price among models'),
                ('Max Price', stats.get('max', 0), 'Maximum price among models'),
                ('Price Range', stats.get('range', 0), 'Difference between max and min prices'),
                ('Coefficient of Variation', stats.get('cv', 0), 'Std dev / mean (measure of relative variability)')
            ]
            
            for stat_name, value, description in stat_items:
                if isinstance(value, (int, float)):
                    if 'Price' in stat_name or stat_name in ['Mean Price', 'Price Std Dev', 'Min Price', 'Max Price', 'Price Range']:
                        value_str = f"${value:.4f}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                self.comparison_tree.insert('', 'end', values=(stat_name, value_str, description))
        
        # Update differences text
        self.update_differences_text(comparison)
    
    def update_differences_text(self, comparison):
        """Update price differences analysis"""
        self.differences_text.delete('1.0', 'end')
        
        if 'prices' not in comparison or len(comparison['prices']) < 2:
            self.differences_text.insert('end', "Need at least 2 models for comparison.")
            return
        
        prices = comparison['prices']
        model_names = list(prices.keys())
        price_values = list(prices.values())
        
        analysis = "PRICE DIFFERENCES ANALYSIS\n"
        analysis += "=" * 40 + "\n\n"
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    price1 = prices[model1]
                    price2 = prices[model2]
                    diff = abs(price1 - price2)
                    rel_diff = (diff / max(price1, price2)) * 100
                    
                    analysis += f"{model1.title()} vs {model2.title()}:\n"
                    analysis += f"  Absolute Difference: ${diff:.4f}\n"
                    analysis += f"  Relative Difference: {rel_diff:.2f}%\n"
                    analysis += f"  Assessment: {'Small' if rel_diff < 1 else 'Moderate' if rel_diff < 5 else 'Large'}\n\n"
        
        # Overall assessment
        if 'statistics' in comparison:
            cv = comparison['statistics'].get('cv', 0)
            analysis += "OVERALL ASSESSMENT:\n"
            if cv < 0.02:
                analysis += "Excellent model agreement (CV < 2%)"
            elif cv < 0.05:
                analysis += "Good model agreement (CV < 5%)"
            elif cv < 0.10:
                analysis += "Moderate model agreement (CV < 10%)"
            else:
                analysis += "Poor model agreement (CV > 10%)"
        
        self.differences_text.insert('end', analysis)
    
    def display_statistics(self, stats_data):
        """
        Display detailed statistics
        
        Args:
            stats_data (dict): Statistics data
        """
        # Clear existing statistics
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add statistics to table
        if isinstance(stats_data, dict):
            for stat_name, value in stats_data.items():
                if isinstance(value, (int, float)):
                    value_str = f"{value:.6f}"
                    
                    # Add interpretation
                    interpretation = self.get_statistic_interpretation(stat_name, value)
                    
                    self.stats_tree.insert('', 'end', values=(
                        stat_name.replace('_', ' ').title(),
                        value_str,
                        interpretation
                    ))
        
        # Update performance analysis
        self.update_performance_analysis(stats_data)
    
    def get_statistic_interpretation(self, stat_name, value):
        """Get interpretation for a statistic"""
        interpretations = {
            'mean': f"Average option price across all models",
            'std': f"Price variability - {'Low' if value < 1 else 'Moderate' if value < 5 else 'High'} dispersion",
            'min': f"Lowest price estimate",
            'max': f"Highest price estimate",
            'range': f"Price spread - {'Narrow' if value < 1 else 'Moderate' if value < 5 else 'Wide'} range",
            'cv': f"Relative variability - {'Low' if value < 0.05 else 'Moderate' if value < 0.1 else 'High'} variation"
        }
        
        return interpretations.get(stat_name.lower(), "Statistical measure")
    
    def update_performance_analysis(self, stats_data):
        """Update model performance analysis"""
        self.performance_text.delete('1.0', 'end')
        
        if not isinstance(stats_data, dict):
            self.performance_text.insert('end', "No performance data available.")
            return
        
        analysis = "MODEL PERFORMANCE ANALYSIS\n"
        analysis += "=" * 40 + "\n\n"
        
        # Convergence analysis
        if 'std' in stats_data and 'mean' in stats_data:
            cv = stats_data['std'] / stats_data['mean'] if stats_data['mean'] != 0 else 0
            
            analysis += "CONVERGENCE ASSESSMENT:\n"
            if cv < 0.01:
                analysis += "• Excellent convergence - models show very high agreement\n"
                analysis += "• Price estimates are highly reliable\n"
            elif cv < 0.05:
                analysis += "• Good convergence - models show reasonable agreement\n"
                analysis += "• Price estimates are reliable\n"
            elif cv < 0.10:
                analysis += "• Moderate convergence - some model disagreement\n"
                analysis += "• Price estimates should be interpreted with caution\n"
            else:
                analysis += "• Poor convergence - significant model disagreement\n"
                analysis += "• Price estimates may be unreliable\n"
                analysis += "• Consider parameter recalibration\n"
            
            analysis += f"• Coefficient of Variation: {cv:.2%}\n\n"
        
        # Model reliability assessment
        analysis += "RELIABILITY RECOMMENDATIONS:\n"
        
        if 'range' in stats_data:
            price_range = stats_data['range']
            if price_range < 0.50:
                analysis += "• High reliability - narrow price range suggests robust estimates\n"
            elif price_range < 2.00:
                analysis += "• Moderate reliability - reasonable price range\n"
            else:
                analysis += "• Low reliability - wide price range suggests model uncertainty\n"
        
        analysis += "• Use ensemble average for final price estimate\n"
        analysis += "• Consider model-specific strengths for different market conditions\n"
        analysis += "• Monitor Greek sensitivities for risk management\n"
        
        self.performance_text.insert('end', analysis)
    
    def clear_all_results(self):
        """Clear all displayed results"""
        # Clear pricing results
        for item in self.pricing_tree.get_children():
            self.pricing_tree.delete(item)
        self.summary_text.delete('1.0', 'end')
        
        # Clear comparison results
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        self.differences_text.delete('1.0', 'end')
        
        # Clear statistics
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        self.performance_text.delete('1.0', 'end')
    
    def export_results_to_dataframe(self):
        """Export current results to pandas DataFrame"""
        # Export pricing results
        pricing_data = []
        for item in self.pricing_tree.get_children():
            values = self.pricing_tree.item(item)['values']
            pricing_data.append(values)
        
        if pricing_data:
            pricing_df = pd.DataFrame(pricing_data, columns=[
                'Model', 'Price', 'Std Error', 'Confidence Interval', 'Method'
            ])
        else:
            pricing_df = pd.DataFrame()
        
        # Export comparison results
        comparison_data = []
        for item in self.comparison_tree.get_children():
            values = self.comparison_tree.item(item)['values']
            comparison_data.append(values)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data, columns=[
                'Metric', 'Value', 'Description'
            ])
        else:
            comparison_df = pd.DataFrame()
        
        return {
            'pricing_results': pricing_df,
            'comparison_results': comparison_df
        }