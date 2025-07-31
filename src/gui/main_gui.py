"""
Main GUI for Derivative Pricing Application - Version corrigée
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from datetime import datetime
import threading

from ..data.data_manager import DataManager
from ..pricing.pricing_engine import PricingEngine
from .parameter_forms import ParameterForms
from .results_display import ResultsDisplay
from .charts import ChartManager

class DerivativePricingGUI:
    """
    Main GUI application for derivative pricing
    """
    
    def __init__(self, root):
        """
        Initialize the main GUI
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Monte Carlo Derivative Pricing Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.data_manager = DataManager()
        self.pricing_engine = PricingEngine()
        self.parameter_forms = None
        self.results_display = None
        self.chart_manager = None
        
        # Data storage
        self.current_market_data = None
        self.pricing_results = None
        
        # Setup GUI
        self.setup_menu()
        self.setup_main_interface()
        self.setup_status_bar()
        
        # Load default market data
        self.load_default_data()
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Market Data", command=self.load_market_data)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_command(label="Export to Excel", command=self.export_to_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Setup Black-Scholes", command=self.setup_black_scholes)
        models_menu.add_command(label="Setup Heston", command=self.setup_heston)
        models_menu.add_command(label="Setup Dupire", command=self.setup_dupire)
        models_menu.add_separator()
        models_menu.add_command(label="Calibrate Models", command=self.calibrate_models)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Price All Models", command=self.price_options)
        analysis_menu.add_command(label="Calculate Greeks", command=self.calculate_greeks)
        analysis_menu.add_command(label="Sensitivity Analysis", command=self.sensitivity_analysis)
        analysis_menu.add_command(label="Model Comparison", command=self.compare_models)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_main_interface(self):
        """Setup main interface with notebook tabs"""
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Market Data Tab
        self.market_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.market_tab, text="Market Data")
        self.setup_market_data_tab()
        
        # Parameters Tab
        self.params_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.params_tab, text="Model Parameters")
        self.parameter_forms = ParameterForms(self.params_tab, self.pricing_engine)
        
        # Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Pricing Results")
        self.results_display = ResultsDisplay(self.results_tab)
        
        # Charts Tab
        self.charts_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.charts_tab, text="Charts & Analysis")
        self.chart_manager = ChartManager(self.charts_tab)
        # IMPORTANT: Connecter le chart manager au pricing engine
        self.chart_manager.set_pricing_engine(self.pricing_engine)
        
        # Greeks Tab
        self.greeks_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.greeks_tab, text="Greeks Analysis")
        self.setup_greeks_tab()
    
    def setup_market_data_tab(self):
        """Setup market data tab"""
        # Market data input frame
        input_frame = ttk.LabelFrame(self.market_tab, text="Market Data Input", padding="10")
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Symbol input
        ttk.Label(input_frame, text="Stock Symbol:").grid(row=0, column=0, sticky='w', pady=2)
        self.symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(input_frame, textvariable=self.symbol_var, width=15).grid(row=0, column=1, pady=2)
        
        # Period selection
        ttk.Label(input_frame, text="Time Period:").grid(row=0, column=2, sticky='w', padx=(20,5), pady=2)
        self.period_var = tk.StringVar(value="5y")
        period_combo = ttk.Combobox(input_frame, textvariable=self.period_var, 
                                  values=["1y", "2y", "5y", "10y"], width=10)
        period_combo.grid(row=0, column=3, pady=2)
        
        # Load button
        ttk.Button(input_frame, text="Load Data", 
                  command=self.load_market_data_symbol).grid(row=0, column=4, padx=(20,0), pady=2)
        
        # Market data display frame
        display_frame = ttk.LabelFrame(self.market_tab, text="Market Data Summary", padding="10")
        display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for market data
        columns = ('Metric', 'Value')
        self.market_tree = ttk.Treeview(display_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=200)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(display_frame, orient='vertical', command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=scrollbar.set)
        
        self.market_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Price chart frame
        chart_frame = ttk.LabelFrame(self.market_tab, text="Price Chart", padding="10")
        chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Matplotlib figure for price chart
        self.price_fig, self.price_ax = plt.subplots(figsize=(12, 4))
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, chart_frame)
        self.price_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_greeks_tab(self):
        """Setup Greeks analysis tab"""
        # Greeks control frame
        control_frame = ttk.LabelFrame(self.greeks_tab, text="Greeks Calculation", padding="10")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Calculate All Greeks", 
                  command=self.calculate_greeks).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Greeks Sensitivity", 
                  command=self.greeks_sensitivity).pack(side='left', padx=5)
        
        # Greeks display frame
        self.greeks_frame = ttk.LabelFrame(self.greeks_tab, text="Greeks Results", padding="10")
        self.greeks_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
    
    def load_default_data(self):
        """Load default market data"""
        try:
            self.update_status("Loading default market data...")
            self.current_market_data = self.data_manager.get_market_data_summary("AAPL")
            self.update_market_data_display()
            self.plot_price_chart()
            self.update_status("Default market data loaded successfully")
        except Exception as e:
            self.update_status(f"Error loading default data: {e}")
            messagebox.showerror("Error", f"Failed to load default data: {e}")
    
    def load_market_data_symbol(self):
        """Load market data for specified symbol"""
        symbol = self.symbol_var.get().upper()
        period = self.period_var.get()
        
        if not symbol:
            messagebox.showwarning("Warning", "Please enter a stock symbol")
            return
        
        def load_data():
            try:
                self.update_status(f"Loading data for {symbol}...")
                self.current_market_data = self.data_manager.get_market_data_summary(symbol)
                
                # Update GUI in main thread
                self.root.after(0, self.update_market_data_display)
                self.root.after(0, self.plot_price_chart)
                self.root.after(0, lambda: self.update_status(f"Data loaded for {symbol}"))
                
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"Error: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load data for {symbol}: {e}"))
        
        # Run in background thread
        threading.Thread(target=load_data, daemon=True).start()
    
    def update_market_data_display(self):
        """Update market data display"""
        if not self.current_market_data:
            return
        
        # Clear existing data
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)
        
        # Add market data
        data = self.current_market_data
        items = [
            ("Symbol", data['symbol']),
            ("Current Price", f"${data['current_price']:.2f}"),
            ("Volatility (Annual)", f"{data['volatility']:.2%}"),
            ("Risk-Free Rate", f"{data['risk_free_rate']:.2%}"),
            ("Data Period", f"{data['start_date']} to {data['end_date']}"),
            ("Observations", str(data['num_observations']))
        ]
        
        for item, value in items:
            self.market_tree.insert('', 'end', values=(item, value))
    
    def plot_price_chart(self):
        """Plot price chart"""
        if not self.current_market_data:
            return
        
        try:
            data = self.current_market_data['data']
            
            self.price_ax.clear()
            self.price_ax.plot(data.index, data['Close'], linewidth=1.5, color='blue')
            self.price_ax.set_title(f"{self.current_market_data['symbol']} Stock Price")
            self.price_ax.set_xlabel("Date")
            self.price_ax.set_ylabel("Price ($)")
            self.price_ax.grid(True, alpha=0.3)
            
            # Format x-axis
            self.price_fig.autofmt_xdate()
            
            self.price_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting chart: {e}")
    
    def setup_black_scholes(self):
        """Setup Black-Scholes model with current market data"""
        if not self.current_market_data:
            messagebox.showwarning("Warning", "Please load market data first")
            return
        
        # Switch to parameters tab
        self.notebook.select(self.params_tab)
        
        # Setup model with market data
        self.parameter_forms.setup_black_scholes_from_market(self.current_market_data)
        self.update_status("Black-Scholes model setup completed")
    
    def setup_heston(self):
        """Setup Heston model"""
        if not self.current_market_data:
            messagebox.showwarning("Warning", "Please load market data first")
            return
        
        self.notebook.select(self.params_tab)
        self.parameter_forms.setup_heston_from_market(self.current_market_data)
        self.update_status("Heston model setup completed")
    
    def setup_dupire(self):
        """Setup Dupire model"""
        if not self.current_market_data:
            messagebox.showwarning("Warning", "Please load market data first")
            return
        
        self.notebook.select(self.params_tab)
        self.parameter_forms.setup_dupire_from_market(self.current_market_data)
        self.update_status("Dupire model setup completed")
    
    def price_options(self):
        """Price options using all models"""
        try:
            if not self.pricing_engine.models:
                messagebox.showwarning("Warning", "No models configured. Please setup models first.")
                return
            
            self.update_status("Pricing options...")
            
            # Get pricing results
            self.pricing_results = self.pricing_engine.price_all_models()
            
            # Update results display
            self.results_display.display_pricing_results(self.pricing_results)
            
            # Switch to results tab
            self.notebook.select(self.results_tab)
            
            self.update_status("Option pricing completed")
            
        except Exception as e:
            self.update_status(f"Pricing error: {e}")
            messagebox.showerror("Error", f"Pricing failed: {e}")
    
    def calculate_greeks(self):
        """Calculate Greeks for all models"""
        try:
            if not self.pricing_engine.models:
                messagebox.showwarning("Warning", "No models configured. Please setup models first.")
                return
                
            self.update_status("Calculating Greeks...")
            
            greeks_results = self.pricing_engine.calculate_greeks_all_models()
            
            # Display Greeks results
            self.display_greeks_results(greeks_results)
            
            # Switch to Greeks tab
            self.notebook.select(self.greeks_tab)
            
            self.update_status("Greeks calculation completed")
            
        except Exception as e:
            self.update_status(f"Greeks calculation error: {e}")
            messagebox.showerror("Error", f"Greeks calculation failed: {e}")
    
    def display_greeks_results(self, greeks_results):
        """Display Greeks results"""
        # Clear existing widgets
        for widget in self.greeks_frame.winfo_children():
            widget.destroy()
        
        # Create treeview for Greeks
        columns = ('Model', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho')
        greeks_tree = ttk.Treeview(self.greeks_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            greeks_tree.heading(col, text=col)
            greeks_tree.column(col, width=120)
        
        # Add data
        for model_name, greeks in greeks_results.items():
            if 'error' not in greeks:
                values = [model_name.title()]
                for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                    if greek in greeks:
                        values.append(f"{greeks[greek]:.4f}")
                    else:
                        values.append("N/A")
                greeks_tree.insert('', 'end', values=values)
        
        greeks_tree.pack(fill='both', expand=True)
    
    def sensitivity_analysis(self):
        """Perform sensitivity analysis"""
        # Create sensitivity analysis dialog
        self.sensitivity_dialog()
    
    def sensitivity_dialog(self):
        """Create sensitivity analysis dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Sensitivity Analysis")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Parameter selection
        ttk.Label(dialog, text="Select Parameter:").pack(pady=10)
        
        param_var = tk.StringVar(value="S0")
        params = ["S0", "K", "T", "r", "sigma"]
        
        for param in params:
            ttk.Radiobutton(dialog, text=param, variable=param_var, value=param).pack(anchor='w', padx=20)
        
        # Range selection
        ttk.Label(dialog, text="Range (±%):").pack(pady=(20,5))
        range_var = tk.DoubleVar(value=20.0)
        ttk.Scale(dialog, from_=5, to=50, variable=range_var, orient='horizontal').pack(fill='x', padx=20)
        
        range_label = ttk.Label(dialog, text="20%")
        range_label.pack()
        
        def update_range_label(*args):
            range_label.config(text=f"{range_var.get():.0f}%")
        
        range_var.trace('w', update_range_label)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def run_analysis():
            try:
                if not self.pricing_engine.models:
                    messagebox.showwarning("Warning", "No models configured. Please setup models first.")
                    return
                    
                param = param_var.get()
                range_pct = range_var.get() / 100
                
                results = self.pricing_engine.perform_sensitivity_analysis(param, range_pct)
                
                # Plot results
                self.chart_manager.plot_sensitivity_analysis(results)
                self.notebook.select(self.charts_tab)
                
                dialog.destroy()
                self.update_status(f"Sensitivity analysis for {param} completed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Sensitivity analysis failed: {e}")
        
        ttk.Button(button_frame, text="Run Analysis", command=run_analysis).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)
    
    def greeks_sensitivity(self):
        """Greeks sensitivity analysis"""
        try:
            if not self.pricing_engine.models:
                messagebox.showwarning("Warning", "No models configured. Please setup models first.")
                return
                
            # This would create charts showing how Greeks change with underlying parameters
            self.chart_manager.plot_greeks_sensitivity(self.pricing_engine)
            self.notebook.select(self.charts_tab)
            self.update_status("Greeks sensitivity analysis completed")
        except Exception as e:
            messagebox.showerror("Error", f"Greeks sensitivity analysis failed: {e}")
    
    def compare_models(self):
        """Compare all models"""
        try:
            if not self.pricing_engine.models:
                messagebox.showwarning("Warning", "No models configured. Please setup models first.")
                return
            
            # S'assurer qu'on a des résultats de pricing
            if not self.pricing_engine.results:
                self.update_status("No pricing results found. Running pricing first...")
                self.pricing_engine.price_all_models()
            
            comparison = self.pricing_engine.compare_models()
            
            # Display comparison dans Results tab
            self.results_display.display_model_comparison(comparison)
            
            # ET afficher le graphique de comparaison
            self.chart_manager.plot_model_comparison(comparison)
            
            # Aller aux résultats d'abord, puis l'utilisateur peut voir les charts
            self.notebook.select(self.results_tab)
            
            self.update_status("Model comparison completed. Check Charts tab for visualization.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {e}")
    
    def calibrate_models(self):
        """Calibrate models to market data"""
        try:
            if not self.pricing_engine.models:
                messagebox.showwarning("Warning", "No models configured. Please setup models first.")
                return
                
            self.update_status("Calibrating models...")
            
            # For demonstration, create synthetic market data
            # In practice, this would come from real option market data
            market_data = self.generate_synthetic_market_data()
            
            results = self.pricing_engine.calibrate_models(market_data)
            
            # Display calibration results
            self.display_calibration_results(results)
            
            self.update_status("Model calibration completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Model calibration failed: {e}")
    
    def generate_synthetic_market_data(self):
        """Generate synthetic market data for calibration"""
        if not self.current_market_data:
            raise ValueError("No market data available")
        
        S0 = self.current_market_data['current_price']
        
        # Create synthetic option data
        strikes = np.array([S0 * k for k in [0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Generate implied volatilities (synthetic)
        implied_vols = np.array([
            [0.25, 0.22, 0.20, 0.22, 0.25],  # 3 months
            [0.24, 0.21, 0.19, 0.21, 0.24],  # 6 months
            [0.23, 0.20, 0.18, 0.20, 0.23]   # 1 year
        ])
        
        return {
            'strikes': strikes,
            'maturities': maturities,
            'implied_vols': implied_vols
        }
    
    def display_calibration_results(self, results):
        """Display calibration results"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibration Results")
        dialog.geometry("500x400")
        
        text_widget = tk.Text(dialog, wrap='word')
        scrollbar = ttk.Scrollbar(dialog, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Display results
        for model_name, result in results.items():
            text_widget.insert('end', f"{model_name.upper()} MODEL:\n")
            text_widget.insert('end', "-" * 30 + "\n")
            
            if result.get('success', False):
                text_widget.insert('end', "Calibration successful!\n")
                if 'error' in result:
                    text_widget.insert('end', f"Final error: {result['error']:.6f}\n")
                
                # Display parameters
                for key, value in result.items():
                    if key not in ['success', 'error', 'message']:
                        text_widget.insert('end', f"{key}: {value:.4f}\n")
            else:
                text_widget.insert('end', "Calibration failed!\n")
                text_widget.insert('end', f"Error: {result.get('message', 'Unknown error')}\n")
            
            text_widget.insert('end', "\n")
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def load_market_data(self):
        """Load market data from file"""
        filename = filedialog.askopenfilename(
            title="Load Market Data",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Implementation for loading market data from file
                self.update_status(f"Loaded market data from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load market data: {e}")
    
    def save_results(self):
        """Save pricing results"""
        if not self.pricing_results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                saved_file = self.pricing_engine.export_results(filename)
                self.update_status(f"Results saved to {saved_file}")
                messagebox.showinfo("Success", f"Results saved to {saved_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def export_to_excel(self):
        """Export results to Excel"""
        if not self.pricing_results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Create Excel export
                self.create_excel_export(filename)
                self.update_status(f"Results exported to {filename}")
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export to Excel: {e}")
    
    def create_excel_export(self, filename):
        """Create Excel export of results"""
        import pandas as pd
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Market data sheet
            if self.current_market_data:
                market_df = pd.DataFrame([
                    ["Symbol", self.current_market_data['symbol']],
                    ["Current Price", self.current_market_data['current_price']],
                    ["Volatility", self.current_market_data['volatility']],
                    ["Risk-Free Rate", self.current_market_data['risk_free_rate']]
                ], columns=['Metric', 'Value'])
                market_df.to_excel(writer, sheet_name='Market Data', index=False)
            
            # Pricing results sheet
            if self.pricing_results:
                results_data = []
                for model, result in self.pricing_results.items():
                    if isinstance(result, dict) and 'price' in result:
                        results_data.append([model, result['price']])
                
                if results_data:
                    results_df = pd.DataFrame(results_data, columns=['Model', 'Price'])
                    results_df.to_excel(writer, sheet_name='Pricing Results', index=False)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Monte Carlo Derivative Pricing Tool
Version 1.0

This application implements three advanced option pricing models:
• Black-Scholes Model
• Heston Stochastic Volatility Model  
• Dupire Local Volatility Model

Features:
• Real-time market data integration
• Monte Carlo simulation
• Greeks calculation
• Model calibration
• Sensitivity analysis
• Professional reporting

Developed for academic and professional use.
        """
        
        messagebox.showinfo("About", about_text)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()