"""
Parameter Forms for Model Setup
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class ParameterForms:
    """
    Forms for setting up model parameters
    """
    
    def __init__(self, parent, pricing_engine):
        """
        Initialize parameter forms
        
        Args:
            parent: Parent widget
            pricing_engine: PricingEngine instance
        """
        self.parent = parent
        self.pricing_engine = pricing_engine
        
        # Parameter variables
        self.bs_vars = {}
        self.heston_vars = {}
        self.dupire_vars = {}
        
        self.setup_forms()
    
    def setup_forms(self):
        """Setup all parameter forms"""
        # Create notebook for different models
        self.forms_notebook = ttk.Notebook(self.parent)
        self.forms_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Black-Scholes form
        self.bs_frame = ttk.Frame(self.forms_notebook)
        self.forms_notebook.add(self.bs_frame, text="Black-Scholes")
        self.setup_black_scholes_form()
        
        # Heston form
        self.heston_frame = ttk.Frame(self.forms_notebook)
        self.forms_notebook.add(self.heston_frame, text="Heston")
        self.setup_heston_form()
        
        # Dupire form
        self.dupire_frame = ttk.Frame(self.forms_notebook)
        self.forms_notebook.add(self.dupire_frame, text="Dupire")
        self.setup_dupire_form()
    
    def setup_black_scholes_form(self):
        """Setup Black-Scholes parameter form"""
        # Main frame
        main_frame = ttk.LabelFrame(self.bs_frame, text="Black-Scholes Parameters", padding="15")
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize variables
        self.bs_vars = {
            'S0': tk.DoubleVar(value=100.0),
            'K': tk.DoubleVar(value=100.0),
            'T': tk.DoubleVar(value=0.25),
            'r': tk.DoubleVar(value=0.05),
            'sigma': tk.DoubleVar(value=0.2),
            'option_type': tk.StringVar(value='call')
        }
        
        # Create form fields
        row = 0
        
        # Current Stock Price
        ttk.Label(main_frame, text="Current Stock Price (S₀):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.bs_vars['S0'], width=15).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(main_frame, text="$", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        # Strike Price
        ttk.Label(main_frame, text="Strike Price (K):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.bs_vars['K'], width=15).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(main_frame, text="$", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        # Time to Maturity
        ttk.Label(main_frame, text="Time to Maturity (T):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.bs_vars['T'], width=15).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(main_frame, text="years", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        # Risk-free Rate
        ttk.Label(main_frame, text="Risk-free Rate (r):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.bs_vars['r'], width=15).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(main_frame, text="decimal (e.g., 0.05 = 5%)", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        # Volatility
        ttk.Label(main_frame, text="Volatility (σ):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.bs_vars['sigma'], width=15).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(main_frame, text="decimal (e.g., 0.2 = 20%)", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        # Option Type
        ttk.Label(main_frame, text="Option Type:").grid(row=row, column=0, sticky='w', pady=5)
        type_frame = ttk.Frame(main_frame)
        type_frame.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Radiobutton(type_frame, text="Call", variable=self.bs_vars['option_type'], value='call').pack(side='left')
        ttk.Radiobutton(type_frame, text="Put", variable=self.bs_vars['option_type'], value='put').pack(side='left', padx=(10,0))
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Setup Model", command=self.apply_black_scholes).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Price Option", command=self.price_black_scholes).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_black_scholes).pack(side='left', padx=5)
        
        # Monte Carlo settings
        mc_frame = ttk.LabelFrame(self.bs_frame, text="Monte Carlo Settings", padding="10")
        mc_frame.pack(fill='x', padx=10, pady=5)
        
        self.bs_vars['num_simulations'] = tk.IntVar(value=100000)
        self.bs_vars['num_steps'] = tk.IntVar(value=252)
        
        ttk.Label(mc_frame, text="Number of Simulations:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(mc_frame, textvariable=self.bs_vars['num_simulations'], width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(mc_frame, text="Number of Steps:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(mc_frame, textvariable=self.bs_vars['num_steps'], width=15).grid(row=1, column=1, padx=5, pady=2)
    
    def setup_heston_form(self):
        """Setup Heston parameter form"""
        main_frame = ttk.LabelFrame(self.heston_frame, text="Heston Parameters", padding="15")
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize variables
        self.heston_vars = {
            'S0': tk.DoubleVar(value=100.0),
            'K': tk.DoubleVar(value=100.0),
            'T': tk.DoubleVar(value=0.25),
            'r': tk.DoubleVar(value=0.05),
            'v0': tk.DoubleVar(value=0.04),
            'kappa': tk.DoubleVar(value=2.0),
            'theta': tk.DoubleVar(value=0.04),
            'sigma_v': tk.DoubleVar(value=0.3),
            'rho': tk.DoubleVar(value=-0.5),
            'option_type': tk.StringVar(value='call')
        }
        
        # Create form fields
        row = 0
        
        # Basic parameters
        ttk.Label(main_frame, text="Current Stock Price (S₀):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['S0'], width=15).grid(row=row, column=1, padx=5, pady=3)
        row += 1
        
        ttk.Label(main_frame, text="Strike Price (K):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['K'], width=15).grid(row=row, column=1, padx=5, pady=3)
        row += 1
        
        ttk.Label(main_frame, text="Time to Maturity (T):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['T'], width=15).grid(row=row, column=1, padx=5, pady=3)
        row += 1
        
        ttk.Label(main_frame, text="Risk-free Rate (r):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['r'], width=15).grid(row=row, column=1, padx=5, pady=3)
        row += 1
        
        # Heston-specific parameters
        ttk.Label(main_frame, text="Initial Variance (v₀):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['v0'], width=15).grid(row=row, column=1, padx=5, pady=3)
        ttk.Label(main_frame, text="variance (σ²)", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        ttk.Label(main_frame, text="Mean Reversion Speed (κ):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['kappa'], width=15).grid(row=row, column=1, padx=5, pady=3)
        ttk.Label(main_frame, text="rate of mean reversion", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        ttk.Label(main_frame, text="Long-term Variance (θ):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['theta'], width=15).grid(row=row, column=1, padx=5, pady=3)
        ttk.Label(main_frame, text="long-term variance", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        ttk.Label(main_frame, text="Vol of Vol (σᵥ):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['sigma_v'], width=15).grid(row=row, column=1, padx=5, pady=3)
        ttk.Label(main_frame, text="volatility of volatility", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        ttk.Label(main_frame, text="Correlation (ρ):").grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(main_frame, textvariable=self.heston_vars['rho'], width=15).grid(row=row, column=1, padx=5, pady=3)
        ttk.Label(main_frame, text="between -1 and 1", foreground='gray').grid(row=row, column=2, sticky='w')
        row += 1
        
        # Option Type
        ttk.Label(main_frame, text="Option Type:").grid(row=row, column=0, sticky='w', pady=3)
        type_frame = ttk.Frame(main_frame)
        type_frame.grid(row=row, column=1, sticky='w', padx=5, pady=3)
        ttk.Radiobutton(type_frame, text="Call", variable=self.heston_vars['option_type'], value='call').pack(side='left')
        ttk.Radiobutton(type_frame, text="Put", variable=self.heston_vars['option_type'], value='put').pack(side='left', padx=(10,0))
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=15)
        
        ttk.Button(button_frame, text="Setup Model", command=self.apply_heston).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Price Option", command=self.price_heston).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_heston).pack(side='left', padx=5)
        
        # Feller condition check
        feller_frame = ttk.LabelFrame(self.heston_frame, text="Feller Condition Check", padding="10")
        feller_frame.pack(fill='x', padx=10, pady=5)
        
        self.feller_label = ttk.Label(feller_frame, text="Feller condition: 2κθ ≥ σᵥ²")
        self.feller_label.pack()
        
        ttk.Button(feller_frame, text="Check Feller Condition", command=self.check_feller_condition).pack(pady=5)
    
    def setup_dupire_form(self):
        """Setup Dupire parameter form"""
        main_frame = ttk.LabelFrame(self.dupire_frame, text="Dupire Parameters", padding="15")
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize variables
        self.dupire_vars = {
            'S0': tk.DoubleVar(value=100.0),
            'r': tk.DoubleVar(value=0.05),
            'dividend_yield': tk.DoubleVar(value=0.0),
            'K': tk.DoubleVar(value=100.0),
            'T': tk.DoubleVar(value=0.25),
            'option_type': tk.StringVar(value='call')
        }
        
        # Create form fields
        row = 0
        
        ttk.Label(main_frame, text="Current Stock Price (S₀):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.dupire_vars['S0'], width=15).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Risk-free Rate (r):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.dupire_vars['r'], width=15).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Dividend Yield (q):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.dupire_vars['dividend_yield'], width=15).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Strike Price (K):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.dupire_vars['K'], width=15).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        ttk.Label(main_frame, text="Time to Maturity (T):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(main_frame, textvariable=self.dupire_vars['T'], width=15).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        # Option Type
        ttk.Label(main_frame, text="Option Type:").grid(row=row, column=0, sticky='w', pady=5)
        type_frame = ttk.Frame(main_frame)
        type_frame.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Radiobutton(type_frame, text="Call", variable=self.dupire_vars['option_type'], value='call').pack(side='left')
        ttk.Radiobutton(type_frame, text="Put", variable=self.dupire_vars['option_type'], value='put').pack(side='left', padx=(10,0))
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=15)
        
        ttk.Button(button_frame, text="Setup Model", command=self.apply_dupire).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Price Option", command=self.price_dupire).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_dupire).pack(side='left', padx=5)
        
        # Local volatility surface info
        surface_frame = ttk.LabelFrame(self.dupire_frame, text="Local Volatility Surface", padding="10")
        surface_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(surface_frame, text="Note: Local volatility surface will be constructed from market data.").pack()
        ttk.Button(surface_frame, text="Construct Surface", command=self.construct_local_vol_surface).pack(pady=5)
    
    def apply_black_scholes(self):
        """Apply Black-Scholes parameters"""
        try:
            model = self.pricing_engine.setup_black_scholes(
                S0=self.bs_vars['S0'].get(),
                K=self.bs_vars['K'].get(),
                T=self.bs_vars['T'].get(),
                r=self.bs_vars['r'].get(),
                sigma=self.bs_vars['sigma'].get(),
                option_type=self.bs_vars['option_type'].get()
            )
            messagebox.showinfo("Success", "Black-Scholes model setup successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup Black-Scholes model: {e}")
    
    def apply_heston(self):
        """Apply Heston parameters"""
        try:
            # Validate Feller condition
            kappa = self.heston_vars['kappa'].get()
            theta = self.heston_vars['theta'].get()
            sigma_v = self.heston_vars['sigma_v'].get()
            
            if 2 * kappa * theta < sigma_v**2:
                response = messagebox.askyesno("Warning", 
                    "Feller condition not satisfied! This may lead to negative variance.\n"
                    "Do you want to continue anyway?")
                if not response:
                    return
            
            model = self.pricing_engine.setup_heston(
                S0=self.heston_vars['S0'].get(),
                K=self.heston_vars['K'].get(),
                T=self.heston_vars['T'].get(),
                r=self.heston_vars['r'].get(),
                v0=self.heston_vars['v0'].get(),
                kappa=self.heston_vars['kappa'].get(),
                theta=self.heston_vars['theta'].get(),
                sigma_v=self.heston_vars['sigma_v'].get(),
                rho=self.heston_vars['rho'].get(),
                option_type=self.heston_vars['option_type'].get()
            )
            messagebox.showinfo("Success", "Heston model setup successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup Heston model: {e}")
    
    def apply_dupire(self):
        """Apply Dupire parameters"""
        try:
            model = self.pricing_engine.setup_dupire(
                S0=self.dupire_vars['S0'].get(),
                r=self.dupire_vars['r'].get(),
                dividend_yield=self.dupire_vars['dividend_yield'].get()
            )
            
            # Set additional parameters for pricing
            model.K = self.dupire_vars['K'].get()
            model.T = self.dupire_vars['T'].get()
            model.option_type = self.dupire_vars['option_type'].get()
            
            messagebox.showinfo("Success", "Dupire model setup successfully!\nNote: Construct local volatility surface before pricing.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup Dupire model: {e}")
    
    def price_black_scholes(self):
        """Price option using Black-Scholes"""
        try:
            self.apply_black_scholes()
            model = self.pricing_engine.models['black_scholes']
            
            # Analytical price
            analytical_price = model.price()
            
            # Monte Carlo price
            mc_result = model.monte_carlo_price(
                num_simulations=self.bs_vars['num_simulations'].get(),
                num_steps=self.bs_vars['num_steps'].get()
            )
            
            # Display results
            result_text = f"""
Black-Scholes Pricing Results:

Analytical Price: ${analytical_price:.4f}
Monte Carlo Price: ${mc_result['price']:.4f}
Standard Error: ${mc_result['std_error']:.4f}
95% Confidence Interval: [${mc_result['confidence_interval'][0]:.4f}, ${mc_result['confidence_interval'][1]:.4f}]

Greeks:
Delta: {model.delta():.4f}
Gamma: {model.gamma():.4f}
Theta: {model.theta():.4f} (per day)
Vega: {model.vega():.4f}
Rho: {model.rho():.4f}
            """
            
            self.show_results_dialog("Black-Scholes Results", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Pricing failed: {e}")
    
    def price_heston(self):
        """Price option using Heston model"""
        try:
            self.apply_heston()
            model = self.pricing_engine.models['heston']
            
            # Monte Carlo price (primary method for Heston)
            mc_result = model.monte_carlo_price(num_simulations=50000)
            
            result_text = f"""
Heston Model Pricing Results:

Monte Carlo Price: ${mc_result['price']:.4f}
Standard Error: ${mc_result['std_error']:.4f}
95% Confidence Interval: [${mc_result['confidence_interval'][0]:.4f}, ${mc_result['confidence_interval'][1]:.4f}]

Greeks (finite difference):
Delta: {model.delta():.4f}
Gamma: {model.gamma():.4f}
Vega: {model.vega():.4f}
            """
            
            self.show_results_dialog("Heston Results", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Pricing failed: {e}")
    
    def price_dupire(self):
        """Price option using Dupire model"""
        try:
            if 'dupire' not in self.pricing_engine.models:
                messagebox.showwarning("Warning", "Please setup Dupire model first")
                return
            
            model = self.pricing_engine.models['dupire']
            
            if model.local_vol_surface is None:
                messagebox.showwarning("Warning", "Please construct local volatility surface first")
                return
            
            # Monte Carlo price
            mc_result = model.monte_carlo_price(
                K=self.dupire_vars['K'].get(),
                T=self.dupire_vars['T'].get(),
                option_type=self.dupire_vars['option_type'].get(),
                num_simulations=50000
            )
            
            result_text = f"""
Dupire Model Pricing Results:

Monte Carlo Price: ${mc_result['price']:.4f}
Standard Error: ${mc_result['std_error']:.4f}
95% Confidence Interval: [${mc_result['confidence_interval'][0]:.4f}, ${mc_result['confidence_interval'][1]:.4f}]
            """
            
            self.show_results_dialog("Dupire Results", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Pricing failed: {e}")
    
    def construct_local_vol_surface(self):
        """Construct local volatility surface"""
        try:
            if 'dupire' not in self.pricing_engine.models:
                messagebox.showwarning("Warning", "Please setup Dupire model first")
                return
            
            # Generate synthetic market data for demonstration
            S0 = self.dupire_vars['S0'].get()
            strikes = np.array([S0 * k for k in [0.8, 0.9, 1.0, 1.1, 1.2]])
            maturities = np.array([0.25, 0.5, 1.0])
            
            # Synthetic implied volatilities
            implied_vols = np.array([
                [0.25, 0.22, 0.20, 0.22, 0.25],
                [0.24, 0.21, 0.19, 0.21, 0.24],
                [0.23, 0.20, 0.18, 0.20, 0.23]
            ])
            
            market_data = {
                'strikes': strikes,
                'maturities': maturities,
                'implied_vols': implied_vols
            }
            
            model = self.pricing_engine.models['dupire']
            model.construct_local_vol_surface(market_data)
            
            messagebox.showinfo("Success", "Local volatility surface constructed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to construct surface: {e}")
    
    def check_feller_condition(self):
        """Check Feller condition for Heston model"""
        try:
            kappa = self.heston_vars['kappa'].get()
            theta = self.heston_vars['theta'].get()
            sigma_v = self.heston_vars['sigma_v'].get()
            
            feller_left = 2 * kappa * theta
            feller_right = sigma_v**2
            
            if feller_left >= feller_right:
                status = "✓ Satisfied"
                color = "green"
            else:
                status = "✗ Not Satisfied"
                color = "red"
            
            result_text = f"""
Feller Condition Check:

2κθ = 2 × {kappa} × {theta} = {feller_left:.4f}
σᵥ² = {sigma_v}² = {feller_right:.4f}

Condition: {status}

The Feller condition (2κθ ≥ σᵥ²) ensures that the variance process remains positive.
            """
            
            dialog = tk.Toplevel(self.parent)
            dialog.title("Feller Condition Check")
            dialog.geometry("400x250")
            
            text_widget = tk.Text(dialog, wrap='word', height=12)
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)
            text_widget.insert('1.0', result_text)
            text_widget.config(state='disabled')
            
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check Feller condition: {e}")
    
    def show_results_dialog(self, title, text):
        """Show results in a dialog"""
        dialog = tk.Toplevel(self.parent)
        dialog.title(title)
        dialog.geometry("500x400")
        
        text_widget = tk.Text(dialog, wrap='word', font=('Courier', 10))
        scrollbar = ttk.Scrollbar(dialog, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
        
        text_widget.insert('1.0', text)
        text_widget.config(state='disabled')
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
    
    def reset_black_scholes(self):
        """Reset Black-Scholes parameters to defaults"""
        self.bs_vars['S0'].set(100.0)
        self.bs_vars['K'].set(100.0)
        self.bs_vars['T'].set(0.25)
        self.bs_vars['r'].set(0.05)
        self.bs_vars['sigma'].set(0.2)
        self.bs_vars['option_type'].set('call')
        self.bs_vars['num_simulations'].set(100000)
        self.bs_vars['num_steps'].set(252)
    
    def reset_heston(self):
        """Reset Heston parameters to defaults"""
        self.heston_vars['S0'].set(100.0)
        self.heston_vars['K'].set(100.0)
        self.heston_vars['T'].set(0.25)
        self.heston_vars['r'].set(0.05)
        self.heston_vars['v0'].set(0.04)
        self.heston_vars['kappa'].set(2.0)
        self.heston_vars['theta'].set(0.04)
        self.heston_vars['sigma_v'].set(0.3)
        self.heston_vars['rho'].set(-0.5)
        self.heston_vars['option_type'].set('call')
    
    def reset_dupire(self):
        """Reset Dupire parameters to defaults"""
        self.dupire_vars['S0'].set(100.0)
        self.dupire_vars['r'].set(0.05)
        self.dupire_vars['dividend_yield'].set(0.0)
        self.dupire_vars['K'].set(100.0)
        self.dupire_vars['T'].set(0.25)
        self.dupire_vars['option_type'].set('call')
    
    def setup_black_scholes_from_market(self, market_data):
        """Setup Black-Scholes from market data"""
        self.bs_vars['S0'].set(market_data['current_price'])
        self.bs_vars['r'].set(market_data['risk_free_rate'])
        self.bs_vars['sigma'].set(market_data['volatility'])
        
        # Switch to Black-Scholes tab
        self.forms_notebook.select(self.bs_frame)
    
    def setup_heston_from_market(self, market_data):
        """Setup Heston from market data"""
        self.heston_vars['S0'].set(market_data['current_price'])
        self.heston_vars['r'].set(market_data['risk_free_rate'])
        
        # Set initial variance from historical volatility
        vol = market_data['volatility']
        self.heston_vars['v0'].set(vol**2)
        self.heston_vars['theta'].set(vol**2)
        
        # Switch to Heston tab
        self.forms_notebook.select(self.heston_frame)
    
    def setup_dupire_from_market(self, market_data):
        """Setup Dupire from market data"""
        self.dupire_vars['S0'].set(market_data['current_price'])
        self.dupire_vars['r'].set(market_data['risk_free_rate'])
        
        # Switch to Dupire tab
        self.forms_notebook.select(self.dupire_frame)