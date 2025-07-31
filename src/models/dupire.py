"""
Dupire Local Volatility Model Implementation
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d
from scipy.optimize import minimize
import pandas as pd

class DupireModel:
    """
    Dupire local volatility model for option pricing
    """
    
    def __init__(self, S0, r, dividend_yield=0):
        """
        Initialize Dupire model
        
        Args:
            S0 (float): Current stock price
            r (float): Risk-free rate
            dividend_yield (float): Dividend yield
        """
        self.S0 = S0
        self.r = r
        self.q = dividend_yield
        self.local_vol_surface = None
        self.strikes = None
        self.maturities = None
    
    def construct_local_vol_surface(self, market_data):
        """
        Construct local volatility surface from market data
        
        Args:
            market_data (dict): Market data with strikes, maturities, and implied vols
                Format: {'strikes': [...], 'maturities': [...], 'implied_vols': [[...]]}
        """
        strikes = np.array(market_data['strikes'])
        maturities = np.array(market_data['maturities'])
        implied_vols = np.array(market_data['implied_vols'])
        
        self.strikes = strikes
        self.maturities = maturities
        
        # Create mesh grid
        K_grid, T_grid = np.meshgrid(strikes, maturities)
        
        # Calculate option prices from implied volatilities
        prices = np.zeros_like(implied_vols)
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                sigma_iv = implied_vols[i, j]
                prices[i, j] = self._black_scholes_price(self.S0, K, T, self.r, sigma_iv, 'call')
        
        # Calculate derivatives using finite differences
        dC_dK = np.zeros_like(prices)
        d2C_dK2 = np.zeros_like(prices)
        dC_dT = np.zeros_like(prices)
        
        # First derivative with respect to strike
        for i in range(len(maturities)):
            for j in range(1, len(strikes) - 1):
                dK = strikes[j+1] - strikes[j-1]
                dC_dK[i, j] = (prices[i, j+1] - prices[i, j-1]) / dK
        
        # Second derivative with respect to strike
        for i in range(len(maturities)):
            for j in range(1, len(strikes) - 1):
                dK = strikes[1] - strikes[0]  # Assuming uniform grid
                d2C_dK2[i, j] = (prices[i, j+1] - 2*prices[i, j] + prices[i, j-1]) / (dK**2)
        
        # First derivative with respect to time
        for i in range(1, len(maturities) - 1):
            for j in range(len(strikes)):
                dT = maturities[i+1] - maturities[i-1]
                dC_dT[i, j] = (prices[i+1, j] - prices[i-1, j]) / dT
        
        # Calculate local volatility using Dupire formula
        local_vols = np.zeros_like(prices)
        
        for i in range(1, len(maturities) - 1):
            for j in range(1, len(strikes) - 1):
                K = strikes[j]
                T = maturities[i]
                
                numerator = dC_dT[i, j] + (self.r - self.q) * K * dC_dK[i, j] + self.q * prices[i, j]
                denominator = 0.5 * K**2 * d2C_dK2[i, j]
                
                if denominator > 1e-10:
                    local_vol_squared = 2 * numerator / denominator
                    if local_vol_squared > 0:
                        local_vols[i, j] = np.sqrt(local_vol_squared)
                    else:
                        local_vols[i, j] = 0.2  # Default volatility
                else:
                    local_vols[i, j] = 0.2
        
        # Handle boundary conditions
        local_vols[0, :] = local_vols[1, :]
        local_vols[-1, :] = local_vols[-2, :]
        local_vols[:, 0] = local_vols[:, 1]
        local_vols[:, -1] = local_vols[:, -2]
        
        # Create interpolation function
        self.local_vol_surface = RectBivariateSpline(maturities, strikes, local_vols, 
                                                   bbox=[maturities.min(), maturities.max(),
                                                         strikes.min(), strikes.max()],
                                                   kx=1, ky=1, s=0)
    
    def get_local_volatility(self, S, T):
        """
        Get local volatility at given spot and time
        
        Args:
            S (float or array): Stock price(s)
            T (float or array): Time(s) to maturity
            
        Returns:
            float or array: Local volatility
        """
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not constructed. Call construct_local_vol_surface first.")
        
        # Ensure inputs are within bounds
        T_bounded = np.clip(T, self.maturities.min(), self.maturities.max())
        S_bounded = np.clip(S, self.strikes.min(), self.strikes.max())
        
        return self.local_vol_surface(T_bounded, S_bounded, grid=False)
    
    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        """
        Calculate Black-Scholes price
        """
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-self.q * T) * norm.cdf(-d1)
        
        return price
    
    def monte_carlo_price(self, K, T, option_type='call', num_simulations=100000, num_steps=252):
        """
        Monte Carlo pricing using local volatility
        
        Args:
            K (float): Strike price
            T (float): Time to maturity
            option_type (str): 'call' or 'put'
            num_simulations (int): Number of simulations
            num_steps (int): Number of time steps
            
        Returns:
            dict: Pricing results
        """
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not constructed.")
        
        dt = T / num_steps
        
        # Initialize stock price paths
        S = np.zeros((num_simulations, num_steps + 1))
        S[:, 0] = self.S0
        
        # Generate random numbers
        dW = np.random.standard_normal((num_simulations, num_steps)) * np.sqrt(dt)
        
        # Simulate paths using local volatility
        for t in range(num_steps):
            current_time = (t + 1) * dt
            remaining_time = T - current_time
            
            if remaining_time > 0:
                # Get local volatility for current stock prices and time
                local_vols = np.array([self.get_local_volatility(s, remaining_time) 
                                     for s in S[:, t]])
                
                # Update stock prices
                S[:, t + 1] = S[:, t] * np.exp((self.r - self.q - 0.5 * local_vols**2) * dt + 
                                              local_vols * dW[:, t])
            else:
                S[:, t + 1] = S[:, t]
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
        
        # Discount and calculate price
        price = np.exp(-self.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'stock_paths': S,
            'payoffs': payoffs
        }
    
    def delta(self, K, T, option_type='call', bump_size=0.01):
        """
        Calculate Delta using finite difference
        """
        original_S0 = self.S0
        
        # Bump up
        self.S0 = original_S0 * (1 + bump_size)
        price_up = self.monte_carlo_price(K, T, option_type, num_simulations=50000)['price']
        
        # Bump down
        self.S0 = original_S0 * (1 - bump_size)
        price_down = self.monte_carlo_price(K, T, option_type, num_simulations=50000)['price']
        
        # Restore
        self.S0 = original_S0
        
        return (price_up - price_down) / (2 * original_S0 * bump_size)
    
    def gamma(self, K, T, option_type='call', bump_size=0.01):
        """
        Calculate Gamma using finite difference
        """
        original_S0 = self.S0
        
        # Center
        price_center = self.monte_carlo_price(K, T, option_type, num_simulations=50000)['price']
        
        # Bump up
        self.S0 = original_S0 * (1 + bump_size)
        price_up = self.monte_carlo_price(K, T, option_type, num_simulations=50000)['price']
        
        # Bump down
        self.S0 = original_S0 * (1 - bump_size)
        price_down = self.monte_carlo_price(K, T, option_type, num_simulations=50000)['price']
        
        # Restore
        self.S0 = original_S0
        
        bump_abs = original_S0 * bump_size
        return (price_up - 2 * price_center + price_down) / (bump_abs**2)
    
    def theta(self, K, T, option_type='call', bump_size=1/365):
        """
        Calculate Theta using finite difference
        """
        if T <= bump_size:
            return 0
        
        # Current price
        price_current = self.monte_carlo_price(K, T, option_type, num_simulations=50000)['price']
        
        # Price with time decay
        price_decay = self.monte_carlo_price(K, T - bump_size, option_type, num_simulations=50000)['price']
        
        return (price_decay - price_current) / bump_size
    
    def calibrate_to_market(self, market_data, lambda_smooth=0.01, max_iterations=100):
        """
        Calibrate local volatility surface to market data
        
        Args:
            market_data (dict): Market option data
            lambda_smooth (float): Smoothing parameter
            max_iterations (int): Maximum iterations
            
        Returns:
            dict: Calibration results
        """
        def objective(params):
            # Reshape parameters to volatility surface
            vol_surface = params.reshape(len(self.maturities), len(self.strikes))
            
            error = 0
            count = 0
            
            # Calculate pricing error
            for i, T in enumerate(self.maturities):
                for j, K in enumerate(self.strikes):
                    if 'market_prices' in market_data:
                        market_price = market_data['market_prices'][i][j]
                        
                        # Temporary local vol surface
                        temp_surface = RectBivariateSpline(self.maturities, self.strikes, vol_surface)
                        self.local_vol_surface = temp_surface
                        
                        try:
                            model_price = self.monte_carlo_price(K, T, 'call', num_simulations=10000)['price']
                            error += (model_price - market_price)**2
                            count += 1
                        except:
                            error += 1e6
            
            # Add smoothing penalty
            smoothing_penalty = lambda_smooth * np.sum(np.diff(vol_surface, axis=0)**2) + \
                              lambda_smooth * np.sum(np.diff(vol_surface, axis=1)**2)
            
            return error / max(count, 1) + smoothing_penalty
        
        # Initial guess
        initial_vol = 0.2 * np.ones((len(self.maturities), len(self.strikes)))
        initial_params = initial_vol.flatten()
        
        # Bounds
        bounds = [(0.01, 2.0) for _ in range(len(initial_params))]
        
        # Optimize
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': max_iterations})
        
        if result.success:
            optimal_surface = result.x.reshape(len(self.maturities), len(self.strikes))
            self.local_vol_surface = RectBivariateSpline(self.maturities, self.strikes, optimal_surface)
            
            return {
                'success': True,
                'error': result.fun,
                'surface': optimal_surface,
                'message': result.message
            }
        else:
            return {
                'success': False,
                'error': result.fun,
                'message': result.message
            }
    
    def get_volatility_smile(self, T, strikes=None):
        """
        Get volatility smile for a given maturity
        
        Args:
            T (float): Time to maturity
            strikes (array): Strike prices (optional)
            
        Returns:
            dict: Volatility smile data
        """
        if strikes is None:
            strikes = self.strikes
        
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not constructed.")
        
        local_vols = [self.get_local_volatility(K, T) for K in strikes]
        
        return {
            'strikes': strikes,
            'maturities': T,
            'local_volatilities': local_vols
        }
    
    def get_term_structure(self, K, maturities=None):
        """
        Get volatility term structure for a given strike
        
        Args:
            K (float): Strike price
            maturities (array): Times to maturity (optional)
            
        Returns:
            dict: Term structure data
        """
        if maturities is None:
            maturities = self.maturities
        
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not constructed.")
        
        local_vols = [self.get_local_volatility(K, T) for T in maturities]
        
        return {
            'strikes': K,
            'maturities': maturities,
            'local_volatilities': local_vols
        }