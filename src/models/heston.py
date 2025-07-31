"""
Heston Stochastic Volatility Model Implementation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import math

class HestonModel:
    """
    Heston model for option pricing with stochastic volatility
    """
    
    def __init__(self, S0, K, T, r, v0, kappa, theta, sigma_v, rho, option_type='call'):
        """
        Initialize Heston model parameters
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            v0 (float): Initial variance
            kappa (float): Mean reversion speed
            theta (float): Long-term variance
            sigma_v (float): Volatility of volatility
            rho (float): Correlation between stock and volatility
            option_type (str): 'call' or 'put'
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.option_type = option_type.lower()
    
    def characteristic_function(self, u, j):
        """
        Heston characteristic function
        
        Args:
            u (complex): Integration variable
            j (int): 1 or 2 for different formulations
            
        Returns:
            complex: Characteristic function value
        """
        if j == 1:
            b = self.kappa - self.rho * self.sigma_v
        else:
            b = self.kappa
        
        a = self.kappa * self.theta
        
        d = np.sqrt((self.rho * self.sigma_v * u * 1j - b)**2 - 
                   self.sigma_v**2 * (2 * u * 1j - u**2))
        
        g = (b - self.rho * self.sigma_v * u * 1j + d) / \
            (b - self.rho * self.sigma_v * u * 1j - d)
        
        exp_term = np.exp(-d * self.T)
        
        C = self.r * u * 1j * self.T + \
            (a / self.sigma_v**2) * \
            ((b - self.rho * self.sigma_v * u * 1j + d) * self.T - 
             2 * np.log((1 - g * exp_term) / (1 - g)))
        
        D = (b - self.rho * self.sigma_v * u * 1j + d) / self.sigma_v**2 * \
            ((1 - exp_term) / (1 - g * exp_term))
        
        return np.exp(C + D * self.v0 + 1j * u * np.log(self.S0))
    
    def P_function(self, j):
        """
        Calculate probability function P_j
        
        Args:
            j (int): 1 or 2
            
        Returns:
            float: Probability
        """
        def integrand(u):
            cf = self.characteristic_function(u - 1j, j)
            return np.real(np.exp(-1j * u * np.log(self.K)) * cf / (1j * u))
        
        integral, _ = quad(integrand, 1e-8, 100)
        return 0.5 + (1/np.pi) * integral
    
    def price(self):
        """
        Calculate Heston option price using semi-analytical formula
        
        Returns:
            float: Option price
        """
        try:
            P1 = self.P_function(1)
            P2 = self.P_function(2)
            
            if self.option_type == 'call':
                price = self.S0 * P1 - self.K * np.exp(-self.r * self.T) * P2
            else:  # put
                price = self.K * np.exp(-self.r * self.T) * (1 - P2) - self.S0 * (1 - P1)
            
            return max(price, 0)
        except:
            # Fallback to Monte Carlo if semi-analytical fails
            return self.monte_carlo_price()['price']
    
    def monte_carlo_price(self, num_simulations=50000, num_steps=252):
        """
        Monte Carlo pricing for Heston model
        
        Args:
            num_simulations (int): Number of simulations
            num_steps (int): Number of time steps
            
        Returns:
            dict: Pricing results
        """
        dt = self.T / num_steps
        
        # Initialize arrays
        S = np.zeros((num_simulations, num_steps + 1))
        v = np.zeros((num_simulations, num_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        # Generate correlated random numbers
        for t in range(num_steps):
            Z1 = np.random.standard_normal(num_simulations)
            Z2 = np.random.standard_normal(num_simulations)
            
            # Correlated Brownian motions
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Ensure variance stays positive (Full Truncation scheme)
            v_pos = np.maximum(v[:, t], 0)
            
            # Update variance using Euler scheme
            v[:, t + 1] = v[:, t] + self.kappa * (self.theta - v_pos) * dt + \
                         self.sigma_v * np.sqrt(v_pos * dt) * W2
            
            # Update stock price
            S[:, t + 1] = S[:, t] * np.exp((self.r - 0.5 * v_pos) * dt + 
                                          np.sqrt(v_pos * dt) * W1)
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(S[:, -1] - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S[:, -1], 0)
        
        # Discount and calculate price
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'stock_paths': S,
            'variance_paths': v,
            'payoffs': payoffs
        }
    
    def delta(self, bump_size=0.01):
        """
        Calculate Delta using finite difference
        
        Args:
            bump_size (float): Size of the bump
            
        Returns:
            float: Delta
        """
        original_S0 = self.S0
        
        # Bump up
        self.S0 = original_S0 * (1 + bump_size)
        price_up = self.price()
        
        # Bump down
        self.S0 = original_S0 * (1 - bump_size)
        price_down = self.price()
        
        # Restore original value
        self.S0 = original_S0
        
        return (price_up - price_down) / (2 * original_S0 * bump_size)
    
    def gamma(self, bump_size=0.01):
        """
        Calculate Gamma using finite difference
        
        Args:
            bump_size (float): Size of the bump
            
        Returns:
            float: Gamma
        """
        original_S0 = self.S0
        
        # Center price
        price_center = self.price()
        
        # Bump up
        self.S0 = original_S0 * (1 + bump_size)
        price_up = self.price()
        
        # Bump down
        self.S0 = original_S0 * (1 - bump_size)
        price_down = self.price()
        
        # Restore original value
        self.S0 = original_S0
        
        bump_abs = original_S0 * bump_size
        return (price_up - 2 * price_center + price_down) / (bump_abs**2)
    
    def vega(self, bump_size=0.01):
        """
        Calculate Vega (sensitivity to initial volatility)
        
        Args:
            bump_size (float): Size of the bump
            
        Returns:
            float: Vega
        """
        original_v0 = self.v0
        
        # Bump up
        self.v0 = original_v0 * (1 + bump_size)
        price_up = self.price()
        
        # Bump down
        self.v0 = original_v0 * (1 - bump_size)
        price_down = self.price()
        
        # Restore original value
        self.v0 = original_v0
        
        return (price_up - price_down) / (2 * np.sqrt(original_v0) * bump_size) / 100
    
    def calibrate_to_market(self, market_prices, strikes, maturities, initial_guess=None):
        """
        Calibrate Heston parameters to market prices
        
        Args:
            market_prices (list): Market option prices
            strikes (list): Strike prices
            maturities (list): Times to maturity
            initial_guess (list): Initial parameter guess [v0, kappa, theta, sigma_v, rho]
            
        Returns:
            dict: Calibration results
        """
        if initial_guess is None:
            initial_guess = [0.04, 2.0, 0.04, 0.3, -0.5]
        
        def objective(params):
            v0, kappa, theta, sigma_v, rho = params
            
            # Parameter constraints
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0:
                return 1e6
            if abs(rho) >= 1:
                return 1e6
            if 2 * kappa * theta <= sigma_v**2:  # Feller condition
                return 1e6
            
            error = 0
            original_params = (self.K, self.T, self.v0, self.kappa, 
                             self.theta, self.sigma_v, self.rho)
            
            try:
                for i, (market_price, K, T) in enumerate(zip(market_prices, strikes, maturities)):
                    self.K = K
                    self.T = T
                    self.v0 = v0
                    self.kappa = kappa
                    self.theta = theta
                    self.sigma_v = sigma_v
                    self.rho = rho
                    
                    model_price = self.monte_carlo_price(num_simulations=10000)['price']
                    error += (model_price - market_price)**2
                
                # Restore original parameters
                (self.K, self.T, self.v0, self.kappa, 
                 self.theta, self.sigma_v, self.rho) = original_params
                
                return error
            except:
                return 1e6
        
        # Bounds for parameters
        bounds = [(1e-6, 1), (1e-6, 10), (1e-6, 1), (1e-6, 2), (-0.99, 0.99)]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            v0_opt, kappa_opt, theta_opt, sigma_v_opt, rho_opt = result.x
            return {
                'success': True,
                'v0': v0_opt,
                'kappa': kappa_opt,
                'theta': theta_opt,
                'sigma_v': sigma_v_opt,
                'rho': rho_opt,
                'error': result.fun,
                'message': result.message
            }
        else:
            return {
                'success': False,
                'error': result.fun,
                'message': result.message
            }
    
    def get_all_greeks(self):
        """
        Get all Greeks
        
        Returns:
            dict: All Greeks
        """
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega()
        }