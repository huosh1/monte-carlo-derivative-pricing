"""
Black-Scholes Model Implementation
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import math

class BlackScholesModel:
    """
    Black-Scholes model for option pricing and Greeks calculation
    """
    
    def __init__(self, S0, K, T, r, sigma, option_type='call'):
        """
        Initialize Black-Scholes model parameters
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
    
    def _d1(self):
        """Calculate d1 parameter"""
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def _d2(self):
        """Calculate d2 parameter"""
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def price(self):
        """
        Calculate option price using Black-Scholes formula
        
        Returns:
            float: Option price
        """
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        
        return price
    
    def delta(self):
        """
        Calculate Delta (price sensitivity to underlying price)
        
        Returns:
            float: Delta
        """
        d1 = self._d1()
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    def gamma(self):
        """
        Calculate Gamma (Delta sensitivity to underlying price)
        
        Returns:
            float: Gamma
        """
        d1 = self._d1()
        return norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
    
    def theta(self):
        """
        Calculate Theta (price sensitivity to time decay)
        
        Returns:
            float: Theta (per day)
        """
        d1 = self._d1()
        d2 = self._d2()
        
        theta_term1 = -(self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if self.option_type == 'call':
            theta_term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            theta = theta_term1 + theta_term2
        else:  # put
            theta_term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            theta = theta_term1 + theta_term2
        
        return theta / 365  # Convert to per day
    
    def vega(self):
        """
        Calculate Vega (price sensitivity to volatility)
        
        Returns:
            float: Vega
        """
        d1 = self._d1()
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T) / 100  # Per 1% change in volatility
    
    def rho(self):
        """
        Calculate Rho (price sensitivity to interest rate)
        
        Returns:
            float: Rho
        """
        d2 = self._d2()
        
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:  # put
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    def monte_carlo_price(self, num_simulations=100000, num_steps=252):
        """
        Calculate option price using Monte Carlo simulation
        
        Args:
            num_simulations (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps
            
        Returns:
            dict: Price and confidence interval
        """
        dt = self.T / num_steps
        
        # Generate random paths
        Z = np.random.standard_normal((num_simulations, num_steps))
        
        # Initialize price paths
        S = np.zeros((num_simulations, num_steps + 1))
        S[:, 0] = self.S0
        
        # Generate stock price paths
        for t in range(1, num_steps + 1):
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + 
                                        self.sigma * np.sqrt(dt) * Z[:, t-1])
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(S[:, -1] - self.K, 0)
        else:  # put
            payoffs = np.maximum(self.K - S[:, -1], 0)
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        # Calculate price and confidence interval
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        confidence_interval = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': confidence_interval,
            'paths': S,
            'payoffs': payoffs
        }
    
    def implied_volatility(self, market_price, max_iterations=100, tolerance=1e-6):
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price (float): Market price of the option
            max_iterations (int): Maximum iterations
            tolerance (float): Convergence tolerance
            
        Returns:
            float: Implied volatility
        """
        def objective(sigma):
            self.sigma = sigma
            return (self.price() - market_price)**2
        
        # Use Brent's method for robust optimization
        result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
        
        if result.success:
            return result.x
        else:
            return None
    
    def get_all_greeks(self):
        """
        Get all Greeks in a dictionary
        
        Returns:
            dict: All Greeks
        """
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'theta': self.theta(),
            'vega': self.vega(),
            'rho': self.rho()
        }
    
    def sensitivity_analysis(self, parameter, range_pct=0.2, num_points=21):
        """
        Perform sensitivity analysis on a parameter
        
        Args:
            parameter (str): Parameter name ('S0', 'K', 'T', 'r', 'sigma')
            range_pct (float): Range as percentage of current value
            num_points (int): Number of points in the analysis
            
        Returns:
            dict: Sensitivity analysis results
        """
        original_value = getattr(self, parameter)
        
        # Create range of values
        min_val = original_value * (1 - range_pct)
        max_val = original_value * (1 + range_pct)
        values = np.linspace(min_val, max_val, num_points)
        
        prices = []
        deltas = []
        gammas = []
        
        for value in values:
            # Temporarily change parameter
            setattr(self, parameter, value)
            
            # Calculate metrics
            prices.append(self.price())
            deltas.append(self.delta())
            gammas.append(self.gamma())
        
        # Restore original value
        setattr(self, parameter, original_value)
        
        return {
            'parameter': parameter,
            'values': values,
            'prices': prices,
            'deltas': deltas,
            'gammas': gammas,
            'original_value': original_value
        }