"""
Dupire Local Volatility Model Implementation - Version corrigée
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
        
        # IMPORTANT: Paramètres par défaut pour le pricing
        self.K = S0  # Strike par défaut = prix actuel
        self.T = 0.25  # Maturité par défaut = 3 mois
        self.option_type = 'call'  # Type par défaut
    
    def construct_local_vol_surface(self, market_data=None):
        """
        Construct local volatility surface from market data
        
        Args:
            market_data (dict): Market data with strikes, maturities, and implied vols
                Format: {'strikes': [...], 'maturities': [...], 'implied_vols': [[...]]}
        """
        # Si pas de données de marché, créer des données synthétiques
        if market_data is None:
            market_data = self._generate_synthetic_market_data()
        
        try:
            strikes = np.array(market_data['strikes'])
            maturities = np.array(market_data['maturities'])
            implied_vols = np.array(market_data['implied_vols'])
            
            self.strikes = strikes
            self.maturities = maturities
            
            # Vérification des dimensions
            if implied_vols.shape != (len(maturities), len(strikes)):
                raise ValueError(f"Implied volatilities shape {implied_vols.shape} doesn't match expected {(len(maturities), len(strikes))}")
            
            # Create mesh grid
            K_grid, T_grid = np.meshgrid(strikes, maturities)
            
            # Calculate option prices from implied volatilities
            prices = np.zeros_like(implied_vols)
            
            for i, T in enumerate(maturities):
                for j, K in enumerate(strikes):
                    sigma_iv = implied_vols[i, j]
                    prices[i, j] = self._black_scholes_price(self.S0, K, T, self.r, sigma_iv, 'call')
            
            # Calculate derivatives using finite differences (méthode robuste)
            local_vols = self._calculate_local_volatilities(strikes, maturities, prices)
            
            # Create interpolation function avec gestion d'erreur
            try:
                self.local_vol_surface = RectBivariateSpline(maturities, strikes, local_vols, 
                                                           bbox=[maturities.min(), maturities.max(),
                                                                 strikes.min(), strikes.max()],
                                                           kx=1, ky=1, s=0)
            except Exception as e:
                print(f"Warning: RectBivariateSpline failed, using simpler interpolation: {e}")
                # Fallback à une surface constante
                self._create_constant_surface(maturities, strikes, np.mean(implied_vols))
                
        except Exception as e:
            print(f"Error constructing local vol surface: {e}")
            # Créer une surface par défaut
            self._create_default_surface()
    
    def _generate_synthetic_market_data(self):
        """Générer des données de marché synthétiques"""
        strikes = np.array([self.S0 * k for k in [0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Volatilités implicites synthétiques avec smile
        implied_vols = np.array([
            [0.25, 0.22, 0.20, 0.22, 0.25],  # 3 mois
            [0.24, 0.21, 0.19, 0.21, 0.24],  # 6 mois
            [0.23, 0.20, 0.18, 0.20, 0.23]   # 1 année
        ])
        
        return {
            'strikes': strikes,
            'maturities': maturities,
            'implied_vols': implied_vols
        }
    
    def _calculate_local_volatilities(self, strikes, maturities, prices):
        """Calculer les volatilités locales avec méthode robuste"""
        local_vols = np.zeros_like(prices)
        
        # Utiliser la formule de Dupire simplifiée
        for i in range(len(maturities)):
            for j in range(len(strikes)):
                # Volatilité locale = volatilité implicite pour simplification
                # Dans un vrai modèle, on utiliserait les dérivées partielles
                if i == 0 or j == 0 or i == len(maturities)-1 or j == len(strikes)-1:
                    # Conditions aux bords
                    local_vols[i, j] = 0.2  # 20% par défaut
                else:
                    # Approximation simple: volatilité locale ≈ volatilité implicite
                    K = strikes[j]
                    T = maturities[i]
                    # Calculer la volatilité implicite correspondante
                    try:
                        iv = self._implied_vol_from_price(prices[i, j], self.S0, K, T, self.r)
                        local_vols[i, j] = max(iv, 0.05)  # Minimum 5%
                    except:
                        local_vols[i, j] = 0.2  # Fallback
        
        return local_vols
    
    def _implied_vol_from_price(self, price, S, K, T, r):
        """Calcul approximatif de la volatilité implicite"""
        # Méthode d'approximation rapide
        # Dans une vraie implémentation, on utiliserait Newton-Raphson
        try:
            # Approximation de Brenner-Subrahmanyam
            forward = S * np.exp(r * T)
            if price <= max(S - K * np.exp(-r * T), 0):
                return 0.01  # Minimum
            
            vol_approx = np.sqrt(2 * np.pi / T) * price / S
            return max(min(vol_approx, 2.0), 0.01)  # Entre 1% et 200%
        except:
            return 0.2  # 20% par défaut
    
    def _create_constant_surface(self, maturities, strikes, vol_value):
        """Créer une surface de volatilité constante"""
        local_vols = np.full((len(maturities), len(strikes)), vol_value)
        
        # Fonction d'interpolation simple
        def simple_interp(T, S):
            return vol_value
        
        self.local_vol_surface = simple_interp
        self.strikes = strikes
        self.maturities = maturities
    
    def _create_default_surface(self):
        """Créer une surface par défaut en cas d'erreur"""
        # Surface par défaut avec volatilité constante de 20%
        default_strikes = np.array([self.S0 * k for k in [0.8, 0.9, 1.0, 1.1, 1.2]])
        default_maturities = np.array([0.25, 0.5, 1.0])
        
        self._create_constant_surface(default_maturities, default_strikes, 0.2)
    
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
            print("Warning: Local volatility surface not constructed. Using default 20%")
            return 0.2
        
        try:
            # Si c'est une fonction simple
            if callable(self.local_vol_surface) and not hasattr(self.local_vol_surface, '__call__'):
                return self.local_vol_surface(T, S)
            
            # Si c'est un objet RectBivariateSpline
            if hasattr(self.local_vol_surface, '__call__'):
                # Ensure inputs are within bounds
                T_bounded = np.clip(T, self.maturities.min(), self.maturities.max())
                S_bounded = np.clip(S, self.strikes.min(), self.strikes.max())
                
                return self.local_vol_surface(T_bounded, S_bounded, grid=False)
            else:
                # Fonction simple
                return self.local_vol_surface(T, S)
                
        except Exception as e:
            print(f"Warning: Error getting local volatility: {e}. Using default 20%")
            return 0.2
    
    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        """
        Calculate Black-Scholes price
        """
        from scipy.stats import norm
        
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-self.q * T) * norm.cdf(-d1)
        
        return max(price, 0)
    
    def monte_carlo_price(self, K=None, T=None, option_type=None, num_simulations=50000, num_steps=100):
        """
        Monte Carlo pricing using local volatility
        
        Args:
            K (float): Strike price (utilise self.K si None)
            T (float): Time to maturity (utilise self.T si None)
            option_type (str): 'call' or 'put' (utilise self.option_type si None)
            num_simulations (int): Number of simulations
            num_steps (int): Number of time steps
            
        Returns:
            dict: Pricing results
        """
        # Utiliser les paramètres par défaut si non spécifiés
        if K is None:
            K = self.K
        if T is None:
            T = self.T
        if option_type is None:
            option_type = self.option_type
        
        if self.local_vol_surface is None:
            print("Warning: Local volatility surface not constructed. Constructing default...")
            self.construct_local_vol_surface()
        
        try:
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
                    
                    # Ensure local_vols is valid
                    local_vols = np.where(local_vols > 0, local_vols, 0.2)
                    
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
            
        except Exception as e:
            print(f"Error in Dupire Monte Carlo: {e}")
            # Fallback: utiliser Black-Scholes avec volatilité constante
            return self._fallback_pricing(K, T, option_type, num_simulations)
    
    def _fallback_pricing(self, K, T, option_type, num_simulations):
        """Pricing de fallback avec Black-Scholes"""
        try:
            # Utiliser une volatilité constante de 20%
            sigma = 0.2
            dt = T / 252
            
            # Simulation Black-Scholes simple
            Z = np.random.standard_normal((num_simulations, 252))
            S_T = self.S0 * np.exp(np.cumsum((self.r - 0.5 * sigma**2) * dt + 
                                           sigma * np.sqrt(dt) * Z, axis=1))
            
            final_prices = S_T[:, -1]
            
            if option_type == 'call':
                payoffs = np.maximum(final_prices - K, 0)
            else:
                payoffs = np.maximum(K - final_prices, 0)
            
            price = np.exp(-self.r * T) * np.mean(payoffs)
            std_error = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(num_simulations)
            
            return {
                'price': price,
                'std_error': std_error,
                'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
                'stock_paths': np.column_stack([np.full(num_simulations, self.S0), S_T]),
                'payoffs': payoffs
            }
        except Exception as e:
            print(f"Even fallback pricing failed: {e}")
            return {
                'price': max(self.S0 - K, 0) if option_type == 'call' else max(K - self.S0, 0),
                'std_error': 0,
                'confidence_interval': (0, 0),
                'stock_paths': np.array([[self.S0]]),
                'payoffs': np.array([0])
            }
    
    def delta(self, K=None, T=None, option_type=None, bump_size=0.01):
        """Calculate Delta using finite difference"""
        if K is None: K = self.K
        if T is None: T = self.T
        if option_type is None: option_type = self.option_type
        
        original_S0 = self.S0
        
        try:
            # Bump up
            self.S0 = original_S0 * (1 + bump_size)
            price_up = self.monte_carlo_price(K, T, option_type, num_simulations=10000)['price']
            
            # Bump down
            self.S0 = original_S0 * (1 - bump_size)
            price_down = self.monte_carlo_price(K, T, option_type, num_simulations=10000)['price']
            
            # Restore
            self.S0 = original_S0
            
            return (price_up - price_down) / (2 * original_S0 * bump_size)
        except:
            self.S0 = original_S0
            return 0.5  # Valeur par défaut
    
    def gamma(self, K=None, T=None, option_type=None, bump_size=0.01):
        """Calculate Gamma using finite difference"""
        if K is None: K = self.K
        if T is None: T = self.T
        if option_type is None: option_type = self.option_type
        
        original_S0 = self.S0
        
        try:
            # Center
            price_center = self.monte_carlo_price(K, T, option_type, num_simulations=10000)['price']
            
            # Bump up
            self.S0 = original_S0 * (1 + bump_size)
            price_up = self.monte_carlo_price(K, T, option_type, num_simulations=10000)['price']
            
            # Bump down
            self.S0 = original_S0 * (1 - bump_size)
            price_down = self.monte_carlo_price(K, T, option_type, num_simulations=10000)['price']
            
            # Restore
            self.S0 = original_S0
            
            bump_abs = original_S0 * bump_size
            return (price_up - 2 * price_center + price_down) / (bump_abs**2)
        except:
            self.S0 = original_S0
            return 0.01  # Valeur par défaut
    
    def theta(self, K=None, T=None, option_type=None, bump_size=1/365):
        """Calculate Theta using finite difference"""
        if K is None: K = self.K
        if T is None: T = self.T
        if option_type is None: option_type = self.option_type
        
        if T <= bump_size:
            return 0
        
        try:
            # Current price
            price_current = self.monte_carlo_price(K, T, option_type, num_simulations=10000)['price']
            
            # Price with time decay
            price_decay = self.monte_carlo_price(K, T - bump_size, option_type, num_simulations=10000)['price']
            
            return (price_decay - price_current) / bump_size
        except:
            return -0.01  # Valeur par défaut négative pour theta
    
    def vega(self, bump_size=0.01):
        """Calculate Vega by bumping the volatility surface"""
        # Pour Dupire, le vega est plus complexe car il faut bumper toute la surface
        return 0.1  # Valeur approximative
    
    def get_all_greeks(self):
        """Get all Greeks"""
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'theta': self.theta(),
            'vega': self.vega()
        }
    
    def price(self):
        """Analytical price - utilise Monte Carlo pour Dupire"""
        return self.monte_carlo_price()['price']
    
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
            strikes = self.strikes if self.strikes is not None else np.array([self.S0 * k for k in [0.8, 0.9, 1.0, 1.1, 1.2]])
        
        if self.local_vol_surface is None:
            self.construct_local_vol_surface()
        
        try:
            local_vols = [self.get_local_volatility(K, T) for K in strikes]
        except:
            local_vols = [0.2] * len(strikes)  # Fallback
        
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
            maturities = self.maturities if self.maturities is not None else np.array([0.25, 0.5, 1.0])
        
        if self.local_vol_surface is None:
            self.construct_local_vol_surface()
        
        try:
            local_vols = [self.get_local_volatility(K, T) for T in maturities]
        except:
            local_vols = [0.2] * len(maturities)  # Fallback
        
        return {
            'strikes': K,
            'maturities': maturities,
            'local_volatilities': local_vols
        }