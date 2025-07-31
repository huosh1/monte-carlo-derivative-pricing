"""
Pricing Engine - Orchestrates all pricing models
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json

from ..models.black_scholes import BlackScholesModel
from ..models.heston import HestonModel
from ..models.dupire import DupireModel

class PricingEngine:
    """
    Main pricing engine that coordinates all models
    """
    
    def __init__(self):
        """Initialize pricing engine"""
        self.models = {}
        self.results = {}
        self.market_data = None
    
    def set_market_data(self, market_data):
        """
        Set market data for pricing
        
        Args:
            market_data (dict): Market data dictionary
        """
        self.market_data = market_data
    
    def setup_black_scholes(self, S0, K, T, r, sigma, option_type='call'):
        """
        Setup Black-Scholes model
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): Option type
        """
        self.models['black_scholes'] = BlackScholesModel(S0, K, T, r, sigma, option_type)
        return self.models['black_scholes']
    
    def setup_heston(self, S0, K, T, r, v0, kappa, theta, sigma_v, rho, option_type='call'):
        """
        Setup Heston model
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            v0 (float): Initial variance
            kappa (float): Mean reversion speed
            theta (float): Long-term variance
            sigma_v (float): Volatility of volatility
            rho (float): Correlation
            option_type (str): Option type
        """
        self.models['heston'] = HestonModel(S0, K, T, r, v0, kappa, theta, sigma_v, rho, option_type)
        return self.models['heston']
    
    def setup_dupire(self, S0, r, dividend_yield=0):
        """
        Setup Dupire model
        
        Args:
            S0 (float): Current stock price
            r (float): Risk-free rate
            dividend_yield (float): Dividend yield
        """
        self.models['dupire'] = DupireModel(S0, r, dividend_yield)
        return self.models['dupire']
    
    def price_all_models(self, use_monte_carlo=False):
        """
        Price option using all available models
        
        Args:
            use_monte_carlo (bool): Use Monte Carlo for all models
            
        Returns:
            dict: Pricing results from all models
        """
        results = {}
        
        for model_name, model in self.models.items():
            try:
                if use_monte_carlo or model_name == 'dupire':
                    if model_name == 'black_scholes':
                        result = model.monte_carlo_price()
                    elif model_name == 'heston':
                        result = model.monte_carlo_price()
                    elif model_name == 'dupire':
                        # For Dupire, we need K and T parameters
                        if hasattr(model, 'K') and hasattr(model, 'T'):
                            result = model.monte_carlo_price(model.K, model.T)
                        else:
                            result = {'price': 'N/A', 'error': 'Missing parameters'}
                else:
                    result = {'price': model.price()}
                
                results[model_name] = result
                
            except Exception as e:
                results[model_name] = {'price': 'Error', 'error': str(e)}
        
        self.results = results
        return results
    
    def calculate_greeks_all_models(self):
        """
        Calculate Greeks for all models
        
        Returns:
            dict: Greeks from all models
        """
        greeks_results = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'get_all_greeks'):
                    greeks = model.get_all_greeks()
                    greeks_results[model_name] = greeks
                else:
                    # Calculate individual Greeks
                    greeks = {}
                    if hasattr(model, 'delta'):
                        greeks['delta'] = model.delta()
                    if hasattr(model, 'gamma'):
                        greeks['gamma'] = model.gamma()
                    if hasattr(model, 'theta'):
                        greeks['theta'] = model.theta()
                    if hasattr(model, 'vega'):
                        greeks['vega'] = model.vega()
                    if hasattr(model, 'rho'):
                        greeks['rho'] = model.rho()
                    
                    greeks_results[model_name] = greeks
                    
            except Exception as e:
                greeks_results[model_name] = {'error': str(e)}
        
        return greeks_results
    
    def perform_sensitivity_analysis(self, parameter, range_pct=0.2, num_points=21):
        """
        Perform sensitivity analysis across all models
        
        Args:
            parameter (str): Parameter to analyze
            range_pct (float): Range as percentage
            num_points (int): Number of points
            
        Returns:
            dict: Sensitivity analysis results
        """
        sensitivity_results = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'black_scholes' and hasattr(model, 'sensitivity_analysis'):
                    result = model.sensitivity_analysis(parameter, range_pct, num_points)
                    sensitivity_results[model_name] = result
                else:
                    # Manual sensitivity analysis for other models
                    result = self._manual_sensitivity_analysis(model, parameter, range_pct, num_points)
                    sensitivity_results[model_name] = result
                    
            except Exception as e:
                sensitivity_results[model_name] = {'error': str(e)}
        
        return sensitivity_results
    
    def _manual_sensitivity_analysis(self, model, parameter, range_pct, num_points):
        """
        Manual sensitivity analysis for models without built-in method
        """
        if not hasattr(model, parameter):
            return {'error': f'Parameter {parameter} not found in model'}
        
        original_value = getattr(model, parameter)
        min_val = original_value * (1 - range_pct)
        max_val = original_value * (1 + range_pct)
        values = np.linspace(min_val, max_val, num_points)
        
        prices = []
        
        for value in values:
            setattr(model, parameter, value)
            try:
                if hasattr(model, 'monte_carlo_price'):
                    price = model.monte_carlo_price()['price']
                else:
                    price = model.price()
                prices.append(price)
            except:
                prices.append(np.nan)
        
        # Restore original value
        setattr(model, parameter, original_value)
        
        return {
            'parameter': parameter,
            'values': values,
            'prices': prices,
            'original_value': original_value
        }
    
    def compare_models(self):
        """
        Compare results across all models
        
        Returns:
            dict: Comparison results
        """
        if not self.results:
            self.price_all_models()
        
        comparison = {
            'prices': {},
            'greeks': self.calculate_greeks_all_models(),
            'statistics': {}
        }
        
        # Extract prices
        prices = []
        for model_name, result in self.results.items():
            if isinstance(result, dict) and 'price' in result:
                price = result['price']
                if isinstance(price, (int, float)) and not np.isnan(price):
                    comparison['prices'][model_name] = price
                    prices.append(price)
        
        # Calculate statistics
        if prices:
            comparison['statistics'] = {
                'mean': np.mean(prices),
                'std': np.std(prices),
                'min': np.min(prices),
                'max': np.max(prices),
                'range': np.max(prices) - np.min(prices),
                'cv': np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0
            }
        
        return comparison
    
    def calibrate_models(self, market_data):
        """
        Calibrate models to market data
        
        Args:
            market_data (dict): Market option data
            
        Returns:
            dict: Calibration results
        """
        calibration_results = {}
        
        # Calibrate Heston model
        if 'heston' in self.models and 'option_prices' in market_data:
            try:
                heston_model = self.models['heston']
                result = heston_model.calibrate_to_market(
                    market_data['option_prices'],
                    market_data['strikes'],
                    market_data['maturities']
                )
                calibration_results['heston'] = result
            except Exception as e:
                calibration_results['heston'] = {'success': False, 'error': str(e)}
        
        # Calibrate Dupire model
        if 'dupire' in self.models and 'implied_vols' in market_data:
            try:
                dupire_model = self.models['dupire']
                dupire_model.construct_local_vol_surface(market_data)
                calibration_results['dupire'] = {'success': True, 'message': 'Local vol surface constructed'}
            except Exception as e:
                calibration_results['dupire'] = {'success': False, 'error': str(e)}
        
        return calibration_results
    
    def generate_monte_carlo_paths(self, model_name, num_paths=1000):
        """
        Generate Monte Carlo paths for visualization
        
        Args:
            model_name (str): Model name
            num_paths (int): Number of paths to generate
            
        Returns:
            dict: Path data
        """
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        
        try:
            if model_name == 'black_scholes':
                result = model.monte_carlo_price(num_simulations=num_paths)
                return {
                    'paths': result['paths'][:num_paths],
                    'final_prices': result['paths'][:num_paths, -1],
                    'payoffs': result['payoffs'][:num_paths]
                }
            elif model_name == 'heston':
                result = model.monte_carlo_price(num_simulations=num_paths)
                return {
                    'stock_paths': result['stock_paths'][:num_paths],
                    'variance_paths': result['variance_paths'][:num_paths],
                    'final_prices': result['stock_paths'][:num_paths, -1],
                    'payoffs': result['payoffs'][:num_paths]
                }
            elif model_name == 'dupire':
                if hasattr(model, 'K') and hasattr(model, 'T'):
                    result = model.monte_carlo_price(model.K, model.T, num_simulations=num_paths)
                    return {
                        'paths': result['stock_paths'][:num_paths],
                        'final_prices': result['stock_paths'][:num_paths, -1],
                        'payoffs': result['payoffs'][:num_paths]
                    }
                else:
                    return {'error': 'Dupire model missing K and T parameters'}
        except Exception as e:
            return {'error': str(e)}
    
    def export_results(self, filename=None):
        """
        Export results to file
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Filename of exported results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pricing_results_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'market_data': self.market_data,
            'pricing_results': self.results,
            'model_comparison': self.compare_models(),
            'greeks': self.calculate_greeks_all_models()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        export_data = convert_numpy(export_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def get_model_summary(self):
        """
        Get summary of all models
        
        Returns:
            dict: Model summary
        """
        summary = {}
        
        for model_name, model in self.models.items():
            model_info = {
                'type': model_name,
                'parameters': {}
            }
            
            # Extract parameters based on model type
            if model_name == 'black_scholes':
                model_info['parameters'] = {
                    'S0': model.S0,
                    'K': model.K,
                    'T': model.T,
                    'r': model.r,
                    'sigma': model.sigma,
                    'option_type': model.option_type
                }
            elif model_name == 'heston':
                model_info['parameters'] = {
                    'S0': model.S0,
                    'K': model.K,
                    'T': model.T,
                    'r': model.r,
                    'v0': model.v0,
                    'kappa': model.kappa,
                    'theta': model.theta,
                    'sigma_v': model.sigma_v,
                    'rho': model.rho,
                    'option_type': model.option_type
                }
            elif model_name == 'dupire':
                model_info['parameters'] = {
                    'S0': model.S0,
                    'r': model.r,
                    'dividend_yield': model.q,
                    'local_vol_surface_constructed': model.local_vol_surface is not None
                }
            
            summary[model_name] = model_info
        
        return summary