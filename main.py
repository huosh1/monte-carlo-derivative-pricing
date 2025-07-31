"""
Monte Carlo Derivative Pricing Tool
Main Application Entry Point - Version corrigée pour Python 3.13
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Vérifier que toutes les dépendances sont installées"""
    required_modules = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 
        'seaborn', 'yfinance', 'openpyxl', 'tkinter'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Modules manquants détectés:")
        for module in missing_modules:
            print(f"   - {module}")
        return False
    
    print("✅ Toutes les dépendances sont installées!")
    return True

def main():
    """
    Main entry point for the derivative pricing application
    """
    print("Lancement de Monte Carlo Derivative Pricing Tool")
    print("=" * 55)
    
    # Vérifier les dépendances
    if not check_dependencies():
        input("\nAppuyez sur Entrée pour quitter...")
        sys.exit(1)
    
    try:
        # Importer tkinter en premier pour vérifier la disponibilité GUI
        import tkinter as tk
        
        # Importer les composants de l'application
        from src.gui.main_gui import DerivativePricingGUI
        
        print("📊 Initialisation de l'interface graphique...")
        
        # Create the main window
        root = tk.Tk()
        
        # Initialize the GUI
        app = DerivativePricingGUI(root)
        
        print("✅ Application prête! Fenêtre en cours d'ouverture...")
        
        # Start the application
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()