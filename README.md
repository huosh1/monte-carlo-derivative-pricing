# Monte Carlo Derivative Pricing Tool

Projet Python de finance quantitative dédié au **pricing d’options par Monte Carlo** et à l’analyse de risque associée. L’application implémente et compare trois modèles de volatilité standards et avancés : **Black-Scholes** (référence analytique), **Heston** (volatilité stochastique) et **Dupire** (volatilité locale). L’objectif est de fournir un outil cohérent de bout en bout (modélisation → simulation → greeks → calibration/validation) avec une architecture propre et exploitable.

Points clés :

* Moteur **Monte Carlo vectorisé** (NumPy), analyse de convergence et intervalles de confiance, réduction de variance.
* **Black-Scholes** : pricing analytique, Greeks analytiques, volatilité implicite (Newton-Raphson), validation MC.
* **Heston** : dynamique conjointe prix/variance, corrélation, schémas de simulation robustes (positivité de la variance), pricing semi-analytique via fonction caractéristique, calibration sur données de marché.
* **Dupire** : construction d’une **surface de volatilité locale** à partir de volatilités implicites, dérivées numériques et interpolation, simulation sous volatilité dépendante de (S,t).
* Calcul des **Greeks** (Delta, Gamma, Vega, Theta, Rho) et analyses de sensibilité pour des usages risk/hedging.
* Intégration de données de marché (récupération, nettoyage, mise en cache) et **export** des résultats (Excel/JSON).
* Interface graphique pour paramétrer les modèles, lancer les simulations et comparer les résultats.

Exécution :

```bash
pip install -r requirements.txt
python main.py
```
