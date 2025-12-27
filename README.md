# Black–Scholes Option Dashboard

A **Python Streamlit app** for pricing options and visualizing their sensitivity using the **Black–Scholes model**.  
Interactively explore option prices, Greeks, and put-call parity across different spot prices and volatilities.

---

## 🚀 Features

- **Single-Point Valuation**  
  Calculate the price and Greeks (Delta, Gamma, Vega, Theta, Rho) for **call** and **put** options.

- **Interactive Sensitivity Heatmap**  
  Visualize option prices across a range of spot prices and volatilities.

- **Put-Call Parity Validation**  
  Automatically computes and displays parity error to verify model consistency.

- **Customizable Inputs**  
  Adjust spot price, strike price, volatility, time to maturity, interest rate, and grid resolution via sliders.

- **Professional Visualizations**  
  Heatmaps with hover info showing call price, put price, parity RHS, and parity error.

---

## 🛠️ Tech Stack

- **Python 3.13+**  
- **Streamlit** – Web dashboard  
- **NumPy** – Efficient numerical computation  
- **SciPy** – Statistical functions  
- **Plotly** – Interactive charts  

---

## ⚡ Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/black-scholes-dashboard.git
cd black-scholes-dashboard
