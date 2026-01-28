# CSP Thermal Energy Storage — Transient Discharge Model

This repository contains the Python numerical model developed to analyze the transient discharge behavior of a thermal energy storage (TES) system for a Concentrated Solar Power (CSP) application.

The model compares two storage configurations:
- Sensible heat storage using limestone rocks
- Latent heat storage using a KCl–LiCl phase change material (PCM)

The implementation solves the coupled transient energy balance equations for the heat transfer fluid (HTF) and the storage medium during off-sunshine operation.

---

## Physical Model Overview

- Storage configuration: vertical shell-and-tube heat exchanger  
- Heat transfer fluid (HTF): Syltherm 800 flowing inside tubes  
- Storage medium: rocks or PCM on the shell side  
- Discharge duration: 6 hours  
- Nominal storage target: 1 MW for 4 hours  
- Ambient heat losses included  

Latent heat effects in the PCM are modeled using an effective heat capacity method centered around the melting temperature.

---

## Numerical Method

- Governing equations: coupled first-order ordinary differential equations (ODEs)  
- Time integration: explicit variable-step Runge–Kutta method  
- Python solver: scipy.integrate.solve_ivp  
- Outputs:
  - HTF outlet temperature as a function of time  
  - Released thermal power during discharge  

A parametric sensitivity study on the overall heat transfer conductance (UA) is also included to assess heat-transfer-limited operation.

---

## Repository Structure

csp-tes-transient-model/
├── tes_model.py
├── requirements.txt
├── README.md
└── figures/

---

## How to Run the Code

1. Install Python (version ≥ 3.9 recommended)
2. Install dependencies:
   pip install -r requirements.txt
3. Run the simulation:
   python tes_model.py

The script generates figures showing the HTF outlet temperature and the released thermal power versus time.

---

## Relation to the Report

This code supports the numerical resolution and performance analysis presented in Part II of the associated academic report.
The report itself is fully self-contained; this repository is provided solely for transparency and reproducibility.

---

## License

This repository is intended for academic use only.

