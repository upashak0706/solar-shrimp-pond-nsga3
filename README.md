# Multi-Objective Optimization of Solar PV-Battery Aeration for Semi-Intensive Shrimp Ponds

## Overview

A GPU-accelerated multi-objective optimization framework for designing a solar PV-battery powered aeration system for semi-intensive shrimp ponds.

## Methodology

- Developed a JAX/GPU-accelerated digital twin for dissolved oxygen dynamics, temperature-dependent oxygen demand, PV generation, battery SOC, and aerator power flow.
- Formulated a 6-variable, 6-objective optimization problem using NSGA-III.
- Optimized PV capacity, battery capacity, MPPT rating, DO threshold, load-shedding SOC, and maximum battery C-rate.
- Evaluated LCOE, CapEx, battery stress, SOC deficit, MPPT clipping, and surplus energy.
- Performed Phase-2 robustness validation using TMY, 2026 P10, and 25-year lowest-solar scenarios.

## Results

- 24 Pareto-optimal architectures generated.
- 10,000 evaluations completed in approximately 73 seconds using GPU acceleration.
- 2 of 24 architectures satisfied the defined survivability criteria across all three validation scenarios.
- Surviving configurations used approximately 7.87–9.24 kW PV and 23.74–29.96 kWh battery capacity.
- Surviving configurations achieved 0 lethal-DO hours across the three tested scenarios.

## Technologies

Python • JAX • NumPy • Pandas • pymoo • NSGA-III • GPU Computing • Multi-Objective Optimization