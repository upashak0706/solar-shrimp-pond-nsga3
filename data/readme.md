# Dataset Description

This directory contains the datasets used for the solar PV-battery optimization and robustness validation of the semi-intensive shrimp pond aeration system.

## 1. cleaned_GHI_and_WetBulb.csv

Primary historical meteorological dataset used by the digital twin.

Relevant variables include:

* `ALLSKY_SFC_SW_DWN` — solar irradiance / GHI
* `T2MWET` — wet-bulb temperature
* `MO` — month
* `DY` — day
* `HR` — hour

Used for Phase 1 optimization and for constructing the empirical temperature and solar profiles used in Phase 2.

## 2. Solar_GHI_Forecast_2026.csv

Solar forecast dataset used to construct the 2026 P10 lower-bound validation scenario.

Required column:

* `P10_Lower_Bound`

## 3. ALLSKY_SFC_SW_DWN_365_days_stats_v2.csv

Daily solar statistics used to construct annual stress-test scenarios.

Required columns:

* `Median` — used for the TMY/median solar scenario
* `Lowest` — used for the 25-year absolute-lowest solar scenario

## Usage

The datasets are read by the Python scripts in `src/`.


