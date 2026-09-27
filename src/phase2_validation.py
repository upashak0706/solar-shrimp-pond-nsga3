import jax
import jax.numpy as jnp
from jax.lax import scan
import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

device_backend = jax.lib.xla_bridge.get_backend().platform
print(f"Executing Phase 2 Validation on: {device_backend.upper()}")

# ==========================================
# 1. CONSTANTS
# ==========================================
POND_VOL_LITERS = 207888.0
SALINITY_PPT = 15.0
THERMAL_LAG_GAMMA = 0.15
K_PHOTO = 0.003
R_BIO_20 = 1.40
Q10_THETA = 1.072
CRITICAL_DO = 4.0
LETHAL_DO = 1.5
MAX_ANNUAL_DOSE = 10.0          # Sub-lethal hypoxic tolerance (10.0 mg/L-hrs per year)

MOTOR_POWER_W = 864.0
SOTR = 1.443
ALPHA_FOULING = 0.85
THETA_O2 = 1.024
C_SAT_20_FRESH = 9.09
SYS_VOLTAGE = 48.0
AUX_LOAD_W = 50.0
TIMESTEPS = 8760                # 1 Standard Year (365 days)

# ==========================================
# 2. METEOROLOGICAL PRE-PROCESSING
# ==========================================
print("Extracting 10-Year Empirical Diurnal Profile from NASA Data...")
df_master = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "cleaned_GHI_and_WetBulb.csv")
# Remove leap days to ensure exactly 365 days (8760 hours)
df_no_leap = df_master[~((df_master['MO'] == 2) & (df_master['DY'] == 29))]

# Extract the 10-Year average Wet Bulb Temperature per hour
tmy_temp_df = df_no_leap.groupby(['MO', 'DY', 'HR'])['T2MWET'].mean().reset_index()
tmy_temp_df = tmy_temp_df.sort_values(by=['MO', 'DY', 'HR'])
t_water_raw = tmy_temp_df['T2MWET'].ewm(alpha=THERMAL_LAG_GAMMA, adjust=False).mean().values
t_water_tensor = jnp.array(t_water_raw, dtype=jnp.float32)

# Extract the Empirical 12x24 Diurnal Fraction Matrix (Seasonal Solar Shapes)
monthly_hourly_ghi = df_no_leap.groupby(['MO', 'HR'])['ALLSKY_SFC_SW_DWN'].mean().unstack(level='HR')
monthly_diurnal_fractions = monthly_hourly_ghi.div(monthly_hourly_ghi.sum(axis=1), axis=0).values

# Pre-compute month lookup once outside the hot loop (Optimization)
month_per_day = np.array([pd.to_datetime(f'2023-{d+1:03}', format='%Y-%j').month - 1 for d in range(365)])
def synthesize_hourly_ghi(daily_ghi_array):
    """Maps daily kWh/m2/day targets onto true empirical Kharagpur seasonal curves"""
    hourly_ghi = np.zeros(365 * 24)

    for day in range(365):
        # O(1) array lookup instead of per-iteration Timestamp parsing
        month_idx = month_per_day[day]

        daily_total_wh = daily_ghi_array[day] * 1000.0
        hourly_profile = daily_total_wh * monthly_diurnal_fractions[month_idx]
        hourly_ghi[day*24 : (day+1)*24] = hourly_profile

    return jnp.array(hourly_ghi, dtype=jnp.float32)

print("Synthesizing 8760-hour Extreme Weather Tensors...")
df_p10 = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "Solar_GHI_Forecast_2026.csv")
df_stats = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "ALLSKY_SFC_SW_DWN_365_days_stats_v2.csv")

# Generate the 3 Stress Test Solar Environments using the real seasonal matrices
ghi_tmy = synthesize_hourly_ghi(df_stats['Median'].values)
ghi_25yr_lowest = synthesize_hourly_ghi(df_stats['Lowest'].values)
ghi_p10 = synthesize_hourly_ghi(df_p10['P10_Lower_Bound'].values)


# ==========================================
# 3. JAX KERNEL: DIGITAL TWIN FORWARD PASS
# ==========================================
@jax.jit
def calc_c_sat_saline(t_celsius, salinity):
    tk = t_celsius + 273.15
    c_fresh = jnp.exp(-139.34411 + (1.575701e5 / tk) - (6.642308e7 / tk**2) + (1.243800e10 / tk**3) - (8.621949e11 / tk**4))
    salinity_factor = jnp.exp(-salinity * (0.017674 - (10.754 / tk) + (2140.7 / tk**2)))
    return c_fresh * salinity_factor

@jax.jit
def run_stress_test(x_tensor, ghi_tensor):
    pv_kw, batt_kwh, mppt_amps, do_thresh, load_shed_soc, max_c_rate = x_tensor
    batt_max_wh = batt_kwh * 1000.0
    mppt_max_w = mppt_amps * SYS_VOLTAGE
    max_charge_power = batt_max_wh * max_c_rate

    init_c_sat = calc_c_sat_saline(t_water_tensor[0], SALINITY_PPT)
    init_carry = (
        jnp.float32(init_c_sat), batt_max_wh, jnp.float32(0.0), jnp.float32(0.0)
    )

    def scan_step(carry, step_data):
        current_do, batt_wh, hypoxic_dose, lethal_crashes = carry
        ghi, t_water = step_data

        c_sat = calc_c_sat_saline(t_water, SALINITY_PPT)
        r_bio = R_BIO_20 * (Q10_THETA ** (t_water - 20.0))
        photo_gen = ghi * K_PHOTO

        c_pred = jnp.maximum(0.0, current_do - r_bio + photo_gen)
        c_pred = jnp.minimum(c_pred, c_sat)

        deficit_mg_l = jnp.maximum(0.0, do_thresh - c_pred)
        deficit_kg = deficit_mg_l * (POND_VOL_LITERS / 1e6)

        theta_correction = THETA_O2 ** (t_water - 20.0)
        aotr = SOTR * ((c_sat - c_pred) / C_SAT_20_FRESH) * ALPHA_FOULING * theta_correction
        aotr = jnp.maximum(aotr, 0.1)

        d_req = jnp.clip(deficit_kg / aotr, 0.0, 1.0)
        motor_demand_w = d_req * MOTOR_POWER_W

        current_soc = (batt_wh / batt_max_wh) * 100.0
        active_aux_w = jnp.where(current_soc > load_shed_soc, AUX_LOAD_W, 0.0)
        total_demand_w = motor_demand_w + active_aux_w

        gen_w = pv_kw * ghi * 0.85
        gen_clipped = jnp.minimum(gen_w, mppt_max_w)

        max_deliverable_w = gen_clipped + jnp.minimum(batt_wh, max_charge_power)
        power_avail_for_motor = jnp.maximum(0.0, max_deliverable_w - active_aux_w)
        power_shortfall = total_demand_w > max_deliverable_w

        d_actual = jnp.where(power_shortfall, power_avail_for_motor / MOTOR_POWER_W, d_req)
        d_actual = jnp.clip(d_actual, 0.0, 1.0)

        net_w_final = gen_clipped - (d_actual * MOTOR_POWER_W) - active_aux_w
        net_w_final = jnp.clip(net_w_final, -max_charge_power, max_charge_power)
        e_final = batt_wh + net_w_final
        new_batt_wh = jnp.clip(e_final, 0.0, batt_max_wh)

        mg_l_injected = (aotr * d_actual * (1e6 / POND_VOL_LITERS))
        new_do = jnp.clip(c_pred + mg_l_injected, 0.0, c_sat)

        # Dual-Biological Constraints Tracking
        deficit = jnp.maximum(0.0, CRITICAL_DO - new_do)
        new_hypoxic_dose = hypoxic_dose + deficit
        new_lethal_crashes = lethal_crashes + jnp.where(new_do < LETHAL_DO, 1.0, 0.0)

        return (new_do, new_batt_wh, new_hypoxic_dose, new_lethal_crashes), None

    step_data = (ghi_tensor, t_water_tensor)
    final_carry, _ = scan(scan_step, init_carry, step_data)
    _, _, total_dose, total_lethal_crashes = final_carry

    return total_dose, total_lethal_crashes

# Vmap across the whole population
batch_test_tmy = jax.jit(jax.vmap(lambda x: run_stress_test(x, ghi_tmy)))
batch_test_lowest = jax.jit(jax.vmap(lambda x: run_stress_test(x, ghi_25yr_lowest)))
batch_test_p10 = jax.jit(jax.vmap(lambda x: run_stress_test(x, ghi_p10)))

# ==========================================
# 4. EXECUTE VALIDATION ON PHASE 1 DESIGNS
# ==========================================
if __name__ == "__main__":
    start_time = time.time()

    print("\nLoading Phase 1 Unbounded Pareto Front...")
    df_designs = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "phase1_final_pareto_front_unbounded.csv")

    X_matrix = df_designs[['PV_kW', 'Batt_kWh', 'MPPT_A', 'DO_Thresh', 'Shed_SOC_%', 'Max_C_Rate']].values
    X_jax = jnp.array(X_matrix, dtype=jnp.float32)

    print("Executing Tier 1: Typical Meteorological Year (TMY)...")
    dose_tmy, crash_tmy = batch_test_tmy(X_jax)

    print("Executing Tier 2a: 2026 P-10 Forecast...")
    dose_p10, crash_p10 = batch_test_p10(X_jax)

    print("Executing Tier 2b: 25-Year Absolute Lowest...")
    dose_lowest, crash_lowest = batch_test_lowest(X_jax)

    # Append results to the dataframe
    df_designs['TMY_Dose'] = np.array(dose_tmy)
    df_designs['TMY_Lethal_Hrs'] = np.array(crash_tmy)

    df_designs['P10_Dose'] = np.array(dose_p10)
    df_designs['P10_Lethal_Hrs'] = np.array(crash_p10)

    df_designs['Lowest_Dose'] = np.array(dose_lowest)
    df_designs['Lowest_Lethal_Hrs'] = np.array(crash_lowest)

    # Categorize Survivability based on BOTH constraints
    def categorize_tier(row):
        # Check TMY Failure
        if row['TMY_Lethal_Hrs'] > 0 or row['TMY_Dose'] > MAX_ANNUAL_DOSE:
            return "Failed TMY (Non-Viable)"

        # Check Extreme Failures
        if (row['P10_Lethal_Hrs'] > 0 or row['P10_Dose'] > MAX_ANNUAL_DOSE or
            row['Lowest_Lethal_Hrs'] > 0 or row['Lowest_Dose'] > MAX_ANNUAL_DOSE):
            return "Survived TMY Only (Extreme Fail)"

        return "Ultimate Survivor"

    df_designs['Validation_Tier'] = df_designs.apply(categorize_tier, axis=1)

    elapsed = time.time() - start_time
    print(f"\nPhase 2 Completed in {elapsed:.2f} seconds.")
    print("\nValidation Summary:")
    print(df_designs['Validation_Tier'].value_counts())

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "phase2_validated_architectures.csv"
    df_designs.to_csv(output_path, index=False)
    print(f"\nResults exported to '{output_path}'.")