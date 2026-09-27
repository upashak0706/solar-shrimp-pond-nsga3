import jax
import jax.numpy as jnp
from jax.lax import scan
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

device_backend = jax.lib.xla_bridge.get_backend().platform
print(f"Executing High-Performance Final Digital Twin on: {device_backend.upper()}")

# ==========================================
# 1. EMPIRICAL CONSTANTS & CITATIONS
# ==========================================
# Biology (Litopenaeus vannamei & Pond Dynamics)
POND_VOL_LITERS = 207888.0
SALINITY_PPT = 15.0
THERMAL_LAG_GAMMA = 0.15
K_PHOTO = 0.003
R_BIO_20 = 1.40
Q10_THETA = 1.072

CRITICAL_DO = 4.0
LETHAL_DO = 1.5
MAX_DOSE_LIMIT = 100.0

# Electrical & Physical Hardware
MOTOR_POWER_W = 864.0
SOTR = 1.443
ALPHA_FOULING = 0.85
THETA_O2 = 1.024
C_SAT_20_FRESH = 9.09

SYS_VOLTAGE = 48.0
AUX_LOAD_W = 50.0
AUX_SURPLUS_LIMIT = 300.0

# Financial Parameters (INR)
COST_PV_PER_KW = 28000.0
COST_BATT_PER_KWH = 18000.0
COST_MPPT_PER_AMP = 150.0
DISCOUNT_RATE = 0.08            # 8% standard discount rate
OM_FRAC = 0.015                 # 1.5% of CapEx for annual O&M
LIFESPAN_YRS = 10.0
BATT_CYCLE_LIMIT = 4000.0       # EFC threshold for battery replacement

TIMESTEPS = 87672

# ==========================================
# 2. DATA PRE-PROCESSING
# ==========================================
print("Processing Climatological Datasets...")
try:
    df_master = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "cleaned_GHI_and_WetBulb.csv")
    df_master['T_Water'] = df_master['T2MWET'].ewm(alpha=THERMAL_LAG_GAMMA, adjust=False).mean()
    ghi_tensor = jnp.array(df_master['ALLSKY_SFC_SW_DWN'].values, dtype=jnp.float32)
    t_water_tensor = jnp.array(df_master['T_Water'].values, dtype=jnp.float32)
except FileNotFoundError:
    print("Warning: Dataset not found. Using synthetic arrays.")
    ghi_tensor = jnp.zeros(TIMESTEPS, dtype=jnp.float32)
    t_water_tensor = jnp.ones(TIMESTEPS, dtype=jnp.float32) * 25.0

# Pre-calculate 10-year discount factor sum for fixed annual flows
DISCOUNT_SUM = sum([1.0 / ((1.0 + DISCOUNT_RATE) ** t) for t in range(1, int(LIFESPAN_YRS) + 1)])

# ==========================================
# 3. JAX KERNEL: THERMODYNAMICS
# ==========================================
@jax.jit
def calc_c_sat_saline(t_celsius, salinity):
    tk = t_celsius + 273.15
    c_fresh = jnp.exp(-139.34411 + (1.575701e5 / tk) -
                      (6.642308e7 / tk**2) +
                      (1.243800e10 / tk**3) -
                      (8.621949e11 / tk**4))
    salinity_factor = jnp.exp(-salinity * (0.017674 - (10.754 / tk) + (2140.7 / tk**2)))
    return c_fresh * salinity_factor

# ==========================================
# 4. JAX KERNEL: XLA SCAN LOOP
# ==========================================
@jax.jit
def run_microgrid_scan(x_tensor):
    pv_kw, batt_kwh, mppt_amps, do_thresh, load_shed_soc, max_c_rate = x_tensor

    batt_max_wh = batt_kwh * 1000.0
    mppt_max_w = mppt_amps * SYS_VOLTAGE
    max_charge_power = batt_max_wh * max_c_rate

    init_t_water = t_water_tensor[0]
    init_c_sat = calc_c_sat_saline(init_t_water, SALINITY_PPT)

    init_carry = (
        jnp.float32(init_c_sat),
        batt_max_wh,
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.zeros(72, dtype=jnp.float32) + 100.0,
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0)
    )

    def scan_step(carry, step_data):
        (current_do, batt_wh, hypoxic_dose, lethal_flag, soc_hist,
         total_served, stress_idx, clip_loss, usable_surplus) = carry

        ghi, t_water = step_data

        # Thermodynamics & Biology
        c_sat = calc_c_sat_saline(t_water, SALINITY_PPT)
        r_bio = R_BIO_20 * (Q10_THETA ** (t_water - 20.0))
        photo_gen = ghi * K_PHOTO

        c_pred = jnp.maximum(0.0, current_do - r_bio + photo_gen)
        c_pred = jnp.minimum(c_pred, c_sat)

        # Aerator Demand
        deficit_mg_l = jnp.maximum(0.0, do_thresh - c_pred)
        deficit_kg = deficit_mg_l * (POND_VOL_LITERS / 1e6)

        theta_correction = THETA_O2 ** (t_water - 20.0)
        aotr = SOTR * ((c_sat - c_pred) / C_SAT_20_FRESH) * ALPHA_FOULING * theta_correction
        aotr = jnp.maximum(aotr, 0.1)

        d_req = jnp.clip(deficit_kg / aotr, 0.0, 1.0)
        motor_demand_w = d_req * MOTOR_POWER_W

        # Microgrid Power Flow
        current_soc = (batt_wh / batt_max_wh) * 100.0
        active_aux_w = jnp.where(current_soc > load_shed_soc, AUX_LOAD_W, 0.0)
        total_demand_w = motor_demand_w + active_aux_w

        gen_w = pv_kw * ghi * 0.85
        gen_clipped = jnp.minimum(gen_w, mppt_max_w)

        # C-Rate Limits and Battery Dead Fallback
        max_deliverable_w = gen_clipped + jnp.minimum(batt_wh, max_charge_power)
        power_avail_for_motor = jnp.maximum(0.0, max_deliverable_w - active_aux_w)
        power_shortfall = total_demand_w > max_deliverable_w

        d_actual = jnp.where(power_shortfall, power_avail_for_motor / MOTOR_POWER_W, d_req)
        d_actual = jnp.clip(d_actual, 0.0, 1.0)

        # State Updates
        net_w_final = gen_clipped - (d_actual * MOTOR_POWER_W) - active_aux_w
        net_w_final = jnp.clip(net_w_final, -max_charge_power, max_charge_power)
        e_final = batt_wh + net_w_final
        new_batt_wh = jnp.clip(e_final, 0.0, batt_max_wh)

        # Advanced Metrics
        mg_l_injected = (aotr * d_actual * (1e6 / POND_VOL_LITERS))
        new_do = jnp.clip(c_pred + mg_l_injected, 0.0, c_sat)

        spillage = jnp.maximum(0.0, e_final - batt_max_wh)
        usable_now = jnp.where(current_soc > load_shed_soc, jnp.minimum(spillage, AUX_SURPLUS_LIMIT), 0.0)
        step_clip_loss = (gen_w - gen_clipped)

        dod = 1.0 - (new_batt_wh / batt_max_wh)
        dod_penalty = 1.0 + 1.5 * (dod ** 2.0)
        c_rate_actual = jnp.abs(net_w_final) / batt_max_wh
        thermal_penalty = 1.0 + 1.2 * (c_rate_actual ** 2.0)
        step_stress = jnp.abs(net_w_final) * dod_penalty * thermal_penalty

        # Constraints Tracking
        deficit = jnp.maximum(0.0, CRITICAL_DO - new_do)
        new_hypoxic_dose = hypoxic_dose + deficit
        new_lethal_flag = jnp.maximum(lethal_flag, jnp.where(new_do < LETHAL_DO, 1.0, 0.0))

        new_soc_hist = jnp.roll(soc_hist, shift=1)
        new_soc_hist = new_soc_hist.at[0].set((new_batt_wh / batt_max_wh) * 100.0)

        new_carry = (new_do, new_batt_wh, new_hypoxic_dose, new_lethal_flag, new_soc_hist,
                     total_served + (d_actual * MOTOR_POWER_W),
                     stress_idx + step_stress,
                     clip_loss + step_clip_loss,
                     usable_surplus + usable_now)

        avg_72h_soc = jnp.mean(new_soc_hist)

        return new_carry, avg_72h_soc

    step_data = (ghi_tensor, t_water_tensor)
    final_carry, soc_history_outputs = scan(scan_step, init_carry, step_data)

    # Unpack Results
    _, _, final_dose, lethal_flag, _, total_served, stress_idx, clip_loss, surplus = final_carry

    # -----------------------------------------------------
    # FINANCIAL ENGINEERING: TRUE LCOE CALCULATION
    # -----------------------------------------------------
    capex = (pv_kw * COST_PV_PER_KW) + (batt_kwh * COST_BATT_PER_KWH) + (mppt_amps * COST_MPPT_PER_AMP)

    # Discounted O&M
    npv_om = (OM_FRAC * capex) * DISCOUNT_SUM

    # Dynamic Multi-Replacement NPV Engine
    batt_efc = stress_idx / batt_max_wh
    efc_per_year = batt_efc / LIFESPAN_YRS
    years_to_replace = BATT_CYCLE_LIMIT / (efc_per_year + 1e-6)

    replacement_cost = batt_kwh * COST_BATT_PER_KWH
    npv_replacement = jnp.float32(0.0)

    # Statically unroll up to 5 replacements
    for k in range(1, 6):
        rep_year = k * years_to_replace
        cost_discounted = jnp.where(
            rep_year < LIFESPAN_YRS,
            replacement_cost / ((1.0 + DISCOUNT_RATE) ** rep_year),
            0.0
        )
        npv_replacement += cost_discounted

    total_npv_cost = capex + npv_om + npv_replacement

    # Discounted Energy
    annual_energy_kwh = (total_served / 1000.0) / LIFESPAN_YRS
    discounted_energy = annual_energy_kwh * DISCOUNT_SUM

    true_lcoe = total_npv_cost / (discounted_energy + 1e-6)

    # Extract survivability metric
    valid_soc_outputs = soc_history_outputs[72:]
    max_soc_deficit = 100.0 - jnp.min(valid_soc_outputs)

    # Return Objectives (F)
    F = jnp.array([
        true_lcoe,
        capex,
        max_soc_deficit,
        batt_efc,
        clip_loss / 1000.0,
        -surplus / 1000.0
    ])

    # -----------------------------------------------------
    # CONSTRAINTS (G)
    # -----------------------------------------------------
    mppt_kw = (mppt_amps * SYS_VOLTAGE) / 1000.0
    oversize_violation = pv_kw - (1.3 * mppt_kw) # Must be <= 0

    G = jnp.array([
        final_dose - MAX_DOSE_LIMIT,    # G1: Sub-lethal dose
        lethal_flag,                    # G2: Strict survival
        oversize_violation              # G3: MPPT Physical Limiting
    ])

    return F, G

# JAX Vmap wrapper
batch_simulate = jax.jit(jax.vmap(run_microgrid_scan))
