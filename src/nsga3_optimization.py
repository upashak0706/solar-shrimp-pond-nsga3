import jax
import jax.numpy as jnp
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

try:
    from .digital_twin import batch_simulate, SYS_VOLTAGE, MAX_DOSE_LIMIT
except ImportError:
    from digital_twin import batch_simulate, SYS_VOLTAGE, MAX_DOSE_LIMIT

class JAXAquacultureMicrogrid(Problem):
    def __init__(self):
        # Expanded xu for Shed_SOC_% to 95.0 to fully uncork the load-shedding bounds
        super().__init__(n_var=6, n_obj=6, n_constr=3,
                         xl=np.array([1.0, 2.0, 10.0, 4.1, 10.0, 0.2]),
                         xu=np.array([10.0, 30.0, 165.0, 7.0, 95.0, 1.0]))

    def _evaluate(self, X, out, *args, **kwargs):
        X_jax = jnp.array(X, dtype=jnp.float32)
        F_jax, G_jax = batch_simulate(X_jax)
        out["F"] = np.array(F_jax)
        out["G"] = np.array(G_jax)

# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\nInitializing NSGA-III XLA Optimization (Final Release)...")
    ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=3)

    algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs, eliminate_duplicates=True)
    problem = JAXAquacultureMicrogrid()

    start_time = time.time()
    res = minimize(problem, algorithm, seed=42, termination=('n_gen', 100), verbose=True)
    elapsed = time.time() - start_time

    print(f"\nOptimization Completed in {elapsed:.2f} seconds.")

    if res.X is not None:
        print(f"Discovered {len(res.X)} Bankable Architectures.")

        x_cols = ['PV_kW', 'Batt_kWh', 'MPPT_A', 'DO_Thresh', 'Shed_SOC_%', 'Max_C_Rate']
        f_cols = ['True_LCOE_INR', 'CapEx_INR', 'Max_72h_SOC_Deficit_%', 'Batt_Stress_EFC', 'MPPT_Clip_kWh', 'Surplus_kWh']

        df_x = pd.DataFrame(res.X, columns=x_cols)
        df_f = pd.DataFrame(res.F, columns=f_cols)
        df_f['Surplus_kWh'] = df_f['Surplus_kWh'] * -1.0

        master_df = pd.concat([df_x, df_f], axis=1).round(2)
        results_dir = Path(__file__).resolve().parents[1] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / "phase1_final_pareto_front_unbounded.csv"
        master_df.to_csv(output_path, index=False)
        print(f"Exported to '{output_path}'.")
    else:
        print("ERROR: Hardware limits too constrained.")