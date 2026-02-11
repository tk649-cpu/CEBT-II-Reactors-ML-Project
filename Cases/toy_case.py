import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from .benchmark import Benchmark


class ToyCase(Benchmark):
    """Benchmark representing a simple reaction performed in a batch reactor.
    reactact A is converted to product B.

    Parameters
    ----------
    random_seed: int, optional
        The Random seed to generate noises. Default is 0.

    Parameter units :
        - 'Batch_Time'                          : min
        - 'Activation_Energy'                   : kJ/mol
        - 'Referenced_Reaction_Rate_Constant'   : 
        - 'Concentration'                       : mol/L
        - 'Temperature'                         : oC

    """

    solids = []
    gases = []
    streams = ["Batch_Feed"]
    reactions = ["A > B"]
    species = ["A", "B"]

    def __init__(self, phenos, random_seed=0):
        structure_params = self._setup_structure_params()
        physics_params = self._setup_physics_params()
        reaction_params = self._setup_reaction_params()
        transport_params = self._setup_transport_params()
        operation_params = self._setup_operation_params()
        operation_name2ind = self._setup_operation_name2ind()
        measure_ind2name = self._setup_measure_ind2name()
        var2unit = self._setup_var2unit()
        super().__init__(
            structure_params,
            physics_params,
            reaction_params,
            transport_params,
            operation_params,
            operation_name2ind,
            measure_ind2name,
            var2unit
        )
        self._validate_phenos(phenos)
        self.phenos = phenos
        self.random_seed = random_seed

    def _validate_phenos(self, phenos):
        assert isinstance(phenos, dict), "phenos should be a dictionary"
        assert "Mass accumulation" in phenos and phenos["Mass accumulation"] == "Batch", \
            "This case runs a 'Batch' reactor"
        assert "Flow pattern" in phenos and phenos["Flow pattern"] == "Well_Mixed", \
            "This case is operated with 'Well_Mixed' flow in batch"
        assert "Mass transport" in phenos and phenos["Mass transport"] == [], \
            "Substances are well mixed in the batch reactor"

    def _setup_structure_params(self):
        structure_params = {}
        return structure_params

    def _setup_physics_params(self):
        physics_params = {}
        return physics_params

    def _setup_reaction_params(self):
        reaction_params = {
            ("Activation_Energy", None, "Batch_Feed", "A > B", None): 80.0,  # kJ/mol
            ("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "A > B", None): 0.0001,  # L/mol s
            ("Stoichiometric_Coefficient", None, None, "A > B", "A"): -1,
            ("Stoichiometric_Coefficient", None, None, "A > B", "B"): 1,
            ("Partial_Order", None, None, "A > B", "A"): 1,
        }
        return reaction_params

    def _setup_transport_params(self):
        transport_params = {}
        return transport_params

    def _setup_operation_params(self):
        operation_params = {
            ("Concentration", None, "Batch_Feed", None, "A"):   None, # mol/L
            ("Temperature", None, None, None, None):            None, # oC
            ("Batch_Time", None, None, None, None):             None, # min
        }
        return operation_params

    def _setup_operation_name2ind(self):
        operation_name2ind = {
            "A_conc":   ("Concentration", None, "Batch_Feed", None, "A"),
            "temp":     ("Temperature", None, None, None, None),
            "t_b":      ("Batch_Time", None, None, None, None),
        }
        return operation_name2ind

    def _setup_measure_ind2name(self):
        measure_ind2name = {
            ("Concentration", None, "Batch_Feed", None, "B"): "outlet_B_conc",
        }
        return measure_ind2name

    def _setup_var2unit(self):
        var2unit = {
            "Activation_Energy": "kJ/mol",
            "Referenced_Reaction_Rate_Constant": None,
            "Temperature": "oC",
            "Concentration": "mol/L",
            "Batch_Time": "min",
        }
        return var2unit

    def _simulate(self, params):
        R = 8.314
        t_b = params[("Batch_Time", None, None, None, None)]
        t_b *= 60
        T = params[("Temperature", None, None, None, None)]
        T += 273.15
        c_0 = np.zeros((1, 2), dtype=np.float64)
        c_0[0, 0] = params[("Concentration", None, "Batch_Feed", None, "A")]
        nu = np.zeros((1, 2), dtype=np.float64)
        nu[0, 0] = params[("Stoichiometric_Coefficient", None, None, "A > B", "A")]
        nu[0, 1] = params[("Stoichiometric_Coefficient", None, None, "A > B", "B")]
        n = np.zeros((1, 2), dtype=np.float64)
        n[0, 0] = params[("Partial_Order", None, None, "A > B", "A")]
        A = np.zeros((1, 1), dtype=np.float64)
        A[0, 0] = params[("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "A > B", None)]
        E_a = np.zeros((1, 1), dtype=np.float64)
        E_a[0, 0] = params[("Activation_Energy", None, "Batch_Feed", "A > B", None)]
        E_a *= 1000

        def _derivative(t, c):
            c = c.reshape((1, 2))
            r_r = np.zeros((1, 1), dtype=np.float64)
            r_r[0, 0] = A[0, 0] * np.exp(-E_a[0, 0] / R * (1 / T - 1 / 298.15)) * np.prod(c[0] ** n[0])
            dc_dt = np.matmul(r_r, nu)
            return dc_dt

        t_eval = np.linspace(0, t_b, 201)
        res = solve_ivp(_derivative, (0, t_b), c_0.reshape(-1, ),
                        method="LSODA", t_eval=t_eval, atol=1e-8)
        t_eval /= 60
        return t_eval, res.y

    def _run(self, operation_params, reaction_params, transport_params):
        assert set(operation_params.keys()).issubset(
            set(self._operation_params.keys())), "Unknown operation parameters included"
        assert set(reaction_params.keys()).issubset(
            set(self._reaction_params.keys())), "Unknown kinetics parameters included"
        assert set(transport_params.keys()).issubset(
            set(self._transport_params.keys())), "Unknown mole transport parameters included"
        params = self._param_list2dict(self.params())
        params.update(operation_params)
        if reaction_params is not None:
            params.update(reaction_params)
        if transport_params is not None:
            params.update(transport_params)
        return self._simulate(params)

    def run(self, operation_params, reaction_params=None, transport_params=None):
        if reaction_params is None:
            reaction_params = self._reaction_params
        if transport_params is None:
            transport_params = self._transport_params
        t, cs = self._run(operation_params, reaction_params, transport_params)
        return t, cs

    def run_dataset(self, dataset):
        dataset = dataset.copy()
        res = {name: [] for name in self._measure_ind2name.values()}
        for i in range(len(dataset)):
            operation_params = {}
            for name, ind in self._operation_name2ind.items():
                operation_params[ind] = dataset.loc[i, name]
            _, cs = self.run(operation_params)
            for ind, name in self._measure_ind2name.items():
                res[name].append(cs[self.species.index(ind[-1]), -1])
        for name, val in res.items():
            dataset[name] = val
        return dataset

    def calibrate(self, cal_param_bounds, dataset):
        cal_param_bounds = self._param_list2dict(cal_param_bounds)
        fixed_cal_param_bounds = {k: round(v[0], 6) for k, v in cal_param_bounds.items() if v[0] == v[1]}
        cal_param_bounds = {k: v for k, v in cal_param_bounds.items() if v[0] != v[1]}
        params = self._param_list2dict(self.params())
        for param_tuple, v in cal_param_bounds.items():
            if v[0] == v[1]:
                params[param_tuple] = v[0]
                cal_param_bounds[param_tuple] = v[0]
        def calc_mse(p):
            mse = 0
            cal_params = {cal_param_ind: _p for cal_param_ind, _p in zip(cal_param_bounds.keys(), p)}
            params.update(cal_params)
            for i in range(len(dataset)):
                operation_params = {ind: dataset.loc[i, name] for name, ind in self._operation_name2ind.items()}
                params.update(operation_params)
                t, cs = self._simulate(params)
                for ind, name in self._measure_ind2name.items():
                    mse += (cs[self.species.index(ind[-1]), -1] - dataset.loc[i, name])**2
            return mse
        res = minimize(
            fun=calc_mse, 
            x0=[np.mean(v).item() for v in cal_param_bounds.values()], 
            method='L-BFGS-B',
            bounds=list(cal_param_bounds.values()),
        )
        cal_params = {ind: round(v.item(), 6) for ind, v in zip(cal_param_bounds.keys(), res.x)}
        cal_params.update(fixed_cal_param_bounds)
        return self._param_dict2list(cal_params)
    
    def plot_simulation_profiles(self, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        t, cs = self.run(operation_params, reaction_params, transport_params)
        data = {"Time (min)": [], "Concentration (mol/L)": [], "Species": []}
        for i, s in enumerate(self.species):
            for _t, _c in zip(t, cs[i]):
                data["Time (min)"].append(_t)
                data["Concentration (mol/L)"].append(_c)
                data["Species"].append(f"{s}")
        df = pd.DataFrame(data)
        fig = px.line(df, x="Time (min)", y="Concentration (mol/L)", color="Species")
        fig.update_layout(width=800, height=500, title="Concentration Profiles")
        fig.show()

    def plot_product_profile_with_temperatures(self, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        temperatures = operation_params[("Temperature", None, None, None, None)]
        batch_time = operation_params[("Batch_Time", None, None, None, None)]
        A_conc = operation_params[("Concentration", None, "Batch_Feed", None, "A")]
        data = {"Time (min)": [], "B concentration (mol/L)": [], "Temperature (oC)": []}
        for temperature in temperatures:
            _operation_params = {
                ("Temperature", None, None, None, None): temperature,
                ("Batch_Time", None, None, None, None): batch_time,
                ("Concentration", None, "Batch_Feed", None, "A"): A_conc,
            }
            t, cs = self.run(_operation_params, reaction_params, transport_params)
            for _t, _c in zip(t, cs[self.species.index("B")]):
                data["Time (min)"].append(_t)
                data["B concentration (mol/L)"].append(_c)
                data["Temperature (oC)"].append(temperature)
        df = pd.DataFrame(data)
        fig = px.line(df, x="Time (min)", y="B concentration (mol/L)", color="Temperature (oC)")
        fig.update_layout(width=800, height=500, title="Product Concentration Profiles under Varied Temperatures")
        fig.show()

    def plot_product_profile_with_A_concs(self, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        temperature = operation_params[("Temperature", None, None, None, None)]
        batch_time = operation_params[("Batch_Time", None, None, None, None)]
        A_concs = operation_params[("Concentration", None, "Batch_Feed", None, "A")]
        data = {"Time (min)": [], "B concentration (mol/L)": [], "A concentration (mol/L)": []}
        for A_conc in A_concs:
            _operation_params = {
                ("Temperature", None, None, None, None): temperature,
                ("Batch_Time", None, None, None, None): batch_time,
                ("Concentration", None, "Batch_Feed", None, "A"): A_conc,
            }
            t, cs = self.run(_operation_params, reaction_params, transport_params)
            for _t, _c in zip(t, cs[self.species.index("B")]):
                data["Time (min)"].append(_t)
                data["B concentration (mol/L)"].append(_c)
                data["A concentration (mol/L)"].append(round(A_conc, 1))
        df = pd.DataFrame(data)
        fig = px.line(df, x="Time (min)", y="B concentration (mol/L)", color="A concentration (mol/L)")
        fig.update_layout(width=900, height=500, title="Product Concentration Profiles under Varied A Concentrations")
        fig.show()