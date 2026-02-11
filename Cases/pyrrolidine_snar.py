import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from .benchmark import Benchmark


class PyrrolidineSNAr(Benchmark):
    """Benchmark representing a nucleophilic aromatic substitution (SNAr) reaction.
    Pyrrolidine is the nucleophile to attack the electron-deficient aromatic ring.

    The SNAr reaction occurs in a plug flow reactor where residence time, nucleophile 
    concentration, and temperature can be adjusted.

    Parameters
    ----------
    random_seed: int, optional
        The Random seed to generate noises. Default is 0.

    Notes
    -----
    This benchmark relies on the kinetics observerd by [Hone] et al. The mechanistic 
    model is integrated using scipy to find outlet concentrations of all species.

    Parameter units :
        - 'Radius'                              : m
        - 'Length'                              : m
        - 'Activation_Energy'                   : kJ/mol
        - 'Referenced_Reaction_Rate_Constant'   : 
        - 'Concentration'                       : mol/L
        - 'Temperature'                         : oC
        - 'Residence Time'                      : min

    References
    ----------
    .. [Hone] C. A. Hone et al., React. Chem. Eng., 2017, 2, 103–108. DOI:
       `10.1039/C6RE00109B <https://doi.org/10.1039/C6RE00109B>`_

    """

    solids = []
    gases = []
    streams = ["Continuous_Flow"]
    reactions = [
        "dfnb + prld > ortho",
        "dfnb + prld > para",
        "ortho + prld > bis",
        "para + prld > bis",
    ]
    species = ["dfnb", "prld", "ortho", "para", "bis"]

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
        assert "Mass accumulation" in phenos and phenos["Mass accumulation"] == "Continuous", \
            "SNAr reaction is operated in a 'Continuous' flow reactor"
        assert "Flow pattern" in phenos and phenos["Flow pattern"] == "Tubular_Flow", \
            "SNAr reaction is operated with 'Tubular Flow'"
        assert "Mass transport" in phenos and phenos["Mass transport"] == [], \
            "The tubular flow is well mixed along the radical direction"

    def _setup_structure_params(self):
        structure_params = {}
        return structure_params

    def _setup_physics_params(self):
        physics_params = {}
        return physics_params

    def _setup_reaction_params(self):
        reaction_params = {
            ("Activation_Energy", None, "Continuous_Flow", "dfnb + prld > ortho", None):    33.3, # kJ/mol
            ("Activation_Energy", None, "Continuous_Flow", "dfnb + prld > para", None):     35.3, # kJ/mol
            ("Activation_Energy", None, "Continuous_Flow", "ortho + prld > bis", None):     38.9, # kJ/mol
            ("Activation_Energy", None, "Continuous_Flow", "para + prld > bis", None):      44.8, # kJ/mol
            ("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "dfnb + prld > ortho", None):    0.57900, # L/mol s
            ("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "dfnb + prld > para", None):     0.02700, # L/mol s
            ("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "ortho + prld > bis", None):     0.00865, # L/mol s
            ("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "para + prld > bis", None):      0.01630, # L/mol s
            ("Stoichiometric_Coefficient", None, None, "dfnb + prld > ortho", "dfnb"):  -1,
            ("Stoichiometric_Coefficient", None, None, "dfnb + prld > ortho", "prld"):  -1,
            ("Stoichiometric_Coefficient", None, None, "dfnb + prld > ortho", "ortho"): 1,
            ("Stoichiometric_Coefficient", None, None, "dfnb + prld > para", "dfnb"):   -1,
            ("Stoichiometric_Coefficient", None, None, "dfnb + prld > para", "prld"):   -1,
            ("Stoichiometric_Coefficient", None, None, "dfnb + prld > para", "para"):   1,
            ("Stoichiometric_Coefficient", None, None, "ortho + prld > bis", "ortho"):  -1,
            ("Stoichiometric_Coefficient", None, None, "ortho + prld > bis", "prld"):   -1,
            ("Stoichiometric_Coefficient", None, None, "ortho + prld > bis", "bis"):    1,
            ("Stoichiometric_Coefficient", None, None, "para + prld > bis", "para"):    -1,
            ("Stoichiometric_Coefficient", None, None, "para + prld > bis", "prld"):    -1,
            ("Stoichiometric_Coefficient", None, None, "para + prld > bis", "bis"):     1,
            ("Partial_Order", None, None, "dfnb + prld > ortho", "dfnb"):   1,
            ("Partial_Order", None, None, "dfnb + prld > ortho", "prld"):   1,
            ("Partial_Order", None, None, "dfnb + prld > para", "dfnb"):    1,
            ("Partial_Order", None, None, "dfnb + prld > para", "prld"):    1,
            ("Partial_Order", None, None, "ortho + prld > bis", "ortho"):   1,
            ("Partial_Order", None, None, "ortho + prld > bis", "prld"):    1,
            ("Partial_Order", None, None, "para + prld > bis", "para"):     1,
            ("Partial_Order", None, None, "para + prld > bis", "prld"):     1,
        }
        return reaction_params

    def _setup_transport_params(self):
        transport_params = {}
        return transport_params

    def _setup_operation_params(self):
        operation_params = {
            ("Concentration", None, "Continuous_Flow", None, "dfnb"): 0.2,  # mol/L
            ("Concentration", None, "Continuous_Flow", None, "prld"): None,  # mol/L
            ("Temperature", None, None, None, None): None,  # oC
            ("Residence_Time", None, None, None, None): None,  # min
        }
        return operation_params

    def _setup_operation_name2ind(self):
        operation_name2ind = {
            "prld_conc": ("Concentration", None, "Continuous_Flow", None, "prld"),
            "temp": ("Temperature", None, None, None, None),
            "t_r": ("Residence_Time", None, None, None, None),
        }
        return operation_name2ind

    def _setup_measure_ind2name(self):
        measure_ind2name = {
            ("Concentration", None, "Continuous_Flow", None, "ortho"): "outlet_ortho_conc",
        }
        return measure_ind2name

    def _setup_var2unit(self):
        var2unit = {
            "Activation_Energy": "kJ/mol",
            "Referenced_Reaction_Rate_Constant": None,
            "Temperature": "oC",
            "Concentration": "mol/L",
            "Residence_Time": "min",
        }
        return var2unit

    def _simulate(self, params):
        R = 8.314
        t_r = params[("Residence_Time", None, None, None, None)]
        t_r *= 60
        T = params[("Temperature", None, None, None, None)]
        T += 273.15
        c_0 = np.zeros((1, 5), dtype=np.float64)
        c_0[0, 0] = params[("Concentration", None, "Continuous_Flow", None, "dfnb")]
        c_0[0, 1] = params[("Concentration", None, "Continuous_Flow", None, "prld")]
        nu = np.zeros((4, 5), dtype=np.float64)
        nu[0, 0] = params[("Stoichiometric_Coefficient", None, None, "dfnb + prld > ortho", "dfnb")]
        nu[0, 1] = params[("Stoichiometric_Coefficient", None, None, "dfnb + prld > ortho", "prld")]
        nu[0, 2] = params[("Stoichiometric_Coefficient", None, None, "dfnb + prld > ortho", "ortho")]
        nu[1, 0] = params[("Stoichiometric_Coefficient", None, None, "dfnb + prld > para", "dfnb")]
        nu[1, 1] = params[("Stoichiometric_Coefficient", None, None, "dfnb + prld > para", "prld")]
        nu[1, 3] = params[("Stoichiometric_Coefficient", None, None, "dfnb + prld > para", "para")]
        nu[2, 1] = params[("Stoichiometric_Coefficient", None, None, "ortho + prld > bis", "prld")]
        nu[2, 2] = params[("Stoichiometric_Coefficient", None, None, "ortho + prld > bis", "ortho")]
        nu[2, 4] = params[("Stoichiometric_Coefficient", None, None, "ortho + prld > bis", "bis")]
        nu[3, 1] = params[("Stoichiometric_Coefficient", None, None, "para + prld > bis", "prld")]
        nu[3, 3] = params[("Stoichiometric_Coefficient", None, None, "para + prld > bis", "para")]
        nu[3, 4] = params[("Stoichiometric_Coefficient", None, None, "para + prld > bis", "bis")]
        n = np.zeros((4, 5), dtype=np.float64)
        n[0, 0] = params[("Partial_Order", None, None, "dfnb + prld > ortho", "dfnb")]
        n[0, 1] = params[("Partial_Order", None, None, "dfnb + prld > ortho", "prld")]
        n[1, 0] = params[("Partial_Order", None, None, "dfnb + prld > para", "dfnb")]
        n[1, 1] = params[("Partial_Order", None, None, "dfnb + prld > para", "prld")]
        n[2, 1] = params[("Partial_Order", None, None, "ortho + prld > bis", "prld")]
        n[2, 2] = params[("Partial_Order", None, None, "ortho + prld > bis", "ortho")]
        n[3, 1] = params[("Partial_Order", None, None, "para + prld > bis", "prld")]
        n[3, 3] = params[("Partial_Order", None, None, "para + prld > bis", "para")]
        A = np.zeros((1, 4), dtype=np.float64)
        A[0, 0] = params[("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "dfnb + prld > ortho", None)]
        A[0, 1] = params[("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "dfnb + prld > para", None)]
        A[0, 2] = params[("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "ortho + prld > bis", None)]
        A[0, 3] = params[("Referenced_Reaction_Rate_Constant", None, "Continuous_Flow", "para + prld > bis", None)]
        E_a = np.zeros((1, 4), dtype=np.float64)
        E_a[0, 0] = params[("Activation_Energy", None, "Continuous_Flow", "dfnb + prld > ortho", None)]
        E_a[0, 1] = params[("Activation_Energy", None, "Continuous_Flow", "dfnb + prld > para", None)]
        E_a[0, 2] = params[("Activation_Energy", None, "Continuous_Flow", "ortho + prld > bis", None)]
        E_a[0, 3] = params[("Activation_Energy", None, "Continuous_Flow", "para + prld > bis", None)]
        E_a *= 1000

        def _derivative(t, c):
            c = c.reshape((1, 5))
            r_r = np.zeros((1, 4), dtype=np.float64)
            r_r[0, 0] = A[0, 0] * np.exp(-E_a[0, 0] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[0])
            r_r[0, 1] = A[0, 1] * np.exp(-E_a[0, 1] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[1])
            r_r[0, 2] = A[0, 2] * np.exp(-E_a[0, 2] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[2])
            r_r[0, 3] = A[0, 3] * np.exp(-E_a[0, 3] / R * (1 / T - 1 / 363.15)) * np.prod(c[0] ** n[3])
            dc_dt = np.matmul(r_r, nu)
            return dc_dt

        t_eval = np.linspace(0, t_r, 201)
        res = solve_ivp(_derivative, (0, t_r), c_0.reshape(-1, ),
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

    def generate_lhs_dataset(self, operation_param_ranges, num_points):
        _operation_param_ranges = {
            self._operation_name2ind[n]: v for n, v in operation_param_ranges.items()
        }
        lhs = LatinHypercube(len(_operation_param_ranges), rng=self.random_seed)
        data = lhs.random(num_points)
        for i, (_, param_range) in enumerate(_operation_param_ranges.items()):
            data[:, i] = data[:, i] * (param_range[1] - param_range[0]) + param_range[0]
        dataset = pd.DataFrame(data, columns=list(operation_param_ranges.keys()))
        return self.run_dataset(dataset)

    def calibrate(self, cal_param_bounds, dataset):
        cal_param_bounds = self._param_list2dict(cal_param_bounds)
        cal_param_bounds = {k: v for k, v in cal_param_bounds.items() if v[0] != v[1]}
        params = self._param_list2dict(self.params())
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
                data["Species"].append(f"{i+1} {s}")
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
        residence_time = operation_params[("Residence_Time", None, None, None, None)]
        prld_conc = operation_params[("Concentration", None, "Continuous_Flow", None, "prld")]
        data = {"Time (min)": [], "3 Ortho concentration (mol/L)": [], "Temperature (oC)": []}
        for temperature in temperatures:
            _operation_params = {
                ("Temperature", None, None, None, None): temperature,
                ("Residence_Time", None, None, None, None): residence_time,
                ("Concentration", None, "Continuous_Flow", None, "prld"): prld_conc,
            }
            t, cs = self.run(_operation_params, reaction_params, transport_params)
            for _t, _c in zip(t, cs[self.species.index("ortho")]):
                data["Time (min)"].append(_t)
                data["3 Ortho concentration (mol/L)"].append(_c)
                data["Temperature (oC)"].append(temperature)
        df = pd.DataFrame(data)
        fig = px.line(df, x="Time (min)", y="3 Ortho concentration (mol/L)", color="Temperature (oC)")
        fig.update_layout(width=800, height=500, title="Product Concentration Profiles under Varied Temperatures")
        fig.show()

    def plot_product_profile_with_prld_concs(self, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        temperature = operation_params[("Temperature", None, None, None, None)]
        residence_time = operation_params[("Residence_Time", None, None, None, None)]
        prld_concs = operation_params[("Concentration", None, "Continuous_Flow", None, "prld")]
        data = {"Time (min)": [], "3 Ortho concentration (mol/L)": [], "Pyrrolidine concentration (mol/L)": []}
        for prld_conc in prld_concs:
            _operation_params = {
                ("Temperature", None, None, None, None): temperature,
                ("Residence_Time", None, None, None, None): residence_time,
                ("Concentration", None, "Continuous_Flow", None, "prld"): prld_conc,
            }
            t, cs = self.run(_operation_params, reaction_params, transport_params)
            for _t, _c in zip(t, cs[self.species.index("ortho")]):
                data["Time (min)"].append(_t)
                data["3 Ortho concentration (mol/L)"].append(_c)
                data["Pyrrolidine concentration (mol/L)"].append(round(prld_conc, 1))
        df = pd.DataFrame(data)
        fig = px.line(df, x="Time (min)", y="3 Ortho concentration (mol/L)", color="Pyrrolidine concentration (mol/L)")
        fig.update_layout(width=900, height=500, title="Product Concentration Profiles under Varied Pyrrolidine Concentrations")
        fig.show()
    
    def plot_product_conc_landscapes(self, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        temperatures = operation_params[("Temperature", None, None, None, None)]
        residence_times = operation_params[("Residence_Time", None, None, None, None)]
        prld_concs = operation_params[("Concentration", None, "Continuous_Flow", None, "prld")]
        shape = (len(residence_times), len(prld_concs))
        
        data = []
        for temperature in temperatures:
            d = {
                "Temperature (oC)": temperature,
                "Residence time (min)": [], 
                "2 Pyrrolidine concentration (mol/L)": [], 
                "3 Ortho concentration (mol/L)": [], 
            }
            for residence_time in residence_times:
                for prld_conc in prld_concs:
                    _operation_params = {
                        ("Temperature", None, None, None, None): temperature,
                        ("Residence_Time", None, None, None, None): residence_time,
                        ("Concentration", None, "Continuous_Flow", None, "prld"): prld_conc,
                    }
                    t, cs = self.run(_operation_params, reaction_params, transport_params)
                    d["Residence time (min)"].append(residence_time)
                    d["2 Pyrrolidine concentration (mol/L)"].append(prld_conc)
                    d["3 Ortho concentration (mol/L)"].append(cs[2][-1])
            d["Residence time (min)"] = np.array(d["Residence time (min)"]).reshape(shape)
            d["2 Pyrrolidine concentration (mol/L)"] = np.array(d["2 Pyrrolidine concentration (mol/L)"]).reshape(shape)
            d["3 Ortho concentration (mol/L)"] = np.array(d["3 Ortho concentration (mol/L)"]).reshape(shape)
            data.append(d)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=d["Residence time (min)"], 
                    y=d["2 Pyrrolidine concentration (mol/L)"], 
                    z=d["3 Ortho concentration (mol/L)"], 
                    coloraxis="coloraxis",
                ) for d in data
            ] + [
                go.Scatter3d(
                    x=[d["Residence time (min)"][0, -1]], 
                    y=[d["2 Pyrrolidine concentration (mol/L)"][0, -1]], 
                    z=[d["3 Ortho concentration (mol/L)"][0, -1]], 
                    mode="text",
                    text=[f"T = {d['Temperature (oC)']} oC"],
                    textposition="bottom right",
                    textfont=dict(size=12, color="red"),
                    showlegend=False,
                ) for d in data
            ]
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(tickmode="array", tickvals=[0.5, 1, 1.5, 2], title="Residence time (min)"),
                yaxis=dict(tickmode="array", tickvals=[0.1, 0.2, 0.3, 0.4, 0.5], title="2 Pyrrolidine concentration (M)"),
                zaxis=dict(tickmode="array",tickvals=[0.05, 0.1, 0.15, 0.2], title="3 Ortho concentration (M)"),
            ),
            coloraxis=dict(colorscale="Viridis", cmin=0.06, cmax=0.18),
            width=900, height=700,
            scene_camera = dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            title="Product Concentration Landscapes"
        )
        fig.show()
    
    def plot_product_conc_landscape_with_ground_truth(
            self, 
            operation_params, 
            cal_reaction_params=None, 
            cal_transport_params=None,
            reaction_params=None, 
            transport_params=None
        ):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        if cal_reaction_params:
            cal_reaction_params = self._param_list2dict(cal_reaction_params)
        if cal_transport_params:
            cal_transport_params = self._param_list2dict(cal_transport_params)
        temperature = operation_params[("Temperature", None, None, None, None)]
        residence_times = operation_params[("Residence_Time", None, None, None, None)]
        prld_concs = operation_params[("Concentration", None, "Continuous_Flow", None, "prld")]
        shape = (len(residence_times), len(prld_concs))

        gt_data = {
            "Temperature (oC)": temperature,
            "Residence time (min)": [], 
            "2 Pyrrolidine concentration (mol/L)": [], 
            "3 Ortho concentration (mol/L)": [], 
        }
        for residence_time in residence_times:
            for prld_conc in prld_concs:
                _operation_params = {
                    ("Temperature", None, None, None, None): temperature,
                    ("Residence_Time", None, None, None, None): residence_time,
                    ("Concentration", None, "Continuous_Flow", None, "prld"): prld_conc,
                }
                t, cs = self.run(_operation_params, reaction_params, transport_params)
                gt_data["Residence time (min)"].append(residence_time)
                gt_data["2 Pyrrolidine concentration (mol/L)"].append(prld_conc)
                gt_data["3 Ortho concentration (mol/L)"].append(cs[2][-1])
        gt_data["Residence time (min)"] = np.array(gt_data["Residence time (min)"]).reshape(shape)
        gt_data["2 Pyrrolidine concentration (mol/L)"] = np.array(gt_data["2 Pyrrolidine concentration (mol/L)"]).reshape(shape)
        gt_data["3 Ortho concentration (mol/L)"] = np.array(gt_data["3 Ortho concentration (mol/L)"]).reshape(shape)

        cal_data = {
            "Temperature (oC)": temperature,
            "Residence time (min)": [], 
            "2 Pyrrolidine concentration (mol/L)": [], 
            "3 Ortho concentration (mol/L)": [], 
        }
        for residence_time in residence_times:
            for prld_conc in prld_concs:
                _operation_params = {
                    ("Temperature", None, None, None, None): temperature,
                    ("Residence_Time", None, None, None, None): residence_time,
                    ("Concentration", None, "Continuous_Flow", None, "prld"): prld_conc,
                }
                t, cs = self.run(_operation_params, cal_reaction_params, cal_transport_params)
                cal_data["Residence time (min)"].append(residence_time)
                cal_data["2 Pyrrolidine concentration (mol/L)"].append(prld_conc)
                cal_data["3 Ortho concentration (mol/L)"].append(cs[2][-1])
        cal_data["Residence time (min)"] = np.array(cal_data["Residence time (min)"]).reshape(shape)
        cal_data["2 Pyrrolidine concentration (mol/L)"] = np.array(cal_data["2 Pyrrolidine concentration (mol/L)"]).reshape(shape)
        cal_data["3 Ortho concentration (mol/L)"] = np.array(cal_data["3 Ortho concentration (mol/L)"]).reshape(shape)

        fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=gt_data["Residence time (min)"], 
                y=gt_data["2 Pyrrolidine concentration (mol/L)"], 
                z=gt_data["3 Ortho concentration (mol/L)"], 
                colorscale='Viridis',
                colorbar=dict(title='Ground-truth', len=0.8, x=1.05),
                cmin=0.1,
                cmax=0.185,
                name='Ground-truth',
                showscale=True,
            )
        )
        fig.add_trace(
            go.Surface(
                x=cal_data["Residence time (min)"], 
                y=cal_data["2 Pyrrolidine concentration (mol/L)"], 
                z=cal_data["3 Ortho concentration (mol/L)"], 
                colorscale='Thermal',
                colorbar=dict(title='Calibrated model', len=0.8, x=1.25),
                cmin=0.1,
                cmax=0.185,
                name='Calibrated model',
                showscale=True
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(tickmode="array", tickvals=[0.5, 1, 1.5, 2], title="Residence time (min)"),
                yaxis=dict(tickmode="array", tickvals=[0.1, 0.2, 0.3, 0.4, 0.5], title="2 Pyrrolidine concentration (M)"),
                zaxis=dict(tickmode="array",tickvals=[0.05, 0.1, 0.15, 0.2], title="3 Ortho concentration (M)"),
            ),
            # coloraxis=dict(colorscale="Viridis", cmin=0.06, cmax=0.18),
            width=900, height=700,
            scene_camera = dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            title="Modelled vs Ground-truth Product Concentration Landscapes"
        )
        fig.show()