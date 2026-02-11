import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.stats.qmc import LatinHypercube

from .benchmark import Benchmark


operation_name2ind = {
    "m_rdy":        ("Initial_Mass_Solid", "Solid_Feedstock", None, None, "RDY"),   # kg
    "t_b":          ("Batch_Time", None, None, None, None),                         # min
    "temp":         ("Temperature", None, None, None, None),                        # oC
    "pressure":     ("Pressure", None, None, None, None),                           # bar
    "agitation":    ("Agitation", None, None, None, None),                          # rpm
}

measure_ind2name = {
    ("Mass", None, "Batch_Feed", None, "RDEt"):           "m_rdet",       # kg
    ("Mass", None, "Batch_Feed", None, "dimer"):          "m_dimer",      # kg
    ("Mass_Gas_Used", "Gas_Flow", None, None, "H2"):        "h2_used",      # kg
    ("Mass_Solid", "Solid_Feedstock", None, None, "RDY"):   "m_rdy_left",   # kg
}


def _set_params(params):
    R = 8.314
    t_b = params[("Batch_Time", None, None, None, None)]
    t_b *= 60
    T = params[("Temperature", None, None, None, None)]
    T += 273.15
    P = params[("Pressure", None, None, None, None)]
    P *= 101325
    N = params[("Agitation", None, None, None, None)]
    m_0 = np.zeros((1, 13), dtype=np.float64)
    m_0[0, 0] = params[("Initial_Mass", None, "Batch_Feed", None, "water")]
    m_0[0, 1] = params[("Initial_Mass", None, "Batch_Feed", None, "solvent")]
    m_0[0, 9] = params[("Initial_Mass", None, "Batch_Feed", None, "cat2")]
    m_0[0, 11] = params[("Initial_Mass", None, "Batch_Feed", None, "catX")]
    m_s_0 = np.zeros((1, 13), dtype=np.float64)
    m_s_0[0, 2] = params[("Initial_Mass_Solid", "Solid_Feedstock", None, None, "RDY")]
    x_g = np.zeros((1, 13), dtype=np.float64)
    x_g[0, 8] = params[("Mass_Gas_Fraction", "Gas_Flow", None, None, "H2")]
    
    M = np.zeros(13, dtype=np.float64)
    M[0] = params[("Molecular_Weight", None, None, None, "water")]
    M[1] = params[("Molecular_Weight", None, None, None, "solvent")]
    M[2] = params[("Molecular_Weight", None, None, None, "RDY")]
    M[3] = params[("Molecular_Weight", None, None, None, "RYE")]
    M[4] = params[("Molecular_Weight", None, None, None, "RYA")]
    M[5] = params[("Molecular_Weight", None, None, None, "REA")]
    M[6] = params[("Molecular_Weight", None, None, None, "RDEt")]
    M[7] = params[("Molecular_Weight", None, None, None, "dimer")]
    M[8] = params[("Molecular_Weight", None, None, None, "H2")]
    M[9] = params[("Molecular_Weight", None, None, None, "cat2")]
    M[10] = params[("Molecular_Weight", None, None, None, "cat-H")]
    M[11] = params[("Molecular_Weight", None, None, None, "catX")]
    M[12] = params[("Molecular_Weight", None, None, None, "catX-RDEt")]
    ρ = np.zeros(1, dtype=np.float64)
    ρ[0] = params[("Density", None, "Batch_Feed", None, None)]
    ρ_s = np.zeros(1, dtype=np.float64)
    ρ_s[0] = params[("Density_Solid", "Solid_Feedstock", None, None, None)]

    nu = np.zeros((10, 13), dtype=np.float64)
    nu[0, 8] = params[("Stoichiometric_Coefficient", None, None, "H2 + cat2 > 2 cat-H", "H2")]
    nu[0, 9] = params[("Stoichiometric_Coefficient", None, None, "H2 + cat2 > 2 cat-H", "cat2")]
    nu[0, 10] = params[("Stoichiometric_Coefficient", None, None, "H2 + cat2 > 2 cat-H", "cat-H")]
    nu[1, 8] = params[("Stoichiometric_Coefficient", None, None, "2 cat-H > H2 + cat2", "H2")]
    nu[1, 9] = params[("Stoichiometric_Coefficient", None, None, "2 cat-H > H2 + cat2", "cat2")]
    nu[1, 10] = params[("Stoichiometric_Coefficient", None, None, "2 cat-H > H2 + cat2", "cat-H")]
    nu[2, 2] = params[("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "RDY")]
    nu[2, 3] = params[("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "RYE")]
    nu[2, 9] = params[("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "cat2")]
    nu[2, 10] = params[("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "cat-H")]
    nu[3, 3] = params[("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "RYE")]
    nu[3, 4] = params[("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "RYA")]
    nu[3, 9] = params[("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "cat2")]
    nu[3, 10] = params[("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "cat-H")]
    nu[4, 4] = params[("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "RYA")]
    nu[4, 5] = params[("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "REA")]
    nu[4, 9] = params[("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "cat2")]
    nu[4, 10] = params[("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "cat-H")]
    nu[5, 5] = params[("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "REA")]
    nu[5, 6] = params[("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "RDEt")]
    nu[5, 9] = params[("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "cat2")]
    nu[5, 10] = params[("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "cat-H")]
    nu[6, 6] = params[("Stoichiometric_Coefficient", None, None, "RDEt + catX > catX-RDEt", "RDEt")]
    nu[6, 11] = params[("Stoichiometric_Coefficient", None, None, "RDEt + catX > catX-RDEt", "catX")]
    nu[6, 12] = params[("Stoichiometric_Coefficient", None, None, "RDEt + catX > catX-RDEt", "catX-RDEt")]
    nu[7, 6] = params[("Stoichiometric_Coefficient", None, None, "catX-RDEt > RDEt + catX", "catX-RDEt")]
    nu[7, 11] = params[("Stoichiometric_Coefficient", None, None, "catX-RDEt > RDEt + catX", "catX-RDEt")]
    nu[7, 12] = params[("Stoichiometric_Coefficient", None, None, "catX-RDEt > RDEt + catX", "catX-RDEt")]
    nu[8, 3] = params[("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "RYE")]
    nu[8, 7] = params[("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "dimer")]
    nu[8, 11] = params[("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "catX")]
    nu[8, 12] = params[("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "catX-RDEt")]
    nu[9, 5] = params[("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "REA")]
    nu[9, 7] = params[("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "dimer")]
    nu[9, 8] = params[("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "H2")]
    nu[9, 11] = params[("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "catX")]
    nu[9, 12] = params[("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "catX-RDEt")]
    n = np.zeros((10, 13), dtype=np.float64)
    n[0, 8] = params[("Partial_Order", None, None, "H2 + cat2 > 2 cat-H", "H2")]
    n[0, 9] = params[("Partial_Order", None, None, "H2 + cat2 > 2 cat-H", "cat2")]
    n[1, 10] = params[("Partial_Order", None, None, "2 cat-H > H2 + cat2", "cat-H")]
    n[2, 2] = params[("Partial_Order", None, None, "RDY + 2 cat-H > RYE + cat2", "RDY")]
    n[2, 10] = params[("Partial_Order", None, None, "RDY + 2 cat-H > RYE + cat2", "cat-H")]
    n[3, 3] = params[("Partial_Order", None, None, "RYE + 2 cat-H > RYA + cat2", "RYE")]
    n[3, 10] = params[("Partial_Order", None, None, "RYE + 2 cat-H > RYA + cat2", "cat-H")]
    n[4, 4] = params[("Partial_Order", None, None, "RYA + 2 cat-H > REA + cat2", "RYA")]
    n[4, 10] = params[("Partial_Order", None, None, "RYA + 2 cat-H > REA + cat2", "cat-H")]
    n[5, 5] = params[("Partial_Order", None, None, "REA + 2 cat-H > RDEt + cat2", "REA")]
    n[5, 10] = params[("Partial_Order", None, None, "REA + 2 cat-H > RDEt + cat2", "cat-H")]
    n[6, 6] = params[("Partial_Order", None, None, "RDEt + catX > catX-RDEt", "RDEt")]
    n[6, 11] = params[("Partial_Order", None, None, "RDEt + catX > catX-RDEt", "catX")]
    n[7, 12] = params[("Partial_Order", None, None, "catX-RDEt > RDEt + catX", "catX-RDEt")]
    n[8, 3] = params[("Partial_Order", None, None, "RYE + catX-RDEt > dimer + catX", "RYE")]
    n[8, 12] = params[("Partial_Order", None, None, "RYE + catX-RDEt > dimer + catX", "catX-RDEt")]
    n[9, 5] = params[("Partial_Order", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "REA")]
    n[9, 12] = params[("Partial_Order", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "catX-RDEt")]
    k_ref = np.zeros((1, 10), dtype=np.float64)
    k_ref[0, 2] = params[("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "RDY + 2 cat-H > RYE + cat2", None)]
    k_ref[0, 4] = params[("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "RYA + 2 cat-H > REA + cat2", None)]
    k_ref[0, 8] = params[("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "RYE + catX-RDEt > dimer + catX", None)]
    k_ref[0, 9] = params[("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "REA + catX-RDEt > 2 H2 + dimer + catX", None)]
    E_a = np.zeros((1, 10), dtype=np.float64)
    E_a[0, 2] = params[("Activation_Energy", None, "Batch_Feed", "RDY + 2 cat-H > RYE + cat2", None)]
    E_a[0, 4] = params[("Activation_Energy", None, "Batch_Feed", "RYA + 2 cat-H > REA + cat2", None)]
    E_a[0, 8] = params[("Activation_Energy", None, "Batch_Feed", "RYE + catX-RDEt > dimer + catX", None)]
    E_a[0, 9] = params[("Activation_Energy", None, "Batch_Feed", "REA + catX-RDEt > 2 H2 + dimer + catX", None)]
    E_a *= 1000
    k = np.zeros((1, 10), dtype=np.float64)
    k[0, 0] = params[("Rate_Constant", None, "Batch_Feed", "H2 + cat2 > 2 cat-H", None)]
    k[0, 1] = params[("Rate_Constant", None, "Batch_Feed", "2 cat-H > H2 + cat2", None)]
    k[0, 3] = params[("Rate_Constant", None, "Batch_Feed", "RYE + 2 cat-H > RYA + cat2", None)]
    k[0, 5] = params[("Rate_Constant", None, "Batch_Feed", "REA + 2 cat-H > RDEt + cat2", None)]
    k[0, 6] = params[("Rate_Constant", None, "Batch_Feed", "RDEt + catX > catX-RDEt", None)]
    k[0, 7] = params[("Rate_Constant", None, "Batch_Feed", "catX-RDEt > RDEt + catX", None)]
    
    k_Sa_D = np.zeros((1, 1, 13), dtype=np.float64)
    k_Sa_D[0, 0, 2] = params[("Solid-Liquid_Volumetric_Mass_Transfer_Coefficient_D", "Solid_Feedstock", "Batch_Feed", None, "RDY")]
    k_Sa_alpha = np.zeros((1, 1, 13), dtype=np.float64)
    k_Sa_alpha[0, 0, 2] = params[("Solid-Liquid_Volumetric_Mass_Transfer_Coefficient_alpha", "Solid_Feedstock", "Batch_Feed", None, "RDY")]
    k_La_D = np.zeros((1, 1, 13), dtype=np.float64)
    k_La_D[0, 0, 8] = params[("Gas-Liquid_Volumetric_Mass_Transfer_Coefficient_D", "Gas_Flow", "Batch_Feed", None, "H2")]
    k_La_alpha = np.zeros((1, 1, 13), dtype=np.float64)
    k_La_alpha[0, 0, 8] = params[("Gas-Liquid_Volumetric_Mass_Transfer_Coefficient_alpha", "Gas_Flow", "Batch_Feed", None, "H2")]
    k_H = np.zeros((1, 1, 13), dtype=np.float64)
    k_H[0, 0, 8] = params[("Henry's_Constant", "Gas_Flow", "Batch_Feed", None, "H2")]
    c_g_star_const = np.zeros((1, 1, 13), dtype=np.float64)
    c_g_star_const[0, 0, 8] = params[("Constant_Gas_Saturated_Concentration", "Gas_Flow", "Batch_Feed", None, "H2")]
    c_s_star_const = np.zeros((1, 1, 13), dtype=np.float64)
    c_s_star_const[0, 0, 2] = params[("Constant_Solid_Saturated_Concentration", "Solid_Feedstock", "Batch_Feed", None, "RDY")]
    
    return R, t_b, T, P, N, m_0, m_s_0, x_g, M, ρ, ρ_s, nu, n, k_ref, E_a, k, k_Sa_D, k_Sa_alpha, k_La_D, k_La_alpha, k_H, c_g_star_const, c_s_star_const

def _derivative(t, m, R, t_b, T, P, N, m_0, m_s_0, x_g, M, ρ, ρ_s, nu, n, k_ref, E_a, k, k_Sa_D, k_Sa_alpha, k_La_D, k_La_alpha, k_H, c_g_star_const, c_s_star_const, phenos):
    m_g = np.zeros((1, 13), dtype=np.float64)
    m_g[0, 8] = m[13]
    m_s = np.zeros((1, 13), dtype=np.float64)
    m_s[0, 2] = m[14]
    m = np.array(m[:13], dtype=np.float64)
    m = m.reshape((1, 13))
    
    # concentration
    c = np.zeros((1, 13), dtype=np.float64)
    c[0] = (m[0] * ρ[0]) / (np.sum(m[0]) * M)

    # definition

    # reaction
    r_r = np.zeros((1, 10), dtype=np.float64)
    # stream: Batch stream, reaction: H2 + cat2 > 2 cat-H
    r_r[0, 0] = k[0, 0] * np.prod(c[0] ** n[0])

    # stream: Batch stream, reaction: 2 cat-H > H2 + cat2
    r_r[0, 1] = k[0, 1] * np.prod(c[0] ** n[1])

    # stream: Batch stream, reaction: RDY + 2 cat-H > RYE + cat2
    r_r[0, 2] = k_ref[0, 2] * np.exp(- E_a[0, 2] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[2])

    # stream: Batch stream, reaction: RYE + 2 cat-H > RYA + cat2
    r_r[0, 3] = k[0, 3] * np.prod(c[0] ** n[3])

    # stream: Batch stream, reaction: RYA + 2 cat-H > REA + cat2
    r_r[0, 4] = k_ref[0, 4] * np.exp(- E_a[0, 4] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[4])

    # stream: Batch stream, reaction: REA + 2 cat-H > RDEt + cat2
    r_r[0, 5] = k[0, 5] * np.prod(c[0] ** n[5])

    # stream: Batch stream, reaction: RDEt + catX > catX-RDEt
    r_r[0, 6] = k[0, 6] * np.prod(c[0] ** n[6])

    # stream: Batch stream, reaction: catX-RDEt > RDEt + catX
    r_r[0, 7] = k[0, 7] * np.prod(c[0] ** n[7])

    # stream: Batch stream, reaction: RYE + catX-RDEt > dimer + catX
    r_r[0, 8] = k_ref[0, 8] * np.exp(- E_a[0, 8] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[8])

    # stream: Batch stream, reaction: REA + catX-RDEt > 2 H2 + dimer + catX
    r_r[0, 9] = k_ref[0, 9] * np.exp(- E_a[0, 9] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[9])
    
    r_r = np.maximum(r_r, 0)
    
    # mass equilibrium
    c_g_star = np.zeros((1, 1, 13), dtype=np.float64)
    if phenos["param law"]["Gas_Dissolution_Saturated_Concentration"] == "Henry's_Law":
        c_g_star[0, 0, 8] = (P * x_g[0, 8]) / (1000 * R * T * k_H[0, 0, 8])
    if phenos["param law"]["Gas_Dissolution_Saturated_Concentration"] == "Constant":
        c_g_star[0, 0, 8] = c_g_star_const[0, 0, 8]

    c_s_star = np.zeros((1, 1, 13), dtype=np.float64)
    if "Solid_Dissolution_Saturated_Concentration" in phenos["param law"] and \
        phenos["param law"]["Solid_Dissolution_Saturated_Concentration"] == "Fitted":
        def calc_solid_dissolution_equilibrium_term(m):
            return 0.00159 * np.exp(-0.1126 * (100 * np.maximum(m[6], 0) / np.sum(m)) ** 0.83)
        c_s_star[0, 0, 2] = calc_solid_dissolution_equilibrium_term(m[0])
    if phenos["param law"]["Solid_Dissolution_Saturated_Concentration"] == "Constant":
        c_s_star[0, 0, 2] = c_s_star_const[0, 0, 2]

    # mass transport
    r_t_g = np.zeros((1, 1, 13), dtype=np.float64)
    r_t_g[0, 0, 8] = k_La_D[0, 0, 8] * N ** k_La_alpha[0, 0, 8] * (c_g_star[0, 0, 8] - c[0, 8])

    r_t_s = np.zeros((1, 1, 13), dtype=np.float64)
    r_t_s[0, 0, 2] = k_Sa_D[0, 0, 2] * N ** k_Sa_alpha[0, 0, 2] * (m_s[0, 2] * ρ[0]) / (np.sum(m[0]) * ρ_s[0]) * (c_s_star[0, 0, 2] - c[0, 2])
    if m_s[0, 2] < 1e-6:
        r_t_s[0, 0, 2] = 0
    
    # accumulation
    dm = np.zeros((1, 13), dtype=np.float64)
    dm[0, 0] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 0]) + np.sum(r_t_g[:, 0, 0]) + np.sum(r_t_s[:, 0, 0])) * M[0]
    dm[0, 1] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 1]) + np.sum(r_t_g[:, 0, 1]) + np.sum(r_t_s[:, 0, 1])) * M[1]
    dm[0, 2] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 2]) + np.sum(r_t_g[:, 0, 2]) + np.sum(r_t_s[:, 0, 2])) * M[2]
    dm[0, 3] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 3]) + np.sum(r_t_g[:, 0, 3]) + np.sum(r_t_s[:, 0, 3])) * M[3]
    dm[0, 4] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 4]) + np.sum(r_t_g[:, 0, 4]) + np.sum(r_t_s[:, 0, 4])) * M[4]
    dm[0, 5] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 5]) + np.sum(r_t_g[:, 0, 5]) + np.sum(r_t_s[:, 0, 5])) * M[5]
    dm[0, 6] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 6]) + np.sum(r_t_g[:, 0, 6]) + np.sum(r_t_s[:, 0, 6])) * M[6]
    dm[0, 7] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 7]) + np.sum(r_t_g[:, 0, 7]) + np.sum(r_t_s[:, 0, 7])) * M[7]
    dm[0, 8] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 8]) + np.sum(r_t_g[:, 0, 8]) + np.sum(r_t_s[:, 0, 8])) * M[8]
    dm[0, 9] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 9]) + np.sum(r_t_g[:, 0, 9]) + np.sum(r_t_s[:, 0, 9])) * M[9]
    dm[0, 10] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 10]) + np.sum(r_t_g[:, 0, 10]) + np.sum(r_t_s[:, 0, 10])) * M[10]
    dm[0, 11] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 11]) + np.sum(r_t_g[:, 0, 11]) + np.sum(r_t_s[:, 0, 11])) * M[11]
    dm[0, 12] = np.sum(m[0]) / ρ[0] * (np.matmul(r_r[0], nu[:, 12]) + np.sum(r_t_g[:, 0, 12]) + np.sum(r_t_s[:, 0, 12])) * M[12]
    dm_g = np.zeros((1, 1, 13), dtype=np.float64)
    dm_g[0, 0, 8] = np.sum(m[0]) / ρ[0] * np.sum(r_t_g[0, 0, 8]) * M[8]
    dm_s = np.zeros((1, 1, 13), dtype=np.float64)
    dm_s[0, 0, 2] = -np.sum(m[0]) / ρ[0] * np.sum(r_t_s[0, 0, 2]) * M[2]

    dm = dm.reshape(-1, ).tolist()
    dm.append(dm_g[0, 0, 8])
    dm.append(dm_s[0, 0, 2])
    dm = np.array(dm, dtype=np.float64)
    
    return dm

def _simulate(params, phenos):
    (R, t_b, T, P, N, m_0, m_s_0, x_g, M, ρ, ρ_s, nu, n, k_ref, E_a, k, k_Sa_D, k_Sa_alpha,
     k_La_D, k_La_alpha, k_H, c_g_star_const, c_s_star_const) = _set_params(params)

    m_0 = np.concatenate([m_0.reshape(-1), np.array([0, m_s_0[0, 2]], dtype=np.float64)])
    t_eval = np.linspace(0, t_b, 201, dtype=np.float64)
    res = solve_ivp(_derivative, (0, t_b), m_0, args=(R, t_b, T, P, N, m_0, m_s_0, x_g, M, ρ, ρ_s, nu, n, k_ref, E_a, k, k_Sa_D,
                    k_Sa_alpha, k_La_D, k_La_alpha, k_H, c_g_star_const, c_s_star_const, phenos), t_eval=t_eval, method='LSODA', atol=1e-12)
    t_eval /= 60
    return t_eval, res.y

def calc_se(p, params, cal_param_bounds, phenos, dataset):
    se = 0
    cal_params = {cal_param_ind: _p for cal_param_ind, _p in zip(cal_param_bounds.keys(), p)}
    params.update(cal_params)
    for i in range(len(dataset)):
        operation_params = {ind: dataset.loc[i, name] for name, ind in operation_name2ind.items()}
        params.update(operation_params)
        t, ms = _simulate(params, phenos)
        for ind, name in measure_ind2name.items():
            if ind[0] == "Mass":
                se += (ms[Hydrogenation.species.index(ind[-1]), -1] - dataset.loc[i, name])**2
            if ind == ("Mass_Gas_Used", "Gas_Flow", None, None, "H2"):
                se += (ms[-2, -1] - dataset.loc[i, name])**2
            if ind == ("Mass_Solid", "Solid_Feedstock", None, None, "RDY"):
                se += (ms[-1, -1] - dataset.loc[i, name])**2
    return se


class Hydrogenation(Benchmark):
    """Benchmark representing a R-di-yne hydrogenation process.
    The process targets at producing R-di-ethyl, with dimer as the side product.

    The hydrogenation reaction occurs in a batch reactor with an agitator and a cooling jacket.

    Parameters
    ----------
    random_seed: int, optional
        The Random seed to generate noises. Default is 0.

    Notes
    -----
    This benchmark relies on experimental data from Syngenta. The mechanistic 
    model is integrated using scipy to find outlet concentrations of all species.

    Parameter units :
        - 'Mass'                                                : kg
        - 'Mass_Gas'                                            : kg
        - 'Mass_Gas_Fraction'                                   : 
        - 'Mass_Solid'                                          : kg
        - 'Density'                                             : kg/m3
        - 'Density_Solid'                                       : kg/m3
        - 'Concentration'                                       : mol/L
        - 'Activation_Energy'                                   : kJ/mol
        - 'Referenced_Reaction_Rate_Constant'                   : 
        - 'Gas-Liquid_Volumetric_Mass_Transfer_Coefficient'     : /s
        - 'Solid-Liquid_Volumetric_Mass_Transfer_Coefficient'   : /s
        - 'Saturated_Concentration'                             : mol/L
        - 'Henry's_Constant'                                    : 
        - 'Temperature'                                         : oC
        - 'Pressure'                                            : bar
        - 'Batch_Time'                                          : min

    """

    solids = ["Solid_Feedstock"]
    gases = ["Gas_Flow"]
    streams = ["Batch_Feed"]
    reactions = [
        "H2 + cat2 > 2 cat-H",
        "2 cat-H > H2 + cat2",
        "RDY + 2 cat-H > RYE + cat2",
        "RYE + 2 cat-H > RYA + cat2",
        "RYA + 2 cat-H > REA + cat2",
        "REA + 2 cat-H > RDEt + cat2",
        "RDEt + catX > catX-RDEt",
        "catX-RDEt > RDEt + catX",
        "RYE + catX-RDEt > dimer + catX",
        "REA + catX-RDEt > 2 H2 + dimer + catX"
    ]
    species = ["water", "solvent", "RDY", "RYE", "RYA", "REA", "RDEt", "dimer", "H2", "cat2", "cat-H", "catX", "catX-RDEt"]

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
            "Hydrogenation is operated in a 'Batch' reactor"
        assert "Flow pattern" in phenos and phenos["Flow pattern"] == "Well_Mixed", \
            "Hydrogenation is operated with 'Well_Mixed' stream in batch"
        assert "Mass transport" in phenos and all([
            pheno in ["Gas-Liquid_Mass_Transfer", "Solid-Liquid_Mass_Transfer"] 
            for pheno in phenos["Mass transport"]]), "Unknown mass transport phenomena"
        if "Gas-Liquid_Mass_Transfer" in phenos["Mass transport"]:
            assert "param law" in phenos and \
                "Gas_Dissolution_Saturated_Concentration" in phenos["param law"] and \
                phenos["param law"]["Gas_Dissolution_Saturated_Concentration"] in ["Henry's_Law", "Constant"], \
                "Gas_Dissolution_Saturated_Concentration can only be 'Henry's_Law' or 'Constant'"
        if "Solid-Liquid_Mass_Transfer" in phenos["Mass transport"]:
            assert "param law" in phenos and \
                "Solid_Dissolution_Saturated_Concentration" in phenos["param law"] and \
                phenos["param law"]["Solid_Dissolution_Saturated_Concentration"] in ["Fitted", "Constant"], \
                "Solid_Dissolution_Saturated_Concentration can only be 'Fitted' or 'Constant'"

    def _setup_structure_params(self):
        structure_params = {}
        return structure_params

    def _setup_physics_params(self):
        physics_params = {
            ("Molecular_Weight", None, None, None, "water"):        18,     # g/mol
            ("Molecular_Weight", None, None, None, "solvent"):      60,     # g/mol
            ("Molecular_Weight", None, None, None, "RDY"):          150,    # g/mol
            ("Molecular_Weight", None, None, None, "RYE"):          152,    # g/mol
            ("Molecular_Weight", None, None, None, "RYA"):          154,    # g/mol
            ("Molecular_Weight", None, None, None, "REA"):          156,    # g/mol
            ("Molecular_Weight", None, None, None, "RDEt"):         158,    # g/mol
            ("Molecular_Weight", None, None, None, "dimer"):        310,    # g/mol
            ("Molecular_Weight", None, None, None, "H2"):           2,      # g/mol
            ("Molecular_Weight", None, None, None, "cat2"):         100,    # g/mol
            ("Molecular_Weight", None, None, None, "cat-H"):        51,     # g/mol
            ("Molecular_Weight", None, None, None, "catX"):         100,    # g/mol
            ("Molecular_Weight", None, None, None, "catX-RDEt"):    258,    # g/mol

            ("Density", None, "Batch_Feed", None, None):          1100,   # kg/m3
            
            ("Density_Solid", "Solid_Feedstock", None, None, None): 1000,   # kg/m3
        }
        return physics_params

    def _setup_reaction_params(self):
        reaction_params = {
            ("Rate_Constant", None, "Batch_Feed", "H2 + cat2 > 2 cat-H", None):           130,    # L/mol s
            ("Rate_Constant", None, "Batch_Feed", "2 cat-H > H2 + cat2", None):           10,     # L/mol s
            ("Rate_Constant", None, "Batch_Feed", "RYE + 2 cat-H > RYA + cat2", None):    50,     # L/mol s
            ("Rate_Constant", None, "Batch_Feed", "REA + 2 cat-H > RDEt + cat2", None):   50,     # L/mol s
            ("Rate_Constant", None, "Batch_Feed", "RDEt + catX > catX-RDEt", None):       1,      # L/mol s
            ("Rate_Constant", None, "Batch_Feed", "catX-RDEt > RDEt + catX", None):       0.001,  # /s

            ("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "RDY + 2 cat-H > RYE + cat2", None):            3, # L/mol s
            ("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "RYA + 2 cat-H > REA + cat2", None):            3, # L/mol s
            ("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "RYE + catX-RDEt > dimer + catX", None):        1, # L/mol s
            ("Referenced_Reaction_Rate_Constant", None, "Batch_Feed", "REA + catX-RDEt > 2 H2 + dimer + catX", None): 1, # L/mol s
            
            ("Activation_Energy", None, "Batch_Feed", "RDY + 2 cat-H > RYE + cat2", None):            30, # kJ/mol
            ("Activation_Energy", None, "Batch_Feed", "RYA + 2 cat-H > REA + cat2", None):            30, # kJ/mol
            ("Activation_Energy", None, "Batch_Feed", "RYE + catX-RDEt > dimer + catX", None):        38, # kJ/mol
            ("Activation_Energy", None, "Batch_Feed", "REA + catX-RDEt > 2 H2 + dimer + catX", None): 38, # kJ/mol

            ("Stoichiometric_Coefficient", None, None, "H2 + cat2 > 2 cat-H", "H2"):                            -1.0,
            ("Stoichiometric_Coefficient", None, None, "H2 + cat2 > 2 cat-H", "cat2"):                          -1.0,
            ("Stoichiometric_Coefficient", None, None, "H2 + cat2 > 2 cat-H", "cat-H"):                         2.0,
            ("Stoichiometric_Coefficient", None, None, "2 cat-H > H2 + cat2", "cat-H"):                         -2.0,
            ("Stoichiometric_Coefficient", None, None, "2 cat-H > H2 + cat2", "H2"):                            1.0,
            ("Stoichiometric_Coefficient", None, None, "2 cat-H > H2 + cat2", "cat2"):                          1.0,
            ("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "RDY"):                    -1.0,
            ("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "cat-H"):                  -2.0,
            ("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "RYE"):                    1.0,
            ("Stoichiometric_Coefficient", None, None, "RDY + 2 cat-H > RYE + cat2", "cat2"):                   1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "RYE"):                    -1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "cat-H"):                  -2.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "RYA"):                    1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + 2 cat-H > RYA + cat2", "cat2"):                   1.0,
            ("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "RYA"):                    -1.0,
            ("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "cat-H"):                  -2.0,
            ("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "REA"):                    1.0,
            ("Stoichiometric_Coefficient", None, None, "RYA + 2 cat-H > REA + cat2", "cat2"):                   1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "REA"):                   -1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "cat-H"):                 -2.0,
            ("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "RDEt"):                  1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + 2 cat-H > RDEt + cat2", "cat2"):                  1.0,
            ("Stoichiometric_Coefficient", None, None, "RDEt + catX > catX-RDEt", "RDEt"):                      -1.0,
            ("Stoichiometric_Coefficient", None, None, "RDEt + catX > catX-RDEt", "catX"):                      -1.0,
            ("Stoichiometric_Coefficient", None, None, "RDEt + catX > catX-RDEt", "catX-RDEt"):                 1.0,
            ("Stoichiometric_Coefficient", None, None, "catX-RDEt > RDEt + catX", "catX-RDEt"):                 -1.0,
            ("Stoichiometric_Coefficient", None, None, "catX-RDEt > RDEt + catX", "RDEt"):                      1.0,
            ("Stoichiometric_Coefficient", None, None, "catX-RDEt > RDEt + catX", "catX"):                      1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "RYE"):                -1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "catX-RDEt"):          -1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "dimer"):              1.0,
            ("Stoichiometric_Coefficient", None, None, "RYE + catX-RDEt > dimer + catX", "catX"):               1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "REA"):         -1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "catX-RDEt"):   -1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "H2"):          2.0,
            ("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "dimer"):       1.0,
            ("Stoichiometric_Coefficient", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "catX"):        1.0,
            
            ("Partial_Order", None, None, "H2 + cat2 > 2 cat-H", "H2"):                             1,
            ("Partial_Order", None, None, "H2 + cat2 > 2 cat-H", "cat2"):                           1,
            ("Partial_Order", None, None, "2 cat-H > H2 + cat2", "cat-H"):                          2,
            ("Partial_Order", None, None, "RDY + 2 cat-H > RYE + cat2", "RDY"):                     1,
            ("Partial_Order", None, None, "RDY + 2 cat-H > RYE + cat2", "cat-H"):                   1,
            ("Partial_Order", None, None, "RYE + 2 cat-H > RYA + cat2", "RYE"):                     1,
            ("Partial_Order", None, None, "RYE + 2 cat-H > RYA + cat2", "cat-H"):                   1,
            ("Partial_Order", None, None, "RYA + 2 cat-H > REA + cat2", "RYA"):                     1,
            ("Partial_Order", None, None, "RYA + 2 cat-H > REA + cat2", "cat-H"):                   1,
            ("Partial_Order", None, None, "REA + 2 cat-H > RDEt + cat2", "REA"):                    1,
            ("Partial_Order", None, None, "REA + 2 cat-H > RDEt + cat2", "cat-H"):                  1,
            ("Partial_Order", None, None, "RDEt + catX > catX-RDEt", "RDEt"):                       1,
            ("Partial_Order", None, None, "RDEt + catX > catX-RDEt", "catX"):                       1,
            ("Partial_Order", None, None, "catX-RDEt > RDEt + catX", "catX-RDEt"):                  1,
            ("Partial_Order", None, None, "RYE + catX-RDEt > dimer + catX", "RYE"):                 1,
            ("Partial_Order", None, None, "RYE + catX-RDEt > dimer + catX", "catX-RDEt"):           1,
            ("Partial_Order", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "REA"):          1,
            ("Partial_Order", None, None, "REA + catX-RDEt > 2 H2 + dimer + catX", "catX-RDEt"):    1,
        }
        return reaction_params

    def _setup_transport_params(self):
        transport_params = {
            ("Henry's_Constant", "Gas_Flow", "Batch_Feed", None, "H2"):                                                   42,
            ("Constant_Gas_Saturated_Concentration", "Gas_Flow", "Batch_Feed", None, "H2"):                               0.01,   # mol/L
            ("Gas-Liquid_Volumetric_Mass_Transfer_Coefficient_D", "Gas_Flow", "Batch_Feed", None, "H2"):                  0.014,  # /s
            ("Gas-Liquid_Volumetric_Mass_Transfer_Coefficient_alpha", "Gas_Flow", "Batch_Feed", None, "H2"):              0.6,
            ("Constant_Solid_Saturated_Concentration", "Solid_Feedstock", "Batch_Feed", None, "RDY"):                     0.001,  # mol/L
            ("Solid-Liquid_Volumetric_Mass_Transfer_Coefficient_D", "Solid_Feedstock", "Batch_Feed", None, "RDY"):        0.28,   # /s
            ("Solid-Liquid_Volumetric_Mass_Transfer_Coefficient_alpha", "Solid_Feedstock", "Batch_Feed", None, "RDY"):    0.6,
        }
        return transport_params

    def _setup_operation_params(self):
        operation_params = {
            ("Initial_Mass", None, "Batch_Feed", None, "water"):          0.2574,     # kg
            ("Initial_Mass", None, "Batch_Feed", None, "solvent"):        0.194,      # kg
            ("Initial_Mass", None, "Batch_Feed", None, "cat2"):           0.00235,    # kg
            ("Initial_Mass", None, "Batch_Feed", None, "catX"):           0.00235,    # kg
            ("Mass_Gas_Fraction", "Gas_Flow", None, None, "H2"):            1.0,
            ("Initial_Mass_Solid", "Solid_Feedstock", None, None, "RDY"):   None,       # kg
            ("Temperature", None, None, None, None):                        None,       # oC
            ("Pressure", None, None, None, None):                           None,       # bar
            ("Agitation", None, None, None, None):                          None,       # rpm
            ("Batch_Time", None, None, None, None):                         None,       # min
        }
        return operation_params

    def _setup_operation_name2ind(self):
        operation_name2ind = {
            "m_rdy":        ("Initial_Mass_Solid", "Solid_Feedstock", None, None, "RDY"),   # kg
            "t_b":          ("Batch_Time", None, None, None, None),                         # min
            "temp":         ("Temperature", None, None, None, None),                        # oC
            "pressure":     ("Pressure", None, None, None, None),                           # bar
            "agitation":    ("Agitation", None, None, None, None),                          # rpm
        }
        return operation_name2ind

    def _setup_measure_ind2name(self):
        measure_ind2name = {
            ("Mass", None, "Batch_Feed", None, "RDEt"):           "m_rdet",       # kg
            ("Mass", None, "Batch_Feed", None, "dimer"):          "m_dimer",      # kg
            ("Mass_Gas_Used", "Gas_Flow", None, None, "H2"):        "h2_used",      # kg
            ("Mass_Solid", "Solid_Feedstock", None, None, "RDY"):   "m_rdy_left",   # kg
        }
        return measure_ind2name

    def _setup_var2unit(self):
        var2unit = {
            "Initial_Mass": "kg",
            "Initial_Mass_Solid": "kg",
            "Mass_Gas_Fraction": None,
            "Density": "kg/m3",
            "Density_Solid": "kg/m3",
            "Activation_Energy": "kJ/mol",
            "Referenced_Reaction_Rate_Constant": None,
            "Gas-Liquid_Volumetric_Mass_Transfer_Coefficient_D": "s-1",
            "Solid-Liquid_Volumetric_Mass_Transfer_Coefficient_D": "s-1",
            "Gas-Liquid_Volumetric_Mass_Transfer_Coefficient_alpha": None,
            "Solid-Liquid_Volumetric_Mass_Transfer_Coefficient_alph": None,
            "Henry's_Constant": None,
            "Saturated_Concentration": "mol/L",
            "Temperature": "oC",
            "Pressure": "bar",
            "Agitation": "rpm",
            "Concentration": "mol/L",
            "Batch_Time": "min",
        }
        return var2unit

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
        return _simulate(params, self.phenos)

    def run(self, operation_params, reaction_params=None, transport_params=None):
        if reaction_params is None:
            reaction_params = self._reaction_params
        if transport_params is None:
            transport_params = self._transport_params
        t, ms = self._run(operation_params, reaction_params, transport_params)
        return t, ms

    def run_dataset(self, dataset, reaction_params=None, transport_params=None):
        dataset = dataset.copy()
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        res = {name: [] for name in self._measure_ind2name.values()}
        for i in range(len(dataset)):
            operation_params = {}
            for name, ind in self._operation_name2ind.items():
                operation_params[ind] = dataset.loc[i, name]
            _, ms = self.run(operation_params, reaction_params, transport_params)
            for ind, name in self._measure_ind2name.items():
                if ind[0] == "Mass":
                    res[name].append(ms[self.species.index(ind[-1]), -1])
                if ind == ("Mass_Gas_Used", "Gas_Flow", None, None, "H2"):
                    res[name].append(ms[-2, -1])
                if ind == ("Mass_Solid", "Solid_Feedstock", None, None, "RDY"):
                    res[name].append(ms[-1, -1])
        for name, val in res.items():
            dataset[name] = val
        return dataset
    
    def calibrate(self, cal_param_bounds, dataset):
        cal_param_bounds = self._param_list2dict(cal_param_bounds)
        params = self._param_list2dict(self.params())
        from functools import partial
        partial_calc_se = partial(
            calc_se, params=params, cal_param_bounds=cal_param_bounds, phenos=self.phenos, dataset=dataset)
        res = differential_evolution(func=partial_calc_se, bounds=list(cal_param_bounds.values(
        )), updating='deferred', workers=4, popsize=4, maxiter=8, disp=True, polish=False, rng=0)
        cal_params = {ind: round(v.item(), 6) for ind, v in zip(cal_param_bounds.keys(), res.x)}
        return self._param_dict2list(cal_params)
    
    def plot_simulation_profiles(self, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        t, ms = self.run(operation_params, reaction_params, transport_params)
        measure_data = {"Time (min)": [], "Mass (kg)": [], "Species": []}
        intermediate_data = {"Time (min)": [], "Mass (kg)": [], "Species": []}
        for i, s in enumerate(self.species):
            if s in ["RDEt", "dimer"]:
                for _t, _m in zip(t, ms[i]):
                    measure_data["Time (min)"].append(_t)
                    measure_data["Mass (kg)"].append(_m)
                    measure_data["Species"].append(f"{s}")
            if s in ["RDY", "RYE", "RYA", "REA"]:
                for _t, _m in zip(t, ms[i]):
                    intermediate_data["Time (min)"].append(_t)
                    intermediate_data["Mass (kg)"].append(_m)
                    intermediate_data["Species"].append(f"{s}")
        # solid RDY
        for _t, _m in zip(t, ms[-1]):
            measure_data["Time (min)"].append(_t)
            measure_data["Mass (kg)"].append(_m)
            measure_data["Species"].append(f"RDY (Solid)")
        # gas H2 used
        for _t, _m in zip(t, ms[-2]):
            measure_data["Time (min)"].append(_t)
            measure_data["Mass (kg)"].append(_m)
            measure_data["Species"].append(f"H2 used (Gas)")

        fig = make_subplots(rows=2, cols=1)
        fig1 = px.line(
            pd.DataFrame(measure_data), x="Time (min)", y="Mass (kg)", color="Species"
        )
        colors = px.colors.qualitative.Plotly
        shifted_colors = colors[len(fig1.data):] + colors[:len(fig1.data)]
        fig2 = px.line(
            pd.DataFrame(intermediate_data), 
            x="Time (min)", 
            y="Mass (kg)", 
            color="Species", 
            color_discrete_sequence=shifted_colors
        )

        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig.add_trace(trace, row=2, col=1)
        fig.update_xaxes(title_text="Time (min)", row=1, col=1)
        fig.update_yaxes(title_text="Mass (kg)", row=1, col=1)
        fig.update_xaxes(title_text="Time (min)", row=2, col=1)
        fig.update_yaxes(title_text="Mass (kg)", row=2, col=1)
        fig.update_layout(width=900, height=700, title="Mass Profiles")
        fig.show()
    
    def plot_simulation_rates(self, steps, operation_params, reaction_params=None, transport_params=None):
        # set parameters
        operation_params = self._param_list2dict(operation_params)
        if reaction_params is None:
            reaction_params = self._reaction_params
        if transport_params is None:
            transport_params = self._transport_params
        
        params = self._param_list2dict(self.params())
        params.update(operation_params)
        if reaction_params is not None:
            params.update(reaction_params)
        if transport_params is not None:
            params.update(transport_params)

        (R, t_b, T, P, N, m_0, m_s_0, x_g, M, ρ, ρ_s, nu, n, k_ref, E_a, k, k_Sa_D, k_Sa_alpha,
         k_La_D, k_La_alpha, k_H, c_g_star_const, c_s_star_const) = _set_params(params)

        # run simulation results
        t, ms = self.run(operation_params, reaction_params, transport_params)
        
        # define derivative function
        
        def _rate(t, m):
            m_g = np.zeros((1, 13), dtype=np.float64)
            m_g[0, 8] = m[13]
            m_s = np.zeros((1, 13), dtype=np.float64)
            m_s[0, 2] = m[14]
            m = np.array(m[:13], dtype=np.float64)
            m = m.reshape((1, 13))
            
            # concentration
            c = np.zeros((1, 13), dtype=np.float64)
            c[0] = (m[0] * ρ[0]) / (np.sum(m[0]) * M)

            # definition

            # reaction
            r_r = np.zeros((1, 10), dtype=np.float64)
            # stream: Batch stream, reaction: H2 + cat2 > 2 cat-H
            r_r[0, 0] = k[0, 0] * np.prod(c[0] ** n[0])

            # stream: Batch stream, reaction: 2 cat-H > H2 + cat2
            r_r[0, 1] = k[0, 1] * np.prod(c[0] ** n[1])

            # stream: Batch stream, reaction: RDY + 2 cat-H > RYE + cat2
            r_r[0, 2] = k_ref[0, 2] * np.exp(- E_a[0, 2] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[2])

            # stream: Batch stream, reaction: RYE + 2 cat-H > RYA + cat2
            r_r[0, 3] = k[0, 3] * np.prod(c[0] ** n[3])

            # stream: Batch stream, reaction: RYA + 2 cat-H > REA + cat2
            r_r[0, 4] = k_ref[0, 4] * np.exp(- E_a[0, 4] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[4])

            # stream: Batch stream, reaction: REA + 2 cat-H > RDEt + cat2
            r_r[0, 5] = k[0, 5] * np.prod(c[0] ** n[5])

            # stream: Batch stream, reaction: RDEt + catX > catX-RDEt
            r_r[0, 6] = k[0, 6] * np.prod(c[0] ** n[6])

            # stream: Batch stream, reaction: catX-RDEt > RDEt + catX
            r_r[0, 7] = k[0, 7] * np.prod(c[0] ** n[7])

            # stream: Batch stream, reaction: RYE + catX-RDEt > dimer + catX
            r_r[0, 8] = k_ref[0, 8] * np.exp(- E_a[0, 8] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[8])

            # stream: Batch stream, reaction: REA + catX-RDEt > 2 H2 + dimer + catX
            r_r[0, 9] = k_ref[0, 9] * np.exp(- E_a[0, 9] / R * (1 / T - 1 / 293.15)) * np.prod(c[0] ** n[9])
            
            # mass equilibrium
            c_g_star = np.zeros((1, 1, 13), dtype=np.float64)
            if "Gas_Dissolution_Saturated_Concentration" in self.phenos["param law"] and \
                self.phenos["param law"]["Gas_Dissolution_Saturated_Concentration"] == "Henry's_Law":
                c_g_star[0, 0, 8] = (P * x_g[0, 8]) / (1000 * R * T * k_H[0, 0, 8])
            if "Gas_Dissolution_Saturated_Concentration" in self.phenos["param law"] and \
                self.phenos["param law"]["Gas_Dissolution_Saturated_Concentration"] == "Constant":
                c_g_star[0, 0, 8] = c_g_star_const[0, 0, 8]

            c_s_star = np.zeros((1, 1, 13), dtype=np.float64)
            if "Solid_Dissolution_Saturated_Concentration" in self.phenos["param law"] and \
                self.phenos["param law"]["Solid_Dissolution_Saturated_Concentration"] == "Fitted":
                def calc_solid_dissolution_equilibrium_term(m):
                    return 0.00159 * np.exp(-0.1126 * (100 * np.maximum(m[6], 0) / np.sum(m)) ** 0.83)
                c_s_star[0, 0, 2] = calc_solid_dissolution_equilibrium_term(m[0])
            if "Solid_Dissolution_Saturated_Concentration" in self.phenos["param law"] and \
                self.phenos["param law"]["Solid_Dissolution_Saturated_Concentration"] == "Constant":
                c_s_star[0, 0, 2] = c_s_star_const[0, 0, 2]

            # mass transport
            r_t_g = np.zeros((1, 1, 13), dtype=np.float64)
            r_t_g[0, 0, 8] = k_La_D[0, 0, 8] * N ** k_La_alpha[0, 0, 8] * (c_g_star[0, 0, 8] - c[0, 8])

            r_t_s = np.zeros((1, 1, 13), dtype=np.float64)
            r_t_s[0, 0, 2] = k_Sa_D[0, 0, 2] * N ** k_Sa_alpha[0, 0, 2] * (m_s[0, 2] * ρ[0]) / (np.sum(m[0]) * ρ_s[0]) * (c_s_star[0, 0, 2] - c[0, 2])
            if m_s[0, 2] < 1e-6:
                r_t_s[0, 0, 2] = 0
            
            # rate
            r = np.zeros(12, dtype=np.float64)
            r[:10] = np.maximum(r_r[0], 0)
            r[10] = r_t_g[0, 0, 8]
            r[11] = r_t_s[0, 0, 2]
            
            return r

        # calculate rate data
        labels = self.reactions + ["H2 dissolution", "RDY dissolution"]
        rs = np.stack([_rate(t[i], ms[:, i]) for i in range(len(t))])
        data = {"Time (min)": [], "Rate (mol/L s)": [], "Reaction/Mass transport": []}
        for _t, r in zip(t, rs):
            for i, label in enumerate(labels):
                if label in steps:
                    data["Time (min)"].append(_t)
                    data["Rate (mol/L s)"].append(r[i] if _t > 0 else None)
                    data["Reaction/Mass transport"].append(label)
        
        # plot rate
        df = pd.DataFrame(data)
        fig = px.line(df, x="Time (min)", y="Rate (mol/L s)", color="Reaction/Mass transport")
        fig.update_layout(width=1000, height=500, title="Rate Profiles")
        fig.show()
        
    def plot_sensitivity_analysis(self, varied_mechanistic_params, operation_params, reaction_params=None, transport_params=None):
        operation_params = self._param_list2dict(operation_params)
        if reaction_params:
            reaction_params = self._param_list2dict(reaction_params)
        if transport_params:
            transport_params = self._param_list2dict(transport_params)
        fold = 1.5
        rdet_data = {"Time (min)": [], "RDEt (kg)": [], "Label": []}
        dimer_data = {"Time (min)": [], "Dimer (kg)": [], "Label": []}
        used_h2_data = {"Time (min)": [], "Used H2 (kg)": [], "Label": []}
        left_rdy_data = {"Time (min)": [], "Left RDY (kg)": [], "Label": []}
        for label, param in varied_mechanistic_params.items():
            _reaction_params = self._reaction_params.copy()
            if reaction_params is not None:
                _reaction_params.update(reaction_params)
            _transport_params = self._transport_params.copy()
            if transport_params is not None:
                _transport_params.update(transport_params)
            if param:
                param_tuple = []
                param_tuple.append(param["param"])
                if "solid" in param:
                    param_tuple.append(param["solid"])
                elif "gas" in param:
                    param_tuple.append(param["gas"])
                else:
                    param_tuple.append(None)
                param_tuple.append(param.get("stream", None))
                param_tuple.append(param.get("reaction", None))
                param_tuple.append(param.get("species", None))
                param_tuple = tuple(param_tuple)
                if param_tuple in _reaction_params:
                    _reaction_params[param_tuple] *= fold
                if param_tuple in _transport_params:
                    _transport_params[param_tuple] *= fold
            t, ms = self.run(operation_params, reaction_params=_reaction_params, transport_params=_transport_params)
            for _t, _m in zip(t, ms[self.species.index("RDEt")]):
                rdet_data["Time (min)"].append(_t)
                rdet_data["RDEt (kg)"].append(_m)
                rdet_data["Label"].append(label + " [RDEt]" if param is None else label + f" x {fold} [RDEt]")
            for _t, _m in zip(t, ms[self.species.index("dimer")]):
                dimer_data["Time (min)"].append(_t)
                dimer_data["Dimer (kg)"].append(_m)
                dimer_data["Label"].append(label + " [Dimer]" if param is None else label + f" x {fold} [Dimer]")
            for _t, _m in zip(t, ms[-1]):
                left_rdy_data["Time (min)"].append(_t)
                left_rdy_data["Left RDY (kg)"].append(_m)
                left_rdy_data["Label"].append(label + " [Left RDY]" if param is None else label + f" x {fold} [Left RDY]")
            for _t, _m in zip(t, ms[-2]):
                used_h2_data["Time (min)"].append(_t)
                used_h2_data["Used H2 (kg)"].append(_m)
                used_h2_data["Label"].append(label + " [Used H2]" if param is None else label + f" x {fold} [Used H2]")
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("RDEt (kg)", "Left RDY (kg)", "Dimer (kg)", "Used H2 (kg)"))
        fig1 = px.line(pd.DataFrame(rdet_data), x="Time (min)", y="RDEt (kg)", color="Label")
        fig2 = px.line(pd.DataFrame(left_rdy_data), x="Time (min)", y="Left RDY (kg)", color="Label")
        fig3 = px.line(pd.DataFrame(dimer_data), x="Time (min)", y="Dimer (kg)", color="Label")
        fig4 = px.line(pd.DataFrame(used_h2_data), x="Time (min)", y="Used H2 (kg)", color="Label")
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in fig3.data:
            fig.add_trace(trace, row=2, col=1)
        for trace in fig4.data:
            fig.add_trace(trace, row=2, col=2)
        fig.update_xaxes(title_text="Time (min)", row=1, col=1)
        fig.update_yaxes(title_text="RDEt (kg)", row=1, col=1)
        fig.update_xaxes(title_text="Time (min)", row=1, col=2)
        fig.update_yaxes(title_text="Left RDY (kg)", row=1, col=2)
        fig.update_xaxes(title_text="Time (min)", row=2, col=1)
        fig.update_yaxes(title_text="Dimer (kg)", row=2, col=1)
        fig.update_xaxes(title_text="Time (min)", row=2, col=2)
        fig.update_yaxes(title_text="Used H2 (kg)", row=2, col=2)
        fig.update_layout(width=1200, height=900, title="Sensitivity Analysis")
        fig.show()
    
    def plot_simulation_parity(self, dataset, reaction_params=None, transport_params=None):
        pred_dataset = self.run_dataset(dataset, reaction_params, transport_params)
        rdet_data = {
            "Experiment": dataset["m_rdet"].tolist(), 
            "Prediction": pred_dataset["m_rdet"].tolist()
        }
        left_rdy_data = {
            "Experiment": dataset["m_rdy_left"].tolist(), 
            "Prediction": pred_dataset["m_rdy_left"].tolist()
        }
        dimer_data = {
            "Experiment": dataset["m_dimer"].tolist(), 
            "Prediction": pred_dataset["m_dimer"].tolist()
        }
        used_h2_data = {
            "Experiment": dataset["h2_used"].tolist(), 
            "Prediction": pred_dataset["h2_used"].tolist()
        }
        se = ((dataset["m_rdet"] - pred_dataset["m_rdet"]) ** 2).sum()
        se += ((dataset["m_rdy_left"] - pred_dataset["m_rdy_left"]) ** 2).sum()
        se += ((dataset["m_dimer"] - pred_dataset["m_dimer"]) ** 2).sum()
        se += ((dataset["h2_used"] - pred_dataset["h2_used"]) ** 2).sum()
        print(f"Squared error: {se:.6f}")
    
        fig = make_subplots(rows=2, cols=2, subplot_titles=("RDEt (kg)", "Left RDY (kg)", "Dimer (kg)", "Used H2 (kg)"))
        fig1 = px.scatter(pd.DataFrame(rdet_data), x="Experiment", y="Prediction")
        fig2 = px.scatter(pd.DataFrame(left_rdy_data), x="Experiment", y="Prediction")
        fig3 = px.scatter(pd.DataFrame(dimer_data), x="Experiment", y="Prediction")
        fig4 = px.scatter(pd.DataFrame(used_h2_data), x="Experiment", y="Prediction")
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in fig3.data:
            fig.add_trace(trace, row=2, col=1)
        for trace in fig4.data:
            fig.add_trace(trace, row=2, col=2)
        fig.update_xaxes(title_text="Experiment", row=1, col=1)
        fig.update_yaxes(title_text="Prediction", row=1, col=1)
        fig.update_xaxes(title_text="Experiment", row=1, col=2)
        fig.update_yaxes(title_text="Prediction", row=1, col=2)
        fig.update_xaxes(title_text="Experiment", row=2, col=1)
        fig.update_yaxes(title_text="Prediction", row=2, col=1)
        fig.update_xaxes(title_text="Experiment", row=2, col=2)
        fig.update_yaxes(title_text="Prediction", row=2, col=2)
        fig.update_layout(width=900, height=900, title="Parity Plot")
        fig.show()
