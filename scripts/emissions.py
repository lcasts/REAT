### This file includes all functions to process the data with nasa cea
#region: Imports & Co.
from input_data.config import *
from scripts.data_processing import *
import subprocess
import shutil
import os
import re
import glob
import math
import warnings
import pandas as pd
import ast
from scripts.trajectory import calculate_atmosphere_traj
from ast import literal_eval
import numpy as np
from scipy.integrate import odeint
import cantera as ct

# Suppress warnings from openpyxl
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Path to the NASA CEA executable; normpath because Windows rejects forward slashes in a
# relative program path (WinError 2)
nasa_cea_executable = os.path.normpath(nasa_cea_folder_path + nasa_cea_exe)

# Cantera only searches its own data folder and the working directory, cea_species.yaml is
# in input_data/
ct.add_directory(input_data_folder_path)

#region: Additional subfunctions to process the main functions

# Find a species in a loaded Cantera phase, tolerant to capitalisation ('AR' vs 'Ar')
def resolve_cantera_species_name(gas, name):
    if name in gas.species_names:
        return name
    for variant in (name.upper(), name.capitalize(), name.lower()):
        if variant in gas.species_names:
            return variant
    return None

# Mole-fraction dict for gas.TPX, restricted to the species the loaded phase defines and
# renormalised over them
def build_cantera_atm_composition(gas, df_atm_molarfractions):
    candidate_species = {
        'N2': df_atm_molarfractions.get('N2', 0),
        'O2': df_atm_molarfractions.get('O2', 0),
        'O': df_atm_molarfractions.get('O', 0),
        'N': df_atm_molarfractions.get('N', 0),
        'NO': df_atm_molarfractions.get('NO', 0),
        'Ar': df_atm_molarfractions.get('Ar', 0),
    }
    atm_composition = {}
    skipped = []
    for species, mole_frac in candidate_species.items():
        value = float(mole_frac.iloc[0]) if hasattr(mole_frac, 'iloc') else float(mole_frac)
        resolved = resolve_cantera_species_name(gas, species)
        if resolved is not None:
            atm_composition[resolved] = atm_composition.get(resolved, 0.0) + value
        elif value > 0:
            skipped.append(species)
    if skipped:
        print(f"Warning: species {skipped} not found in the '{file_name_ct_species}' Cantera mechanism "
              f"(checked upper/lower/capitalized variants); excluding from atmospheric composition.")
    total_moles = sum(atm_composition.values())
    if total_moles > 0:
        atm_composition = {species: mole_frac / total_moles for species, mole_frac in atm_composition.items()}
    return atm_composition

# Function to calcualte the NASA CEA
def pivot_species(df):
    """Pivot a species/EI_species/row_idx table; an empty input used to raise KeyError 'row_idx'."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)


def rof_ab_calc(x_ab, a_ref, massflow, rho_atm, velocity, timestep):
    # Air massflow [kg/s] times the row's timestep [s] against the material mass d_m [kg] ablated in that step
    rof_ab = velocity*rho_atm*(a_ref*x_ab - a_ref)*timestep/massflow
    return rof_ab

def row_timesteps(df):
    """Timestep [s] of each row, taken per object so that object boundaries do not break it.

    After compression a row covers a whole bin and d_m is summed over it, so the air mass of
    the row has to be integrated over the same span. The first row of an object and rows
    without a positive difference fall back to delta_t.
    """
    object_col = next((c for c in ('ObjectID', 'FragmentID') if c in df.columns), None)
    if object_col:
        dt = df.sort_values([object_col, 'Time [s]']).groupby(object_col)['Time [s]'].diff()
    else:
        dt = df['Time [s]'].sort_values().diff()
    dt = dt.reindex(df.index)
    return dt.where(dt > 0, delta_t)

# Function to create the NASA CEA .inp file
def create_inp_file(filename, df=None, *args, problem_override=None):
    t_atm, p_atm, t_mat, t_stag, p_stag, t_air, t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions = args

    # problem_override forces the problem type; the quench run always needs 'tp'
    problem = problem_override or problem_material_combustion

    line = "problem o/f={},\n".format(rof_ab)

    # Input based on problem type
    if problem == "tp":
        line += "  " + problem + "  t,k={} p,bar={} \n".format(t_ab, p_ab)
    elif problem == "hp":
        line += "  " + problem + "  p,bar={} \n".format(p_ab)

    # Separate fuel and air species into dictionaries with their mass fractions
    material_species = {species: df_material_massfractions[species].values[0] for species in df_material_massfractions.columns}
    air_species = {species: df_air_massfractions[species].values[0] for species in df_air_massfractions.columns}

    # Check and limit total species count
    total_species = len(material_species) + len(air_species)
    #print('df_air_massfractions',df_air_massfractions)
    #print('df_material_massfractions',df_material_massfractions)

    if total_species > 24:
        # Remove He and Ar from air species (hot-gas output carries '*', ambient air does not)
        air_species = {
            k: v for k, v in air_species.items()
            if k.lstrip('*') not in ["He", "Ar"]
        }

        # Rank by mass per kg of material: material_data holds percent (sum 100), the air mass
        # fractions (sum 1) times rof_ab give kg of air per kg of material. Kept as
        # (side, name, mass) so a species on both sides ('O', 'N', 'H') is not overwritten
        ranked = ([('fuel', k, v / 100.0) for k, v in material_species.items()] +
                  [('oxid', k, v * rof_ab) for k, v in air_species.items()])
        kept = sorted(ranked, key=lambda x: x[2], reverse=True)[:24]

        # Split back; CEA gets the original wt values, it normalises fuel and oxidant separately
        kept_fuel = {k for side, k, _ in kept if side == 'fuel'}
        kept_oxid = {k for side, k, _ in kept if side == 'oxid'}
        material_species = {k: v for k, v in material_species.items() if k in kept_fuel}
        air_species = {k: v for k, v in air_species.items() if k in kept_oxid}

        total_species = len(material_species) + len(air_species)

    # Include Fuel and Air Species
    line += "react \n"
    for fuel, mass_fraction in material_species.items():
        if fuel == "AL(cr)" and t_mat > 933.61:
            fuel = "AL(L)"
        if fuel == "Zn(cr)" and t_mat > 692.73:
            fuel = "Zn(L)"
        elif fuel == "Li(cr)" and t_mat > 453.69:
            fuel = "Li(L)"
        elif fuel == "Ag(cr)" and t_mat > 1235.08:
            fuel = "Ag(L)"
        elif fuel == "Sn(cr)" and t_mat > 505.12:
            fuel = "Sn(L)"
        elif fuel == "Mg(cr)" and t_mat > 923.00:
            fuel = "Mg(L)"
        elif fuel == "Mn(a)" and t_mat >= 980.00 and t_mat < 1361.00:
            fuel = "Mn(b)"
        elif fuel == "Mn(a)" and t_mat >= 1361.00 and t_mat < 1412.00:
            fuel = "Mn(c)"
        elif fuel == "Mn(a)" and t_mat >= 1412.00 and t_mat < 1519.00:
            fuel = "Mn(d)"
        elif fuel == "Mn(a)" and t_mat > 1519.00:
            fuel = "Mn(L)"
        elif fuel == "Fe(a)" and t_mat >= 1184.00 and t_mat < 1665.00:
            fuel = "Fe(c)"
        elif fuel == "Fe(a)" and t_mat >= 1665.00 and t_mat < 1809.00:
            fuel = "Fe(d)"
        elif fuel == "Fe(a)" and t_mat > 1809.00:
            fuel = "Fe(L)"
        elif fuel == "Co(a)" and t_mat >= 700.1007 and t_mat < 1768.000:
            fuel = "Co(b)"
        elif fuel == "Co(a)" and t_mat > 1768.000:
            fuel = "Co(L)"
        elif fuel == "Cr(cr)" and t_mat > 2130.00:
            fuel = "Cr(L)"
        elif fuel == "Cu(cr)" and t_mat > 1358.0007:
            fuel = "Cu(L)"
        elif fuel == "P(cr)" and t_mat > 317.30:
            fuel = "P(L)"
        elif fuel == "Pb(cr)" and t_mat > 600.65:
            fuel = "Pb(L)"
        elif fuel == "S(a)" and t_mat >= 368.3007 and t_mat < 388.3607:
            fuel = "S(b)"
        elif fuel == "S(a)" and t_mat > 388.3607:
            fuel = "S(L)"
        elif fuel == "Si(cr)" and t_mat > 1690.00:
            fuel = "Si(L)"
        elif fuel == "Ti(a)" and t_mat >= 1156.000 and t_mat < 1944.000:
            fuel = "Ti(b)"
        elif fuel == "Ti(a)" and t_mat > 1944.000:
            fuel = "Ti(L)"
        elif fuel == "Zr(a)" and t_mat > 1135.0007 and t_mat < 2125.0007:
            fuel = "Zr(b)"
        elif fuel == "Zr(a)" and t_mat > 2125.0007:
            fuel = "Zr(L)"
        line += "  fuel={} wt={} t,k={} \n".format(fuel, mass_fraction, t_mat)

    for air, mass_fraction in air_species.items():
        if air == "NO2" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        elif air == "N2O" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        elif air == "NO3" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        elif air == "N3" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        elif air == "O3" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        elif air == "HNO" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        elif air == "HO2" and t_air >= 7200:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, 7199)
        else:
            line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, t_air)
            
    # Ending Lines
    line += "output \n"
    line += "  siunits massf trace={} \n".format(trace)
    line += "  plot p t mach H rho s \n"
    line += "end \n"

    inp_string = line
    with open(filename + ".inp", "w") as file:
        file.write(inp_string)

# --- NASA CEA shock problem ------------------------------------------------
# Station -> (calculation basis, block heading) in the .out file; only the heading prefix is
# matched because CEA2 misspells the frozen one ("CONDITI1ONS")
SHOCK_BASIS_HEADER = {
    "eq": "EQUILIBRIUM COMPOSITION FOR INCIDENT SHOCKED CONDITI",
    "fz": "FROZEN COMPOSITION FOR INCIDENT SHOCKED CONDITI",
}
SHOCK_STATIONS = {
    "inc_eq": ("eq", "SHOCKED GAS (2)--INCIDENT--EQUILIBRIUM"),
    "inc_fz": ("fz", "SHOCKED GAS (2)--INCIDENT--FROZEN"),
    "ref_eq": ("eq", "SHOCKED GAS (5)--REFLECTED--EQUILIBRIUM"),
    "ref_fz": ("fz", "SHOCKED GAS (5)--REFLECTED--FROZEN"),
}


def cea_float(token):
    """Parse CEA's compact number notation ('1.8832-4' -> 1.8832e-4, '1.4154 0' -> 1.4154)."""
    t = str(token).strip().replace(" ", "")
    if not t:
        return float("nan")
    if "*" in t:                      # CEA prints '******' when a field overflows
        return 1.0
    try:
        return float(t)
    except ValueError:
        pass
    m = re.match(r"^([+-]?\d*\.?\d+)([+-]\d+)$", t)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    return float("nan")


def read_shock_out(filename, station=None):
    """Read one station of a CEA shock problem in the same shape readCEA() returns."""
    station = station or shock_state
    if station not in SHOCK_STATIONS:
        raise ValueError("Unknown shock_state '{}'. Valid: {}".format(
            station, sorted(SHOCK_STATIONS)))
    basis, header = SHOCK_STATIONS[station]

    with open(filename + ".out") as f:
        text = f.read()

    # Find the section of the matching calculation basis
    chunks = text.split("SHOCK WAVE PARAMETERS ASSUMING")
    chunk = next((c for c in chunks[1:] if SHOCK_BASIS_HEADER[basis] in c), None)
    if chunk is None:
        raise ValueError("No '{}' section in {}.out - CEA computed no shock solution "
                         "(Mach 1 too low, or '{}' missing from shock_flags).".format(
                             basis, filename, basis))

    # Block of the requested station, up to the next station heading
    lines = chunk.splitlines()
    try:
        i0 = next(i for i, l in enumerate(lines) if header in l)
    except StopIteration:
        raise ValueError("Station '{}' ({}) not in {}.out - check shock_flags.".format(
            station, header, filename))
    i1 = next((i for i in range(i0 + 1, len(lines)) if "SHOCKED GAS" in lines[i]),
              len(lines))
    block = lines[i0:i1]

    # State values; 'mach' stays empty and is only kept to match the six 'plot' header rows
    wanted = [("p", r"^\s*P, BAR\s+(.+)$"),
              ("t", r"^\s*T, K\s+(.+)$"),
              ("rho", r"^\s*RHO, KG/CU M\s+(.+)$"),
              ("H", r"^\s*H, KJ/KG\s+(.+)$"),
              ("s", r"^\s*S, KJ/\(KG\)\(K\)\s+(.+)$")]
    params = {}
    for key, pattern in wanted:
        for l in block:
            m = re.match(pattern, l)
            if m:
                params[key] = cea_float(m.group(1))
                break
    params.setdefault("mach", float("nan"))

    # Mass fractions; without 'massf' CEA writes MOLE FRACTIONS instead, so check explicitly
    try:
        j0 = next(i for i, l in enumerate(block) if "MASS FRACTIONS" in l)
    except StopIteration:
        if any("MOLE FRACTIONS" in l for l in block):
            raise ValueError("{}.out contains MOLE FRACTIONS instead of MASS FRACTIONS "
                             "- the CEA input needs 'massf' in its output line.".format(filename))
        raise ValueError("No MASS FRACTIONS block for station '{}' in {}.out.".format(
            station, filename))

    fractions = {}
    for l in block[j0 + 1:]:
        stripped = l.strip()
        if not stripped:
            continue
        if "THERMODYNAMIC PROPERTIES" in stripped:
            break
        if stripped.startswith(("PRODUCTS", "NOTE.")):
            break
        parts = stripped.split()
        if len(parts) < 2:
            continue
        fractions[parts[0]] = cea_float(parts[1])

    # Post-shock Mach number from CEA's velocity and speed of sound, stagnation state from the
    # isentropic deceleration of station 2 to rest
    extra = {}
    for key, pattern in [("u2", r"^\s*U[25], M/SEC\s+(.+)$"),
                         ("son_vel", r"^\s*SON VEL,M/SEC\s+(.+)$"),
                         ("gammas", r"^\s*GAMMAs\s+(.+)$")]:
        for l in block:
            m = re.match(pattern, l)
            if m:
                extra[key] = cea_float(m.group(1))
                break

    mach_2 = float("nan")
    if extra.get("son_vel"):
        mach_2 = extra["u2"] / extra["son_vel"]
    g = extra.get("gammas", float("nan"))
    if np.isfinite(mach_2) and np.isfinite(g) and g > 1.0:
        isentropic = 1.0 + 0.5 * (g - 1.0) * mach_2 ** 2
        p_stag = params.get("p", float("nan")) * isentropic ** (g / (g - 1.0))
        t_stag = params.get("t", float("nan")) * isentropic
    else:
        p_stag = params.get("p", float("nan"))
        t_stag = params.get("t", float("nan"))

    params["mach"] = mach_2

    order = ["p", "t", "mach", "H", "rho", "s"]
    df = pd.DataFrame({"exit": [params.get(k, float("nan")) for k in order]}, index=order)
    out = pd.concat([df, pd.DataFrame({"exit": fractions})])
    # The stagnation state travels as a DataFrame attribute: it does not fit the six header
    # rows the downstream df.iloc[6:] relies on
    out.attrs["shock_state"] = {
        "station": station,
        "p": params.get("p", float("nan")),
        "t": params.get("t", float("nan")),
        "rho": params.get("rho", float("nan")),
        "u2": extra.get("u2", float("nan")),
        "son_vel": extra.get("son_vel", float("nan")),
        "gammas": g,
        "mach_2": mach_2,
        "p_stag": p_stag,
        "t_stag": t_stag,
    }
    return out


def run_material_quench(filename, t_atm, p_atm, t_mat, t_stag, p_stag, t_air,
                       p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react):
    """Re-equilibrate the combustion products at material_quench_temperature (one tp run)."""
    quench_filename = f"{filename}_quench"
    create_inp_file(quench_filename, None, t_atm, p_atm, t_mat, t_stag, p_stag, t_air,
                    material_quench_temperature, p_ab, rof_ab,
                    df_material_massfractions, df_air_massfractions_react,
                    problem_override="tp")

    process = subprocess.Popen(
        [nasa_cea_executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    process.stdin.write(f"{quench_filename}\n".encode())
    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(process.stderr.read().decode())

    df_quench = readCEA(quench_filename)
    if all_scenarios == False:
        save_files(quench_filename)
        save_results_excel(quench_filename, df_quench)
    clean_directory(quench_filename)
    return df_quench


def cantera_postshock_nox_state(gas, atm_composition, t_atm, p_atm, velocity, p_2):
    """Set 'gas' to the post-shock state the NOx formation is computed in; p in bar, v in m/s."""
    # 'shock' solves Rankine-Hugoniot with equilibrium chemistry, 'tp'/'hp' equilibrate at the
    # total enthalpy h1 + u1^2/2 and the post-shock pressure from compute_shockwave()
    pa_per_bar = 1e5
    gas.TPX = t_atm, p_atm * pa_per_bar, atm_composition
    h_1 = gas.enthalpy_mass
    rho_1 = gas.density

    if problem_hotgas_calculation == "shock":
        # x = rho1/rho2, damped because direct substitution oscillates around the solution
        x = 0.15
        for _ in range(60):
            p_shock = p_atm * pa_per_bar + rho_1 * velocity ** 2 * (1.0 - x)
            h_shock = h_1 + 0.5 * velocity ** 2 * (1.0 - x * x)
            gas.HP = h_shock, p_shock
            gas.equilibrate('HP')
            x_new = rho_1 / gas.density
            if abs(x_new - x) < 1e-8:
                break
            x = 0.5 * (x + x_new)
    else:
        gas.HP = h_1 + 0.5 * velocity ** 2, p_2 * pa_per_bar
        gas.equilibrate('HP')

    return gas.T


def create_inp_file_bal(filename, df=None, *args, problem_override=None):
    """Write the NASA CEA input for the hot-gas (air/NOx) calculation."""
    # 'tp'/'hp' equilibrate at the perfect-gas post-shock state from compute_shockwave(); 'shock' lets CEA do the jump itself from the undisturbed gas plus mach_1
    # Call sites always pass mach_1; velocity is optional as the 7th value
    t_ab, p_ab, t_atm, p_atm, df_air_massfractions = args[:5]
    mach_1 = args[5] if len(args) > 5 else None
    velocity_1 = args[6] if len(args) > 6 else None
    # problem_override lets run_hotgas_cea() fall back from 'shock' to 'hp'
    problem = problem_override or problem_hotgas_calculation

    if problem == "shock":
        # Below Mach 1 CEA writes an .out file without a MASS FRACTIONS block instead of
        # failing, so catch it here; the call sites fall back to the emission factors
        if not np.isfinite(mach_1) or mach_1 < shock_mach_min:
            raise ValueError(
                "Mach 1 = {} is below shock_mach_min = {}; NASA CEA computes no shock "
                "solution here.".format(mach_1, shock_mach_min))
        if shock_input == "u1" and (velocity_1 is None or not np.isfinite(velocity_1)):
            raise ValueError(
                "shock_input = 'u1' requires the free-stream velocity, but this call site "
                "does not provide it.")

    air_species = {species: df_air_massfractions[species].values[0] for species in df_air_massfractions.columns}

    # Drop reactants without mass: CEA still takes them into the element system and aborts the
    # shock problem with 'M6202: MATH - log: SING error'
    air_species = {k: v for k, v in air_species.items() if float(v) > 0.0}

    if len(air_species) > 24:
        air_species = {k: v for k, v in air_species.items() if k not in ["*He", "*Ar"]}

    if problem == "shock":
        # The inflow state is the undisturbed atmosphere, NOT the post-shock state
        line = "problem  case=shock{} \n".format(" ions" if shock_ions else "")
        # With 'u1', CEA derives the speed of sound from the reactant mixture itself, using the local NRLMSIS composition instead of an external kappa/R
        if shock_input == "u1":
            free_stream = "u1={}".format(velocity_1)
        elif shock_input == "mach1":
            free_stream = "mach1={}".format(mach_1)
        else:
            raise ValueError(f"Unknown shock_input: {shock_input!r}")
        line += "  shock  {}{}  {}, t,k={}, p,bar={}, \n".format(
            shock_flags, "  deb" if shock_debug else "", free_stream, t_atm, p_atm)
        react_temperature = t_atm
    else:
        line = "problem,\n"
        if problem == "tp":
            line += "  " + problem + "  t,k={} p,bar={} \n".format(t_ab, p_ab)
        elif problem == "hp":
            line += "  " + problem + "  p,bar={} \n".format(p_ab)
        react_temperature = t_ab

    line += "react \n"
    for air, mass_fraction in air_species.items():
        line += "  name={} wt={} t,k={} \n".format(air, mass_fraction, react_temperature)

    line += "output \n"
    line += "  siunits massf trace={} \n".format(trace)
    line += "  plot p t mach H rho s \n"
    line += "end \n"

    with open(filename + ".inp", "w") as file:
        file.write(line)

def extract_cea_hotgas_temperature(df_output, fallback_temperature):
    """Gas temperature from a NASA CEA output, fallback_temperature if none can be found."""
    try:
        if df_output is None or df_output.empty:
            return fallback_temperature

        # Try to find temperature in the first metadata/result rows
        search_df = df_output.copy()

        # Search in index labels first
        for idx in search_df.index:
            idx_str = str(idx).strip().lower()
            if idx_str in ["t", "temp", "temperature", "t,k", "t(k)"]:
                value = pd.to_numeric(search_df.loc[idx].iloc[0], errors="coerce")
                if pd.notna(value):
                    return float(value)

        # Search within the first rows for any temperature-like row
        for i in range(min(10, len(search_df))):
            row_name = str(search_df.index[i]).strip().lower()
            if "temp" in row_name or row_name in ["t", "t,k", "temperature"]:
                value = pd.to_numeric(search_df.iloc[i, 0], errors="coerce")
                if pd.notna(value):
                    return float(value)

        return fallback_temperature

    except Exception:
        return fallback_temperature


def reference_area_bal(row, scenario):
    """Reference area [m^2] the ballistic NOx mass flow is based on.

    Preference order:
      1. 'ReferenceArea [m^2]' of the row - attitude dependent, written by
         build_ownballistic_trajectory.py or by ballistic_traj()
      2. 'A_ref' of the scenario sheet - constant over the trajectory
      3. pi * R_n**2 - the nose cap cross section, the historic fallback
    """
    area = row.get('ReferenceArea [m^2]', np.nan) if hasattr(row, 'get') else np.nan
    if pd.notna(area) and float(area) > 0:
        return float(area)

    area = scenario.get('A_ref', np.nan) if hasattr(scenario, 'get') else np.nan
    if pd.notna(area) and float(area) > 0:
        return float(area)

    return np.pi * float(scenario['R_n']) ** 2


def nox_formation_temperature(row, t_2, t_atm):
    """NOx formation temperature for ballistic entries.

    The hottest of shock-heated air and ambient air, and - only if the trajectory
    actually carries it - the radiative-equilibrium wall temperature. Trajectory
    sources without 'T_rad_equil_comb' (or with a NaN in it) fall back to t_2/t_atm.
    """
    temperatures = [t_2, t_atm]
    t_wall = row.get('T_rad_equil_comb', np.nan) if hasattr(row, 'get') else np.nan
    if pd.notna(t_wall):
        try:
            temperatures.append(float(t_wall))
        except (TypeError, ValueError):
            pass
    return min(23999, max(temperatures))

# Column for the material-combustion temperature
COL_T_AB = "Temp ab [K]"


def format_t_ab(t_used, t_first=None):
    """Zellwert fuer 'Temp ab [K]'.

    Ohne Wiederholungslauf steht dort die verwendete Temperatur als Zahl. Musste
    t_ab beim zweiten Versuch ueberschrieben werden, bleiben beide Werte als
    'T_neu (T_alt)' erhalten.
    """
    try:
        t_used = float(t_used)
    except (TypeError, ValueError):
        return pd.NA
    if t_first is None:
        return t_used
    try:
        t_first = float(t_first)
    except (TypeError, ValueError):
        return t_used
    if abs(t_used - t_first) < 1e-9:
        return t_used
    return f"{t_used:.2f} ({t_first:.2f})"


def insert_t_ab_column(df, records):
    """Haengt 'Temp ab [K]' direkt hinter 'Temp [K]' in das Ergebnis."""
    if df is None or getattr(df, "empty", True) or not records:
        return df
    values = [records.get(i, pd.NA) for i in df.index]
    df = df.drop(columns=[COL_T_AB], errors="ignore")
    pos = df.columns.get_loc("Temp [K]") + 1 if "Temp [K]" in df.columns else len(df.columns)
    df.insert(pos, COL_T_AB, values)
    return df


def get_material_data(material, df_material_data):
    """Species and their mass fractions for one material, as a single-row DataFrame."""
    try:
        # Filter the row corresponding to the material
        material_row = df_material_data[df_material_data['material'] == material]

        if material_row.empty:
            print(f"No data found for material: {material}")
            return pd.DataFrame()

        # Drop the 'material', 'comp', and 'note' columns if they exist
        species_data = material_row.drop(columns=['material', 'comp', 'note'], errors='ignore')

        # Filter out columns with zero or near-zero values
        species_data = species_data.loc[:, (species_data != 0).any(axis=0)]

        # Set species as the header and values as the first row
        df_result = pd.DataFrame([species_data.iloc[0].values], columns=species_data.columns)

        return df_result

    except Exception as e:
        print(f"Error extracting data for material {material}: {e}")
        return pd.DataFrame()

# Copy necessary files to run NASA CEA in main directory
def init_directory():
    # Preparation
    # Copy thermo.lib & trans.lib, because they are necessary in main directory for NASA CEA
    files_to_copy = ["thermo.lib", "trans.lib"]
    for file in files_to_copy:
        try:
            shutil.copy("NASA_CEA" + "/" + file, ".")
        except FileNotFoundError:
            pass

# Save files to output data as backup
def save_files(filename):
    filetype_to_move = [".inp", ".out", ".csv", ".plt"]
    for filetype in filetype_to_move:
        try:
            folder = output_data_folder_path_raw
            new_filename = f"{filename}"
            # Adding (counter) and filetype to new_filename
            new_filename, file_path = get_unique_filename(new_filename, folder, filetype)
            # Renaming raw data file and moving it to storage folder
            os.rename(filename + filetype, new_filename)
            shutil.move(new_filename, os.path.join(folder, new_filename))
        except FileNotFoundError:
            pass

# Save formated Results to Excel
def save_results_excel(filename, df_output):
    base_name = filename 
    folder = output_data_folder_path_excel
    filename, file_path = get_unique_filename(base_name, folder, ".xlsx")
    
    with pd.ExcelWriter(folder + filename, engine='openpyxl') as writer:
        df_output.to_excel(writer, sheet_name=filename)
                
# Move and delete all files created by NASA CEA
def clean_directory(filename):
    ## Clean up - Delete (.xlsx, .inp, .out, .lib) in the main directory   
    #filetype_to_remove = [".xlsx", ".inp", ".out", ".csv", ".plt"]
    filetype_to_remove = [".inp", ".out", ".csv", ".plt"]
    for filetype in filetype_to_remove:
        try:
            os.remove(filename + filetype)
        except FileNotFoundError:
            pass
   
# Delete NASA CEA init files
def clean_up():
    # The condition used to be all_scenarios == 'Test', which is never true for a bool, so the
    # sweep never ran; save_files() has already kept what was meant to be kept
    file_extensions = ["*.lib", "*.inp", "*.out", "*.plt"]
    for ext in file_extensions:
        for filename in glob.glob(ext):
            if os.path.isfile(filename):
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"Could not delete {filename}: {e}")

def compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix):
    # Perfect-gas normal shock with the isentropic exponent and gas constant of the local
    # NRLMSIS mixture, consistent with c_atm; p_atm may be given in any unit, p_2 follows it
    kappa = kappa_mix
    R = r_mix

    mach_1 = velocity / c_atm

    # Before the shock (always valid)
    isentropic_factor_1 = 1.0 + 0.5 * (kappa - 1.0) * mach_1**2
    t_total_1 = t_atm * isentropic_factor_1
    p_total_1 = p_atm * isentropic_factor_1 ** (kappa / (kappa - 1.0))

    # No shock (Mach <= 1)
    if mach_1 <= 1.0:
        mach_2 = mach_1
        p_2 = p_atm
        rho_2 = rho_atm
        t_2 = t_atm

        t_total_2 = t_total_1
        p_total_2 = p_total_1

        velocity_2 = velocity

        return (
            mach_1,
            t_total_1,
            p_total_1,
            mach_2,
            p_2,
            rho_2,
            t_2,
            t_total_2,
            p_total_2,
            velocity_2,
        )

    # Normal shock (Mach > 1)

    mach_2 = math.sqrt(
        (1.0 + 0.5 * (kappa - 1.0) * mach_1**2)
        / (kappa * mach_1**2 - 0.5 * (kappa - 1.0))
    )

    pressure_ratio = 1.0 + (2.0 * kappa / (kappa + 1.0)) * (mach_1**2 - 1.0)
    density_ratio = ((kappa + 1.0) * mach_1**2) / ((kappa - 1.0) * mach_1**2 + 2.0)

    p_2 = p_atm * pressure_ratio
    rho_2 = rho_atm * density_ratio
    t_2 = t_atm * (pressure_ratio / density_ratio)

    # Total values after the shock
    t_total_2 = t_total_1
    p_total_2 = p_2 * (1.0 + 0.5 * (kappa - 1.0) * mach_2**2) ** (kappa / (kappa - 1.0))

    # Velocity after the shock
    c_2 = math.sqrt(kappa * R * t_2)
    velocity_2 = mach_2 * c_2

    return (
        mach_1,
        t_total_1,
        p_total_1,
        mach_2,
        p_2,
        rho_2,
        t_2,
        t_total_2,
        p_total_2,
        velocity_2,
    )

def clean_species_name(species_name):
    return species_name.strip('*').strip()

# Function to read in the plt file
def read_plt_file(plt_filename):
    with open(plt_filename, "r") as file:
        lines = file.readlines()

    header_line = None
    for line in lines:
        if line.startswith("#"):
            header_line = line.strip().lstrip("#").split()
            break

    if header_line is None:
        raise ValueError("No header line found in the .plt file.")

    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]

    num_lines = len(data_lines)

    if num_lines == 3:
        row_names = ["chamber", "throat", "exit"]
    elif num_lines == 1:
        row_names = ["exit"]
    else:
        raise ValueError(f"Unexpected number of lines: {num_lines}. Expected: 1 or 3.")

    data = [list(map(float, line.split())) for line in data_lines]

    df = pd.DataFrame(data, columns=header_line)
    df.index = row_names
    df = df.transpose()
    
    return df

# Function to extract data from the .out and .csv file
def readCEA(filename):  
    def read_problem():
        # Read .out file
        with open(filename + ".out") as f:
            data = f.read() 
        
        match = re.search(r"MASS FRACTIONS(.*)THERMODYNAMIC PROPERTIES", data, re.DOTALL)
        if match is None:
            # Without this check a failed CEA run only raised "NoneType has no attribute group"
            if "MOLE FRACTIONS" in data:
                raise ValueError(f"{filename}.out holds MOLE FRACTIONS instead of MASS "
                                 f"FRACTIONS - 'massf' is missing from the output line of "
                                 f"the CEA input.")
            raise ValueError(f"{filename}.out holds no MASS FRACTIONS block, NASA CEA wrote "
                             f"no result. Typical causes: a temperature outside the range of "
                             f"thermo.inp, or reactants with zero weight.")
        lines = match.group(1).split("\n")
        lines2 = [line for line in lines if len(line) > 10]
        cleaned_lines = [line.strip().split() for line in lines2]
        mass_fractions_df = pd.DataFrame(columns=['exit'])
        
        for item in cleaned_lines:

            if '******' in item[1]:
                item[1] = item[1].replace('******', '1.00')
                print(f"CEA error workaround: Replaced '******' with '1.00' in item: {item}")

            if '*****' in item[1]:
                item[1] = item[1].replace('*****', '1.00')
                print(f"CEA error workaround: Replaced '*****' with '1.00' in item: {item}")

            parts = item[1].split('-') if '-' in item[1] else item[1].split('+')
            result = float(parts[0]) * 10**(-int(parts[1])) if '-' in item[1] else float(parts[0]) * 10**(int(parts[1]))
            new_df = pd.DataFrame({'exit': result}, index=[item[0]])
            mass_fractions_df = pd.concat([mass_fractions_df, new_df])
        mass_fractions_df.index = [item[0] for item in cleaned_lines]
        
        ##### Extract remaining data from .csv #####
        other_values_df = read_plt_file(filename + ".plt")
        
        ##### Combine both df #####
        combined_df = pd.concat([other_values_df, mass_fractions_df])

        return combined_df
    
    # Read .out file
    with open(filename + ".out") as f:
        data = f.read() 
    # A shock problem has one MASS FRACTIONS block per station, which read_problem() would
    # match across in one greedy pattern
    if "SHOCK WAVE PARAMETERS ASSUMING" in data:
        return read_shock_out(filename)

    output_df = read_problem()

    return output_df

def calculate_NO_formation(temp, O2_conc, N2_conc, NO_conc, N_conc, O_conc, H_conc, OH_conc):
    """Solve the NO formation over time for a given temperature and molar fractions."""
    kf_1 = 1.8 * 10**8 * np.exp(-38370/temp)
    kf_2 = 1.8 * 10**4 * temp * np.exp(-4680/temp)
    kf_3 = 7.1 * 10**7 * np.exp(-450/temp)
    kr_1 = 3.8 * 10**7 * np.exp(-425/temp)
    kr_2 = 3.81 * 10**3 * temp * np.exp(-20820/temp)
    kr_3 = 1.7 * 10**8 * np.exp(-24560/temp)

    NO_conc = kf_1 * O2_conc * N2_conc  + kf_2 * N_conc * O2_conc + kf_3 * N_conc * OH_conc 
    backwards = - kr_1 * NO_conc * N_conc - kr_2 * NO_conc * O_conc - kr_3 * NO_conc * H_conc
    dNO_dt = NO_conc + backwards
    
    return dNO_dt  # Return final NO concentration


# Handle NASA CEA Errors
def handle_nasa_cea_error_message():
    pass
    # *******************************************************************************

    #         NASA-GLENN CHEMICAL EQUILIBRIUM PROGRAM CEA2, MAY 21, 2004
    #                 BY  BONNIE MCBRIDE AND SANFORD GORDON
    #     REFS: NASA RP-1311, PART I, 1994 AND NASA RP-1311, PART II, 1996

    # *******************************************************************************



    # problem
    # rocket equilibrium tcest,k=3000
    # p(bar)=115
    # sup,ae/at=58.2
    # reactant
    # name=LH2 wt=0.19323685424354242 t,k=293
    # name=LOX wt=0.8067631457564576 t,k=293
    # output
    # siunits massf transport
    # plot p t mach H rho s
    # end

    # OPTIONS: TP=F  HP=F  SP=F  TV=F  UV=F  SV=F  DETN=F  SHOCK=F  REFL=F  INCD=F
    # RKT=T  FROZ=F  EQL=T  IONS=F  SIUNIT=T  DEBUGF=F  SHKDBG=F  DETDBG=F  TRNSPT=T

    # TRACE= 0.00E+00  S/R= 0.000000E+00  H/R= 0.000000E+00  U/R= 0.000000E+00

    # Pc,BAR =   115.000000

    # Pc/P =

    # SUBSONIC AREA RATIOS =

    # SUPERSONIC AREA RATIOS =    58.2000

    # NFZ=  1  Mdot/Ac= 0.000000E+00  Ac/At= 0.000000E+00

    # YOUR ASSIGNED TEMPERATURE  293.00K FOR LH2            
    # IS OUTSIDE ITS TEMPERATURE RANGE20000.00 TO     0.00K (REACT)

    # ERROR IN REACTANTS DATASET (INPUT)

    # FATAL ERROR IN DATASET (INPUT)

#endregion

#region: Main Functions
def run_cantera_bal(df_trajaerodata, df_scenarios, scenario_name):
    # Extract nose radius from the scenario
    emission_results = df_trajaerodata.copy()
    scenario = df_scenarios.loc[scenario_name]  # Extract row corresponding to the scenario
    timesteps = row_timesteps(emission_results)

    # Iterate through filtered rows where altitude is greater than 0
    for idx, row in emission_results.loc[emission_results['Altitude [m]'] > 0].iterrows():
        time = row['Time [s]']
        date = row['date']
        lat = row['Lat [deg]']
        lon = row['Lon [deg]']
        alt = row['Altitude [m]']

        # Retrieve atmospheric conditions
        if alt < 0:
            alt = 0
        rho_atm, t_atm, p_atm, c_atm, total_number_density, df_air_massfractions, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_traj(date, lat, lon, alt)
        p_atm = p_atm * 10**(-5)  # Pa to bar, cantera_postshock_nox_state() expects bar
        velocity = row['Velocity [m/s]']

        # Air mass of this step; same interaction area as run_nasa_cea_bal()
        a_ref = reference_area_bal(row, scenario)
        massflow = rho_atm * velocity * (a_ref * x_ab - a_ref) * timesteps[idx]

        (
            mach_1, t_total_1, p_total_1,
            mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
        ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)

        t_stag = t_total_2
        p_stag = p_total_2

        t_atm = max(160.1, t_atm)
        p_ab = p_2

        # NOx formation temperature: shock-heated air, ambient air and - if present - the wall
        t_ab_nox = nox_formation_temperature(row, t_2, t_atm)

        # Load GRI-Mech 3.0 mechanism
        gas = ct.Solution(file_name_ct_species, name=ct_species_phase_name)

        # Convert atmospheric molar fractions to match the loaded mechanism's species
        atm_composition = build_cantera_atm_composition(gas, df_atm_molarfractions)

        # Set gas state and equilibrate at constant enthalpy & pressure
        try:
            cantera_postshock_nox_state(gas, atm_composition, t_atm, p_atm,
                                        velocity, p_ab)
        except ct.CanteraError as e:
            print(f"Cantera error: {e} at time {time}s and temperature {t_ab_nox}. No air-shock emissions for this row.")
            continue

        # Emitted mass of every species that the shocked air gains over ambient air
        for _, species_row in cantera_air_species_ei(gas, df_air_massfractions).iterrows():
            emission_results.at[idx, species_row['species']] = species_row['EI_species'] * massflow

    # Inert gases only change through the renormalisation over the mechanism's species
    return emission_results.drop(columns=['Ar', 'AR', 'He', 'HE'], errors='ignore')

#### Emission Factors - Main Function
def emission_factors(df_trajaerodata, df_emission_factors, scenario_name):
    try:   
        init_directory()
        emission_results = df_trajaerodata

        # Filter rows for the given scenario and d_m > 0
        filtered_rows_all = emission_results[
            (emission_results['scenario'] == scenario_name)]
        filtered_rows = emission_results[
            (emission_results['scenario'] == scenario_name) & 
            (emission_results['d_m [kg]'] > 0)
        ]

        total_emissions = pd.DataFrame()

        df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
            lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
        )

        # Air-shock emissions (all species) for all rows regardless of material ablation
        total_nox_emissions = pd.DataFrame()
        if calculate_nox == True and use_nasa_cea == False:
            if nox_method == "nasa_cea":
                total_nox_emissions = _cea_air_nox_block(filtered_rows_all, scenario_name)
            elif nox_method == "cantera":
                total_nox_emissions = _cantera_air_nox_block(filtered_rows_all)

        # Iterate through the filtered rows
        for idx, row in filtered_rows.iterrows():
            massflow = row['d_m [kg]']
            time = row['Time [s]']
            
            # Filter emission factors for the current material and method
            df_emission_factor_material = df_emission_factors[
                (df_emission_factors['material'] == row['Material']) & 
                (df_emission_factors['method'] == emission_factor_method)
            ]
                    

            # Further filter by altitude interval
            altitude_interval_match = df_emission_factor_material[
                df_emission_factor_material['altitude_interval'].apply(
                    lambda x: x[0] <= row['Altitude [km]'] <= x[1] if isinstance(x, list) else False
                )
            ]

            # Skip if no matching altitude interval
            if altitude_interval_match.empty:
                print(f"No matching emission factors for altitude {row['Altitude [km]']} and material {row['Material']}")
                continue
            
            # Select the first matching row of emission factors
            emission_factors_row = altitude_interval_match.iloc[0]

            # Calculate emissions for each species based on emission factors
            df_emission_factor_material_numeric = emission_factors_row.drop(
                ['material', 'method', 'altitude_interval']
            ).astype(float)

            df_species_mass = df_emission_factor_material_numeric
            df_species_mass = df_species_mass.fillna(0)

            # Initialize a DataFrame to store species and their EI_species for the current row
            combined_species = pd.DataFrame(columns=['species', 'EI_species'])
            # Append calculated species emissions to combined_species
            for species, mass in df_species_mass.items():
                if mass > threshold:
                    combined_species = pd.concat(
                        [combined_species, pd.DataFrame({'species': [species], 'EI_species': [mass]})],
                        ignore_index=True
                    )

            # Update Black Carbon Emissions
            if calculate_black_carbon:
                fuel_type = row['Material']
                primary_bc = df_emission_factors[df_emission_factors['material'] == fuel_type].iloc[0]['BC_prim']
                final_bc = primary_bc  # Additional calculations can be added if needed
                combined_species = pd.concat(
                    [combined_species, pd.DataFrame({'species': ['BC'], 'EI_species': [final_bc]})],
                    ignore_index=True
                )

            # Multiply EI_species by massflow and store in total_emissions
            combined_species['EI_species'] *= massflow
            combined_species['row_idx'] = idx  # Add row index to identify each row's results
            total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)
        
        # Pivot total_emissions to have species as columns and rows identified by row_idx
        total_emissions = pivot_species(total_emissions)

        if calculate_nox == True and use_nasa_cea == False:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        # If some columns are duplicated (like 'NO'), combine them by summing
        combined_emissions = combined_emissions.T.groupby(level=0).sum().T

        # Merge the pivoted emissions with filtered_rows to get the final results
        emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')

        # Drop rows where all values are 0
        emission_results = emission_results[(emission_results != 0).any(axis=1)]

        # Final cleanup and output
        emission_results = emission_results.drop(columns=['*Ar', '*He','Ar', 'He'], errors='ignore')
        print("Emission index calculation completed successfully.")

        # Clean up temporary files
        clean_up()

        return emission_results
        
    except Exception as e:
        handle_error(e, "emission_factors", "Error in Emission Factors main function.")

# Emission factor calculation for scarab input with multiple materials per row
def emission_factors_scarab(df_trajaerodata, df_emission_factors, scenario_name):
    try:
        init_directory()
        emission_results = df_trajaerodata

        # Keep the rows of this scenario with any per-material d_m > 0 ('d_m AA7075 [kg]')
        dm_pattern = re.compile(r'^d_m\s+(?P<mat>.+?)\s+\[kg\]$')
        dm_cols = [c for c in emission_results.columns if dm_pattern.match(c)]

        if not dm_cols:
            print("No per-material massflow columns found (expected columns like 'd_m AA7075 [kg]').")

        scenario_mask = (emission_results['scenario'] == scenario_name) if 'scenario' in emission_results.columns else True
        any_dm_positive = emission_results[dm_cols].fillna(0).gt(0).any(axis=1) if dm_cols else False
        if isinstance(scenario_mask, bool):
            filtered_rows = emission_results.copy()
            filtered_rows_all = emission_results.copy()
        else:
            filtered_rows = emission_results[scenario_mask].copy()
            filtered_rows_all = emission_results[scenario_mask].copy()

        if isinstance(any_dm_positive, pd.Series):
            filtered_rows = filtered_rows.loc[any_dm_positive.reindex(filtered_rows.index, fill_value=False)]

        total_emissions = pd.DataFrame()

        # Air-shock emissions (all species) for all rows regardless of material ablation
        total_nox_emissions = pd.DataFrame()
        if calculate_nox == True and use_nasa_cea == False:
            if nox_method == "nasa_cea":
                total_nox_emissions = _cea_air_nox_block(filtered_rows_all, scenario_name)
            elif nox_method == "cantera":
                total_nox_emissions = _cantera_air_nox_block(filtered_rows_all)

        # parse altitude_interval
        df_emission_factors = df_emission_factors.copy()
        df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
            lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
        )

        # meta/species columns
        meta_cols = [c for c in ['material', 'method', 'altitude_interval'] if c in df_emission_factors.columns]
        species_cols = [c for c in df_emission_factors.columns if c not in meta_cols]
        df_emission_factors[species_cols] = df_emission_factors[species_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Restrict to the configured emission_factor_method, otherwise the first matching row wins
        df_emission_factors_method = df_emission_factors[df_emission_factors['method'] == emission_factor_method]

        # group EF by material for faster access
        ef_by_material = {m: g.reset_index(drop=True) for m, g in df_emission_factors_method.groupby('material', dropna=False)}

        # --- iterate filtered rows
        for idx, row in filtered_rows.iterrows():
            altitude = row['Altitude [km]']

            # accumulate per-species totals across materials for this row
            row_species_totals = {}

            for dm_col in dm_cols:
                mat_name = dm_pattern.match(dm_col).group('mat')
                try:
                    massflow = float(row.get(dm_col, 0.0))
                except Exception:
                    massflow = 0.0
                if massflow <= 0:
                    continue

                # emission factors for this material
                ef_tbl = ef_by_material.get(mat_name)
                if ef_tbl is None or ef_tbl.empty:
                    print(f"No emission factors for material '{mat_name}'.")
                    continue

                # altitude interval match
                altitude_interval_match = ef_tbl[
                    ef_tbl['altitude_interval'].apply(
                        lambda x: x[0] <= altitude <= x[1] if isinstance(x, (list, tuple)) and len(x) == 2 else False
                    )
                ]
                if altitude_interval_match.empty:
                    print(f"No matching emission factors for altitude {altitude} and material {mat_name}")
                    continue

                emission_factors_row = altitude_interval_match.iloc[0]

                # numeric EI series
                ei_series = emission_factors_row[species_cols].astype(float).fillna(0.0)

                # optional thresholding (uses global 'threshold' if you defined it elsewhere)
                try:
                    thr = float(threshold)
                except Exception:
                    thr = 0.0
                if thr > 0:
                    ei_series = ei_series.where(ei_series > thr, other=0.0)

                # optional Black Carbon addition (uses global 'calculate_black_carbon' and 'BC_prim' if present)
                try:
                    use_bc = bool(calculate_black_carbon)
                except Exception:
                    use_bc = False
                if use_bc and 'BC_prim' in ef_tbl.columns:
                    bc_val = float(emission_factors_row.get('BC_prim', 0.0) or 0.0)
                    if 'BC' in ei_series.index:
                        ei_series.loc['BC'] = ei_series.loc['BC'] + bc_val
                    else:
                        ei_series.loc['BC'] = bc_val

                # convert EI to emitted mass for this material at this timestep
                emitted = (ei_series * massflow).to_dict()

                # accumulate into row totals
                for sp, val in emitted.items():
                    if pd.isna(val) or val == 0:
                        continue
                    row_species_totals[sp] = row_species_totals.get(sp, 0.0) + float(val)

            # if anything was computed, append to total_emissions in "long" form (like original)
            if row_species_totals:
                combined_species = pd.DataFrame(
                    {'species': list(row_species_totals.keys()),
                     'EI_species': list(row_species_totals.values())}
                )
                combined_species['row_idx'] = idx
                total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

        # pivot to wide species columns
        if not total_emissions.empty:
            total_emissions = pivot_species(total_emissions)
        else:
            total_emissions = pd.DataFrame()

        if calculate_nox == True and use_nasa_cea == False and not total_nox_emissions.empty:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        if not combined_emissions.empty:
            # If some columns are duplicated (like 'NO'), combine them by summing
            combined_emissions = combined_emissions.T.groupby(level=0).sum().T
            emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')
        else:
            emission_results = filtered_rows_all.copy()

        # final cleanup and output
        emission_results = emission_results.drop(columns=['*Ar', '*He', 'Ar', 'He'], errors='ignore')
        print("Emission index calculation completed successfully.")

        return emission_results

    except Exception as e:
        handle_error(e, "emission_factors", "Error in Emission Factors main function.")
        # re-raise if you want callers to notice
        # raise
    finally:
        # Clean up temporary files regardless of success/failure
        try:
            clean_up()
        except Exception:
            pass

#### NASA CEA Emissions - Main Function
def run_nasa_cea(df_trajaerodata, df_emission_factors, df_material_data, scenario_name):
    try:
        # Initialize results
        init_directory()
        emission_results = df_trajaerodata

        # Filter rows for the given scenario and d_m > 0
        filtered_rows_all = emission_results[
            (emission_results['scenario'] == scenario_name)]
        filtered_rows = emission_results[
            (emission_results['scenario'] == scenario_name) &
            (emission_results['d_m [kg]'] > 0)
        ]
        timesteps = row_timesteps(filtered_rows_all)
        t_ab_records = {}   # idx -> material-combustion temperature
        total_emissions = pd.DataFrame()
        # Air-shock emissions (all species) for rows without material ablation; rows with
        # ablation get the shocked air through the material combustion
        total_nox_emissions = pd.DataFrame()
        if calculate_nox == True:
            if nox_method == "nasa_cea":
                total_nox_emissions = _cea_air_nox_block(filtered_rows_all, scenario_name, skip_idx=filtered_rows_all['d_m [kg]'] > 0)
            elif nox_method == "cantera":
                total_nox_emissions = _cantera_air_nox_block(filtered_rows_all, skip_idx=filtered_rows_all['d_m [kg]'] > 0)

        # Material emissions for rows with d_m > 0
        for idx, row in filtered_rows.iterrows():
            # Atmospheric Data
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions_atm, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_data(row)
            p_atm = p_atm * 10**(-5)
            velocity = row['Velocity [km/s]'] * 1000
            a_ref = row['ReferenceArea [m^2]']
            massflow = row['d_m [kg]']
            time = row['Time [s]']
            # Needed for the file name below; without it 'altitude' leaked from the NOx loop
            altitude = row['Altitude [km]']

            # Calculate rof_ab
            rof_ab = rof_ab_calc(x_ab, a_ref, massflow, rho_atm, velocity, timesteps[idx])

            # Material Data
            df_material_massfractions = get_material_data(row['Material'], df_material_data)

            # Remove species with mass fraction below threshold
            df_material_massfractions = df_material_massfractions.loc[
                :, df_material_massfractions.iloc[0] > threshold]

            # Definition of material temperature from DRAMA file
            t_mat = row['Temp [K]']

            # Calculation of thermodynamic properties before and after shock
            (mach_1, t_total_1, p_total_1, mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)

            # Definition of calculation parameters at stagnation point
            t_stag = t_total_2
            p_stag = p_total_2

            # Definition of hot gas air temperature
            t_air = t_2
            
            # Definition of temperature and pressure of combustion
            t_ab = t_mat
            p_ab = p_stag

            # Reaction input defaults
            df_air_massfractions_react = df_air_massfractions_atm

            # Hot-gas state for the material combustion: supplies t_ab via temp_factor, t_air,
            # p_ab and the reactant composition, so it is computed regardless of nox_method
            compute_hotgas_state = True
            if compute_hotgas_state:
                try:
                    afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
                    # CEA shock first, 'hp' at the perfect-gas post-shock state as backup; if both
                    # fail, the except branch keeps ambient air at the perfect-gas stagnation state
                    df_output, _ = run_hotgas_cea(afterburning_filename, t_2, p_2, t_atm, p_atm,
                                                  df_air_massfractions_atm, mach_1, velocity)
                    
                    # Extract hot gas temperature
                    hotgas_temp_static = df_output.loc['t', 'exit']

                    # Under 'shock' CEA solved the jump itself, so take mach_2, p_2, p_stag and the isentropic exponent from that solution rather than mixing it with compute_shockwave()'s perfect-gas values
                    kappa_hotgas = kappa_mix
                    if problem_hotgas_calculation == "shock" and "shock_state" in df_output.attrs:
                        shock_state_2 = df_output.attrs["shock_state"]
                        if np.isfinite(shock_state_2["mach_2"]):
                            mach_2 = shock_state_2["mach_2"]
                        if np.isfinite(shock_state_2["p"]):
                            p_2 = shock_state_2["p"]
                        if np.isfinite(shock_state_2["p_stag"]):
                            p_stag = shock_state_2["p_stag"]
                        if np.isfinite(shock_state_2["gammas"]) and shock_state_2["gammas"] > 1.0:
                            kappa_hotgas = shock_state_2["gammas"]

                    # Calculate hot gas stagnation point temperature
                    hotgas_t_stag = hotgas_temp_static * (1 + ((kappa_hotgas - 1) / 2.0) * mach_2 ** 2)
                    
                    # Extract hot gas species
                    df_hotgas_massfractions = df_output.iloc[6:, 0].copy()
                    #df_hotgas_massfractions.index = df_hotgas_massfractions.index.str.replace(r'^\*', '', regex=True)

                    # Mean combustion temperature (lower bound: t_mat)
                    t_ab = combustion_temperature(t_mat, hotgas_temp_static, hotgas_t_stag,
                                                  kappa_hotgas, mach_2, p_2, p_atm, rof_ab)

                    # Overwriting of hot gas air temperature
                    t_air = hotgas_air_temp(hotgas_t_stag, t_total_2)
                    
                    p_ab = p_stag

                    # Overwriting input air composition
                    df_air_massfractions_react = pd.DataFrame([df_hotgas_massfractions])

                except Exception as cache_error:
                    print(f"Could not cache hot-air results for row {idx}: {cache_error}")
                    # Neither CEA shock nor hp converged: ambient air without dissociation reacts,
                    # t_ab from the perfect-gas stagnation point of compute_shockwave(), never below t_mat
                    try:
                        t_ab = combustion_temperature(t_mat, t_2, t_total_2, kappa_mix,
                                                      mach_2, p_2, p_atm, rof_ab)
                    except Exception:
                        t_ab = t_mat
                    t_ab = max(t_ab, t_mat)
                    # Oxidiser temperature and retry temperature: perfect-gas stagnation point
                    t_air = min(23999, t_total_2)

            afterburning_filename = f"{scenario_name}_{row['ObjectName']}_{idx}_{int(altitude)}km_{int(time)}s"

            # Retry once with t_ab clamped to max(t_mat, t_air) before using emission factors
            t_ab_first = t_ab          # initial value, in case the second attempt overwrites it
            cea_success = False
            for attempt in range(2):
                input_file_path = str(afterburning_filename)
                output_file_path = str(afterburning_filename)
                try:
                    create_inp_file(afterburning_filename,None, t_atm, p_atm, t_mat, t_stag, p_stag, t_air, t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react)

                    process = subprocess.Popen(
                        [nasa_cea_executable],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )

                    process.stdin.write(f"{input_file_path}\n".encode())
                    process.stdin.close()
                    process.wait()

                    if process.returncode != 0:
                        error_message = process.stderr.read().decode()
                        print("Error while executing NASA CEA:", error_message)


                    df_output = readCEA(output_file_path)
                    if all_scenarios == False:
                        # Saving raw and formated data
                        save_files(afterburning_filename)
                        save_results_excel(afterburning_filename, df_output)

                    clean_directory(afterburning_filename)

                    # Quench run: t_ab gives the reaction zone, not what reaches the atmosphere
                    if material_quench_temperature:
                        try:
                            df_output = run_material_quench(
                                afterburning_filename, t_atm, p_atm, t_mat, t_stag, p_stag,
                                t_air, p_ab, rof_ab, df_material_massfractions,
                                df_air_massfractions_react)
                        except Exception as quench_error:
                            print(f"Quench calculation failed for row {idx} at "
                                  f"{material_quench_temperature} K, keeping the combustion "
                                  f"composition: {quench_error}")

                    df_output_filtered = df_output.iloc[6:]
                    df_output_filtered.columns = ['ab_output_filtered']
                    df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                    # For the EI always compare against atmospheric air, not against hot air
                    df_air_massfractions_ref = df_air_massfractions_atm.transpose()
                    df_air_massfractions_ref.columns = ['air_massfractions']
                    df_air_massfractions_ref.index = df_air_massfractions_ref.index.str.replace(r'^\*', '', regex=True)

                    combined_df = pd.concat(
                        [df_output_filtered, df_air_massfractions_ref],
                        axis=1,
                        join='outer'
                    ).fillna(0)

                    combined_species = pd.DataFrame(columns=['species', 'EI_species'])

                    for species, row2 in combined_df.iterrows():
                        x_species = row2['ab_output_filtered']
                        x_species_atm = row2['air_massfractions']
                        EI_species = (x_species * (rof_ab + 1)) - (x_species_atm * rof_ab)
                        if EI_species > threshold:
                            combined_species = pd.concat(
                                [
                                    combined_species,
                                    pd.DataFrame({'species': [species], 'EI_species': [EI_species]})
                                ],
                                ignore_index=True
                            )

                    combined_species['EI_species'] *= massflow
                    combined_species['row_idx'] = idx
                    total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

                    clean_directory(afterburning_filename)

                    cea_success = True
                    break

                except Exception as e:
                    mat_name = row.get('Material', 'Unknown')
                    if attempt == 0:
                        print(f"NASA CEA error: {e} at time {time}s and material {mat_name}. Retrying with t_ab = max(t_mat, t_air).")
                        print('t_ab old',t_ab)
                        t_ab = t_air
                        t_ab = max(t_mat, t_ab)
                        print('t_ab new', t_ab)
                    else:
                        print(f"NASA CEA error: {e} at time {time}s and material {mat_name}. Using {emission_factor_method} emission factors.")

            t_ab_records[idx] = format_t_ab(t_ab, t_ab_first)

            if not cea_success:
                mat_name = row.get('Material', 'Unknown')
                if 'altitude_interval' in df_emission_factors.columns:
                    df_emission_factor_material = df_emission_factors[
                        (df_emission_factors['material'] == mat_name) &
                        (df_emission_factors['method'] == emission_factor_method)
                    ]
                    print(f"Using emission factors for material {mat_name} at altitude {row['Altitude [km]']}")

                    altitude_interval_match = df_emission_factor_material[
                        df_emission_factor_material['altitude_interval'].apply(
                            lambda x: (
                                ast.literal_eval(x)[0] <= float(row['Altitude [km]']) <= ast.literal_eval(x)[1]
                                if isinstance(x, str)
                                else x[0] <= float(row['Altitude [km]']) <= x[1]
                            )
                        )
                    ]

                    if altitude_interval_match.empty:
                        print(f"No matching emission factors for altitude {row['Altitude [km]']} and material {mat_name}")
                        print('df_emission_factor_material\n', df_emission_factor_material)
                        continue

                emission_factors_row = altitude_interval_match.iloc[0]

                df_emission_factor_material_numeric = emission_factors_row.drop(
                    ['material', 'method', 'altitude_interval']
                ).astype(float)

                df_species_mass = df_emission_factor_material_numeric.fillna(0)

                combined_species = pd.DataFrame(columns=['species', 'EI_species'])
                for species, mass in df_species_mass.items():
                    if mass > threshold:
                        combined_species = pd.concat(
                            [combined_species, pd.DataFrame({'species': [species], 'EI_species': [mass]})],
                            ignore_index=True
                        )

                if calculate_black_carbon:
                    fuel_type = row['Material']
                    primary_bc = df_emission_factors[df_emission_factors['material'] == fuel_type].iloc[0]['BC_prim']
                    final_bc = primary_bc
                    combined_species = pd.concat(
                        [combined_species, pd.DataFrame({'species': ['BC'], 'EI_species': [final_bc]})],
                        ignore_index=True
                    )

                combined_species['EI_species'] *= massflow
                combined_species['row_idx'] = idx
                total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)
                continue

        # Pivot total_emissions to have species as columns and rows identified by row_idx
        if not total_emissions.empty:
            total_emissions = total_emissions.pivot(
                index='row_idx',
                columns='species',
                values='EI_species'
            ).fillna(0)
        else:
            total_emissions = pd.DataFrame()

        if calculate_nox == True and not total_nox_emissions.empty:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        if not combined_emissions.empty:
            combined_emissions = (combined_emissions.T.groupby(level=0).sum().T)

        emission_results = filtered_rows_all.merge(
            combined_emissions,
            left_index=True,
            right_index=True,
            how='left'
        )

        emission_results = emission_results[(emission_results != 0).any(axis=1)]
        emission_results = emission_results.drop(columns=['*Ar', '*He', 'Ar', 'He'], errors='ignore')
        emission_results = insert_t_ab_column(emission_results, t_ab_records)
        print("NASA CEA calculations completed successfully.")

        clean_up()

        return emission_results

    except Exception as e:
        handle_error(e, "run_nasa_cea", "Error in NASA CEA main function.")

def run_nasa_cea_scarab(df_trajaerodata, df_emission_factors, df_material_data, scenario_name):
    try:
        # Initialize results
        init_directory()
        emission_results = df_trajaerodata

        # --- detect per-material massflow columns (e.g., 'd_m AA7075 [kg]')
        dm_pattern = re.compile(r'^d_m\s+(?P<mat>.+?)\s+\[kg\]$')
        dm_cols = [c for c in emission_results.columns if dm_pattern.match(c)]

        # --- Filter rows for the given scenario
        filtered_rows_all = emission_results[(emission_results['scenario'] == scenario_name)]
        timesteps = row_timesteps(filtered_rows_all)

        # any_dm_positive: per-row flag whether any material ablates in this timestep
        if dm_cols:
            any_dm_positive = emission_results[dm_cols].fillna(0).gt(0).any(axis=1).reindex(filtered_rows_all.index, fill_value=False)
        elif 'd_m [kg]' in filtered_rows_all.columns:
            any_dm_positive = filtered_rows_all['d_m [kg]'] > 0
        else:
            any_dm_positive = pd.Series(False, index=filtered_rows_all.index)

        filtered_rows = filtered_rows_all.loc[any_dm_positive].copy()

        t_ab_records = {}   # idx -> material-combustion temperature
        total_emissions = pd.DataFrame()

        # Air-shock emissions (all species) for rows without material ablation; rows with
        # ablation get the shocked air through the material combustion
        total_nox_emissions = pd.DataFrame()
        if calculate_nox == True:
            if nox_method == "nasa_cea":
                total_nox_emissions = _cea_air_nox_block(filtered_rows_all, scenario_name, skip_idx=any_dm_positive)
            elif nox_method == "cantera":
                total_nox_emissions = _cantera_air_nox_block(filtered_rows_all, skip_idx=any_dm_positive)

        df_emission_factors = df_emission_factors.copy()
        if 'altitude_interval' in df_emission_factors.columns:
            df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
                lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
            )
        meta_cols = [c for c in ['material', 'method', 'altitude_interval'] if c in df_emission_factors.columns]
        species_cols = [c for c in df_emission_factors.columns if c not in meta_cols]
        if species_cols:
            df_emission_factors[species_cols] = df_emission_factors[species_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Material emissions for rows with material massflow
        for idx, row in filtered_rows.iterrows():
            # Atmospheric Data (row-level, same for all materials ablating this row)
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions_atm, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_data(row)
            p_atm = p_atm*10**(-5)  # Pa -> bar
            velocity = row['Velocity [km/s]']*1000
            a_ref = row['ReferenceArea [m^2]']
            time = row['Time [s]']
            altitude = row['Altitude [km]']
            t_mat = row['Temp [K]']

            # Material massflows of this row; the air of the row is shared out over them by d_m
            material_massflows = {}
            for dm_col in dm_cols if dm_cols else ['d_m [kg]']:
                mat_name = row.get('Material', 'Unknown') if dm_col == 'd_m [kg]' else dm_pattern.match(dm_col).group('mat')
                try:
                    massflow = float(row.get(dm_col, 0.0))
                except Exception:
                    massflow = 0.0
                if massflow > 0:
                    material_massflows[mat_name] = massflow
            if not material_massflows:
                continue

            # rof_ab from the total material massflow: material i gets the air share d_m_i / sum(d_m),
            # so rof_ab = air / sum(d_m) is the same for every material and the air is counted once
            rof_ab = rof_ab_calc(x_ab, a_ref, sum(material_massflows.values()), rho_atm, velocity, timesteps[idx])

            # Thermodynamic properties before/after shock (same for every material ablating this row)
            (mach_1, t_total_1, p_total_1, mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)

            t_stag = t_total_2
            p_stag = p_total_2
            t_air = t_2
            t_ab_default = t_mat
            p_ab = p_stag

            # Reaction input defaults to ambient air
            df_air_massfractions_react = df_air_massfractions_atm

            # Hot-gas state for the material combustion, computed once per row: it supplies t_ab
            # via temp_factor, t_air, p_ab and the reactant composition, regardless of nox_method
            compute_hotgas_state = True
            if compute_hotgas_state:
                try:
                    afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
                    # CEA shock first, 'hp' at the perfect-gas post-shock state as backup; if both
                    # fail, the except branch keeps ambient air at the perfect-gas stagnation state
                    df_output, _ = run_hotgas_cea(afterburning_filename, t_2, p_2, t_atm, p_atm,
                                                  df_air_massfractions_atm, mach_1, velocity)

                    # Extract hot gas temperature
                    hotgas_temp_static = df_output.loc['t', 'exit']

                    # Under 'shock' CEA solved the jump itself, so take mach_2, p_2, p_stag and the isentropic exponent from that solution rather than mixing it with compute_shockwave()'s perfect-gas values
                    kappa_hotgas = kappa_mix
                    if problem_hotgas_calculation == "shock" and "shock_state" in df_output.attrs:
                        shock_state_2 = df_output.attrs["shock_state"]
                        if np.isfinite(shock_state_2["mach_2"]):
                            mach_2 = shock_state_2["mach_2"]
                        if np.isfinite(shock_state_2["p"]):
                            p_2 = shock_state_2["p"]
                        if np.isfinite(shock_state_2["p_stag"]):
                            p_stag = shock_state_2["p_stag"]
                        if np.isfinite(shock_state_2["gammas"]) and shock_state_2["gammas"] > 1.0:
                            kappa_hotgas = shock_state_2["gammas"]
                    hotgas_t_stag = hotgas_temp_static * (1 + ((kappa_hotgas - 1) / 2.0) * mach_2 ** 2)

                    # Extract hot gas species
                    df_hotgas_massfractions = df_output.iloc[6:, 0].copy()

                    # Mean combustion temperature (lower bound: t_mat)
                    t_ab_default = combustion_temperature(t_mat, hotgas_temp_static, hotgas_t_stag,
                                                          kappa_hotgas, mach_2, p_2, p_atm, rof_ab)

                    t_air = hotgas_air_temp(hotgas_t_stag, t_total_2)
                    p_ab = p_stag
                    df_air_massfractions_react = pd.DataFrame([df_hotgas_massfractions])

                except Exception as cache_error:
                    print(f"Could not cache hot-air results for row {idx}: {cache_error}")
                    # Neither CEA shock nor hp converged: ambient air without dissociation reacts,
                    # t_ab from the perfect-gas stagnation point of compute_shockwave(), never below t_mat
                    try:
                        t_ab_default = combustion_temperature(t_mat, t_2, t_total_2, kappa_mix,
                                                      mach_2, p_2, p_atm, rof_ab)
                    except Exception:
                        t_ab_default = t_mat
                    t_ab_default = max(t_ab_default, t_mat)
                    # Oxidiser temperature and retry temperature: perfect-gas stagnation point
                    t_air = min(23999, t_total_2)

            # iterate each material ablating in this row
            for mat_name, massflow in material_massflows.items():
                # Material Data (for THIS material)
                df_material_massfractions = get_material_data(mat_name, df_material_data)

                obj_name = row.get('ObjectName', row.get('FragmentID', 'Object'))
                safe_mat = str(mat_name).replace(' ', '_')
                afterburning_filename = f"{scenario_name}_{obj_name}_{safe_mat}_{idx}_{int(altitude)}km_{int(time)}s"

                t_ab = t_ab_default
                t_ab_first = t_ab      # initial value, in case the second attempt overwrites it
                cea_success = False

                if df_material_massfractions.empty:
                    # get_material_data() already printed why; use the emission-factor fallback
                    print(f"Skipping NASA CEA for material '{mat_name}' (no material data); using {emission_factor_method} emission factors.")
                    attempt_range = []
                else:
                    # Apply threshold to material species
                    df_material_massfractions = df_material_massfractions.loc[
                        :, df_material_massfractions.iloc[0] > threshold
                    ]
                    attempt_range = range(2)

                # Retry once with t_ab clamped to max(t_mat, t_air) before using emission factors
                for attempt in attempt_range:
                    input_file_path = str(afterburning_filename)
                    output_file_path = str(afterburning_filename)
                    try:
                        create_inp_file(afterburning_filename, None, t_atm, p_atm, t_mat, t_stag, p_stag, t_air, t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react)

                        process = subprocess.Popen(
                            [nasa_cea_executable],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE
                        )
                        process.stdin.write(f"{input_file_path}\n".encode())
                        process.stdin.close()
                        process.wait()

                        if process.returncode != 0:
                            error_message = process.stderr.read().decode()
                            print("Error while executing NASA CEA:", error_message)

                        df_output = readCEA(output_file_path)
                        if all_scenarios == False:
                            save_files(afterburning_filename)
                            save_results_excel(afterburning_filename, df_output)
                        clean_directory(afterburning_filename)

                        # Quench run: t_ab gives the reaction zone, not what reaches the atmosphere
                        if material_quench_temperature:
                            try:
                                df_output = run_material_quench(
                                    afterburning_filename, t_atm, p_atm, t_mat, t_stag, p_stag,
                                    t_air, p_ab, rof_ab, df_material_massfractions,
                                    df_air_massfractions_react)
                            except Exception as quench_error:
                                print(f"Quench calculation failed for row {idx} at "
                                      f"{material_quench_temperature} K, keeping the combustion "
                                      f"composition: {quench_error}")

                        df_output_filtered = df_output.iloc[6:]
                        df_output_filtered.columns = ['ab_output_filtered']
                        df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                        # For the EI always compare against atmospheric air, not against hot air
                        df_air_massfractions_ref = df_air_massfractions_atm.transpose()
                        df_air_massfractions_ref.columns = ['air_massfractions']
                        df_air_massfractions_ref.index = df_air_massfractions_ref.index.str.replace(r'^\*', '', regex=True)

                        combined_df = pd.concat([df_output_filtered, df_air_massfractions_ref], axis=1, join='outer').fillna(0)

                        # Per-material EI
                        combined_species = pd.DataFrame(columns=['species', 'EI_species'])
                        for species, row2 in combined_df.iterrows():
                            x_species = row2['ab_output_filtered']
                            x_species_atm = row2['air_massfractions']
                            EI_species = (x_species * (rof_ab + 1)) - (x_species_atm * rof_ab)
                            if EI_species > threshold:
                                combined_species = pd.concat(
                                    [combined_species, pd.DataFrame({'species': [species], 'EI_species': [EI_species]})],
                                    ignore_index=True
                                )

                        # Multiply EI by this material's massflow and append
                        combined_species['EI_species'] *= massflow
                        combined_species['row_idx'] = idx
                        total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

                        cea_success = True
                        break

                    except Exception as e:
                        if attempt == 0:
                            print(f"NASA CEA error: {e} at time {time}s and material {mat_name}. Retrying with t_ab = max(t_mat, t_air).")
                            t_ab = max(t_mat, t_air)
                        else:
                            print(f"NASA CEA error: {e} at time {time}s and material {mat_name}. Using {emission_factor_method} emission factors.")

                t_ab_records[idx] = format_t_ab(t_ab, t_ab_first)

                if not cea_success:
                    # Fallback: stoichiometric EF for THIS material
                    if 'altitude_interval' in df_emission_factors.columns:
                        df_emission_factor_material = df_emission_factors[(df_emission_factors['material'] == mat_name) & (df_emission_factors['method'] == emission_factor_method)]
                        print(f"Using emission factors for material {mat_name} at altitude {altitude}")

                        altitude_interval_match = df_emission_factor_material[
                            df_emission_factor_material['altitude_interval'].apply(
                                lambda x: x[0] <= altitude <= x[1] if isinstance(x, list) else False
                            )
                        ]

                        if altitude_interval_match.empty:
                            print(f"No matching emission factors for altitude {altitude} and material {mat_name}")
                            continue

                        emission_factors_row = altitude_interval_match.iloc[0]
                        df_ei = emission_factors_row.drop(
                            [c for c in ['material', 'method', 'altitude_interval'] if c in emission_factors_row.index]
                        ).astype(float).fillna(0)

                        combined_species = pd.DataFrame(columns=['species', 'EI_species'])
                        for species, mass in df_ei.items():
                            if mass > threshold:
                                combined_species = pd.concat(
                                    [combined_species, pd.DataFrame({'species': [species], 'EI_species': [mass]})],
                                    ignore_index=True
                                )

                        # Optional BC addition
                        if calculate_black_carbon and 'BC_prim' in df_emission_factors.columns:
                            try:
                                primary_bc = df_emission_factors[df_emission_factors['material'] == mat_name].iloc[0]['BC_prim']
                                combined_species = pd.concat(
                                    [combined_species, pd.DataFrame({'species': ['BC'], 'EI_species': [primary_bc]})],
                                    ignore_index=True
                                )
                            except Exception:
                                pass

                        combined_species['EI_species'] *= massflow
                        combined_species['row_idx'] = idx
                        total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

        # Pivot and combine totals
        if not total_emissions.empty:
            total_emissions = total_emissions.pivot_table(
                index='row_idx', columns='species', values='EI_species', aggfunc='sum', fill_value=0
            )
        else:
            total_emissions = pd.DataFrame()

        if calculate_nox == True and not total_nox_emissions.empty:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        if not combined_emissions.empty:
            # Sum across duplicated columns (e.g. 'NO' from both material and NOx calculations)
            combined_emissions = combined_emissions.T.groupby(level=0).sum().T

        # Merge back to all scenario rows
        emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')

        emission_results = emission_results[(emission_results != 0).any(axis=1)]
        emission_results = emission_results.drop(columns=['*Ar', '*He', 'Ar', 'He'], errors='ignore')
        emission_results = insert_t_ab_column(emission_results, t_ab_records)
        print("NASA CEA calculations completed successfully.")

        clean_up()

        return emission_results

    except Exception as e:
        handle_error(e, "run_nasa_cea_scarab", "Error in NASA CEA main function.")


#### Cantera Material Combustion - shared helpers ##########################################
# Cantera equivalent of the NASA CEA material combustion (create_inp_file() with problem_material_combustion='tp'), using cea_species.yaml instead of a subprocess call; t_ab is t_mat here, the temp_factor refinement is not replicated

# material_data.xlsx column -> its polymorph species, low-to-high temperature; only a tie-breaker, the actual choice comes from each species' own NASA9 ranges
CANTERA_ELEMENT_POLYMORPHS = {
    'AL(cr)': ['AL(cr)', 'AL(L)'],
    'Ag(cr)': ['Ag(cr)', 'Ag(L)'],
    'B(b)': ['B(b)'],
    'Cr(cr)': ['Cr(cr)', 'Cr(L)'],
    'Co(a)': ['Co(a)', 'Co(b)', 'Co(L)'],
    'Cu(cr)': ['Cu(cr)', 'Cu(L)'],
    'Fe(a)': ['Fe(a)', 'Fe(c)', 'Fe(d)', 'Fe(L)'],
    'Li(cr)': ['Li(cr)', 'Li(L)'],
    'Mg(cr)': ['Mg(cr)', 'Mg(L)'],
    'Mo(cr)': ['Mo(cr)'],
    'Mn(a)': ['Mn(a)', 'Mn(b)', 'Mn(c)', 'Mn(d)', 'Mn(L)'],
    'Nb(cr)': ['Nb(cr)'],
    'Ni(cr)': ['Ni(cr)', 'Ni(L)'],
    'P(cr)': ['P(cr)', 'P(L)'],
    'Pb(cr)': ['Pb(cr)', 'Pb(L)'],
    'S(a)': ['S(a)', 'S(b)', 'S(L)'],
    'Si(cr)': ['Si(cr)', 'Si(L)'],
    'Sn(cr)': ['Sn(cr)', 'Sn(L)'],
    'Ta(cr)': ['Ta(cr)'],
    'Ti(a)': ['Ti(a)', 'Ti(b)', 'Ti(L)'],
    'V(cr)': ['V(cr)'],
    'W(cr)': ['W(cr)'],
    'Zr(a)': ['Zr(a)', 'Zr(b)', 'Zr(L)'],
    'Zn(cr)': ['Zn(cr)', 'Zn(L)'],
    'C(gr)': ['C(gr)'],
}

# Elements that go into the shared ideal-solid-solution alloy phase instead of their own
# condensed phase; B/C/P/S/Si stay separate, they form no metallic solid solution
CANTERA_METALLIC_ELEMENTS = {
    'AL(cr)', 'Ag(cr)', 'Co(a)', 'Cr(cr)', 'Cu(cr)', 'Fe(a)', 'Li(cr)', 'Mg(cr)', 'Mo(cr)',
    'Mn(a)', 'Nb(cr)', 'Ni(cr)', 'Pb(cr)', 'Sn(cr)', 'Ta(cr)', 'Ti(a)', 'V(cr)', 'W(cr)',
    'Zr(a)', 'Zn(cr)',
}

# Non-metal polymorph species -> their pre-declared individual cea_species.yaml phase name.
CANTERA_NONMETAL_PHASE_BY_SPECIES = {
    'B(b)': 'b_b', 'C(gr)': 'c_gr',
    'P(cr)': 'p_cr', 'P(L)': 'p_l',
    'S(a)': 's_a', 'S(b)': 's_b', 'S(L)': 's_l',
    'Si(cr)': 'si_cr', 'Si(L)': 'si_l',
}

# Oxide products that thermo.inp only has as condensed species, added whenever their element
# is present
CANTERA_ELEMENT_OXIDE_PHASES = {
    'Cr(cr)': ['cr2o3_i_prime', 'cr2o3_i', 'cr2o3_l'],
    'Fe(a)': ['fe2o3_cr'],
    'V(cr)': ['v2o5_cr', 'v2o5_l'],
}

_cantera_species_db = None


def _get_cantera_species_db():
    """All species of cea_species.yaml keyed by name, loaded once and cached."""
    global _cantera_species_db
    if _cantera_species_db is None:
        _cantera_species_db = {s.name: s for s in ct.Species.list_from_file(file_name_ct_species)}
    return _cantera_species_db


def resolve_cantera_material_species(col, t_ab):
    """Column + t_ab -> the polymorph whose own NASA9 temperature ranges bracket t_ab."""
    # Evaluated against the target equilibrium temperature, not t_mat: Cantera needs every
    # initial species to be valid at the solve temperature, NASA CEA does not
    candidates = CANTERA_ELEMENT_POLYMORPHS.get(col)
    if not candidates:
        return None
    species_db = _get_cantera_species_db()
    chosen = None
    for name in candidates:  # declared low-to-high T
        sp = species_db.get(name)
        if sp is None:
            continue
        chosen = name
        if t_ab <= sp.thermo.max_temp:
            return name
    # t_ab above every declared range: take the highest-T polymorph and let the solver retry
    return chosen


def _equilibrate_cantera_material(t_ab, p_ab, rof_ab, material_cols, df_material_massfractions, df_air_massfractions_react):
    gas = ct.Solution(file_name_ct_species, name='cea_gas')
    species_db = _get_cantera_species_db()

    resolved = {}          # col -> species_name
    alloy_species_names = []
    nonmetal_phase_names = set()
    oxide_phase_names = set()

    for col in material_cols:
        species_name = resolve_cantera_material_species(col, t_ab)
        if species_name is None:
            continue
        resolved[col] = species_name
        if col in CANTERA_METALLIC_ELEMENTS:
            alloy_species_names.append(species_name)
        else:
            phase_name = CANTERA_NONMETAL_PHASE_BY_SPECIES.get(species_name)
            if phase_name:
                nonmetal_phase_names.add(phase_name)
        oxide_phase_names.update(CANTERA_ELEMENT_OXIDE_PHASES.get(col, []))

    # Metallic reactants share one ideal-solid-solution alloy phase: Cantera's multiphase solver
    # does not converge reliably with many near-zero-abundance condensed phases
    solutions = [gas]
    if alloy_species_names:
        alloy = ct.Solution(thermo='ideal-condensed', species=[species_db[n] for n in alloy_species_names])
        solutions.append(alloy)
    for p in sorted(nonmetal_phase_names):
        solutions.append(ct.Solution(file_name_ct_species, name=p))
    for p in sorted(oxide_phase_names):
        solutions.append(ct.Solution(file_name_ct_species, name=p))
    mix = ct.Mixture(solutions)

    mole_terms = []
    for col, species_name in resolved.items():
        mass_frac = float(df_material_massfractions.iloc[0][col])
        mw = species_db[species_name].molecular_weight
        mole_terms.append(f'{species_name}:{(mass_frac / 100.0) / mw}')

    air_terms = {}
    for col in df_air_massfractions_react.columns:
        name = resolve_cantera_species_name(gas, col)
        if name is None or name not in species_db:
            continue
        mass_frac = float(df_air_massfractions_react.iloc[0][col])
        if mass_frac <= 0:
            continue
        mole_terms.append(f'{name}:{(mass_frac * rof_ab) / species_db[name].molecular_weight}')
        air_terms[name] = mass_frac

    mix.T = t_ab
    mix.P = p_ab * 1e5  # bar -> Pa
    mix.species_moles = ', '.join(mole_terms)
    mix.equilibrate('TP', log_level=0)

    total_mass = sum(
        mix.species_moles[i] * species_db[mix.species_name(i)].molecular_weight
        for i in range(mix.n_species) if mix.species_name(i) in species_db
    )
    if total_mass <= 0:
        raise RuntimeError("Cantera material equilibrium returned zero mass.")

    rows = []
    for i in range(mix.n_species):
        n = mix.species_moles[i]
        if n <= 0:
            continue
        name = mix.species_name(i)
        sp = species_db.get(name)
        if sp is None:
            continue
        x_species = (n * sp.molecular_weight) / total_mass
        x_species_atm = air_terms.get(name, 0.0)
        EI_species = (x_species * (rof_ab + 1)) - (x_species_atm * rof_ab)
        if EI_species > threshold:
            rows.append({'species': name, 'EI_species': EI_species})

    combined_species = pd.DataFrame(rows, columns=['species', 'EI_species'])
    if combined_species.empty:
        return combined_species
    return combined_species.groupby('species', as_index=False)['EI_species'].sum()


def hotgas_air_temp(hotgas_t_stag, t_total_2):
    """Temperatur der Oxidator-Reaktanten (t_air), Auswahl ueber config.hotgas_air_temperature.

    'cea_stagnation'         CEA-Gleichgewichtswert hinter dem Stoss, isentrop auf Stillstand
    'perfect_gas_stagnation' Stautemperatur des kalorisch perfekten Gases (t_total_2)

    Die Deckelung bei 23999 K ist das obere Ende der CEA-Thermodatensaetze; sie ersetzt den
    Wert stillschweigend und greift bei der perfekten-Gas-Variante deutlich haeufiger.
    """
    if hotgas_air_temperature == "perfect_gas_stagnation":
        value = t_total_2
    elif hotgas_air_temperature == "cea_stagnation":
        value = hotgas_t_stag
    else:
        raise ValueError(f"Unbekanntes hotgas_air_temperature: {hotgas_air_temperature!r}")
    return min(23999, value)


def combustion_temperature(t_mat, hotgas_temp_static, hotgas_t_stag, kappa_hotgas,
                           mach_2, p_2, p_atm, rof_ab=None):
    """Verbrennungstemperatur t_ab nach der in config.temp_factor gewaehlten Methode.

    Die Referenztemperatur-Korrelationen (Roberts, Eckert) interpolieren zwischen
    Wandtemperatur und Gaszustand und sind fuer den Fall "kalte Wand in heissem Gas"
    aufgestellt. Faellt die Stautemperatur unter die Materialtemperatur - im
    Unterschall-Auslauf der Trajektorie der Normalfall - liefern sie t_ab < t_mat.
    Das ist ausserhalb ihres Gueltigkeitsbereichs: Ablationsprodukte koennen die
    Oberflaeche nicht kaelter verlassen, als diese ist. Deshalb ist t_mat die
    untere Schranke - dieselbe Annahme, die der CEA-Wiederholungslauf mit
    max(t_mat, t_air) bereits trifft.
    """
    if temp_factor == "isentropic":
        t_ab = (hotgas_temp_static * kappa / ((2 * kappa - 1) * (p_atm - p_2)) * p_2 ** (-(kappa - 1) / kappa)
                * (p_atm ** ((2 * kappa - 1) / kappa) - p_2 ** ((2 * kappa - 1) / kappa)))

    elif temp_factor == "mix_temperature":
        if rof_ab is None:
            raise ValueError(
                "temp_factor 'mix_temperature' braucht rof_ab.")
        t_ab = (rof_ab * hotgas_temp_static + t_mat) / (rof_ab + 1.0)

    elif temp_factor == "eckert_lam":
        r = Pr_lam ** (1 / 2)
        t_aw = hotgas_temp_static * (1 + r * ((kappa_hotgas - 1) / 2) * mach_2 ** 2)
        t_ab = hotgas_temp_static + 0.5 * (t_mat - hotgas_temp_static) + 0.22 * (t_aw - hotgas_temp_static)

    elif temp_factor == "eckert_turb":
        r = Pr_lam ** (1 / 3)
        t_aw = hotgas_temp_static * (1 + r * ((kappa_hotgas - 1) / 2) * mach_2 ** 2)
        t_ab = hotgas_temp_static + 0.5 * (t_mat - hotgas_temp_static) + 0.22 * (t_aw - hotgas_temp_static)

    elif temp_factor == "roberts_lam":
        t_ab = t_mat + (1 - (1 / 3) * (Pr_lam) ** (-0.6)) * (hotgas_t_stag - t_mat)

    elif temp_factor == "roberts_turb":
        t_ab = t_mat + (1 - (1 / 3) * (Pr_turb) ** (-0.6)) * (hotgas_t_stag - t_mat)

    elif isinstance(temp_factor, (int, float)):
        t_ab = t_mat + temp_factor * (hotgas_temp_static - t_mat)

    else:
        raise ValueError(f"Unbekannter temp_factor: {temp_factor!r}")

    return max(t_ab, t_mat)


def hotgas_state(scenario_name, idx, altitude, time, t_atm, p_atm, t_mat,
                 t_2, t_total_2, p_2, p_stag, mach_1, velocity, mach_2, rof_ab,
                 df_air_massfractions_atm, kappa_mix=kappa):
    """Hot-gas-Zustand hinter dem Stoss fuer die Materialverbrennung.

    Entspricht dem Block, den run_nasa_cea()/run_nasa_cea_scarab() inline
    ausfuehren: CEA loest den Stoss, daraus folgen die Verbrennungstemperatur
    t_ab (ueber temp_factor aus config.py), die Luft-Temperatur t_air, der
    Druck p_ab und die Zusammensetzung der Reaktionsluft.

    Rueckgabe: (t_ab, t_air, p_ab, df_air_massfractions_react). Schlaegt die
    Berechnung fehl, kommen die Ausgangswerte zurueck (t_ab = t_mat, t_air =
    t_2, p_ab = p_stag, Reaktionsluft = Atmosphaere).
    """
    t_ab = t_mat
    t_air = t_2
    p_ab = p_stag
    df_air_massfractions_react = df_air_massfractions_atm

    try:
        afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
        # CEA shock first, 'hp' at the perfect-gas post-shock state as backup; if both
        # fail, the except branch keeps ambient air at the perfect-gas stagnation state
        df_output, _ = run_hotgas_cea(afterburning_filename, t_2, p_2, t_atm, p_atm,
                                      df_air_massfractions_atm, mach_1, velocity)

        hotgas_temp_static = df_output.loc['t', 'exit']

        # Under 'shock', CEA solved the jump itself; take mach_2, p_2, p_stag and the isentropic exponent from that solution instead of compute_shockwave()
        kappa_hotgas = kappa_mix
        if problem_hotgas_calculation == "shock" and "shock_state" in df_output.attrs:
            shock_state_2 = df_output.attrs["shock_state"]
            if np.isfinite(shock_state_2["mach_2"]):
                mach_2 = shock_state_2["mach_2"]
            if np.isfinite(shock_state_2["p"]):
                p_2 = shock_state_2["p"]
            if np.isfinite(shock_state_2["p_stag"]):
                p_stag = shock_state_2["p_stag"]
            if np.isfinite(shock_state_2["gammas"]) and shock_state_2["gammas"] > 1.0:
                kappa_hotgas = shock_state_2["gammas"]

        hotgas_t_stag = hotgas_temp_static * (1 + ((kappa_hotgas - 1) / 2.0) * mach_2 ** 2)
        df_hotgas_massfractions = df_output.iloc[6:, 0].copy()

        # Mean combustion temperature (lower bound: t_mat)
        t_ab = combustion_temperature(t_mat, hotgas_temp_static, hotgas_t_stag,
                                      kappa_hotgas, mach_2, p_2, p_atm, rof_ab)

        t_air = hotgas_air_temp(hotgas_t_stag, t_total_2)
        p_ab = p_stag
        df_air_massfractions_react = pd.DataFrame([df_hotgas_massfractions])

    except Exception as hotgas_error:
        print(f"Could not cache hot-air results for row {idx}: {hotgas_error}")
        # Neither CEA shock nor hp converged: ambient air without dissociation reacts,
        # t_ab from the perfect-gas stagnation point of compute_shockwave(), never below t_mat
        try:
            t_ab = combustion_temperature(t_mat, t_2, t_total_2, kappa_mix,
                                          mach_2, p_2, p_atm, rof_ab)
        except Exception:
            t_ab = t_mat
        t_ab = max(t_ab, t_mat)

    return t_ab, t_air, p_ab, df_air_massfractions_react


def run_cantera_material_combustion(t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react):
    """Equilibrate material and shocked air at (t_ab, p_ab), the Cantera 'tp' counterpart."""
    # Returns (species, EI_species) as excess-over-ambient mass fractions per unit material
    # mass; on failure the smallest included element is dropped and the call retried
    material_cols = [
        c for c in df_material_massfractions.columns
        if c in CANTERA_ELEMENT_POLYMORPHS and float(df_material_massfractions.iloc[0][c]) > cantera_material_element_threshold_pct
    ]
    material_cols.sort(key=lambda c: float(df_material_massfractions.iloc[0][c]), reverse=True)

    last_error = None
    for n_included in range(len(material_cols), 0, -1):
        try:
            return _equilibrate_cantera_material(
                t_ab, p_ab, rof_ab, material_cols[:n_included], df_material_massfractions, df_air_massfractions_react
            )
        except Exception as e:
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("No reactable material species for Cantera equilibrium.")


def run_cea_case(filename):
    """Run NASA CEA on <filename>.inp and return the parsed output; raises if CEA fails."""
    process = subprocess.Popen(
        [nasa_cea_executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    process.stdin.write(f"{filename}\n".encode())
    process.stdin.close()
    process.wait()
    try:
        if process.returncode != 0:
            raise RuntimeError(f"NASA CEA exited with code {process.returncode}: "
                               f"{process.stderr.read().decode()}")
        df_output = readCEA(filename)
        if all_scenarios == False:
            save_files(filename)
            save_results_excel(filename, df_output)
        return df_output
    finally:
        clean_directory(filename)


def run_hotgas_cea(filename, t_2, p_2, t_atm, p_atm, df_air_massfractions, mach_1, velocity):
    """Equilibrium of the shocked air, returns (df_output, problem used); raises if nothing converges.

    With problem_hotgas_calculation = 'shock' CEA solves the jump itself. If that fails (Mach 1
    below shock_mach_min, no converged shock solution) the perfect-gas post-shock state from
    compute_shockwave() is equilibrated in an 'hp' run instead.
    """
    t_ab_nox = min(23999, t_2)
    t_atm_cea = max(160.1, t_atm)
    problems = [problem_hotgas_calculation]
    if problem_hotgas_calculation == "shock":
        problems.append("hp")

    last_error = None
    for problem in problems:
        try:
            create_inp_file_bal(filename, None, t_ab_nox, p_2, t_atm_cea, p_atm,
                                df_air_massfractions, mach_1, velocity, problem_override=problem)
            return run_cea_case(filename), problem
        except Exception as e:
            last_error = e
            if problem != problems[-1]:
                print(f"NASA CEA '{problem}' failed for {filename}: {e} - retrying with 'hp'.")
    raise last_error


def air_species_ei(df_output, df_air_massfractions):
    """Change of every species' mass fraction from ambient to shocked air, per kg of air."""
    x_out = pd.to_numeric(df_output.iloc[6:, 0], errors='coerce').fillna(0.0)
    x_out.index = x_out.index.str.lstrip('*')
    x_atm = df_air_massfractions.iloc[0].astype(float)
    x_atm.index = x_atm.index.str.lstrip('*')
    combined = pd.concat([x_out.rename('out'), x_atm.rename('atm')], axis=1).fillna(0.0)

    rows = []
    for species, values in combined.iterrows():
        # smart_round: CEA prints the mass fractions with limited precision
        EI_species = smart_round(values['out']) - smart_round(values['atm'])
        if EI_species > threshold:
            rows.append({'species': species, 'EI_species': EI_species})
    return pd.DataFrame(rows, columns=['species', 'EI_species'])


def cantera_air_species_ei(gas, df_air_massfractions):
    """Cantera counterpart of air_species_ei() for a gas already set to the shocked state."""
    atm_by_upper = {str(k).lstrip('*').upper(): (str(k).lstrip('*'), float(v))
                    for k, v in df_air_massfractions.iloc[0].items()}
    rows = []
    for name, y in zip(gas.species_names, gas.Y):
        label, y_atm = atm_by_upper.get(name.upper(), (name, 0.0))
        EI_species = y - y_atm
        if EI_species > threshold:
            rows.append({'species': label, 'EI_species': EI_species})
    return pd.DataFrame(rows, columns=['species', 'EI_species'])


def _cea_air_nox_block(filtered_rows_all, scenario_name, skip_idx=None):
    """Standalone air-shock emissions (NASA CEA) as a pivoted DataFrame(row_idx -> species)."""
    results = []
    timesteps = row_timesteps(filtered_rows_all)
    for idx, row in filtered_rows_all.iterrows():
        if skip_idx is not None and skip_idx.get(idx, False):
            continue
        rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_data(row)
        p_atm = p_atm * 10**(-5)  # Pa -> bar
        velocity = row['Velocity [km/s]'] * 1000
        a_ref = row['ReferenceArea [m^2]']
        time = row['Time [s]']
        altitude = row['Altitude [km]']
        massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timesteps[idx]

        (
            mach_1, t_total_1, p_total_1,
            mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
        ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)

        if min(23999, t_2) <= 1000:
            continue

        filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
        try:
            df_output, _ = run_hotgas_cea(filename, t_2, p_2, t_atm, p_atm, df_air_massfractions, mach_1, velocity)
        except Exception as e:
            print(f"NASA CEA error: {e} at time {time}s and temperature {min(23999, t_2)}. No air-shock emissions for this row.")
            continue

        emission_species = air_species_ei(df_output, df_air_massfractions)
        if emission_species.empty:
            continue
        emission_species['EI_species'] *= massflow
        emission_species['row_idx'] = idx
        results.append(emission_species)

    if not results:
        return pd.DataFrame()
    return pivot_species(pd.concat(results, ignore_index=True))


def _cantera_air_nox_block(filtered_rows_all, skip_idx=None):
    """Standalone air-shock emissions (Cantera) as a pivoted DataFrame(row_idx -> species)."""
    total_nox_emissions = pd.DataFrame()
    timesteps = row_timesteps(filtered_rows_all)
    for idx, row in filtered_rows_all.iterrows():
        if skip_idx is not None and skip_idx.get(idx, False):
            continue
        rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_data(row)
        p_atm = p_atm * 10**(-5)
        velocity = row['Velocity [km/s]'] * 1000
        a_ref = row['ReferenceArea [m^2]']
        time = row['Time [s]']

        timestep = timesteps[idx]
        massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep

        (
            mach_1, t_total_1, p_total_1,
            mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
        ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)

        t_ab_nox = min(23999, t_2)
        p_ab_nox = p_2

        if t_ab_nox <= 1000:
            continue

        gas = ct.Solution(file_name_ct_species, name=ct_species_phase_name)
        atm_composition = build_cantera_atm_composition(gas, df_atm_molarfractions)
        try:
            cantera_postshock_nox_state(gas, atm_composition, t_atm, p_atm,
                                        velocity, p_ab_nox)
        except ct.CanteraError as e:
            print(f"Cantera error: {e} at time {time}s and temperature {t_ab_nox}. No air-shock emissions for this row.")
            continue

        emission_species_nox = cantera_air_species_ei(gas, df_air_massfractions)
        if emission_species_nox.empty:
            continue
        emission_species_nox['EI_species'] *= massflow
        emission_species_nox['row_idx'] = idx
        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)

    if total_nox_emissions.empty:
        return pd.DataFrame()
    return pivot_species(total_nox_emissions)


#### Cantera Material Combustion - DEBRISK/Drama Main Function #############################
def run_cantera(df_trajaerodata, df_emission_factors, df_material_data, scenario_name):
    """Cantera equivalent of run_nasa_cea(), material combustion via cea_species.yaml."""
    try:
        init_directory()
        emission_results = df_trajaerodata

        filtered_rows_all = emission_results[(emission_results['scenario'] == scenario_name)]
        filtered_rows = emission_results[
            (emission_results['scenario'] == scenario_name) &
            (emission_results['d_m [kg]'] > 0)
        ]
        timesteps = row_timesteps(filtered_rows_all)
        t_ab_records = {}   # idx -> material-combustion temperature
        total_emissions = pd.DataFrame()

        # NOx for rows without material ablation (standalone air-shock chemistry)
        skip_idx = (filtered_rows_all['d_m [kg]'] > 0)
        # run_nasa_cea() gates its NOx loop on calculate_nox, this one did not.
        total_nox_emissions = (_cantera_air_nox_block(filtered_rows_all, skip_idx=skip_idx)
                               if calculate_nox else pd.DataFrame())

        # Material emissions for rows with d_m > 0
        for idx, row in filtered_rows.iterrows():
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions_atm, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_data(row)
            p_atm = p_atm * 10**(-5)
            velocity = row['Velocity [km/s]'] * 1000
            a_ref = row['ReferenceArea [m^2]']
            massflow = row['d_m [kg]']
            time = row['Time [s]']
            altitude = row['Altitude [km]']

            rof_ab = rof_ab_calc(x_ab, a_ref, massflow, rho_atm, velocity, timesteps[idx])
            df_material_massfractions = get_material_data(row['Material'], df_material_data)
            t_mat = row['Temp [K]']

            (mach_1, t_total_1, p_total_1, mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)
            p_stag = p_total_2

            # Same hot-gas logic as the CEA path: t_ab follows from temp_factor, using the shocked gas as reaction air instead of the undisturbed atmosphere
            t_ab, t_air, p_ab, df_air_massfractions_react = hotgas_state(
                scenario_name, idx, altitude, time, t_atm, p_atm, t_mat,
                t_2, t_total_2, p_2, p_stag, mach_1, velocity, mach_2, rof_ab,
                df_air_massfractions_atm, kappa_mix)

            t_ab_records[idx] = format_t_ab(t_ab)
            cantera_success = False
            mat_name = row.get('Material', 'Unknown')
            if not df_material_massfractions.empty:
                try:
                    combined_species = run_cantera_material_combustion(
                        t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react
                    )
                    combined_species = combined_species.copy()
                    combined_species['EI_species'] *= massflow
                    combined_species['row_idx'] = idx
                    total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)
                    cantera_success = True
                except Exception as e:
                    print(f"Cantera material equilibrium error: {e} at time {time}s and material {mat_name}. Using {emission_factor_method} emission factors.")
            else:
                print(f"Skipping Cantera for material '{mat_name}' (no material data); using {emission_factor_method} emission factors.")

            if not cantera_success:
                df_emission_factor_material = df_emission_factors[
                    (df_emission_factors['material'] == mat_name) &
                    (df_emission_factors['method'] == emission_factor_method)
                ]
                altitude_interval_match = df_emission_factor_material[
                    df_emission_factor_material['altitude_interval'].apply(
                        lambda x: (
                            ast.literal_eval(x)[0] <= float(altitude) <= ast.literal_eval(x)[1]
                            if isinstance(x, str) else x[0] <= float(altitude) <= x[1]
                        )
                    )
                ]
                if altitude_interval_match.empty:
                    print(f"No matching emission factors for altitude {altitude} and material {mat_name}")
                    continue
                emission_factors_row = altitude_interval_match.iloc[0]
                df_species_mass = emission_factors_row.drop(
                    ['material', 'method', 'altitude_interval']
                ).astype(float).fillna(0)
                combined_species = pd.DataFrame(columns=['species', 'EI_species'])
                for species, mass in df_species_mass.items():
                    if mass > threshold:
                        combined_species = pd.concat(
                            [combined_species, pd.DataFrame({'species': [species], 'EI_species': [mass]})],
                            ignore_index=True
                        )
                combined_species['EI_species'] *= massflow
                combined_species['row_idx'] = idx
                total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

        if not total_emissions.empty:
            total_emissions = pivot_species(total_emissions)
        else:
            total_emissions = pd.DataFrame()

        if not total_nox_emissions.empty:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        if not combined_emissions.empty:
            combined_emissions = combined_emissions.T.groupby(level=0).sum().T

        emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')
        emission_results = emission_results[(emission_results != 0).any(axis=1)]
        emission_results = emission_results.drop(columns=['*Ar', '*He', 'Ar', 'He'], errors='ignore')
        emission_results = insert_t_ab_column(emission_results, t_ab_records)
        print("Cantera material combustion calculations completed successfully.")

        return emission_results

    except Exception as e:
        handle_error(e, "run_cantera", "Error in Cantera material combustion main function.")
    finally:
        try:
            clean_up()
        except Exception:
            pass


#### Cantera Material Combustion - SCARAB Main Function #####################################
def run_cantera_scarab(df_trajaerodata, df_emission_factors, df_material_data, scenario_name):
    """Cantera equivalent of run_nasa_cea_scarab(), see run_cantera()."""
    try:
        init_directory()
        emission_results = df_trajaerodata

        dm_pattern = re.compile(r'^d_m\s+(?P<mat>.+?)\s+\[kg\]$')
        dm_cols = [c for c in emission_results.columns if dm_pattern.match(c)]

        filtered_rows_all = emission_results[(emission_results['scenario'] == scenario_name)]

        if dm_cols:
            any_dm_positive = emission_results[dm_cols].fillna(0).gt(0).any(axis=1).reindex(filtered_rows_all.index, fill_value=False)
        elif 'd_m [kg]' in filtered_rows_all.columns:
            any_dm_positive = filtered_rows_all['d_m [kg]'] > 0
        else:
            any_dm_positive = pd.Series(False, index=filtered_rows_all.index)

        filtered_rows = filtered_rows_all.loc[any_dm_positive].copy()
        timesteps = row_timesteps(filtered_rows_all)
        t_ab_records = {}   # idx -> material-combustion temperature
        total_emissions = pd.DataFrame()

        # NOx for rows without any material ablation (standalone air-shock chemistry)
        total_nox_emissions = (_cantera_air_nox_block(filtered_rows_all, skip_idx=any_dm_positive)
                               if calculate_nox else pd.DataFrame())

        df_emission_factors = df_emission_factors.copy()
        if 'altitude_interval' in df_emission_factors.columns:
            df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
                lambda x: [float(i) for i in ast.literal_eval(x)] if isinstance(x, str) else x
            )

        # Material emissions for rows with material massflow
        for idx, row in filtered_rows.iterrows():
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions_atm, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_data(row)
            p_atm = p_atm * 10**(-5)
            velocity = row['Velocity [km/s]'] * 1000
            a_ref = row['ReferenceArea [m^2]']
            time = row['Time [s]']
            altitude = row['Altitude [km]']
            t_mat = row['Temp [K]']

            (mach_1, t_total_1, p_total_1, mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)
            p_stag = p_total_2

            # Material massflows of this row; the air of the row is shared out over them by d_m
            material_massflows = {}
            for dm_col in dm_cols if dm_cols else ['d_m [kg]']:
                mat_name = row.get('Material', 'Unknown') if dm_col == 'd_m [kg]' else dm_pattern.match(dm_col).group('mat')
                try:
                    massflow = float(row.get(dm_col, 0.0))
                except Exception:
                    massflow = 0.0
                if massflow > 0:
                    material_massflows[mat_name] = massflow
            if not material_massflows:
                continue

            # rof_ab from the total material massflow: material i gets the air share d_m_i / sum(d_m),
            # so rof_ab = air / sum(d_m) is the same for every material and the air is counted once
            rof_ab = rof_ab_calc(x_ab, a_ref, sum(material_massflows.values()), rho_atm, velocity, timesteps[idx])

            # Same hot-gas logic as the CEA path; with one rof_ab per row it is needed once per row
            t_ab, t_air, p_ab, df_air_massfractions_react = hotgas_state(
                scenario_name, idx, altitude, time, t_atm, p_atm, t_mat,
                t_2, t_total_2, p_2, p_stag, mach_1, velocity, mach_2, rof_ab,
                df_air_massfractions_atm, kappa_mix)
            t_ab_records[idx] = format_t_ab(t_ab)

            for mat_name, massflow in material_massflows.items():
                df_material_massfractions = get_material_data(mat_name, df_material_data)

                cantera_success = False
                if not df_material_massfractions.empty:
                    try:
                        combined_species = run_cantera_material_combustion(
                            t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react
                        )
                        combined_species = combined_species.copy()
                        combined_species['EI_species'] *= massflow
                        combined_species['row_idx'] = idx
                        total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)
                        cantera_success = True
                    except Exception as e:
                        print(f"Cantera material equilibrium error: {e} at time {time}s and material {mat_name}. Using {emission_factor_method} emission factors.")
                else:
                    print(f"Skipping Cantera for material '{mat_name}' (no material data); using {emission_factor_method} emission factors.")

                if not cantera_success:
                    df_emission_factor_material = df_emission_factors[
                        (df_emission_factors['material'] == mat_name) &
                        (df_emission_factors['method'] == emission_factor_method)
                    ]
                    altitude_interval_match = df_emission_factor_material[
                        df_emission_factor_material['altitude_interval'].apply(
                            lambda x: x[0] <= altitude <= x[1] if isinstance(x, list) else False
                        )
                    ]
                    if altitude_interval_match.empty:
                        print(f"No matching emission factors for altitude {altitude} and material {mat_name}")
                        continue
                    emission_factors_row = altitude_interval_match.iloc[0]
                    df_ei = emission_factors_row.drop(
                        [c for c in ['material', 'method', 'altitude_interval'] if c in emission_factors_row.index]
                    ).astype(float).fillna(0)
                    combined_species = pd.DataFrame(columns=['species', 'EI_species'])
                    for species, mass in df_ei.items():
                        if mass > threshold:
                            combined_species = pd.concat(
                                [combined_species, pd.DataFrame({'species': [species], 'EI_species': [mass]})],
                                ignore_index=True
                            )
                    combined_species['EI_species'] *= massflow
                    combined_species['row_idx'] = idx
                    total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

        if not total_emissions.empty:
            total_emissions = total_emissions.pivot_table(
                index='row_idx', columns='species', values='EI_species', aggfunc='sum', fill_value=0
            )
        else:
            total_emissions = pd.DataFrame()

        if not total_nox_emissions.empty:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        if not combined_emissions.empty:
            combined_emissions = combined_emissions.T.groupby(level=0).sum().T

        emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')
        emission_results = emission_results[(emission_results != 0).any(axis=1)]
        emission_results = emission_results.drop(columns=['*Ar', '*He', 'Ar', 'He'], errors='ignore')
        emission_results = insert_t_ab_column(emission_results, t_ab_records)
        print("Cantera material combustion calculations completed successfully.")

        return emission_results

    except Exception as e:
        handle_error(e, "run_cantera_scarab", "Error in Cantera material combustion main function.")
    finally:
        try:
            clean_up()
        except Exception:
            pass


#### NASA CEA NOx ballistic emissions - Main Function
def run_nasa_cea_bal(df_trajaerodata, df_scenarios, scenario_name):
    try:
        # Initialize results
        init_directory()
        emission_results = df_trajaerodata
        scenario = df_scenarios.loc[scenario_name]  # Extract row corresponding to the scenario
        total_emissions = pd.DataFrame()
        timesteps = row_timesteps(emission_results)

        # Iterate through the filtered rows
        for idx, row in emission_results.loc[emission_results['Altitude [m]'] > 0].iterrows():
            date = row['date']
            lat = row['Lat [deg]']
            lon = row['Lon [deg]']
            alt = row['Altitude [m]']
            if alt < 0:
                alt = 0
                
            # Atmospheric Data
            rho_atm, t_atm, p_atm, c_atm, total_number_density, df_air_massfractions, df_atm_molarfractions, kappa_mix, r_mix = calculate_atmosphere_traj(date, lat, lon, alt)
            p_atm = p_atm * 10**(-5)  #Pa to bar
            velocity = row['Velocity [m/s]']
            time = row['Time [s]']
            a_ref = reference_area_bal(row, scenario)
            massflow = rho_atm * velocity * (a_ref * x_ab - a_ref) * timesteps[idx]

            (
                mach_1, t_total_1, p_total_1,
                mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
            ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa_mix, r_mix)
            p_ab = p_2
            altitude = alt / 1000.0

            # NOx formation temperature: shock-heated air, ambient air and - if present - the wall
            t_ab_nox = nox_formation_temperature(row, t_2, t_atm)

            # CEA shock first, 'hp' at the perfect-gas post-shock state as backup
            afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
            try:
                df_output, _ = run_hotgas_cea(afterburning_filename, t_ab_nox, p_ab, t_atm, p_atm,
                                              df_air_massfractions, mach_1, velocity)
            except Exception as e:
                print(f"Skipping row {idx} at {altitude:.1f} km, no CEA result: {e}")
                continue

            # Every species the shocked air gains over ambient air, '*' prefixes stripped on both sides
            combined_species = air_species_ei(df_output, df_air_massfractions)
            combined_species['EI_species'] *= massflow
            combined_species['row_idx'] = idx

            total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

        # Pivot total_emissions to have species as columns
        if total_emissions.empty:
            print(colors.BOLD + colors.RED + "No CEA result for any row, no NOx emissions." + colors.END)
            clean_up()
            return emission_results
        pivot_emissions = total_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)
        # astype(float): the running concat onto an empty frame leaves object dtype behind
        pivot_emissions = pivot_emissions.astype(float)

        # Merge with filtered_rows
        emission_results = emission_results.merge(pivot_emissions, left_index=True, right_index=True, how='left')

        # Final cleanup
        emission_results = emission_results.drop(columns=['*Ar', '*He', 'Ar', 'He'], errors='ignore')
        print("NASA CEA calculations completed successfully.")

        clean_up()
        return emission_results

    except Exception as e:
        handle_error(e, "run_nasa_cea_bal", "Error in NASA CEA main function.")

#endregion
