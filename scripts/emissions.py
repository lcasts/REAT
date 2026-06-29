### This file includes all functions to process the data with nasa cea
#Neu
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

#region: Additional subfunctions to process the main functions
# Function to calcualte the NASA CEA
def rof_ab_calc(x_ab, a_ref, massflow, rho_atm, velocity):
    rof_ab = velocity*rho_atm*(a_ref*x_ab - a_ref)/massflow
    return rof_ab

# Function to create the NASA CEA .inp file
def create_inp_file(filename, df=None, *args):   
    t_atm, p_atm, t_mat, t_stag, p_stag, t_air, t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions = args

    line = "problem o/f={},\n".format(rof_ab)

    # Input based on problem type 
    if problem_material_combustion == "tp":
        line += "  " + problem_material_combustion + "  t,k={} p,bar={} \n".format(t_ab, p_ab)
    elif problem_material_combustion == "hp":
        line += "  " + problem_material_combustion + "  p,bar={} \n".format(p_ab)

    # Separate fuel and air species into dictionaries with their mass fractions
    material_species = {species: df_material_massfractions[species].values[0] for species in df_material_massfractions.columns}
    air_species = {species: df_air_massfractions[species].values[0] for species in df_air_massfractions.columns}

    # Check and limit total species count
    total_species = len(material_species) + len(air_species)
    #print('df_air_massfractions',df_air_massfractions)
    #print('df_material_massfractions',df_material_massfractions)

    if total_species > 24:
        # Remove He and Ar from air species
        air_species = {
            k: v for k, v in air_species.items()
            if k not in ["*He", "*Ar"]
        }
        #print('He, Ar entfernt bei',filename)

        # Weight air species with rof_ab
        weighted_air_species = {
            k: v * rof_ab
            for k, v in air_species.items()
        }

        # Combine material and weighted air species
        combined_species = {
            **material_species,
            **weighted_air_species
        }

        # Keep only the 24 largest species
        combined_species = dict(
            sorted(
                combined_species.items(),
                key=lambda x: x[1],
                reverse=True
            )[:24]
        )

        # Split back into material and air species
        material_species = {
            k: v for k, v in combined_species.items()
            if k in material_species
        }

        air_species = {
            k: v / rof_ab   # optional: restore original wt values
            for k, v in combined_species.items()
            if k in weighted_air_species
        }
        #print('Aufteilung angepasst bei',filename)

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

def create_inp_file_bal(filename, df=None, *args):
    t_ab, p_ab, t_atm, p_atm, df_air_massfractions = args

    line = "problem,\n"

    if problem_hotgas_calculation == "tp":
        line += "  " + problem_hotgas_calculation + "  t,k={} p,bar={} \n".format(t_ab, p_ab)
    elif problem_hotgas_calculation == "hp":
        line += "  " + problem_hotgas_calculation + "  p,bar={} \n".format(p_ab)

    air_species = {species: df_air_massfractions[species].values[0] for species in df_air_massfractions.columns}

    if len(air_species) > 24:
        air_species = {k: v for k, v in air_species.items() if k not in ["*He", "*Ar"]}

    line += "react \n"
    for air, mass_fraction in air_species.items():
        line += "  name={} wt={} t,k={} \n".format(air, mass_fraction, t_ab)

    line += "output \n"
    line += "  siunits massf trace={} \n".format(trace)
    line += "  plot p t mach H rho s \n"
    line += "end \n"

    with open(filename + ".inp", "w") as file:
        file.write(line)

def extract_cea_hotgas_temperature(df_output, fallback_temperature):
    """
    Extract gas temperature from NASA CEA output.
    Falls back to fallback_temperature if no temperature can be identified safely.
    """
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


def get_material_data(material, df_material_data):
    """
    Extracts species and their values for a given material from the material data DataFrame.

    Parameters:
    material (str): The material name to search for.
    df_material_data (pd.DataFrame): The DataFrame containing material data.

    Returns:
    pd.DataFrame: A DataFrame containing species as the header and their corresponding values in the first row.
    """
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
    file_extensions = ["*.lib", "*.inp", "*.out", "*.plt"]
    if all_scenarios == 'Test':
        for ext in file_extensions:
            for filename in glob.glob(ext):
                if os.path.isfile(filename):  # only delete files
                    try:
                        os.remove(filename)
                        print(f"Deleted {filename}")
                    except Exception as e:
                        print(f"Could not delete {filename}: {e}")

def compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R):

    mach_1 = velocity / c_atm

    # Before the shock (always valid)
    isentropic_factor_1 = 1.0 + 0.5 * (kappa - 1.0) * mach_1**2
    t_total_1 = t_atm * isentropic_factor_1
    p_total_1 = p_atm * isentropic_factor_1 ** (kappa / (kappa - 1.0))

    # ---------------------------------------
    # CASE: NO SHOCK (Mach <= 1)
    # ---------------------------------------
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

    # ---------------------------------------
    # CASE: NORMAL SHOCK (Mach > 1)
    # ---------------------------------------

    mach_2 = math.sqrt(
        (1.0 + 0.5 * (kappa - 1.0) * mach_1**2)
        / (kappa * mach_1**2 - 0.5 * (kappa - 1.0))
    )

    pressure_ratio = 1.0 + (2.0 * kappa / (kappa + 1.0)) * (mach_1**2 - 1.0)
    density_ratio = ((kappa + 1.0) * mach_1**2) / ((kappa - 1.0) * mach_1**2 + 2.0)

    p_2 = p_atm * pressure_ratio
    rho_2 = rho_atm * density_ratio
    t_2 = t_atm * (pressure_ratio / density_ratio)

    # Total quantities after the shock
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
    nose_radius = df_scenarios.loc[scenario_name]['R_n']
    
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
        rho_atm, t_atm, p_atm, c_atm, total_number_density, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_traj(date, lat, lon, alt)
        velocity = row['Velocity [m/s]']

        # Calculate mass flow rate
        try:
            time_old
        except NameError:
            time_old = time - delta_t
        timestep = time - abs(time_old)
        if timestep < 0:
            timestep = delta_t
        massflow = rho_atm * velocity * np.pi * ((nose_radius * x_ab)**2 - nose_radius**2) * (timestep)
        time_old = time

        (
            mach_1, t_total_1, p_total_1,
            mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
        ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

        t_stag = t_total_2
        p_stag = p_total_2

        t_atm = max(160.1, t_atm)
        t_ab_nox = min(23999, t_2)
        p_ab = p_2

        # Determine temperature for NOx formation
        t_ab_nox = max(row['T_rad_equil_comb'], t_atm)

        # Load GRI-Mech 3.0 mechanism
        gas = ct.Solution(file_name_ct_species)

        # Convert atmospheric molar fractions to match GRI-Mech species
        atm_composition = {
            'N2': df_atm_molarfractions.get('N2', 0),
            'O2': df_atm_molarfractions.get('O2', 0),
            'O': df_atm_molarfractions.get('O', 0),   # Atomic Oxygen
            'N': df_atm_molarfractions.get('N', 0),   # Atomic Nitrogen
            'NO': df_atm_molarfractions.get('NO', 0), # Nitric Oxide
            'Ar': df_atm_molarfractions.get('Ar', 0)  # Argon
        }

        # Convert pandas Series to numeric values
        #atm_composition = {species: float(mole_frac) for species, mole_frac in atm_composition.items()}
        atm_composition = {
            species: float(mole_frac.iloc[0])
            for species, mole_frac in atm_composition.items()
        }
        # Compute total moles as a scalar
        total_moles = sum(atm_composition.values())

        # Fix the conditional check
        if total_moles > 0:
            atm_composition = {species: mole_frac / total_moles for species, mole_frac in atm_composition.items()}

        # Set gas state and equilibrate at constant enthalpy & pressure
        gas.TPX = t_ab_nox, p_ab, atm_composition
        gas.equilibrate('HP')

        # Extract NOx mole and mass fractions
        no_mole_fraction = gas['NO'].X[0]
        no_mass_fraction = gas['NO'].Y[0]
        no2_mole_fraction = gas['NO2'].X[0]
        no2_mass_fraction = gas['NO2'].Y[0]

        # Calculate NO emissions in kg/s
        no_emission_rate = ( no_mass_fraction - df_air_massfractions['NO'].values[0] ) * massflow
        no2_emission_rate = no2_mass_fraction * massflow 

        # Add NO emissions data to the dataframe
        emission_results.at[idx, 'NO'] = no_emission_rate
        emission_results.at[idx, 'NO2'] = no2_emission_rate
    
    return emission_results

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

        if calculate_nox == True and use_nasa_cea == False:
            total_nox_emissions = pd.DataFrame()
            if nox_method == "nasa_cea":
                for idx, row in filtered_rows_all.iterrows():
                    # Initialise atmospheric conditions and massflow
                    rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
                    p_atm = p_atm*10**(-5) #Pa into bar
                    velocity = row['Velocity [km/s]']*1000
                    a_ref = row['ReferenceArea [m^2]']
                    time = row['Time [s]']

                    try:
                        time_old
                    except NameError:
                        time_old = time - delta_t

                    timestep = time - abs(time_old)
                    if timestep < 0:
                        timestep = delta_t

                    massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep
                    time_old = time

                    (
                        mach_1, t_total_1, p_total_1,
                        mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
                    ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                    t_stag = t_total_2
                    p_stag = p_total_2

                    t_atm = max(160.1, t_atm)
                    t_ab_nox = min(23999, t_2)
                    p_ab = p_2

                    if t_ab_nox > 1000:
                        # Create and run NASA CEA input file
                        afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
                        # Estimate burning at t_2 (temperature after shockwave) and p_atm (ambient pressure))
                        create_inp_file_bal(afterburning_filename, None, t_ab_nox, p_ab, t_atm, p_atm, df_air_massfractions)

                        try:
                            # Execute NASA CEA
                            process = subprocess.Popen(
                                [nasa_cea_folder_path + nasa_cea_exe],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,  # Suppresses standard output
                                stderr=subprocess.PIPE      # Catch error output
                            )

                            process.stdin.write(f"{afterburning_filename}\n".encode())
                            process.stdin.close()
                            process.wait()  # Wait until process is finished

                            # Check for errors
                            if process.returncode != 0:
                                error_message = process.stderr.read().decode()
                                print("Error while executing NASA CEA:", error_message)
                                continue

                            # Process the output file
                            df_output = readCEA(afterburning_filename)
                            # Saving raw and formated data
                            if all_scenarios == False:
                                save_files(afterburning_filename)
                                # Save results
                                save_results_excel(afterburning_filename, df_output)

                            # Filter and process emissions data
                            df_output_filtered = df_output.iloc[6:]
                            df_output_filtered.columns = ['ab_output_filtered']
                            df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                            df_air_massfractions = df_air_massfractions.transpose()
                            df_air_massfractions.columns = ['air_massfractions']
                            df_air_massfractions.index = df_air_massfractions.index.str.replace(r'^\*', '', regex=True)

                            combined_df = pd.concat([df_output_filtered, df_air_massfractions], axis=1, join='outer').fillna(0)

                            # Initialize a DataFrame to store species and their EI_species
                            emission_species_nox = pd.DataFrame(columns=['species', 'EI_species'])
                            for species, row2 in combined_df.iterrows():
                                x_species = row2['ab_output_filtered']
                                x_species_atm = row2['air_massfractions']
                                x_species = smart_round(x_species) # Rundung, da CEA max. 5 Nachkommastellen ausgibt
                                x_species_atm = smart_round(x_species_atm) # Rundung, da CEA max. 5 Nachkommastellen ausgibt
                                EI_species = x_species - x_species_atm 
                                if EI_species > threshold:
                                    emission_species_nox = pd.concat([emission_species_nox, pd.DataFrame({'species': [species], 'EI_species': [EI_species]})], ignore_index=True)
                        
                        except Exception as e:
                            print(f"NASA CEA error: {e} at time {time}s and temperature {t_ab_nox}. Setting NOx fractions to 0")

                            emission_species_nox = pd.DataFrame({
                                'NO': [0],
                                'NO2': [0]
                            })

                        # Multiply EI_species by massflow and store in total_emissions
                        emission_species_nox['EI_species'] *= massflow
                        emission_species_nox['row_idx'] = idx  # Add row index to identify each row's results
 
                        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)
                    else:
                        pass
                
                total_nox_emissions = total_nox_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)
                cols_to_keep = [col for col in ['NO', 'NO2'] if col in total_nox_emissions.columns]
                total_nox_emissions = total_nox_emissions[cols_to_keep]
            
            if nox_method == "cantera":   
                # Iterate through filtered rows 
                for idx, row in filtered_rows_all.iterrows():
                    # Initialise atmospheric conditions and massflow
                    rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
                    p_atm = p_atm*10**(-5) #Pa into bar
                    velocity = row['Velocity [km/s]']*1000
                    a_ref = row['ReferenceArea [m^2]']
                    time = row['Time [s]']

                    try:
                        time_old
                    except NameError:
                        time_old = time - delta_t

                    timestep = time - abs(time_old)
                    if timestep < 0:
                        timestep = delta_t

                    massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep
                    time_old = time

                    (
                        mach_1, t_total_1, p_total_1,
                        mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
                    ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                    t_stag = t_total_2
                    p_stag = p_total_2

                    t_atm = max(160.1, t_atm)
                    t_ab_nox = min(23999, t_2)
                    p_ab = p_2

                    if t_ab_nox > 1000:
                        # Load GRI-Mech 3.0 mechanism
                        gas = ct.Solution(file_name_ct_species)

                        # Convert atmospheric molar fractions to match GRI-Mech species
                        atm_composition = {
                            'N2': df_atm_molarfractions.get('N2', 0),
                            'O2': df_atm_molarfractions.get('O2', 0),
                            'O': df_atm_molarfractions.get('O', 0),   # Atomic Oxygen
                            'N': df_atm_molarfractions.get('N', 0),   # Atomic Nitrogen
                            'NO': df_atm_molarfractions.get('NO', 0), # Nitric Oxide
                            'Ar': df_atm_molarfractions.get('Ar', 0)  # Argon
                        }

                        # Convert pandas Series to numeric values
                        #atm_composition = {species: float(mole_frac) for species, mole_frac in atm_composition.items()}
                        atm_composition = {
                            species: float(mole_frac.iloc[0])
                            for species, mole_frac in atm_composition.items()
                        }
                        # Compute total moles as a scalar
                        total_moles = sum(atm_composition.values())

                        # Fix the conditional check
                        if total_moles > 0:
                            atm_composition = {species: mole_frac / total_moles for species, mole_frac in atm_composition.items()}

                        try:
                            # Set gas state and equilibrate at constant enthalpy & pressure
                            gas.TPX = t_ab_nox, p_ab, atm_composition
                            gas.equilibrate('HP')

                            # Extract NOx mole and mass fractions
                            no_mole_fraction = gas['NO'].X[0]
                            no_mass_fraction = gas['NO'].Y[0]
                            no2_mole_fraction = gas['NO2'].X[0]
                            no2_mass_fraction = gas['NO2'].Y[0]
                        
                        except ct.CanteraError as e:
                            print(f"Cantera error: {e} at time {time}s and temperature {t_ab_nox}. Setting NOx fractions to 0")

                            no_mole_fraction = 0
                            no_mass_fraction = 0
                            no2_mole_fraction = 0
                            no2_mass_fraction = 0

                        # Calculate NO emissions in kg/s
                        no_emission_rate = ( no_mass_fraction - df_air_massfractions['NO'].values[0] ) * massflow
                        no2_emission_rate = no2_mass_fraction * massflow

                        # Build emissions dataframe
                        emission_species_nox = pd.DataFrame([
                            {'species': 'NO', 'EI_species': no_emission_rate},
                            {'species': 'NO2', 'EI_species': no2_emission_rate}
                        ])

                        # Multiply EI_species by massflow (already done in the calculation above)
                        emission_species_nox['row_idx'] = idx  # Add row index to identify each row's results

                        # Append current emissions to the total emissions dataframe
                        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)

                    else:
                        pass

                total_nox_emissions = total_nox_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)
                cols_to_keep = [col for col in ['NO', 'NO2'] if col in total_nox_emissions.columns]
                total_nox_emissions = total_nox_emissions[cols_to_keep]

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
        total_emissions = total_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)

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

        # --- filter rows for scenario; keep only rows with any per-material d_m > 0
        # detect per-material massflow columns like: 'd_m AA7075 [kg]'
        dm_pattern = re.compile(r'^d_m\s+(?P<mat>.+?)\s+\[kg\]$')
        dm_cols = [c for c in emission_results.columns if dm_pattern.match(c)]

        if not dm_cols:
            print("No per-material massflow columns found (expected columns like 'd_m AA7075 [kg]').")

        scenario_mask = (emission_results['scenario'] == scenario_name) if 'scenario' in emission_results.columns else True
        any_dm_positive = emission_results[dm_cols].fillna(0).gt(0).any(axis=1) if dm_cols else False
        if isinstance(scenario_mask, bool):
            filtered_rows = emission_results.copy()
        else:
            filtered_rows = emission_results[scenario_mask].copy()

        if isinstance(any_dm_positive, pd.Series):
            filtered_rows = filtered_rows.loc[any_dm_positive.reindex(filtered_rows.index, fill_value=False)]

        total_emissions = pd.DataFrame()

        # parse altitude_interval
        df_emission_factors = df_emission_factors.copy()
        df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
            lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
        )

        # meta/species columns
        meta_cols = [c for c in ['material', 'method', 'altitude_interval'] if c in df_emission_factors.columns]
        species_cols = [c for c in df_emission_factors.columns if c not in meta_cols]
        df_emission_factors[species_cols] = df_emission_factors[species_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # group EF by material for faster access
        ef_by_material = {m: g.reset_index(drop=True) for m, g in df_emission_factors.groupby('material', dropna=False)}

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
            pivot_emissions = total_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)
            emission_results = filtered_rows.merge(pivot_emissions, left_index=True, right_index=True, how='left')
        else:
            emission_results = filtered_rows.copy()

        # final cleanup and output
        emission_results = emission_results.drop(columns=['*Ar', '*He'], errors='ignore')
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
        total_emissions = pd.DataFrame()
        # ------------------------------------------------------------------
        # NOx emission calculation for rows without d_m > 0
        # ------------------------------------------------------------------
        if calculate_nox == True:
            total_nox_emissions = pd.DataFrame()

            if nox_method == "nasa_cea":
                for idx, row in filtered_rows_all.iterrows():
                    # Initialise atmospheric conditions and massflow
                    rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
                    p_atm = p_atm * 10**(-5)  # Pa into bar
                    velocity = row['Velocity [km/s]'] * 1000
                    a_ref = row['ReferenceArea [m^2]']
                    time = row['Time [s]']
                    d_m_current = row['d_m [kg]']
                    altitude = row['Altitude [km]']

                    try:
                        time_old
                    except NameError:
                        time_old = time - delta_t

                    timestep = time - abs(time_old)
                    if timestep < 0:
                        timestep = delta_t

                    massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep
                    time_old = time

                    (
                        mach_1, t_total_1, p_total_1,
                        mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
                    ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                    t_stag = t_total_2
                    p_stag = p_total_2

                    t_atm = max(160.1, t_atm)
                    t_ab_nox = min(23999, t_2)
                    p_ab_nox = p_2

                    if t_ab_nox > 1000 and d_m_current <= 0:

                        afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
                        create_inp_file_bal(afterburning_filename, None, t_ab_nox, p_ab_nox, t_atm, p_atm, df_air_massfractions)

                        try:
                            process = subprocess.Popen(
                                [nasa_cea_folder_path + nasa_cea_exe],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE
                            )

                            process.stdin.write(f"{afterburning_filename}\n".encode())
                            process.stdin.close()
                            process.wait()

                            if process.returncode != 0:
                                error_message = process.stderr.read().decode()
                                print("Error while executing NASA CEA:", error_message)
                                continue

                            # Process the output file
                            df_output = readCEA(afterburning_filename)

                            # Saving raw
                            if all_scenarios == False:
                                # Saving raw and formated data
                                save_files(afterburning_filename)
                                save_results_excel(afterburning_filename, df_output)
                            clean_directory(afterburning_filename)

                            df_output_filtered = df_output.iloc[6:]
                            df_output_filtered.columns = ['ab_output_filtered']
                            df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                            df_air_massfractions_nox = df_air_massfractions.transpose()
                            df_air_massfractions_nox.columns = ['air_massfractions']
                            df_air_massfractions_nox.index = df_air_massfractions_nox.index.str.replace(r'^\*', '', regex=True)

                            combined_df = pd.concat(
                                [df_output_filtered, df_air_massfractions_nox],
                                axis=1,
                                join='outer'
                            ).fillna(0)

                            emission_species_nox = pd.DataFrame(columns=['species', 'EI_species'])
                            for species, row2 in combined_df.iterrows():
                                x_species = row2['ab_output_filtered']
                                x_species_atm = row2['air_massfractions']
                                x_species = smart_round(x_species)
                                x_species_atm = smart_round(x_species_atm)
                                EI_species = x_species - x_species_atm
                                if EI_species > threshold:
                                    emission_species_nox = pd.concat(
                                        [
                                            emission_species_nox,
                                            pd.DataFrame({'species': [species], 'EI_species': [EI_species]})
                                        ],
                                        ignore_index=True
                                    )

                        except Exception as e:
                            print(f"NASA CEA error: {e} at time {time}s and temperature {t_ab_nox}. Setting NOx fractions to 0")

                            emission_species_nox = pd.DataFrame([
                                {'species': 'NO', 'EI_species': 0},
                                {'species': 'NO2', 'EI_species': 0}
                            ])

                        # Only relevant for d_m <= 0 branch
                        emission_species_nox['EI_species'] *= massflow
                        emission_species_nox['row_idx'] = idx

                        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)

                    else:
                        pass

                if not total_nox_emissions.empty:
                    total_nox_emissions = total_nox_emissions.pivot(
                        index='row_idx',
                        columns='species',
                        values='EI_species'
                    ).fillna(0)
                    cols_to_keep = [col for col in ['NO', 'NO2'] if col in total_nox_emissions.columns]
                    total_nox_emissions = total_nox_emissions[cols_to_keep]
                else:
                    total_nox_emissions = pd.DataFrame()

            if nox_method == "cantera":
                # Iterate through filtered rows
                for idx, row in filtered_rows_all.iterrows():
                    rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
                    p_atm = p_atm * 10**(-5)
                    velocity = row['Velocity [km/s]'] * 1000
                    a_ref = row['ReferenceArea [m^2]']
                    time = row['Time [s]']

                    try:
                        time_old
                    except NameError:
                        time_old = time - delta_t

                    timestep = time - abs(time_old)
                    if timestep < 0:
                        timestep = delta_t

                    massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep
                    time_old = time

                    (
                        mach_1, t_total_1, p_total_1,
                        mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
                    ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                    t_stag = t_total_2
                    p_stag = p_total_2

                    t_atm = max(160.1, t_atm)
                    t_ab_nox = min(23999, t_2)
                    p_ab_nox = p_2

                    if t_ab_nox > 1000:
                        gas = ct.Solution(file_name_ct_species)

                        atm_composition = {
                            'N2': df_atm_molarfractions.get('N2', 0),
                            'O2': df_atm_molarfractions.get('O2', 0),
                            'O': df_atm_molarfractions.get('O', 0),
                            'N': df_atm_molarfractions.get('N', 0),
                            'NO': df_atm_molarfractions.get('NO', 0),
                            'Ar': df_atm_molarfractions.get('Ar', 0)
                        }

                        atm_composition = {
                            species: float(mole_frac.iloc[0])
                            for species, mole_frac in atm_composition.items()
                        }

                        total_moles = sum(atm_composition.values())
                        if total_moles > 0:
                            atm_composition = {
                                species: mole_frac / total_moles
                                for species, mole_frac in atm_composition.items()
                            }

                        try:
                            gas.TPX = t_ab_nox, p_ab_nox, atm_composition
                            gas.equilibrate('HP')

                            no_mole_fraction = gas['NO'].X[0]
                            no_mass_fraction = gas['NO'].Y[0]
                            no2_mole_fraction = gas['NO2'].X[0]
                            no2_mass_fraction = gas['NO2'].Y[0]

                        except ct.CanteraError as e:
                            print(f"Cantera error: {e} at time {time}s and temperature {t_ab_nox}. Setting NOx fractions to 0")

                            no_mole_fraction = 0
                            no_mass_fraction = 0
                            no2_mole_fraction = 0
                            no2_mass_fraction = 0

                        no_emission_rate = (no_mass_fraction - df_air_massfractions['NO'].values[0]) * massflow
                        no2_emission_rate = no2_mass_fraction * massflow

                        emission_species_nox = pd.DataFrame([
                            {'species': 'NO', 'EI_species': no_emission_rate},
                            {'species': 'NO2', 'EI_species': no2_emission_rate}
                        ])

                        emission_species_nox['row_idx'] = idx
                        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)

                if not total_nox_emissions.empty:
                    total_nox_emissions = total_nox_emissions.pivot(
                        index='row_idx',
                        columns='species',
                        values='EI_species'
                    ).fillna(0)
                    cols_to_keep = [col for col in ['NO', 'NO2'] if col in total_nox_emissions.columns]
                    total_nox_emissions = total_nox_emissions[cols_to_keep]
                else:
                    total_nox_emissions = pd.DataFrame()

        # ------------------------------------------------------------------
        # Material emission calculation for rows with d_m > 0
        # ------------------------------------------------------------------
        for idx, row in filtered_rows.iterrows():
            # Atmospheric Data
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions_atm, df_atm_molarfractions = calculate_atmosphere_data(row)
            p_atm = p_atm * 10**(-5)
            velocity = row['Velocity [km/s]'] * 1000
            a_ref = row['ReferenceArea [m^2]']
            massflow = row['d_m [kg]']
            time = row['Time [s]']

            # Calculate rof_ab
            rof_ab = rof_ab_calc(x_ab, a_ref, massflow, rho_atm, velocity)

            # Material Data
            df_material_massfractions = get_material_data(row['Material'], df_material_data)

            # Remove species with mass fraction below threshold
            df_material_massfractions = df_material_massfractions.loc[
                :, df_material_massfractions.iloc[0] > threshold]

            # Definition of material temperature from DRAMA file
            t_mat = row['Temp [K]']

            # Calculation of thermodynamic properties before and after shock
            (mach_1, t_total_1, p_total_1, mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

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

            # If available, use hot-air composition for CEA reaction input
            if calculate_nox and nox_method == "nasa_cea":
                try:
                    afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
                    # Definition of NOx calculation values
                    t_atm = max(160.1, t_atm)
                    t_ab_nox = min(23999, t_2)
                    p_ab_nox = p_2
                    create_inp_file_bal(afterburning_filename, None, t_ab_nox, p_ab_nox, t_atm, p_atm, df_air_massfractions)

                    process = subprocess.Popen(
                        [nasa_cea_folder_path + nasa_cea_exe],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )

                    process.stdin.write(f"{afterburning_filename}\n".encode())
                    process.stdin.close()
                    process.wait()

                    if process.returncode != 0:
                        error_message = process.stderr.read().decode()
                        print("Error while executing NASA CEA:", error_message)
                        continue

                    # Process the output file
                    df_output = readCEA(afterburning_filename)

                    # Saving raw and formated data
                    if all_scenarios == False:
                        save_files(afterburning_filename)
                        save_results_excel(afterburning_filename, df_output)
                    clean_directory(afterburning_filename)
                    
                    # Extract hot gas temperature
                    hotgas_temp_static = df_output.loc['t', 'exit']

                    # Calculate hot gas stagnation point temperature
                    hotair_t_stag = hotgas_temp_static * (1 + ((kappa - 1) / 2.0) *mach_2 **2)
                    
                    # Extract hot gas species
                    df_hotair_massfractions = df_output.iloc[6:, 0].copy()
                    #df_hotair_massfractions.index = df_hotair_massfractions.index.str.replace(r'^\*', '', regex=True)

                    # Calculate mean temperature for combustion
                    if temp_factor == "isentropic":
                        t_mean = ( hotgas_temp_static * kappa / ((2 * kappa - 1) * (p_atm - p_2)) * p_2 ** (-(kappa - 1) / kappa) * (p_atm ** ((2 * kappa - 1) / kappa) - p_2 ** ((2 * kappa - 1) / kappa)) )
                        #print(f"t_mean: {t_mean:.2f} K")

                    elif temp_factor == "mix_temperature":
                        # Alternate: Calculate mix temperature for combustion
                        def mix_temperature(T_gas, T_mat, rof_ab):
                            return (rof_ab * T_gas + T_mat) / (rof_ab + 1.0)
                        t_mix = mix_temperature(hotgas_temp_static, t_mat, rof_ab)
                        #print('T_mix:',t_mix)

                    elif temp_factor == "eckert_lam":
                        r = Pr_lam**(1/2)
                        t_aw = hotgas_temp_static * (1 + r * ((kappa-1)/2) * mach_2**2)
                        t_ab = hotgas_temp_static + 0.5 * (t_mat - hotgas_temp_static) + 0.22 * (t_aw - hotgas_temp_static)
                        #print('Eckert:',t_ab,'K')

                    elif temp_factor == "eckert_turb":
                        r = Pr_lam**(1/3)
                        t_aw = hotgas_temp_static * (1 + r * ((kappa-1)/2) * mach_2**2)
                        t_ab = hotgas_temp_static + 0.5 * (t_mat - hotgas_temp_static) + 0.22 * (t_aw - hotgas_temp_static)
                        #print('Eckert:',t_ab,'K')

                    elif temp_factor == "roberts_lam":
                        t_ab = t_mat + (1-(1/3)*(Pr_lam)**(-0.6))*(hotgas_temp_static - t_mat)
                        #print('Roberts:',t_ab,'K')
                    
                    elif temp_factor == "roberts_turb":
                        t_ab = t_mat + (1-(1/3)*(Pr_turb)**(-0.6))*(hotgas_temp_static - t_mat)
                        #print('Roberts:',t_ab,'K')

                    elif isinstance(temp_factor, (int, float)):
                        t_ab = t_mat + temp_factor * (hotgas_temp_static - t_mat)
                        #print('Temperature Factor:',t_ab,'K')

                    else:
                        raise ValueError(f"Unbekannter temp_factor: {temp_factor!r}")

                    # Overwriting of hot gas air temperature
                    t_air = hotair_t_stag
                    t_air = min(23999, t_air)
                    
                    p_ab = p_stag

                    # Overwriting input air composition
                    df_air_massfractions_react = pd.DataFrame([df_hotair_massfractions])

                except Exception as cache_error:
                    print(f"Could not cache hot-air results for row {idx}: {cache_error}")

            afterburning_filename = f"{scenario_name}_{row['ObjectName']}_{idx}_{int(altitude)}km_{int(time)}s"

            # Try the NASA CEA calculation up to twice: if it fails, retry once with
            # t_ab clamped to max(t_mat, t_air) before falling back to emission factors.
            cea_success = False
            for attempt in range(2):
                input_file_path = str(afterburning_filename)
                output_file_path = str(afterburning_filename)
                try:
                    create_inp_file(afterburning_filename,None, t_atm, p_atm, t_mat, t_stag, p_stag, t_air, t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions_react)

                    process = subprocess.Popen(
                        [nasa_cea_folder_path + nasa_cea_exe],
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

                    df_output_filtered = df_output.iloc[6:]
                    df_output_filtered.columns = ['ab_output_filtered']
                    df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                    # IMPORTANT:
                    # For EI calculation, always compare against atmospheric air,
                    # not against hot-air species.
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
        if not dm_cols:
            # fall back to legacy column if present, else no massflow filtering
            legacy_mask = (filtered_rows_all['d_m [kg]'] > 0) if 'd_m [kg]' in filtered_rows_all.columns else True
            filtered_rows = filtered_rows_all[legacy_mask].copy()
        else:
            any_dm_positive = emission_results[dm_cols].fillna(0).gt(0).any(axis=1)
            filtered_rows = filtered_rows_all.loc[any_dm_positive.reindex(filtered_rows_all.index, fill_value=False)].copy()

        total_emissions = pd.DataFrame()

        if calculate_nox == True:
            total_nox_emissions = pd.DataFrame()

            if nox_method == "nasa_cea":
                for idx, row in filtered_rows_all.iterrows():
                    # Initialise atmospheric conditions and massflow
                    rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
                    p_atm = p_atm*10**(-5)  # Pa -> bar
                    velocity = row['Velocity [km/s]']*1000
                    a_ref = row['ReferenceArea [m^2]']
                    time = row['Time [s]']

                    try:
                        time_old
                    except NameError:
                        time_old = time - delta_t

                    timestep = time - abs(time_old)
                    if timestep < 0:
                        timestep = delta_t

                    massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep
                    time_old = time

                    (
                        mach_1, t_total_1, p_total_1,
                        mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
                    ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                    t_stag = t_total_2
                    p_stag = p_total_2

                    t_atm = max(160.1, t_atm)
                    t_ab = min(23999, t_2)
                    p_ab = p_2


                    if t_ab > 1000:
                        afterburning_filename = f"{scenario_name}_nox_{int(time)}"
                        create_inp_file_bal(afterburning_filename, None, t_ab, p_ab, t_atm, p_atm, df_air_massfractions)
                        try:
                            process = subprocess.Popen(
                                [nasa_cea_folder_path + nasa_cea_exe],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE
                            )
                            process.stdin.write(f"{afterburning_filename}\n".encode())
                            process.stdin.close()
                            process.wait()

                            if process.returncode != 0:
                                error_message = process.stderr.read().decode()
                                print("Error while executing NASA CEA:", error_message)
                                continue

                            df_output = readCEA(afterburning_filename)
                            save_files(afterburning_filename)

                            df_output_filtered = df_output.iloc[6:]
                            df_output_filtered.columns = ['ab_output_filtered']
                            df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                            df_air_massfractions = df_air_massfractions.transpose()
                            df_air_massfractions.columns = ['air_massfractions']
                            df_air_massfractions.index = df_air_massfractions.index.str_replace(r'^\*', '', regex=True)

                            combined_df = pd.concat([df_output_filtered, df_air_massfractions], axis=1, join='outer').fillna(0)

                            emission_species_nox = pd.DataFrame(columns=['species', 'EI_species'])
                            for species, row2 in combined_df.iterrows():
                                x_species = smart_round(row2['ab_output_filtered'])
                                x_species_atm = smart_round(row2['air_massfractions'])
                                EI_species = x_species - x_species_atm
                                if EI_species > threshold:
                                    emission_species_nox = pd.concat(
                                        [emission_species_nox,
                                         pd.DataFrame({'species': [species], 'EI_species': [EI_species]})],
                                        ignore_index=True
                                    )
                        except Exception as e:
                            print(f"NASA CEA error: {e} at time {time}s and temperature {t_ab}. Setting NOx fractions to 0")
                            emission_species_nox = pd.DataFrame({'species': ['NO', 'NO2'], 'EI_species': [0.0, 0.0]})

                        # Multiply EI_species by massflow and store
                        emission_species_nox['EI_species'] *= massflow
                        emission_species_nox['row_idx'] = idx
                        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)

                total_nox_emissions = total_nox_emissions.pivot_table(
                    index='row_idx', columns='species', values='EI_species', aggfunc='sum', fill_value=0
                )
                cols_to_keep = [col for col in ['NO', 'NO2'] if col in total_nox_emissions.columns]
                total_nox_emissions = total_nox_emissions[cols_to_keep]

            if nox_method == "cantera":
                for idx, row in filtered_rows_all.iterrows():
                    rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
                    p_atm = p_atm*10**(-5)
                    velocity = row['Velocity [km/s]']*1000
                    a_ref = row['ReferenceArea [m^2]']
                    time = row['Time [s]']
                    
                    try:
                        time_old
                    except NameError:
                        time_old = time - delta_t

                    timestep = time - abs(time_old)
                    if timestep < 0:
                        timestep = delta_t

                    massflow = rho_atm * velocity * ((a_ref * x_ab) - a_ref) * timestep
                    time_old = time

                    (
                        mach_1, t_total_1, p_total_1,
                        mach_2, p_2, rho_2, t_2, t_total_2, p_total_2, velocity_2,
                    ) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                    t_stag = t_total_2
                    p_stag = p_total_2

                    t_atm = max(160.1, t_atm)
                    t_ab = min(23999, t_2)
                    p_ab = p_2

                    if t_ab > 1000:
                        gas = ct.Solution(file_name_ct_species)

                        atm_composition = {
                            'N2': float(df_atm_molarfractions.get('N2', 0)),
                            'O2': float(df_atm_molarfractions.get('O2', 0)),
                            'O':  float(df_atm_molarfractions.get('O', 0)),
                            'N':  float(df_atm_molarfractions.get('N', 0)),
                            'NO': float(df_atm_molarfractions.get('NO', 0)),
                            'Ar': float(df_atm_molarfractions.get('Ar', 0)),
                        }
                        total_moles = sum(atm_composition.values())
                        if total_moles > 0:
                            atm_composition = {k: v/total_moles for k, v in atm_composition.items()}

                        try:
                            gas.TPX = t_ab, p_ab, atm_composition
                            gas.equilibrate('HP')
                            no_mass_fraction = gas['NO'].Y[0]
                            no2_mass_fraction = gas['NO2'].Y[0]
                        except ct.CanteraError as e:
                            print(f"Cantera error: {e} at time {time}s and temperature {t_ab}. Setting NOx fractions to 0")
                            no_mass_fraction = 0.0
                            no2_mass_fraction = 0.0

                        no_emission_rate = (no_mass_fraction - df_air_massfractions['NO'].values[0]) * massflow
                        no2_emission_rate = no2_mass_fraction * massflow

                        emission_species_nox = pd.DataFrame([
                            {'species': 'NO', 'EI_species': no_emission_rate},
                            {'species': 'NO2', 'EI_species': no2_emission_rate}
                        ])
                        emission_species_nox['row_idx'] = idx
                        total_nox_emissions = pd.concat([total_nox_emissions, emission_species_nox], ignore_index=True)

                total_nox_emissions = total_nox_emissions.pivot_table(
                    index='row_idx', columns='species', values='EI_species', aggfunc='sum', fill_value=0
                )
                cols_to_keep = [col for col in ['NO', 'NO2'] if col in total_nox_emissions.columns]
                total_nox_emissions = total_nox_emissions[cols_to_keep]

        df_emission_factors = df_emission_factors.copy()
        if 'altitude_interval' in df_emission_factors.columns:
            df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
                lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
            )
        meta_cols = [c for c in ['material', 'method', 'altitude_interval'] if c in df_emission_factors.columns]
        species_cols = [c for c in df_emission_factors.columns if c not in meta_cols]
        if species_cols:
            df_emission_factors[species_cols] = df_emission_factors[species_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        for idx, row in filtered_rows.iterrows():
            # Atmospheric Data (row-level, same for all materials in this timestep)
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
            p_atm = p_atm*10**(-5)  # Pa -> bar
            velocity = row['Velocity [km/s]']*1000
            a_ref = row['ReferenceArea [m^2]']
            time = row['Time [s]']

            # iterate each material present in this row
            for dm_col in dm_cols if dm_cols else ['d_m [kg]']:
                # Determine material name and massflow for this column
                if dm_col == 'd_m [kg]':
                    mat_name = row.get('Material', 'Unknown')
                else:
                    mat_name = dm_pattern.match(dm_col).group('mat')

                try:
                    massflow = float(row.get(dm_col, 0.0))
                except Exception:
                    massflow = 0.0
                if massflow <= 0:
                    continue  # skip materials with zero massflow

                # rof_ab depends on massflow of THIS material
                rof_ab = rof_ab_calc(x_ab, a_ref, massflow, rho_atm, velocity)

                # Material Data (for THIS material)
                df_material_massfractions = get_material_data(mat_name, df_material_data)

                # Apply threshold to material species
                df_material_massfractions = df_material_massfractions.loc[
                    :, df_material_massfractions.iloc[0] > threshold
                ]

                # Build afterburning case (per material)
                t_mat = row['Temp [K]']
                (mach_1,t_total_1,p_total_1,mach_2,p_2,rho_2,t_2,t_total_2,p_total_2,velocity_2,) = compute_shockwave(t_atm, p_atm, rho_atm, velocity, c_atm, kappa, R)

                obj_name = row.get('ObjectName', row.get('FragmentID', 'Object'))
                safe_mat = str(mat_name).replace(' ', '_')
                afterburning_filename = f"{scenario_name}_{obj_name}_{safe_mat}_{int(time)}"

                create_inp_file(afterburning_filename,None, t_atm, p_atm, t_mat, t_stag, p_stag, t_air, t_ab, p_ab, rof_ab, df_material_massfractions, df_air_massfractions)
                input_file_path = str(afterburning_filename)
                output_file_path = str(afterburning_filename)

                try:
                    # Execute CEA for THIS material
                    process = subprocess.Popen(
                        [nasa_cea_folder_path + nasa_cea_exe],
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
                    # Read & save
                    df_output = readCEA(output_file_path)
                    if all_scenarios == False:
                        save_files(afterburning_filename)

                    df_output_filtered = df_output.iloc[6:]
                    df_output_filtered.columns = ['ab_output_filtered']
                    df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                    df_air_mf = df_air_massfractions.transpose()
                    df_air_mf.columns = ['air_massfractions']
                    df_air_mf.index = df_air_mf.index.str.replace(r'^\*', '', regex=True)

                    combined_df = pd.concat([df_output_filtered, df_air_mf], axis=1, join='outer').fillna(0)

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

                    # Clean up CEA files for this material
                    clean_directory(afterburning_filename)

                except Exception as e:
                    # Fallback: stoichiometric EF for THIS material
                    print(f"NASA CEA error: {e} at time {time}s and material {mat_name}. Using {emission_factor_method} emission factors.")
                    if 'altitude_interval' in df_emission_factors.columns:
                        df_emission_factor_material = df_emission_factors[(df_emission_factors['material'] == mat_name) & (df_emission_factors['method'] == emission_factor_method)]
                        print(f"Using emission factors for material {mat_name} at altitude {row['Altitude [km]']}")

                        altitude_interval_match = df_emission_factor_material[
                            df_emission_factor_material['altitude_interval'].apply(
                                lambda x: x[0] <= row['Altitude [km]'] <= x[1] if isinstance(x, list) else False
                            )
                        ]

                        if altitude_interval_match.empty:
                            print(f"No matching emission factors for altitude {row['Altitude [km]']} and material {mat_name}")
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
                        if 'calculate_black_carbon' in globals() and calculate_black_carbon and 'BC_prim' in df_emission_factors.columns:
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
                        continue

        # =========================
        # 3) Pivot & combine totals
        # =========================
        # Sum across materials per row/species
        total_emissions = total_emissions.pivot_table(
            index='row_idx', columns='species', values='EI_species', aggfunc='sum', fill_value=0
        )

        if calculate_nox == True:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        # If some columns are duplicated (like 'NO'), combine them by summing
        combined_emissions = combined_emissions.groupby(level=0, axis=1).sum()

        # Merge back to all scenario rows
        emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')

        # Drop rows where all added species are zero (keep original cols)
        added_cols = [c for c in emission_results.columns if c not in df_trajaerodata.columns]
        if added_cols:
            nonzero_mask = (emission_results[added_cols].fillna(0) != 0).any(axis=1)
            emission_results = emission_results[nonzero_mask | (~nonzero_mask)]  # keep all if you prefer; else just use nonzero_mask

        # Final cleanup and output
        emission_results = emission_results.drop(columns=['*Ar', '*He','Ar', 'He'], errors='ignore')
        print("NASA CEA calculations completed successfully.")

        # Remove thermo.lib and trans.lib (and any other temp artifacts)
        clean_up()

        return emission_results

    except Exception as e:
        handle_error(e, "run_nasa_cea", "Error in NASA CEA main function.")


#### NASA CEA NOx ballistic emissions - Main Function
def run_nasa_cea_bal(df_trajaerodata, df_scenarios, scenario_name):
    try:
        # Initialize results
        init_directory()
        emission_results = df_trajaerodata
        scenario = df_scenarios.loc[scenario_name]  # Extract row corresponding to the scenario
        nose_radius = scenario['R_n']  # Extract scalar value
        total_emissions = pd.DataFrame()

        # Iterate through the filtered rows
        for idx, row in emission_results.loc[emission_results['Altitude [m]'] > 0].iterrows():
            date = row['date']
            lat = row['Lat [deg]']
            lon = row['Lon [deg]']
            alt = row['Altitude [m]']
            if alt < 0:
                alt = 0
                
            # Atmospheric Data
            rho_atm, t_atm, p_atm, c_atm, total_number_density, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_traj(date, lat, lon, alt)
            p_atm = p_atm * 10**(-5)  #Pa to bar
            velocity = row['Velocity [m/s]']
            time = row['Time [s]']
            try:
                time_old
            except NameError:
                time_old = time - delta_t
            timestep = time - abs(time_old)
            if timestep < 0:
                timestep = delta_t
            massflow = rho_atm * velocity * np.pi * ((nose_radius)**2 * x_ab - nose_radius**2)* (timestep)
            time_old = time

            #t_air = shockwave_temperature(t_atm, velocity, c_atm)
            t_ab_nox = max(row['T_rad_equil_comb'], t_atm)
            if t_ab_nox > 24000:
                t_ab_nox = 23999

            # Create and run NASA CEA input file
            afterburning_filename = f"{scenario_name}_nox_{idx}_{int(altitude)}km_{int(time)}s"
            create_inp_file_bal(afterburning_filename, None, t_ab_nox, p_ab, t_atm, p_atm, df_air_massfractions)
            # Execute NASA CEA
            process = subprocess.Popen(
                [nasa_cea_folder_path + nasa_cea_exe],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,  # Suppresses standard output
                stderr=subprocess.PIPE      # Catch error output
            )

            process.stdin.write(f"{afterburning_filename}\n".encode())
            process.stdin.close()
            process.wait()  # Wait until process is finished

            # Check for errors
            if process.returncode != 0:
                error_message = process.stderr.read().decode()
                print("Error while executing NASA CEA:", error_message)
                continue

            # Process the output file
            df_output = readCEA(afterburning_filename)
            
            if all_scenarios == False:
                # Saving raw and formated data
                save_files(afterburning_filename)
                # Save results
                save_results_excel(afterburning_filename, df_output)

            # Filter and process emissions data
            df_output_filtered = df_output.iloc[6:]
            df_output_filtered.columns = ['ab_output_filtered']
            df_air_massfractions = df_air_massfractions.transpose()
            df_air_massfractions.columns = ['air_massfractions']

            combined_df = pd.concat([df_output_filtered, df_air_massfractions], axis=1, join='outer').fillna(0)

            # Initialize a DataFrame to store species and their EI_species
            combined_species = pd.DataFrame(columns=['species', 'EI_species'])

            for species, row2 in combined_df.iterrows():
                x_species = row2['ab_output_filtered']
                x_species_atm = row2['air_massfractions']
                x_species = smart_round(x_species) # Rundung, da CEA max. 5 Nachkommastellen ausgibt
                x_species_atm = smart_round(x_species_atm) # Rundung, da CEA max. 5 Nachkommastellen ausgibt
                EI_species = x_species - x_species_atm 
                if EI_species > threshold:
                    combined_species = pd.concat([combined_species, pd.DataFrame({'species': [species], 'EI_species': [EI_species]})], ignore_index=True)

            # Multiply EI_species by massflow and store in total_emissions
            combined_species['EI_species'] *= massflow
            combined_species['row_idx'] = idx
            
            total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

            # Clean up CEA files
            clean_directory(afterburning_filename)

        # Pivot total_emissions to have species as columns
        pivot_emissions = total_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)
        pivot_emissions.columns = [col.lstrip('*') for col in pivot_emissions.columns]

        # Keep only NO and NO2
        pivot_emissions = pivot_emissions[['NO', 'NO2']]

        # Merge with filtered_rows
        emission_results = emission_results.merge(pivot_emissions, left_index=True, right_index=True, how='left')

        # Final cleanup
        emission_results = emission_results.drop(columns=['*Ar', '*He'], errors='ignore')
        print("NASA CEA calculations completed successfully.")

        clean_up()
        return emission_results

    except Exception as e:
        handle_error(e, "run_nasa_cea_bal", "Error in NASA CEA main function.")

#endregion
