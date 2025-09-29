### This file includes all functions to process the data with nasa cea

#region: Imports & Co.
from input_data.config import *
from scripts.data_processing import *
import subprocess
import shutil
import os
import re
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
def rof_ab_calc(rof_f, a_ref, massflow, rho_atm, velocity):
    rof_ab = velocity*rho_atm*(a_ref*rof_f - a_ref)/massflow
    return rof_ab

# Function to create the NASA CEA .inp file
def create_inp_file(filename, df=None, *args):
    t_atm, p_atm ,rho_atm, s_ab, t_material, t_air, p_air, rof_ab, df_material_massfractions, df_air_massfractions = args
    line = "problem o/f={},\n".format(rof_ab)
    t_atm = max(t_atm, 160.1)
    t_air = min(23999,t_air)

    #  Input based on problem type 
    if problem_afterburning == "tp":
        line += "  " + problem_afterburning + "  t,k={} p,bar={} \n".format(t_material, p_air)
    elif problem_afterburning == "hp":
        line += "  " + problem_afterburning + "  p,bar={} \n".format(t_air, p_air)

    # Separate fuel and air species into dictionaries with their mass fractions
    material_species = {species: df_material_massfractions[species].values[0] for species in df_material_massfractions.columns}
    air_species = {species: df_air_massfractions[species].values[0] for species in df_air_massfractions.columns}

    # Check and limit total species count
    total_species = len(material_species) + len(air_species)

    if total_species > 24:
        # Drop *He and *Ar from air species if total species are between 24 and 26
        air_species = {k: v for k, v in air_species.items() if k not in ["*He", "*Ar"]}
        total_species = len(material_species) + len(air_species)

        # If still exceeding 24, drop the smallest fuel species by mass fraction
        if total_species > 24:
            material_species = dict(sorted(material_species.items(), key=lambda x: x[1], reverse=True)[:24 - len(air_species)])

    # Include Fuel and Air Species
    line += "react \n"
    for fuel, mass_fraction in material_species.items():
        if fuel == "AL(cr)" and t_material > 933.61:
            fuel = "AL(L)"
        if fuel == "Zn(cr)" and t_material > 692.73:
            fuel = "Zn(L)"
        elif fuel == "Li(cr)" and t_material > 453.69:
            fuel = "Li(L)"
        elif fuel == "Ag(cr)" and t_material > 1235.08:
            fuel = "Ag(L)"
        elif fuel == "Sn(cr)" and t_material > 505.12:
            fuel = "Sn(L)"
        elif fuel == "Mg(cr)" and t_material > 923.00:
            fuel = "Mg(L)"
        elif fuel == "Mn(a)" and t_material >= 980.00 and t_material < 1361.00:
            fuel = "Mn(b)"
        elif fuel == "Mn(a)" and t_material >= 1361.00 and t_material < 1412.00:
            fuel = "Mn(c)"
        elif fuel == "Mn(a)" and t_material >= 1412.00 and t_material < 1519.00:
            fuel = "Mn(d)"
        elif fuel == "Mn(a)" and t_material >= 1519.00:
            fuel = "Mn(L)"
        elif fuel == "Fe(a)" and t_material >= 1184.00 and t_material < 1665.00:
            fuel = "Fe(c)"
        elif fuel == "Fe(a)" and t_material >= 1665.00 and t_material < 1809.00:
            fuel = "Fe(d)"
        elif fuel == "Fe(a)" and t_material >= 1809.00:
            fuel = "Fe(L)"
        line += "  fuel={} wt={} t,k={} \n".format(fuel, mass_fraction, t_material)

    for air, mass_fraction in air_species.items():
        line += "  oxid={} wt={} t,k={} \n".format(air, mass_fraction, t_material)
            
    # Ending Lines
    line += "output \n"
    line += "  siunits massf trace={} \n".format(trace)
    line += "  plot p t mach H rho s \n"
    line += "end \n"

    inp_string = line
    with open(filename + ".inp", "w") as file:
            file.write(inp_string)

def create_inp_file_bal(filename, df=None, *args):
    t_ab, p_ab, rho_ab, s_ab, t_air,  df_air_massfractions = args

    line = "problem,\n"
    
    if problem_afterburning in ["tp", "hp"]:
        line += "  {}  t,k={} p,bar={} \n".format(problem_afterburning, t_ab, p_ab)

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
    filetypes_to_delete = [".inp", ".out", ".csv", ".plt"]
    for filetype in filetypes_to_delete:
        try:
            file_path = filename + filetype
            os.remove(file_path)
            #print(f"Deleted: {file_path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

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
    # Remove thermo.lib & trans.lib
    files_to_remove = ["thermo.lib", "trans.lib", ".inp", ".out", ".plt"]
    for file in files_to_remove:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass  

def shockwave_temperature(t_atm, velocity, c_atm):
    mach = velocity / c_atm
    deltat = ((1 + ((kappa - 1) / 2) * mach**2) * (2 * kappa * mach**2 - (kappa - 1))) / ((kappa + 1)**2 * mach**2)
    t_air = t_atm * deltat
    return t_air

def stagnation_point_pressure(p_atm, rho_atm, velocity):
    p_stag = p_atm + 0.5 * rho_atm * velocity**2
    return p_stag

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
def calculate_NO_emissions(df_trajaerodata, df_scenarios, scenario_name):
    # Extract nose radius from the scenario
    emission_results = df_trajaerodata.copy()
    nose_radius = df_scenarios.loc[scenario_name]['R_n']
    time_old = 0
    
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
        massflow = rho_atm * velocity * np.pi * ((nose_radius * rof_f)**2 - nose_radius**2) * (time - time_old)
        time_old = time

        # Determine temperature for NOx formation
        t_ab = max(row['T_rad_equil_comb'], t_atm)

        # Load GRI-Mech 3.0 mechanism
        gas = ct.Solution('gri30.yaml')

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
        atm_composition = {species: float(mole_frac) for species, mole_frac in atm_composition.items()}

        # Compute total moles as a scalar
        total_moles = sum(atm_composition.values())

        # Fix the conditional check
        if total_moles > 0:
            atm_composition = {species: mole_frac / total_moles for species, mole_frac in atm_composition.items()}

        # Set gas state and equilibrate at constant enthalpy & pressure
        gas.TPX = t_ab, p_atm, atm_composition
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
        filtered_rows = emission_results[
            (emission_results['scenario'] == scenario_name) & 
            (emission_results['d_m [kg]'] > 0)
        ]

        total_emissions = pd.DataFrame()

        df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
            lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
        )

        # Iterate through the filtered rows
        for idx, row in filtered_rows.iterrows():
            massflow = row['d_m [kg]']
            time = row['Time [s]']
            
            # Filter emission factors for the current material
            df_emission_factor_material = df_emission_factors[
                df_emission_factors['material'] == row['Material']
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
        pivot_emissions = total_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)

        # Merge the pivoted emissions with filtered_rows to get the final results
        emission_results = filtered_rows.merge(pivot_emissions, left_index=True, right_index=True, how='left')

        # Final cleanup and output
        emission_results = emission_results.drop(columns=['*Ar', '*He'], errors='ignore')
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

        if calculate_nox == True:
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
                        time_old = time - 1
                    massflow = rho_atm * velocity * ((a_ref * rof_f) - a_ref) * (time - time_old)
                    time_old = time
                    t_air = shockwave_temperature(t_atm, velocity, c_atm)
                    t_atm = max(160.1, t_atm)  # Ensure minimum air temperature
                    t_ab = min(23999, t_air) # Ensure maximum air temperature handled by CEA

                    if t_ab > 1000:
                        # Create and run NASA CEA input file
                        afterburning_filename = f"{scenario_name}_nox_{int(time)}"
                        create_inp_file_bal(afterburning_filename, None, t_ab, p_atm, rho_atm, None, t_atm, df_air_massfractions)
                        
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
                            save_files(afterburning_filename)
                            # Save results
                            #save_results_excel(afterburning_filename, df_output)

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
                            print(f"NASA CEA error: {e} at time {time}s and temperature {t_ab}. Setting NOx fractions to 0")

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
                        time_old = time - 1
                    massflow = rho_atm * velocity * ((a_ref * rof_f) - a_ref) * (time - time_old)
                    time_old = time
                    t_air = shockwave_temperature(t_atm, velocity, c_atm)
                    t_air = max(160.1, t_air)  # Ensure minimum air temperature
                    t_ab = min(10000, t_air) # Ensure maximum air temperature handled by Cantera
                    if t_ab > 1000:
                        # Load GRI-Mech 3.0 mechanism
                        gas = ct.Solution('gri30.yaml')

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
                        atm_composition = {species: float(mole_frac) for species, mole_frac in atm_composition.items()}

                        # Compute total moles as a scalar
                        total_moles = sum(atm_composition.values())

                        # Fix the conditional check
                        if total_moles > 0:
                            atm_composition = {species: mole_frac / total_moles for species, mole_frac in atm_composition.items()}

                        try:
                            # Set gas state and equilibrate at constant enthalpy & pressure
                            gas.TPX = t_ab, p_atm, atm_composition
                            gas.equilibrate('HP')

                            # Extract NOx mole and mass fractions
                            no_mole_fraction = gas['NO'].X[0]
                            no_mass_fraction = gas['NO'].Y[0]
                            no2_mole_fraction = gas['NO2'].X[0]
                            no2_mass_fraction = gas['NO2'].Y[0]
                        
                        except ct.CanteraError as e:
                            print(f"Cantera error: {e} at time {time}s and temperature {t_ab}. Setting NOx fractions to 0")

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
            # Atmospheric Data
            rho_atm, t_atm, p_atm, c_atm, df_air_massfractions, df_atm_molarfractions = calculate_atmosphere_data(row)
            p_atm = p_atm*10**(-5) #Pa into bar
            velocity = row['Velocity [km/s]']*1000
            a_ref = row['ReferenceArea [m^2]']
            massflow = row['d_m [kg]']
            time = row['Time [s]']
            
            # Calculate rof_ab
            rof_ab = rof_ab_calc(rof_f, a_ref, massflow, rho_atm, velocity)
            
            # Material Data
            df_material_massfractions = get_material_data(row['Material'], df_material_data)
                    
            # Remove species with mass fraction below threshold
            df_material_massfractions = df_material_massfractions.loc[
                :, df_material_massfractions.iloc[0] > threshold
            ]

            # Create and run NASA CEA input file
            t_mat = row['Temp [K]']
            t_air = shockwave_temperature(t_atm, velocity, c_atm) #Heating of air due to shockwave
            p_air = stagnation_point_pressure(p_atm, rho_atm, velocity)
            #t_air = t_mat # Assumption burning at material temperature
            #t_air = min(23999,max(t_atm, 160.1))  # Ensure minimum air temperature
            afterburning_filename = f"{scenario_name}_{row['ObjectName']}_{int(time)}"
            create_inp_file(afterburning_filename, None, t_atm, p_atm ,rho_atm, None, t_mat, t_air, p_air,rof_ab, df_material_massfractions, df_air_massfractions)
            input_file_path = str(afterburning_filename)
            output_file_path = str(afterburning_filename)

            try:
                # Execution of FCEA2.exe with the input from input_file_path
                process = subprocess.Popen(
                    [nasa_cea_folder_path + nasa_cea_exe],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,  # Suppresses standard output
                    stderr=subprocess.PIPE      # Catch error output
                )

                # Übergabe des Eingabewerts an die exe-Datei
                process.stdin.write(f"{input_file_path}\n".encode())
                process.stdin.close()
                process.wait()  # Wait until process is finished

                # Check error status messages
                if process.returncode != 0:
                    error_message = process.stderr.read().decode()
                    print("Error while executing NASA CEA:", error_message)
                else:
                    pass

                # Saving raw and formated data
                df_output = readCEA(output_file_path)
                save_files(afterburning_filename)
                #save_results_excel(afterburning_filename, df_output)
                
                df_output_filtered = df_output.iloc[6:]
                df_output_filtered.columns = ['ab_output_filtered']
                df_output_filtered.index = df_output_filtered.index.str.replace(r'^\*', '', regex=True)

                df_air_massfractions = df_air_massfractions.transpose()
                df_air_massfractions.columns = ['air_massfractions']
                df_air_massfractions.index = df_air_massfractions.index.str.replace(r'^\*', '', regex=True)
                
                combined_df = pd.concat([df_output_filtered, df_air_massfractions], axis=1, join='outer').fillna(0)

                # Initialize a DataFrame to store species and their EI_species for the current row
                combined_species = pd.DataFrame(columns=['species', 'EI_species'])
                            
                # Update the species mass fractions (emission factors) in df_engines for the current phase and engine pair
                for species, row2 in combined_df.iterrows():
                    x_species = row2['ab_output_filtered']
                    x_species_atm = row2['air_massfractions']
                    EI_species = (x_species * (rof_ab + 1)) - (x_species_atm * rof_ab)
                    if EI_species > threshold:
                        combined_species = pd.concat([combined_species, pd.DataFrame({'species': [species], 'EI_species': [EI_species]})], ignore_index=True)
                
                # Multiply EI_species by massflow and store in total_emissions
                combined_species['EI_species'] *= massflow
                combined_species['row_idx'] = idx  
                total_emissions = pd.concat([total_emissions, combined_species], ignore_index=True)

                # Clean up CEA files
                clean_directory(afterburning_filename)
            
            except Exception as e:
                print(f"NASA CEA error: {e} at time {time}s and temperature {t_air}. Calculating emissions with stoichiometric factors.")
            
                df_emission_factors['altitude_interval'] = df_emission_factors['altitude_interval'].apply(
                    lambda x: [float(i) for i in literal_eval(x)] if isinstance(x, str) else x
                )
                
                # Filter emission factors for the current material
                df_emission_factor_material = df_emission_factors[
                    df_emission_factors['material'] == row['Material']
                ]
                print(f"Using emission factors for material {row['Material']} at altitude {row['Altitude [km]']}")

                # Further filter by altitude interval
                altitude_interval_match = df_emission_factor_material[
                    df_emission_factor_material['altitude_interval'].apply(
                        lambda x: x[0] <= row['Altitude [km]'] <= x[1] if isinstance(x, list) else False
                    )
                ]
                
                # Skip if no matching altitude interval
                if altitude_interval_match.empty:
                    print(f"No matching emission factors for altitude {row['Altitude [km]']} and material {row['Material']}")
                
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
                continue

        # Pivot total_emissions to have species as columns and rows identified by row_idx
        total_emissions = total_emissions.pivot(index='row_idx', columns='species', values='EI_species').fillna(0)
        
        if calculate_nox == True:
            combined_emissions = pd.concat([total_emissions, total_nox_emissions], axis=1)
        else:
            combined_emissions = total_emissions

        # If some columns are duplicated (like 'NO'), combine them by summing
        combined_emissions = combined_emissions.groupby(combined_emissions.columns, axis=1).sum()

        # Merge the pivoted emissions with filtered_rows to get the final results
        emission_results = filtered_rows_all.merge(combined_emissions, left_index=True, right_index=True, how='left')

        # Drop rows where all values are 0
        emission_results = emission_results[(emission_results != 0).any(axis=1)]

        # Final cleanup and output
        emission_results = emission_results.drop(columns=['Ar', 'He'], errors='ignore')
        print("NASA CEA calculations completed successfully.")
        
        # 4. Clean up - Remove thermo.lib and trans.lib
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

        # =========================
        # 1) NOx CALCULATION (unchanged, material-agnostic)
        # =========================
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
                        time_old = time - 1
                    massflow = rho_atm * velocity * ((a_ref * rof_f) - a_ref) * (time - time_old)
                    time_old = time
                    t_air = shockwave_temperature(t_atm, velocity, c_atm)
                    t_atm = max(160.1, t_atm)
                    t_ab = min(23999, t_air)

                    if t_ab > 1000:
                        afterburning_filename = f"{scenario_name}_nox_{int(time)}"
                        create_inp_file_bal(afterburning_filename, None, t_ab, p_atm, rho_atm, None, t_atm, df_air_massfractions)
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
                        time_old = time - 1
                    massflow = rho_atm * velocity * ((a_ref * rof_f) - a_ref) * (time - time_old)
                    time_old = time
                    t_air = shockwave_temperature(t_atm, velocity, c_atm)
                    t_air = max(160.1, t_air)
                    t_ab = min(10000, t_air)
                    if t_ab > 1000:
                        gas = ct.Solution('gri30.yaml')

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
                            gas.TPX = t_ab, p_atm, atm_composition
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

        # =========================
        # 2) CEA AFTERBURNING PER MATERIAL (THIS IS THE NEW PART)
        # =========================
        # Prepare EF fallback table
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
                rof_ab = rof_ab_calc(rof_f, a_ref, massflow, rho_atm, velocity)

                # Material Data (for THIS material)
                df_material_massfractions = get_material_data(mat_name, df_material_data)

                # Apply threshold to material species
                df_material_massfractions = df_material_massfractions.loc[
                    :, df_material_massfractions.iloc[0] > threshold
                ]

                # Build afterburning case (per material)
                t_mat = row['Temp [K]']
                t_air = shockwave_temperature(t_atm, velocity, c_atm)   # Heating of air due to shockwave
                p_air = stagnation_point_pressure(p_atm, rho_atm, velocity)

                obj_name = row.get('ObjectName', row.get('FragmentID', 'Object'))
                safe_mat = str(mat_name).replace(' ', '_')
                afterburning_filename = f"{scenario_name}_{obj_name}_{safe_mat}_{int(time)}"

                create_inp_file(
                    afterburning_filename,
                    None,
                    t_atm,
                    p_atm,
                    rho_atm,
                    None,
                    t_mat,
                    t_air,
                    p_air,
                    rof_ab,
                    df_material_massfractions,
                    df_air_massfractions
                )
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
                    print(f"NASA CEA error: {e} at time {time}s and material {mat_name}. Using stoichiometric emission factors.")
                    if 'altitude_interval' in df_emission_factors.columns:
                        df_emission_factor_material = df_emission_factors[df_emission_factors['material'] == mat_name]
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
        emission_results = emission_results.drop(columns=['Ar', 'He'], errors='ignore')
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
        time_old = 0

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
            massflow = rho_atm * velocity * np.pi * ((nose_radius)**2 * rof_f - nose_radius**2)* (time - time_old)
            time_old = time

            #t_air = shockwave_temperature(t_atm, velocity, c_atm)
            t_ab = max(row['T_rad_equil_comb'], t_atm)
            if t_ab > 24000:
                t_ab = 23999

            # Create and run NASA CEA input file
            afterburning_filename = f"{scenario_name}_{int(time)}"
            create_inp_file_bal(afterburning_filename, None, t_ab, p_atm, rho_atm, None, t_atm, df_air_massfractions)
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
            save_files(afterburning_filename)
            # Save results
            #save_results_excel(afterburning_filename, df_output)

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
