### This file includes all functions to process the data

### Imports
from input_data.config import *
import os
import re
import pandas as pd
import sys
import numpy as np
from pymsis import msis
from io import StringIO
from datetime import datetime, timedelta
from tqdm import tqdm

#region: Input Data Generation
# Import all required data based on user input and config
def get_input_data():
    # 1. Import and Clean Data 
    try:
        # 1) Import and filter scenarios
        # 1.1) Import Scenarios
        s_path = os.path.join(input_data_folder_path, file_name_scenarios)
        df_scenarios = pd.read_excel(s_path, index_col=0, sheet_name=sheet_name_scenarios)
        df_scenarios.drop('note', axis=1, inplace=True)
        missing_scenarios = [s for s in user_defined_scenarios if s not in df_scenarios.index]

        if missing_scenarios:
            print(f"Warning: The following scenarios are not in the scenarios.xlsx file and will be ignored: {missing_scenarios}")

        # 1.2) Filter Scenarios
        if all_scenarios:
            pass
        else:
            df_scenarios = df_scenarios.loc[df_scenarios.index.isin(user_defined_scenarios)]

        #2) Import Emission Factors
        df_emission_factors = pd.DataFrame()
        if use_emission_factors or use_nasa_cea:
            ef_path = os.path.join(input_data_folder_path, file_name_emission_factors)
            df_emission_factors = pd.read_excel(ef_path, index_col=None, sheet_name=sheet_name_emission_factors)
            df_emission_factors = df_emission_factors.fillna(0)

            if 'notes' in df_emission_factors.columns:
                # Select columns up to (but not including) "notes"
                df_emission_factors = df_emission_factors.loc[:, : "notes"].iloc[:, :-1]

        # 3) Import Material Data
        df_material_data = pd.DataFrame()
        if use_nasa_cea:
            md_path = os.path.join(input_data_folder_path, file_name_material_data)
            df_material_data = pd.read_excel(md_path, index_col=None, sheet_name=sheet_name_material_data)
            df_material_data = df_material_data.fillna(0)

            if 'notes' in df_material_data.columns:
                # Select columns up to (but not including) "notes"
                df_material_data = df_material_data.loc[:, : "notes"].iloc[:, :-1]

        return df_emission_factors, df_scenarios, df_material_data

    except Exception as e:
        handle_error(e, "get_input_data", "Input data could not be processed")     

# Import Drama Files
def drama_input(df_scenarios,scenario_name, pydrama_results_folder=None):
    reentry_vehicle = df_scenarios.loc[scenario_name, 'reentry_vehicle']
    folder = 'SARA/REENTRY/output'
    scenario_folder = os.path.join(input_data_folder_path, reentry_vehicle, folder)

    if pydrama_results_folder:
        scenario_folder = os.path.join(pydrama_results_folder, "reentry")
        print(colors.BOLD + colors.GREEN + "Using PyDrama results folder: ", scenario_folder + colors.END)
    
    if not os.path.exists(scenario_folder):
        print(colors.BOLD + colors.RED + f"Warning: Scenario folder '{scenario_folder}' not found." + colors.END)
    
    # Find all relevant files in the scenario folder
    aerothermal_files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if 'AeroThermalHistory' in f]
    #print('aerothermal_files',aerothermal_files)
    trajectory_files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if 'Trajectory' in f]
    #print('trajectory_files',trajectory_files)
    reentry_time = df_scenarios.loc[scenario_name, 'date']
    result_list = []

    # Process all file pairs and append results
    for aero_file in aerothermal_files:
        try:
            # Extract metadata and data from the aerothermal file
            aero_meta, _ = extract_metadata_and_data(aero_file)
            object_id = aero_meta.get('ObjectID')
            
            # Extract base name for aerothermal file
            aero_filename = os.path.basename(aero_file)
            aero_base_match = re.match(r"(.+)_AeroThermalHistory", aero_filename)
            aero_base_name = aero_base_match.group(1) if aero_base_match else aero_filename
            
            for traj_file in trajectory_files:
                try:
                    # Extract metadata and data from the trajectory file
                    traj_meta, _ = extract_metadata_and_data(traj_file)
                    
                    # Extract base name for trajectory file
                    traj_filename = os.path.basename(traj_file)
                    traj_base_match = re.match(r"(.+)_Trajectory", traj_filename)
                    traj_base_name = traj_base_match.group(1) if traj_base_match else traj_filename  

                    # Ensure base name and ObjectID match
                    if traj_base_name == aero_base_name and traj_meta.get('ObjectID') == object_id:
                        df_processed = process_files(aero_file, traj_file, reentry_time, pydrama_results_folder)
                        df_processed['scenario'] = scenario_name  # Add scenario column for traceability
                        df_processed['reentry_vehicle'] = reentry_vehicle
                        result_list.append(df_processed)
                        print('Processed Data Frame of Object:',traj_base_name)
                    else:
                        pass
                except Exception as e:
                    print(f"Error processing trajectory file {traj_file} with aerothermal file {aero_file}: {e}")
        except Exception as e:
            print(f"Error processing aerothermal file {aero_file}: {e}")

    # 3) Combine All Results into a Single DataFrame
    if result_list:
        df_trajaerodata = pd.concat(result_list, ignore_index=True)
        df_trajaerodata.to_csv(f"output_data/trajectory/TRAJ_RAW_{scenario_name}.csv", index=False)
        return df_trajaerodata
    else:
        df_trajaerodata = pd.DataFrame()
        print("No valid results were generated.")

# Import Scarab Files
def scarab_input(df_scenarios, scenario_name):
    # locate simulation data according to config.py and scenarios.xlsx (file_name_scenarios)
    reentry_vehicle = df_scenarios.loc[scenario_name, 'reentry_vehicle']
    scenario_folder_path = os.path.join(input_data_folder_path, scenario_name)

    if not os.path.exists(scenario_folder_path):
        print(colors.BOLD + colors.RED + f"Warning: Scenario folder '{scenario_folder_path}' not found." + colors.END)
              
    # import list of fragments, clean data
    df_frag_list = pd.read_csv(os.path.join(scenario_folder_path, 'frag.lst'), sep='\s+', skiprows=1, 
                               names=['frag_id', 'frag_description', 'frag_description2', 'parent', 'status', 'child1', 'minus', 'child2'],
                               dtype={'frag_id': str})      # fragment ID as string to correctly capture fragments with e.g. part x.10 or x.20
    df_frag_list.loc[df_frag_list['child1'] == 'Impact', 'child1'] = np.nan
    df_frag_list.loc[df_frag_list['status'] == 'Ground', 'status'] = "Ground Impact"
    df_frag_list['frag_description'] = df_frag_list['frag_description'] + ' ' + df_frag_list['frag_description2']
    df_frag_list.drop(columns=['frag_description2'], inplace=True)
    df_frag_list['children'] = df_frag_list['child1'].astype(str) + df_frag_list['minus'].astype(str) + df_frag_list['child2'].astype(str)
    df_frag_list.drop(columns=['child1', 'minus', 'child2'], inplace=True)
    df_frag_list.loc[df_frag_list['children'] == '-nannan', 'children'] = np.nan
    df_frag_list.loc[df_frag_list['children'] == 'nan-nan', 'children'] = np.nan
    
    # import list of materials
    # df_materials = pd.read_csv(os.path.join(scenario_folder_path, '0.1', 'material.lst'), sep='\s+', names=['material_id', 'material_name'], 
    #                            skiprows=lambda x: x < 2 or (x - 2) % 17 != 0)

    # initialize dataframe containing data of all fragments
    df_fragments = pd.DataFrame()
    
    # import data for each fragment
    # iterate through all folders in the folder of the simulation
    for folder in tqdm(os.listdir(scenario_folder_path), desc="Processing", unit="folder"):         # tqdm to print processing progress in console
        
        # filter only the fragment folders
        if os.path.isdir(os.path.join(scenario_folder_path, folder)):
            if folder in ['gnuplot', 'latex', 'maps', 'pics']:
                continue
            
            # define file paths
            fragment_path = os.path.join(scenario_folder_path, folder)
            masmat_path = os.path.join(fragment_path, 'masmat.hst')
            mas_path = os.path.join(fragment_path, 'mas.hst')
            trj_path = os.path.join(fragment_path, 'trj.hst')
            tqm_path = os.path.join(fragment_path, 'tqm.hst')
            arshad_path = os.path.join(fragment_path, 'arshad.hst')
            
            # import and wrangle masmat data (in masmat.hst)          
            if os.path.isfile(masmat_path):
                with open(masmat_path, 'r') as f:
                    lines = f.readlines()
                    lines = lines[2:]
                    lines[0] = lines[0].split('#')[-1]      
                    if 'cfrp-m56-ud1copper' in lines[0]:    
                        lines[0] = lines[0].replace("cfrp-m56-ud1copper", "cfrp-m56-ud1 copper")
                    if 'cc-sic-cerama316' in lines[0]:
                        lines[0] = lines[0].replace("cc-sic-cerama316", "cc-sic-ceram a316")                        
                    df_masmat = pd.read_csv(StringIO(''.join(lines)), sep="\s+", header=0)
                df_masmat.rename(columns={'TIME_[S]': 'TIME[S]'}, inplace=True)
            else:
                print('No masmat.hst file found in ', fragment_path)
            
            # import and wrangle mas data (in mas.hst)
            if os.path.isfile(mas_path):
                df_mas = pd.read_csv(mas_path, skiprows=18, sep="\s+",
                                     names=['TIME', 'ALT', 'TRK', 'MASS', 'COMX', 'COMY', 'COMZ', 'IXX', 'IYY', 'IZZ', 'IXY', 'IXZ', 'IYZ'])
                df_mas.rename(columns={'TIME': 'TIME[S]'}, inplace=True)
                df_mas.rename(columns={'ALT': 'ALT[KM]'}, inplace=True)
                df_mas.rename(columns={'MASS': 'MASS[KG]'}, inplace=True)
            else:
                print('No mas.hst file found in ', fragment_path)
                
            # import and wrangle trajectory data (in trj.hst)
            if os.path.isfile(trj_path):
                df_trj = pd.read_csv(trj_path, skiprows=28, sep="\s+",
                                     names=['TIME[S]', 'H[M]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'GM[DEG]', 'PS[DEG]', 'TRK[KM]'])
                df_trj['H[M]'] = df_trj['H[M]']/1000
                df_trj.rename(columns={'H[M]': 'ALT[KM]'}, inplace=True)
            else:
                print('No trj.hst file found in ', fragment_path)
            
            # import and wrangle temperature and heat flux data (in tqm.hst)
            if os.path.isfile(tqm_path):
                df_tqm = pd.read_csv(tqm_path, skiprows=3, sep="\s+", 
                                     names=['TIME[S]', 'ALT[KM]', 'DST[KM]', 'T_MAX[K]', 'Q_MAX[W/M2]', 'Q_AVG[W/M2]', 'Q_SUM[W]', 'Q_TOT[J]',
                                            'unknown1', 'unknown2', 'unknown3', 'unknown4'])
                df_tqm = df_tqm[['TIME[S]', 'ALT[KM]', 'DST[KM]', 'T_MAX[K]', 'Q_MAX[W/M2]', 'Q_AVG[W/M2]', 'Q_SUM[W]', 'Q_TOT[J]']]
                df_tqm.rename(columns={'DST[KM]': 'TRK[KM]'}, inplace=True)
            else:
                print('No tqm.hst file found in ', fragment_path)
                
            # import and wrangle data containing information about the projected surface area in the direction of flight (in arshad.hst)
            if os.path.isfile(arshad_path):
                df_area = pd.read_csv(arshad_path, sep="\s+",
                                     names=['TIME[S]', 'UNKNOWN', 'PROJ_AREA[M^2]'])
            else:
                print('No arshad.hst file found in ', fragment_path)
            
            
            # merge all the data for a fragment
            # df_mastqm = pd.merge(df_mas[['TIME[S]', 'MASS[KG]']],
            #                      df_tqm[['TIME[S]' ,'T_MAX[K]', 'Q_MAX[W/M2]']],
            #                      on='TIME[S]')
            df_mastqm = pd.merge(df_mas[['TIME[S]', 'MASS[KG]']],
                                 df_tqm[['TIME[S]' ,'T_MAX[K]']],
                                 on='TIME[S]')
            df_mastqmtrj = pd.merge(df_mastqm[['TIME[S]', 'MASS[KG]', 
                                               'T_MAX[K]']], # 'Q_MAX[W/M2]'
                                    df_trj[['TIME[S]' ,'ALT[KM]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'TRK[KM]']],
                                    on='TIME[S]')
            df_mastqmtrjarea = pd.merge(df_mastqmtrj[['TIME[S]', 'MASS[KG]',
                                                      'T_MAX[K]', # 'Q_MAX[W/M2]'
                                                      'ALT[KM]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'TRK[KM]']],
                                    df_area[['TIME[S]', 'PROJ_AREA[M^2]']],
                                    on='TIME[S]')
            df_frag = pd.merge(df_mastqmtrjarea[['TIME[S]', 'MASS[KG]',
                                                 'T_MAX[K]', # 'Q_MAX[W/M2]'
                                                 'ALT[KM]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'TRK[KM]', 
                                                 'PROJ_AREA[M^2]']], 
                               df_masmat, 
                               on='TIME[S]')
            df_frag['frag_id'] = str(folder)
            
            # compute change in mass (dm: mass of the current time step - mass of the former time step)
            for row in range(1, len(df_frag)):
                df_frag.loc[row, 'DM[KG]'] = abs(df_frag.loc[row, 'MASS[KG]'] - df_frag.loc[row-1, 'MASS[KG]'])
            
            # append single fragment's data to the dataframe containing all fragments
            df_fragments = pd.concat([df_fragments, df_frag], ignore_index=True, join='outer')
            

        # print processing progress in console
        pass
    
    
    # check whether all fragments were evaluated and are represented in df_fragments
    if set(df_frag_list['frag_id']).issubset(set(df_fragments['frag_id'])):
        print('----------------------------\n')
        print('All fragments were evaluated')
        print('----------------------------\n')
    else:
        print('-------------------------------------------\n')
        print('ATTENTION: Not all fragments were evaluated')
        print('-------------------------------------------\n')

    # add fragment's status information (demise, fragmentation, ground impact)
    # df_fragments = pd.merge(df_fragments, df_frag_list[['frag_id', 'status']], on='frag_id')
    
    # add datetime
    reentry_time = df_scenarios.loc[scenario_name, 'date']
    reentry_time = datetime.fromisoformat(reentry_time)
    df_fragments['DATETIME'] = df_fragments['TIME[S]'].apply(lambda x: reentry_time + timedelta(seconds=x))
    
    # rename columns of materials 
    df_fragments.rename(columns={'cfrp-m56-ud1': 'cfrp-m56-ud134',
                                 'cc-sic-ceram': 'cc-sic-ceramic'}, inplace=True)
    
    # rename columns, similar to Drama
    df_fragments['V[M/S]'] = df_fragments['V[M/S]']/1000.0
    df_fragments.rename(columns={'V[M/S]'           : 'V[KM/S]'}, inplace=True)
    df_fragments.rename(columns={'TIME[S]'          : 'Time [s]',
                                 'MASS[KG]'         : 'Total Fragment Mass [kg]',
                                 'T_MAX[K]'         : 'Temp [K]',
                                 # 'Q_MAX[W/M2]'      : 'Heatflux [W/m^2]',
                                 'ALT[KM]'          : 'Altitude [km]',
                                 'LT[DEG]'          : 'Lat [deg]',
                                 'LG[DEG]'          : 'Lon [deg]',
                                 'V[KM/S]'          : 'Velocity [km/s]',
                                 'TRK[KM]'          : 'Downrange [km]',
                                 'PROJ_AREA[M^2]'   : 'ReferenceArea [m^2]',
                                 'frag_id'          : 'FragmentID',
                                 'DM[KG]'           : 'd_m [kg]',
                                 'status'           : 'Status',
                                 'DATETIME'         : 'Datetime'}, inplace=True)
    
    # rename material columns to match the material names in material_data.xlsx
    df_fragments.rename(columns={'aa7075'           : 'AA7075 [kg]',
                                 'cfrp-m56-ud134'   : 'CFRP [kg]',
                                 'aa2195'           : 'AA2195 [kg]',
                                 'copper'           : 'Cu [kg]',
                                 'cc-sic-ceramic'   : 'CCSic [kg]',
                                 'a316'             : 'A316 [kg]',
                                 'inconel718'       : 'Inconel718 [kg]',
                                 }, inplace=True)
    
    # compute change in material mass for every material (dm: material mass of the current time step - material mass of the former time step)
    df_fragments = df_fragments.sort_values(by=['FragmentID', 'Time [s]'])
    for col in df_fragments.columns:
        if col in ['AA7075 [kg]', 'CFRP [kg]', 'AA2195 [kg]', 'Cu [kg]', 'CCSic [kg]', 'A316 [kg]', 'Inconel718 [kg]']:
            df_fragments[f'd_m {col}'] = df_fragments.groupby(['FragmentID'])[col].diff().abs()
    
    # reorder dataframe
    df_fragments = df_fragments[[
        "Time [s]", "Altitude [km]", "Lat [deg]", "Lon [deg]", "Velocity [km/s]", "Downrange [km]", "Total Fragment Mass [kg]",
        "Temp [K]", "Datetime", "d_m [kg]", "FragmentID", "ReferenceArea [m^2]", 
        "AA7075 [kg]", "CFRP [kg]", "AA2195 [kg]", "Cu [kg]", "CCSic [kg]", "A316 [kg]", "Inconel718 [kg]", 
        "d_m AA7075 [kg]", "d_m CFRP [kg]", "d_m AA2195 [kg]", "d_m Cu [kg]", "d_m CCSic [kg]", "d_m A316 [kg]", "d_m Inconel718 [kg]"
        ]]
    
    # add scenario and reentry_vehicle
    df_fragments["scenario"] = scenario_name
    df_fragments["reentry_vehicle"] = reentry_vehicle
    
    if df_fragments.empty:
        print("No valid results were generated.")
    else:
        df_fragments.to_csv(f"output_data/trajectory/TRAJ_RAW_{scenario_name}.csv", index=False)
        return df_fragments
    
# Import Debrisk Files
def debrisk_input(df_scenarios, scenario_name):
    # locate simulation data according to config.py and scenarios.xlsx (file_name_scenarios)
    reentry_vehicle = df_scenarios.loc[scenario_name, 'reentry_vehicle']
    scenario_folder_path = os.path.join(input_data_folder_path, scenario_name)

    if not os.path.exists(scenario_folder_path):
        print(colors.BOLD + colors.RED + f"Warning: Scenario folder '{scenario_folder_path}' not found." + colors.END)
        
    with open(os.path.join(scenario_folder_path, f'{scenario_name}.dat'), "r") as f:
        lines = f.readlines()

    # extract column names
    variable_line = next(line for line in lines if line.startswith("VARIABLES"))
    column_names = variable_line.replace("VARIABLES =", "").strip().split()

    # additional columns for the zone and material
    column_names.append("zone")
    column_names.append("material")

    # initializing
    data = []
    current_zone = None
    current_material = None 

    # go through all the lines of the file
    for line in lines:
        if line.startswith("ZONE"):
            # extract zone and material names
            current_zone = line.split("T='")[1].split("'")[0].split("_-_")[0]
            current_material = line.split("T='")[1].split("'")[0].split("_-_")[1]
        elif line.startswith("#") or line.startswith("VARIABLES") or line.strip() == "":
            continue
        else:
            # extract data/values
            values = line.strip().split()
            values.append(current_zone)
            values.append(current_material)
            data.append(values)

    # create dataframe
    df = pd.DataFrame(data, columns=column_names)

    # convert columns to float (if possible)
    for col in df.columns[:-2]:  # last two columns: 'zone' and 'material'
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # add datetime
    reentry_time = df_scenarios.loc[scenario_name, 'date']
    reentry_time = datetime.fromisoformat(reentry_time)
    df['datetime'] = df['time(s)'].apply(lambda x: reentry_time + timedelta(seconds=x))    
        
    # initialize d_m
    df['d_m(kg)'] = np.nan

    # compute change in mass for every zone
    for zone_name, group in df.groupby('zone'):
        group_sorted = group.sort_values('time(s)')
        mass_diff = group_sorted['aerodynamicMass(kg)'].diff().abs()
        df.loc[group_sorted.index, 'd_m(kg)'] = mass_diff      

    # rename columns, similar to Drama
    df['altitude(m)'] = df['altitude(m)']/1000.0
    df['velocity(m/s)'] = df['velocity(m/s)']/1000.0
    df.rename(columns={'altitude(m)'        : 'altitude(km)'}, inplace=True)
    df.rename(columns={'velocity(m/s)'      : 'velocity(km/s)'}, inplace=True)
    df.rename(columns={'time(s)'                : 'Time [s]',
                       'altitude(km)'           : 'Altitude [km]',
                       'longitude(deg)'         : 'Lon [deg]',
                       'latitude(deg)'          : 'Lat [deg]',
                       'velocity(km/s)'         : 'Velocity [km/s]',                   
                       'downRange(km)'          : 'Downrange [km]',                 
                       'aerodynamicMass(kg)'    : 'Mass [kg]',
                       'referenceAeroArea(m2)'  : 'ReferenceArea [m^2]',
                       'wallTemperature(K)'     : 'Temp [K]',
                       'zone'                   : 'ObjectID',
                       'd_m(kg)'                : 'd_m [kg]',
                       'material'               : 'Material',
                       'datetime'               : 'Datetime'}, inplace=True)

    # add scenario and reentry_vehicle
    df['scenario'] = scenario_name
    df['reentry_vehicle'] = reentry_vehicle
    df['ObjectName'] = df['ObjectID']
    df['Part'] = df['ObjectID']


    df_trajaerodata = df[[
        'Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 'Velocity [km/s]', 'Downrange [km]', 'Mass [kg]', 'Temp [K]',
        'Datetime', 'd_m [kg]', 'ObjectID', 'ObjectName', 'Part', 'Material', 'ReferenceArea [m^2]', 'scenario', 'reentry_vehicle'
        ]]
              
    if df_trajaerodata.empty:
        print("No valid results were generated.")
    else:
        df_trajaerodata.to_csv(f"output_data/trajectory/TRAJ_RAW_{scenario_name}.csv", index=False)
        return df_trajaerodata

# Load External Trajectory Data
def load_external_trajectory_data():
    # Load external trajectory data from CSV file specified in filename_ext_traj
    df_trajaerodata = pd.read_csv(file_name_ext_trajectory)
    print(colors.BOLD + colors.GREEN + f"Successfully loaded trajectory results from external file: '{file_name_ext_trajectory}'" + colors.END)
    return df_trajaerodata
#endregion

def compress_and_save_trajectory_results(output_data, scenario_name):
    try:
        # Initialize compressed dataframe with the first row
        compressed_df = pd.DataFrame(output_data.iloc[[0]])
        
        if compress_method == "time":
            current_time = compress_interval
            
            if compress_atmosphere == "latest":
                for idx in range(1, len(output_data)):
                    if output_data['Time [s]'].iloc[idx] >= current_time - interval_tolerance:
                        row = output_data.iloc[[idx]].copy()
                        compressed_df = pd.concat([compressed_df, row], ignore_index=True)
                        current_time += compress_interval

            elif compress_atmosphere == "averages":
                interval_data = []
                for idx in range(1, len(output_data)):
                    interval_data.append(output_data.iloc[idx])
                    
                    if output_data['Time [s]'].iloc[idx] >= current_time - interval_tolerance:
                        interval_df = pd.DataFrame(interval_data)
                        interval_avg = interval_df[['p_tot (Pa)', 'rho', 'T_rad_equil_comb']].mean()
                        averaged_row = output_data.iloc[[idx]].copy()
                        averaged_row[['p_tot (Pa)', 'rho', 'T_rad_equil_comb']] = interval_avg
                        compressed_df = pd.concat([compressed_df, averaged_row], ignore_index=True)
                        current_time += compress_interval
                        interval_data = []

        elif compress_method == "height":
            last_height = output_data['Altitude [m]'].iloc[-1]
            height = output_data['Altitude [m]'].iloc[0]  # Start from the highest altitude
            
            if compress_atmosphere == "latest":
                while height >= last_height:
                    height_data = output_data.query(f'`Altitude [m]` <= {height} and `Altitude [m]` > {height - compress_interval*1000}')
                    
                    if not height_data.empty:
                        row = height_data.iloc[[-1]].copy()
                        compressed_df = pd.concat([compressed_df, row], ignore_index=True)
                    
                    height -= compress_interval*1000
                
            elif compress_atmosphere == "averages":
                interval_data = []
                while height >= last_height:
                    height_data = output_data.query(f'`Altitude [m]` <= {height} and `Altitude [m]` > {height - compress_interval*1000}')
                    interval_data.append(height_data)
                    
                    if not height_data.empty:
                        interval_df = pd.concat(interval_data, ignore_index=True)
                        interval_avg = interval_df[['p_tot (Pa)', 'rho', 'T_rad_equil_comb']].mean()
                        averaged_row = height_data.iloc[[-1]].copy()
                        averaged_row[['p_tot (Pa)', 'rho', 'T_rad_equil_comb']] = interval_avg
                        compressed_df = pd.concat([compressed_df, averaged_row], ignore_index=True)
                    
                    height -= compress_interval*1000
                    interval_data = []
    
    except Exception as e:
        handle_error(e, "compress_and_save_trajectory_results", "Error compressing the data.")

    
    # Saving Compressed Data to csv file
    try:
        base_name = f"{output_data_trajectory_name}{output_data_compression_name}{scenario_name}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        
        compressed_df.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written compressed trajectory results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + "Compressed Trajectory Data:" + colors.END)
        print(compressed_df.head())
        
    except Exception as e:
        handle_error(e, "compress_and_save_trajectory_results", "Error finding required data.")    
    
    return compressed_df

#region: General Data Handling Functions
# Initial Console Prints
def print_scenario_data(idx, scenario_name, df_scenarios):
    try:
        print(colors.BOLD + colors.BLUE + "\nScenario #{} Data:".format(idx), scenario_name + colors.END)
        print(df_scenarios.loc[scenario_name].to_string() + "\n")
    except Exception as e:
        handle_error(e, "print_scenario_data", "Error while locating scenario data.")

# File naming handling
def get_unique_filename(base_name, folder, type):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    counter = 1
    filename = f"{base_name}{type}"
    file_path = os.path.join(folder, filename)

    while os.path.exists(file_path):
        filename = f"{base_name}({counter}){type}"
        file_path = os.path.join(folder, filename)
        counter += 1
    
    return filename, file_path

# General Error Handling Function
def handle_error(error, function_name, function_message):
    error_messages = {
        FileNotFoundError: "One or more input files not found. Please check the file paths.",
        pd.errors.EmptyDataError: "One or more input files are empty. Please ensure they contain data.",
        pd.errors.ParserError: "Error parsing Excel file. Please ensure the file format is correct.",
        ValueError: "Please check your input files and configurations.",
        Exception: "An unexpected error occurred. Please check your input files and configurations."
    }

    if type(error) in error_messages:
        print(colors.BOLD + colors.RED + f"Error in function '{function_name}': {error}")
        print(colors.BOLD + colors.RED + f"{function_message}: " + error_messages[type(error)] + colors.END)
    else:
        print(colors.BOLD + colors.RED + f"An unexpected error occurred in function '{function_name}': {error}")
        print(colors.BOLD + colors.RED + f"{function_message}: Please check your input files and configurations." + colors.END)

    # Stop Script Execution
    #sys.exit(1)
    # Proceed anyway
    return
    
#endregion

#region: Data Processing for Trajectory
# Extract Metadata from input files
def extract_metadata_and_data(filepath):
    metadata = {}
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        # Extract metadata
        for line in lines[:20]:  # Assuming metadata is in the first 20 lines
            if 'ObjectID' in line:
                metadata['ObjectID'] = line.split(':')[-1].strip()
            elif 'ObjectName' in line:
                metadata['ObjectName'] = line.split(':')[-1].strip()
            elif 'ReferenceArea' in line:
                metadata['ReferenceArea'] = float(re.search(r"[\d\.]+", line.split(':')[-1]).group())
            elif 'trajectory history file' in line:
                metadata['FileType'] = 'trajectory'
            elif 'aerothermal history file' in line:
                metadata['FileType'] = 'aerothermal'
        
        # Extract data starting after header
        data_start = None
        for idx, line in enumerate(lines):
            if re.match(r"^\s*[-]{3,}", line):  # Header separator
                data_start = idx + 1
                break
        
        # Parse data into a list, skipping lines starting with '#'
        for line in lines[data_start:]:
            if line.strip() and not line.startswith('#'):
                data.append([float(x) for x in line.split()])
                
    return metadata, data

#Compress trajectory to easen calculation:
def compress_trajaerodata_dataframe(df_trajaerodata, compress_method, compress_interval, compress_lat, compress_lon, scenario_name):

    df_copy = df_trajaerodata.copy()
    df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'])

    print('df_copy:\n', df_copy)
    result_list = []

    for object_id, group_df in df_copy.groupby('ObjectID'):
        group_df = group_df.copy()

        # Set bin_field and compute it
        if compress_method == "time":
            group_df['TimeBin'] = (
                ((group_df['Time [s]'] - group_df['Time [s]'].min()) // compress_interval) * compress_interval
                + group_df['Time [s]'].min()
            )
            bin_field = 'TimeBin'
        elif compress_method == "height":
            group_df['HeightBin'] = (
                group_df['Altitude [km]'].max() -
                (((group_df['Altitude [km]'].max() - group_df['Altitude [km]']) // compress_interval) * compress_interval)
            )
            bin_field = 'HeightBin'
        else:
            raise ValueError("Invalid compress_method. Use 'time' or 'height'.")

        # Apply spatial binning
        group_df['LatBin'] = (group_df['Lat [deg]'] // compress_lat) * compress_lat
        group_df['LonBin'] = (group_df['Lon [deg]'] // compress_lon) * compress_lon

        # Sort to ensure aggregation uses the latest values
        group_df = group_df.sort_values('Datetime')

        # Define fields for grouping
        grouping_fields = ['ObjectID', bin_field, 'LatBin', 'LonBin']

        # Define aggregation rules
        agg_dict = {
            'Time [s]': 'last',
            'Altitude [km]': 'last',
            'Lat [deg]': 'last',
            'Lon [deg]': 'last',
            'Velocity [km/s]': 'last',
            'Downrange [km]': 'last',
            'Mass [kg]': 'last',
            'Temp [K]': 'last',
            'Datetime': 'last',
            'ObjectName': 'last',
            'Part': 'last',
            'Material': 'last',
            'ReferenceArea [m^2]': 'last',
            'scenario': 'last',
            'reentry_vehicle': 'last',
            'd_m [kg]': 'sum',
        }

        # Perform grouping and aggregation
        compressed_group = (
            group_df.groupby(grouping_fields, as_index=False)
                    .agg(agg_dict)
        )

        result_list.append(compressed_group)

    # Combine all object results
    compressed = pd.concat(result_list, ignore_index=True)

    # Restore original column order
    original_cols = df_trajaerodata.columns.tolist()

    print("Original columns expected:", original_cols)
    print("Compressed columns available:", compressed.columns.tolist())

    # Fill missing columns with NaN
    missing_cols = [col for col in original_cols if col not in compressed.columns]
    for col in missing_cols:
        print(f"Warning: Column '{col}' missing in compressed output. Filling with NaN.")
        compressed[col] = pd.NA

    # Reorder columns
    compressed = compressed[original_cols]

    # Sort by original values
    compressed = compressed.sort_values(
        by=['ObjectID', 'Time [s]', 'Lat [deg]', 'Lon [deg]'],
        ascending=[True, True, True, True]
    )

    # Save and return
    output_path = f"output_data/trajectory/TRAJ_COMP_{scenario_name}.csv"
    compressed.to_csv(output_path, index=False)
    print(f"Compressed trajectory saved to: {output_path}")

    return compressed

#Compress trajectory from Scarab to easen calculation:
def compress_trajaerodata_scarab_dataframe(df_trajaerodata, compress_method, compress_interval, compress_lat, compress_lon, scenario_name):

    df_copy = df_trajaerodata.copy()
    
    print('df_copy:\n', df_copy)
    result_list = []

    for fragment_id, group_df in df_copy.groupby('FragmentID'):
        group_df = group_df.copy()

        # Set bin_field and compute it
        if compress_method == "time":
            group_df['TimeBin'] = (
                ((group_df['Time [s]'] - group_df['Time [s]'].min()) // compress_interval) * compress_interval
                + group_df['Time [s]'].min()
            )
            bin_field = 'TimeBin'
        elif compress_method == "height":
            group_df['HeightBin'] = (
                group_df['Altitude [km]'].max() -
                (((group_df['Altitude [km]'].max() - group_df['Altitude [km]']) // compress_interval) * compress_interval)
            )
            bin_field = 'HeightBin'
        else:
            raise ValueError("Invalid compress_method. Use 'time' or 'height'.")

        # Apply spatial binning
        group_df['LatBin'] = (group_df['Lat [deg]'] // compress_lat) * compress_lat
        group_df['LonBin'] = (group_df['Lon [deg]'] // compress_lon) * compress_lon

        # Sort to ensure aggregation uses the latest values
        group_df = group_df.sort_values('Datetime')

        # Define fields for grouping
        grouping_fields = ['FragmentID', bin_field, 'LatBin', 'LonBin']

        # Define aggregation rules
        agg_dict = {
            'Time [s]': 'last',
            'Altitude [km]': 'last',
            'Lat [deg]': 'last',
            'Lon [deg]': 'last',
            'Velocity [km/s]': 'last',
            'Downrange [km]': 'last',
            'Total Fragment Mass [kg]': 'last',
            'Temp [K]': 'last',
            'Datetime': 'last',
            'd_m [kg]': 'sum',
            'ReferenceArea [m^2]': 'last',
            'AA7075 [kg]': 'last',
            'CFRP [kg]': 'last',
            'AA2195 [kg]': 'last',
            'Cu [kg]': 'last',
            'CCSic [kg]': 'last',
            'A316 [kg]': 'last',
            'Inconel718 [kg]': 'last',
            'd_m AA7075 [kg]': 'sum',
            'd_m CFRP [kg]': 'sum',
            'd_m AA2195 [kg]': 'sum',
            'd_m Cu [kg]': 'sum',
            'd_m CCSic [kg]': 'sum',
            'd_m A316 [kg]': 'sum',
            'd_m Inconel718 [kg]': 'sum',
            'scenario': 'last',
            'reentry_vehicle': 'last',
        }

        # Perform grouping and aggregation
        compressed_group = (
            group_df.groupby(grouping_fields, as_index=False)
                    .agg(agg_dict)
        )

        result_list.append(compressed_group)

    # Combine all object results
    compressed = pd.concat(result_list, ignore_index=True)

    # Restore original column order
    original_cols = df_trajaerodata.columns.tolist()

    print("Original columns expected:", original_cols)
    print("Compressed columns available:", compressed.columns.tolist())

    # Fill missing columns with NaN
    missing_cols = [col for col in original_cols if col not in compressed.columns]
    for col in missing_cols:
        print(f"Warning: Column '{col}' missing in compressed output. Filling with NaN.")
        compressed[col] = pd.NA

    # Reorder columns
    compressed = compressed[original_cols]

    # Sort by original values
    compressed = compressed.sort_values(
        by=['FragmentID', 'Time [s]', 'Lat [deg]', 'Lon [deg]'],
        ascending=[True, True, True, True]
    )

    # Save and return
    output_path = f"output_data/trajectory/TRAJ_COMP_{scenario_name}.csv"
    compressed.to_csv(output_path, index=False)
    print(f"Compressed trajectory saved to: {output_path}")

    return compressed

#Ectxtrat Data from Trajectory and Aerothermal Files
def process_files(aerothermal_file, trajectory_file, reentry_time, pydrama_results_folder=None):
    # Extract data and metadata from files
    aero_meta, aero_data = extract_metadata_and_data(aerothermal_file)
    traj_meta, traj_data = extract_metadata_and_data(trajectory_file)

    # Define column names for both files
    aero_columns = [
        'Time [s]', 'Altitude [km]', 'Temp [K]', 'Mass [kg]', 'Thick [mm]',
        'Convective Heat [W]', 'Radiative Heat [W]', 'Oxidation Heat [W]',
        'Integrated Heat [J]', 'VisibilityFactor [-]'
    ]
    if pydrama_results_folder:
        traj_columns = [
            'Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 'Velocity [km/s]',
            'Downrange [km]', 'Drag', 'Lift', 'Side', 'Knudsen', 'Mach',
            'Flight Path [deg]', 'Heading [deg]', 'Density [kg/m^2]'
        ]
    else:
        traj_columns = [
            'Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 'Velocity [km/s]',
            'Downrange [km]', 'Drag', 'Lift', 'Side', 'Knudsen', 'Mach',
            'Flight Path [deg]', 'Heading [deg]'
        ]

    # Create DataFrames
    df_aero = pd.DataFrame(aero_data, columns=aero_columns)
    df_traj = pd.DataFrame(traj_data, columns=traj_columns)

    # Merge dataframes on 'Time [s]'
    df_merged = pd.merge(df_traj, df_aero, on='Time [s]', suffixes=('_traj', '_aero'))

    # Select only required columns
    final_columns = [
        'Time [s]', 'Altitude [km]_traj', 'Lat [deg]', 'Lon [deg]', 'Velocity [km/s]',
        'Downrange [km]', 'Mass [kg]', 'Temp [K]'
    ]
    df_final = df_merged[final_columns]

    # Rename columns for clarity
    df_final = df_final.rename(columns={
        'Altitude [km]_traj': 'Altitude [km]',
        'Mass [kg]': 'Mass [kg]',
        'Temp [K]': 'Temp [K]'
    })
    
    # Add datetime column based on reentry_time and Time [s]
    reentry_datetime = datetime.strptime(reentry_time, '%Y-%m-%dT%H:%M:%S')
    df_final['Datetime'] = df_final['Time [s]'].apply(lambda t: reentry_datetime + timedelta(seconds=t))

    # Calculate the difference in Mass [kg] and add as a new column
    df_final['d_m [kg]'] = abs(df_final['Mass [kg]'].diff())

    # Add metadata columns
    df_final['ObjectID'] = aero_meta.get('ObjectID')

    # Ensure object_name_series matches the length of df_final
    df_final['ObjectName'] = aero_meta.get('ObjectName', [])
    df_final[['Part', 'Material']] = df_final['ObjectName'].str.split('_-_', expand=True)

    df_final['ReferenceArea [m^2]'] = aero_meta.get('ReferenceArea')

    return df_final

#endregion


#region: Data Processing for Emission Calculations
# Preprocessing Function for Emission Calculation with compressed/uncompressed ODE results
def emission_preprocessing_data(df_trajectory_results, df_phases):
    try:
        print('df_trajectory_results',df_trajectory_results)
        # Filter ODE results and add columns
        df_cleaned = df_trajectory_results.loc[:, ["t", "h", "date", "lon", "lat", "v", "d_m",]]
        df_cleaned["active_phases"] = None
        df_cleaned["time_interval"] = None
        
        # Filter time data from phases - Convert the phase information to a list of tuples
        phase_times = []
        for idx, row in df_phases.iterrows():
            start_time = row['time_start']
            end_time = row['time_end']
            phase_times.append((idx, start_time, end_time))
                
        # Iterate through filtered dataframe and include active phases and their time intervals
        for idx, row in df_cleaned.iterrows():
            if idx == 0:
                df_cleaned.at[idx, "time_interval"] = []
                df_cleaned.at[idx, "active_phases"] = []
                last_time = row['t']
            else:
                current_time = row['t']
                active_phases = []
                time_intervals = []

                for phase_name, start_time, end_time in phase_times:
                    if last_time < end_time and current_time > start_time:
                        interval_start = round(max(last_time, start_time), 7)
                        interval_end = round(min(current_time, end_time), 7)
                        active_phases.append(phase_name)
                        time_intervals.append([interval_start, interval_end])

                df_cleaned.at[idx, "active_phases"] = active_phases
                df_cleaned.at[idx, "time_interval"] = time_intervals
                last_time = current_time    

    except Exception as e:
        handle_error(e, "emission_preprocessing_data", "Error while preprocessing data for Emission Calculation.")
        
    return df_cleaned       

# Saving the Results of the Emission Calculations
def save_emission_results(emission_results, scenario_name):
    # Saving data to csv file
    try:
        if emission_results is None or not isinstance(emission_results, pd.DataFrame):
            raise ValueError("Emission results are not valid DataFrame")
        
        # Saving data to csv file
        base_name = f"{output_data_emissions_name}{scenario_name}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_nasa_cea, ".csv")
        
        emission_results.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written emission results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + "Emission Data:" + colors.END)
        print(emission_results.head())
        
    except Exception as e:
        handle_error(e, "save_emission_results", "Error finding required data.")

def filter_emission_data(df, compress_method, compress_interval, compress_lat, compress_lon):
    required_columns = ['Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 
                        'Velocity [km/s]', 'Downrange [km]', 'Datetime', 'd_m [kg]']
    species_columns = df.columns[df.columns.get_loc('ReferenceArea [m^2]') + 1:].tolist()
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input DataFrame is missing required columns.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Round values based on the specified compression intervals
    if compress_method == "time":
        df_copy['Time [s]'] = (df_copy['Time [s]'] // compress_interval) * compress_interval
    elif compress_method == "height":
        df_copy['Altitude [km]'] = (df_copy['Altitude [km]'] // compress_interval) * compress_interval
    else:
        raise ValueError("Invalid compress_method. Use 'time' or 'height'.")
    
    # Round latitude and longitude based on the specified thresholds
    df_copy['Lat [deg]'] = (df_copy['Lat [deg]'] // compress_lat) * compress_lat
    df_copy['Lon [deg]'] = (df_copy['Lon [deg]'] // compress_lon) * compress_lon
    
    # Group by the rounded values and aggregate the species columns by sum
    df_grouped = df_copy.groupby(['Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]']).agg({
        'Velocity [km/s]': 'mean',
        'Downrange [km]': 'mean',
        'Temp [K]': 'mean',
        'Datetime': 'first',
        'd_m [kg]': 'sum',
        **{col: 'sum' for col in species_columns}
    }).reset_index()
    
    return df_grouped

def filter_emission_data_bal(df, compress_method, compress_interval, compress_lat, compress_lon):
    required_columns = ['Time [s]', 'date', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 'Velocity [m/s]']
    species_columns = df.columns[df.columns.get_loc('T_rad_equil_comb') + 1:].tolist()
    
    # Check if all required columns are in the DataFrame
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input DataFrame is missing required columns.")
    
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Round values based on the specified compression intervals
    if compress_method == "time":
        df_copy['Time [s]'] = (df_copy['Time [s]'] // compress_interval) * compress_interval
    elif compress_method == "height":
        df_copy['Altitude [km]'] = (df_copy['Altitude [km]'] // compress_interval*1000) * compress_interval*1000
    else:
        raise ValueError("Invalid compress_method. Use 'time' or 'height'.")
    
    # Round latitude and longitude based on the specified thresholds
    df_copy['Lat [deg]'] = (df_copy['Lat [deg]'] // compress_lat) * compress_lat
    df_copy['Lon [deg]'] = (df_copy['Lon [deg]'] // compress_lon) * compress_lon
    
    # Group by rounded values and aggregate species columns by sum
    df_grouped = df_copy.groupby(['Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]']).agg({
        'Velocity [m/s]': 'first',  # Keep the first velocity value for each group
        **{col: 'sum' for col in species_columns}  # Sum the species columns
    }).reset_index()
    
    # Add the 'sum' column as the total of all species columns
    df_grouped['sum'] = df_grouped[species_columns].sum(axis=1)
    
    # Merge the 'date' column back into the grouped DataFrame (if 'date' exists)
    if 'date' in df.columns:
        df_grouped = pd.merge(df_grouped, df_copy[['Time [s]', 'date']].drop_duplicates(), on='Time [s]', how='left')
    
    # Keep only the required columns, species columns, and the 'sum' column
    columns_to_keep = required_columns + species_columns + ['sum']

    df_filtered = df_grouped[columns_to_keep]

    return df_filtered

def smart_round(number):
    # Convert the number to a string to check for 'e' (scientific notation)
    if 'e' in f"{number}":
        # Separate the coefficient and the exponent
        coefficient, exponent = f"{number:.4e}".split('e')
        # Convert back to float with adjusted rounding
        return float(f"{float(coefficient):.4f}e{int(exponent)}")
    else:
        # For normal floats, round to 5 decimal places
        return round(number, 5)

# Calculate Athmosphere Data for Afterburning
def calculate_atmosphere_data(row):
    # Helper function to handle NaN values
    def handle_nan(value):
        return 0 if np.isnan(value) else value
    
    # Extract date, longitude, latitude, and height from the row
    date = row['Datetime']
    lon = row['Lon [deg]']
    lat = row['Lat [deg]']
    height = row['Altitude [km]']
    
    # Run msis Atmosphere Model Function
    atmosphere = msis.run(date, lon, lat, height)
    
    # Total mass density (kg/m3)
    rho_atm = handle_nan(atmosphere[0, 0])
    t_atm = handle_nan(atmosphere[0, 10])
    p_atm = rho_atm * t_atm * R
    
    # Raw Density Data of Species (in molecules per m3)
    densities = {
        'N2': handle_nan(atmosphere[0, 1]),
        'O2': handle_nan(atmosphere[0, 2]),
        'O': handle_nan(atmosphere[0, 3]),
        'He': handle_nan(atmosphere[0, 4]),
        'H': handle_nan(atmosphere[0, 5]),
        'Ar': handle_nan(atmosphere[0, 6]),
        'N': handle_nan(atmosphere[0, 7]),
        #'aox': handle_nan(atmosphere[0, 8]),
        'NO': handle_nan(atmosphere[0, 9])
    }
    
    # Convert number densities to mass densities (kg/m3)
    mass_densities = {species: densities[species] * molar_masses[species] / avogadro_number for species in densities}
    
    # Calculate mass fractions (kg/kg)
    mass_fractions = {species: mass_densities[species] / rho_atm for species in mass_densities}

    # Convert mass fractions dictionary to DataFrame
    df_atm_massfractions = pd.DataFrame([mass_fractions])

    # Calculate total number density (sum of all molecular number densities)
    total_number_density = sum(densities.values())
    
    # Calculate molar fractions (mol/mol)
    molar_fractions = {species: densities[species] / total_number_density for species in densities}
    
    # Convert molar fractions dictionary to DataFrame
    df_atm_molarfractions = pd.DataFrame([molar_fractions])

    # Calculate speed of sound (m/s)
    #print(R)
    kappa = 1.4  # Ratio of specific heats for air
    c_atm = np.sqrt(kappa * R * t_atm)
    
    return rho_atm, t_atm, p_atm, c_atm, df_atm_massfractions, df_atm_molarfractions
    
#endregion