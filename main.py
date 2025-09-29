# 0. Import Libraries and Functions
from input_data.config import *
from scripts import trajectory
from scripts import emissions
from scripts import data_processing
import pandas as pd
from ast import literal_eval

def main():
    # 1) Import Data and Initial Data Cleaning
    df_emission_factors, df_scenarios, df_material_data = data_processing.get_input_data()
    print(colors.BOLD + colors.BLUE + "DataFrame: df_scenarios" + colors.END)
    print('df_scenarios',df_scenarios)

    # 2) Iterating through all specified scenarios to process
    for idx, scenario_name in enumerate(df_scenarios.index, start=1):
        data_processing.print_scenario_data(idx, scenario_name, df_scenarios)

        # 2.1) Calculating Trajectory based on input type
        if traj_inp_data == "OwnBallistic":
            df_trajaerodata = data_processing.load_external_trajectory_data()
            print(f"[{scenario_name}] OwnBallistic data loaded.")

        elif traj_inp_data == "Ballistic":
            df_trajaerodata = trajectory.ballistic_traj(df_scenarios, scenario_name, delta_t)
            df_trajaerodata = data_processing.compress_and_save_trajectory_results(df_trajaerodata, scenario_name)
            print(f"[{scenario_name}] Ballistic trajectory computed and compressed.")

        elif traj_inp_data == "OwnDrama":
            df_trajaerodata = data_processing.drama_input(df_scenarios, scenario_name)
            print(f"[{scenario_name}] OwnDrama input loaded.")
            
        elif traj_inp_data == "OwnScarab":
            df_trajaerodata = data_processing.scarab_input(df_scenarios, scenario_name)
            print(f"[{scenario_name}] OwnScarab input loaded.")
            
        elif traj_inp_data == "OwnDebrisk":
            df_trajaerodata = data_processing.debrisk_input(df_scenarios, scenario_name)
            print(f"[{scenario_name}] OwnDebrisk input loaded.")
            
        elif traj_inp_data == "PyDrama":
            pydrama_results_folder = trajectory.pydrama(df_scenarios, scenario_name)
            df_trajaerodata = data_processing.drama_input(df_scenarios, scenario_name, pydrama_results_folder)
            print(colors.BOLD + colors.GREEN + f"[{scenario_name}] PyDrama completely processed" + colors.END)

        else:
            print(f"Unknown input: {traj_inp_data}. Please provide a valid input.")
            continue  # skip to the next scenario

        # Apply compression where applicable (but only if not already compressed inside a branch)
        if traj_inp_data in {"OwnDrama", "PyDrama"} and compress_data:
            df_trajaerodata = data_processing.compress_trajaerodata_dataframe(
                df_trajaerodata,
                compress_method,
                compress_interval,
                compress_lat,
                compress_lon,
                scenario_name
            )
            print(f"[{scenario_name}] Trajectory data compressed.")
        
        if traj_inp_data in {"OwnScarab"} and compress_data:
            df_trajaerodata = data_processing.compress_trajaerodata_scarab_dataframe(
                df_trajaerodata,
                compress_method,
                compress_interval,
                compress_lat,
                compress_lon,
                scenario_name
            )
            print(f"[{scenario_name}] Trajectory data compressed.")
        
        
        print(colors.BOLD + colors.BLUE + "DataFrame: df_trajaerodata" + colors.END)
        print('df_trajaerodata',df_trajaerodata)
        # 2.2) Calculating Emissions with Emission Factors or NASA CEA
        if calculate_emissions:  
            # a) Calculate Emissions by emission factory
            if use_emission_factors and (traj_inp_data == "PyDrama" or traj_inp_data == "OwnDrama" or traj_inp_data == "OwnDebrisk"):
                emission_results = emissions.emission_factors(df_trajaerodata, df_emission_factors, scenario_name)
                filtered_emission_results = data_processing.filter_emission_data(emission_results, compress_method, compress_interval, compress_lat, compress_lon)
            elif use_emission_factors and (traj_inp_data == "OwnScarab"):   
                emission_results = emissions.emission_factors_scarab(df_trajaerodata, df_emission_factors, scenario_name)
                print('emission_results\n',emission_results)
                print('emission_results header\n',emission_results.head(0))
                filtered_emission_results = data_processing.filter_emission_data(emission_results, compress_method, compress_interval, compress_lat, compress_lon)
            # b) Calculate Emissions by CEA
            elif use_nasa_cea and (traj_inp_data == "PyDrama" or traj_inp_data == "OwnDrama" or traj_inp_data == "OwnDebrisk"):   
                #print(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors)
                emission_results = emissions.run_nasa_cea(df_trajaerodata, df_emission_factors, df_material_data, scenario_name)
                filtered_emission_results = data_processing.filter_emission_data(emission_results, compress_method, compress_interval, compress_lat, compress_lon)
            elif use_nasa_cea and (traj_inp_data == "OwnScarab"):   
                #print(df_trajectory_results_cleaned, df_phases, df_launch_vehicles_engines, df_launch_vehicles, scenario_name, df_emission_factors)
                emission_results = emissions.run_nasa_cea_scarab(df_trajaerodata, df_emission_factors, df_material_data, scenario_name)
                filtered_emission_results = data_processing.filter_emission_data(emission_results, compress_method, compress_interval, compress_lat, compress_lon)
            elif traj_inp_data == "Ballistic" or traj_inp_data == "OwnBallistic":
                if nox_method == "nasa_cea": 
                    print('Calculating Emissions with NASA CEA for ballistic re-entries...')
                    emission_results = emissions.run_nasa_cea_bal(df_trajaerodata, df_scenarios, scenario_name)
                    print('emission_results\n,',emission_results)
                    filtered_emission_results = data_processing.filter_emission_data_bal(emission_results, compress_method, compress_interval, compress_lat, compress_lon)
                elif nox_method == "cantera": 
                    print('Calculating Emissions with Cantera for ballistic re-entries...')
                    emission_results = emissions.calculate_NO_emissions(df_trajaerodata, df_scenarios, scenario_name)
                    print('emission_results\n,',emission_results)
                    filtered_emission_results = data_processing.filter_emission_data_bal(emission_results, compress_method, compress_interval, compress_lat, compress_lon)
            else:
                print('Invalid combination. Please check your config settings')

            # c) Save the Emission Calculation Results
            data_processing.save_emission_results(filtered_emission_results, scenario_name)

if __name__ == "__main__":
    main()
