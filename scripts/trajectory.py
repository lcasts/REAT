import math
import numpy as np
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET
from drama import sara
from pymsis import msis
from datetime import datetime, timedelta
from input_data.config import *
from scripts.data_processing import *
from ast import literal_eval

from scripts.data_processing import handle_error

#region: Calculate functions
def calculate_atmosphere_traj(date, lat, lon, alt):
    #Calculate atmospheric density and pressure using MSIS model.
    altitude_km = alt / 1000  # Convert to km
    # Run msis Atmosphere Model Function
    atmosphere = msis.run(date, lon, lat, altitude_km)
    def handle_nan(value):
        return 0 if np.isnan(value) else value
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
    kappa = 1.4  # Ratio of specific heats for air
    c_atm = np.sqrt(kappa * R * t_atm)
    
    return rho_atm, t_atm, p_atm, c_atm, total_number_density, df_atm_massfractions, df_atm_molarfractions

def calculate_gravity(alt):
    return g_0 * (R_0 / (R_0 + alt))**2

def calculate_dynamic_pressure(density, velocity):
    return 0.5 * density * velocity**2

def calculate_drag_acceleration(dynamic_pressure, ballistic_coefficient):
    return dynamic_pressure / ballistic_coefficient

def calculate_velocity_components(velocity, flight_path_angle):
    v_y = velocity * np.sin(-flight_path_angle)
    v_x = np.sqrt(velocity**2 - v_y**2)
    return v_x, v_y

def calculate_heat_flux_convective(c, nose_radius, density, velocity):
    return (c / np.sqrt(nose_radius) * density**0.5 * velocity**3) / 1000

def calculate_heat_flux_radiative(a, b, nose_radius, velocity, density, rho_0):
    return (a * nose_radius * (velocity / b)**8.5 * (density / rho_0)**1.6) / 1000

def calculate_wall_temperature(q_combined, emissivity, sigma):
    return (q_combined * 1000 / (emissivity * sigma))**0.25

def coordinates(inc, lat_1, lon_1, dx):
    inc = np.radians(inc)
    c = ((dx/(R_0*2*np.pi))*360)*np.pi/180

    #spärische Trigonometrie, Regel von Neper
    #a = delta lon b = delta lat

    #tan(a) = tan(c)*cos(beta)
    a = math.atan((math.tan(c)*math.cos(inc)))
    a = a*180/np.pi
    lon_2 = (lon_1+a)

    #cos(b*) = cos(90-b) = sin(b) = sin(c) * sin(beta)
    b = math.asin((math.sin(c)*math.sin(inc)))
    b = b*180/np.pi
    lat_2 = (lat_1+b)

    # Differential does not need to be specified since the data is manual calculated
    dlat = 0
    dlon = 0

    return lat_2, lon_2

def initialize_parameters(args):
    """Initialize the parameters for the first time step."""
    (ballistic_coefficient, nose_radius, velocity, flight_path_angle, surface_emissivity, lift_to_drag_ratio, altitude, date, lat, lon, delta_t) = args 
    v_x, v_y = calculate_velocity_components(velocity, flight_path_angle)
    
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    density, temperature, pressure, soundspeed, total_number_density, df_atm_massfractions, df_atm_molarfractions  = calculate_atmosphere_traj(date, lat, lon, altitude)
    dynamic_pressure = calculate_dynamic_pressure(density, velocity)
    a_drag = calculate_drag_acceleration(dynamic_pressure, ballistic_coefficient)
    g = calculate_gravity(altitude)

    # Initialize dv components
    dv_drag = a_drag * delta_t 
    dv_drag_x = dv_drag * np.cos(flight_path_angle)
    dv_drag_y = -dv_drag * np.sin(flight_path_angle)
    dv_grav_y = g * delta_t
    dv_orb_y = (v_x**2 / (R_0 + altitude)) * delta_t
    dv_lift_x = -dv_drag_y * lift_to_drag_ratio
    dv_lift_y = dv_drag_x * lift_to_drag_ratio
    dv_net_x = dv_drag_x + dv_lift_x
    dv_net_y = dv_drag_y + dv_lift_y + dv_orb_y - dv_grav_y

    q_conv = calculate_heat_flux_convective(
        c_bal, nose_radius, density, velocity
    )
    q_rad = calculate_heat_flux_radiative(
        a_bal, b_bal, nose_radius, velocity, density, rho_0
    )
    q_combined = q_conv + q_rad

    T_wall_eq = calculate_wall_temperature(q_combined, surface_emissivity, sigma_bal)

    return {
        'i': 0,
        'Time [s]': 0,
        'date': date,
        'Altitude [m]': altitude,
        'Altitude [km]': altitude / 1000,
        'Lat [deg]': lat,
        'Lon [deg]': lon,
        'Velocity [k/s]': velocity,
        'Downrange [m]': 0,
        'Downrange [km]': 0,
        'v_x': v_x,
        'v_y': v_y,
        'FPA (rad)': flight_path_angle,
        'FPA (deg)': np.degrees(flight_path_angle),
        'rho': density,
        'a_drag': a_drag,
        'n': a_drag / g_0,
        'dv_drag': dv_drag,
        'dv_drag_x': dv_drag_x,
        'dv_drag_y': dv_drag_y,
        'dv_grav_y': dv_grav_y,
        'dv_orb_y': dv_orb_y,
        'dv_lift_x': dv_lift_x,
        'dv_lift_y': dv_lift_y,
        'dv_net_x': dv_net_x,
        'dv_net_y': dv_net_y,
        'h': 0.5 * velocity**2 / 1e6,
        'p_amb': pressure,
        'Ma': velocity / np.sqrt(kappa * pressure / density),
        'p_dyn (Pa)': dynamic_pressure,
        'p_dyn (hPa)': dynamic_pressure / 100,
        'p_tot (Pa)': pressure + dynamic_pressure,
        'p_tot (hPa)': (pressure + dynamic_pressure) / 100,
        'q_conv': q_conv,
        'q_int_conv': 0,
        'q_rad': q_rad,
        'q_int_rad': 0,
        'q_combined': q_combined,
        'q_int_combined': 0,
        'T_rad_equil_comb': T_wall_eq    
    }
#endregion

#region: Calculate ballistic trajectory
def ballistic_traj(df_scenarios,scenario_name, delta_t):
    """Simulates the ballistic entry based on input parameters."""
    print(f"Calculating ballistic trajectory for scenario: {scenario_name}")
    # Unpack initial parameters
    scenario = df_scenarios.loc[scenario_name]  # Extract row corresponding to the scenario

    ballistic_coefficient = scenario['c_b']  # Extract scalar value
    nose_radius = scenario['R_n']  # Extract scalar value
    velocity = scenario['velocity']*1000  # Velocity in m/s
    flight_path_angle = np.radians(scenario['flight_path_angle'])  # Convert to radians
    surface_emissivity = scenario['eta_sf']  # Extract scalar value
    lift_to_drag_ratio = scenario['ldr']  # Extract scalar value
    altitude = scenario['altitude']*1000  # Extract scalar value
    lat = scenario['latitude']  # Extract scalar value
    lon = scenario['longitude']  # Extract scalar value
    heading_angle = scenario['heading_angle']
    inc = (heading_angle - 90) 
    date = scenario['date']  # Extract scalar value
    simulation_date = date
    args = ballistic_coefficient, nose_radius, velocity, flight_path_angle, surface_emissivity, lift_to_drag_ratio, altitude, date, lat, lon, delta_t
    # Initialize results list and parameters for the first step
    results = [initialize_parameters(args)]
    i = 1
    while   0 < altitude < 125000:
        
        prev = results[-1]  # Use the previous step

        # Update velocity components
        v_x = prev['v_x'] - prev['dv_net_x']
        v_y = prev['v_y'] - prev['dv_net_y']
        velocity = np.sqrt(v_x**2 + v_y**2)

        # Update FPA
        flight_path_angle = - np.arctan(v_y/v_x) # rad
        flight_path_angle_deg = flight_path_angle*180/np.pi

        # Update altitude
        altitude = prev['Altitude [m]'] - prev['v_y'] * delta_t
        if altitude < 0:
            altitude = 0

        # Update date
        date = prev['date'] + timedelta(seconds=delta_t) 

        # Update latitude and longitude
        dx = prev['v_x'] * delta_t
        x = prev['Downrange [m]'] + dx
        lat = prev['Lat [deg]']
        lon = prev['Lon [deg]']
        lat, lon = coordinates(inc, lat, lon, dx)

        # Calculate density using MSIS model
        density, temperature, pressure, soundspeed,total_number_density, df_atm_massfractions, df_atm_molarfractions = calculate_atmosphere_traj(date, lat, lon, altitude)

        # Calculate dynamic pressure (q = 0.5 * rho * v^2)
        p_dyn = 0.5 * density * velocity**2

        # Calculate drag force and acceleration
        drag_force = p_dyn / ballistic_coefficient
        a_drag = drag_force
        n_drag = a_drag / g_0

        # Calculate velocity changes
        dv_drag = delta_t * a_drag
        dv_drag_x = dv_drag * np.cos(flight_path_angle)
        dv_drag_y = -dv_drag * np.sin(flight_path_angle)

        dv_grav_y = calculate_gravity(altitude) * delta_t #prev['Altitude [km]'] replaced by altitude
        dv_orb_y = velocity**2 / (R_0 + altitude) * delta_t

        dv_lift_x = -dv_drag_y * lift_to_drag_ratio
        dv_lift_y = dv_drag_x * lift_to_drag_ratio

        dv_net_x = dv_drag_x + dv_lift_x
        dv_net_y = dv_drag_y + dv_orb_y + dv_lift_y - dv_grav_y 

        # Update dynamic parameters
        h = 0.5 * velocity**2 / 1e6
        p_amb = pressure
        mach = velocity / np.sqrt(kappa * p_amb / density)
        p_tot = p_amb + p_dyn

        # Calculate heat flux (convective and radiative)
        q_conv = (c_bal / np.sqrt(nose_radius) * density**0.5 * velocity**3) / 1000
        q_int_conv = prev['q_int_conv'] + prev['q_conv'] * delta_t / 1000

        q_rad = (a_bal * nose_radius * (velocity / b_bal)**8.5 * (density / rho_0)**1.6) / 1000
        q_int_rad = prev['q_int_rad'] + prev['q_rad'] * delta_t / 1000

        # Combined heat flux and equilibrium wall temperature
        q_combined = q_conv + q_rad
        q_int_combined = prev['q_int_combined'] + prev['q_combined'] * delta_t / 1000
        T_wall_eq = (q_combined * 1000 / (surface_emissivity * sigma_bal))**0.25

        # Save results for this step
        results.append({
            'i': i,
            'Time [s]': i * delta_t,
            'date': date,
            'Altitude [m]': altitude,
            'Altitude [km]': altitude / 1000,
            'Lat [deg]': lat,
            'Lon [deg]': lon,
            'Velocity [m/s]': velocity,
            'Downrange [m]': x,
            'Downrange [km]': x / 1000,
            'v_x': v_x,
            'v_y': v_y,
            'FPA (rad)': flight_path_angle,
            'FPA (deg)': flight_path_angle_deg,
            'rho': density,
            'a_drag': a_drag,
            'n': n_drag,
            'dv_drag': dv_drag,
            'dv_drag_x': dv_drag_x,
            'dv_drag_y': dv_drag_y,
            'dv_grav_y': dv_grav_y,
            'dv_orb_y': dv_orb_y,
            'dv_lift_x': dv_lift_x,
            'dv_lift_y': dv_lift_y,
            'dv_net_x': dv_net_x,
            'dv_net_y': dv_net_y,
            'h': h,
            'p_amb': p_amb,
            'Ma': mach,
            'p_dyn (Pa)': p_dyn,
            'p_dyn (hPa)': p_dyn / 100,
            'p_tot (Pa)': p_tot,
            'p_tot (hPa)': p_tot / 100,
            'q_conv': q_conv,
            'q_int_conv': q_int_conv,
            'q_rad': q_rad,
            'q_int_rad': q_int_rad,
            'q_combined': q_combined,
            'q_int_combined': q_int_combined,
            'T_rad_equil_comb': T_wall_eq
        })

        # Increment step
        i += 1

    # Save results to excel
    
    df_results = pd.DataFrame(results)
    df_results = df_results.drop(columns=['i'])
    try:
        base_name = f"{output_data_trajectory_name}{output_data_raw_name}{scenario_name}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        
        df_results.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written trajectory results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + "Trajectory Data:" + colors.END)
        print(df_results.head())
        
    except Exception as e:
        handle_error(e, "process_and_save_ODE_results", "Error finding required data.")

    return df_results

#endregion

#region: Calculate ballistic trajectory
def ballistic():
    df_ballistic = [0]
    return df_ballistic

#endregion

#region: Calculate DRAMA with python integration
def pydrama(df_scenarios,scenario_name):
    # 01 - Loading Scenario Data
    try:
        # Scenario Variables and Constants from Excel
        scenario_params = [
            "reentry_vehicle", "generic_model", "date", "altitude", "latitude", "longitude", 
            "velocity", "flight_path_angle", "heading_angle", "mass", "height", "length", "width", "c_b", "R_n", "eta_sf", "ldr"
        ]
        scenario_data = {param: df_scenarios.loc[scenario_name, param] for param in scenario_params}
        print('scenario_params',scenario_params)
        print('scenario_data',scenario_data)
        reentry_vehicle, generic_model, sc_begin_date, sc_altitude, sc_latitude, sc_longitude, sc_velocity, sc_flight_path_angle, sc_heading_angle, sc_mass, sc_height, sc_length, sc_width, non, non, non, non = scenario_data.values()
        
        sc_radius = sc_length / 2
        
        # Exit the function early, when no generic model is selected (Nan or "") 
        if pd.isna(generic_model) or generic_model == "":
            print(colors.BOLD + colors.RED + "PyDrama WARNING: No Generic Model selected. Skipping generic model processing." + colors.END)
            return  # Exit the function early
        else:
            print(colors.BOLD + colors.GREEN + "PyDrama: Successfully extracted scenario data." + colors.END)
        
        # Scenario Folder & Files
        scenario_folder_path = os.path.join(input_data_folder_path, reentry_vehicle)
        print('scenario_folder_path',scenario_folder_path)
        print(colors.BOLD + colors.GREEN + "PyDrama: Scenario folder path:" + scenario_folder_path + colors.END)
        input_data_folder = os.path.join(scenario_folder_path, "SARA", "REENTRY", "input")
        print('input_data_folder',input_data_folder)
        output_data_folder = os.path.join(scenario_folder_path, "output")
        print('output_data_folder',output_data_folder)
        sc_input_file_paths = {
            "materials": os.path.join(input_data_folder, "materials.xml"),
            "objects": os.path.join(input_data_folder, "objects.xml"),
            #"sara": os.path.join(input_data_folder, "sara.xml"),
        }
        
        # Ensure the scenario folder exists
        os.makedirs(scenario_folder_path, exist_ok=True)
        os.makedirs(input_data_folder, exist_ok=True)
        os.makedirs(output_data_folder, exist_ok=True)
        
        # Ensure the scenario files exist
        if all(os.path.exists(path) for path in sc_input_file_paths.values()):
            print(colors.BOLD + colors.YELLOW + "PyDrama: All scenario files already exist." + colors.END)
        else:
            for path in sc_input_file_paths.values():
                if not os.path.exists(path):
                    open(path, "w").close()
    except Exception as e:
        handle_error(e, "pydrama", "Error while loading scenario data.")
    
    # 02 - Loading Generic Model Data    
    try:
        # Generic Model Folder & Files
        valid_generic_models = {
            "GM_F9US": "GM_F9US",
            "Falcon_9_US": "Falcon_9_US",
        }

        # Check existence of Generic Model folder and required files
        if generic_model in valid_generic_models:
            gm_folder_path = os.path.join(input_data_folder_path, valid_generic_models[generic_model])
            gm_input_data_folder = os.path.join(gm_folder_path, "SARA", "REENTRY", "input")
            print('gm_folder_path',gm_folder_path)

            gm_input_file_paths = {
                #"materials": os.path.join(gm_input_data_folder, "materials.xml"),
                "objects": os.path.join(gm_input_data_folder, "objects.xml"),
                #"sara": os.path.join(gm_input_data_folder, "sara.xml"),
            }
            
            # Error when missing generic model files
            if not os.path.isdir(gm_input_data_folder) or not all(os.path.isfile(path) for path in gm_input_file_paths.values()):
                missing_items = [path for path in gm_input_file_paths.values() if not os.path.isfile(path)]
                raise FileNotFoundError(
                    colors.BOLD + colors.RED +
                    f"PyDrama ERROR: Generic Model '{generic_model}' input data is missing! Check folder '{gm_input_data_folder}' and files: {missing_items}" +
                    colors.END
                )

        # Error when invalid generic model selected
        else:
            raise ValueError(
                colors.BOLD + colors.RED +
                f"PyDrama ERROR: Unsupported Generic Model '{generic_model}' selected! Please choose a valid model from {list(valid_generic_models.keys())}." +
                colors.END
            )
        
        print(colors.BOLD + colors.GREEN + "PyDrama: Scenario setup completed successfully." + colors.END)       
    except Exception as e:
        handle_error(e, "pydrama", "Error while loading generic model data.")
    
    # 03 - Update Scenario Model Data
    try:
        if generic_model == "GM_F9US":
            gm_total_mass = 4520                 # Generic model total mass
            gm_total_height = 15.2                 # Generic model height
            gm_total_length = 3.66                   # Generic model length
            gm_total_width = 3.66                    # Generic model width
            gm_total_radius = gm_total_length/2     # Generic model radius
                        
            for key, gm_file in gm_input_file_paths.items():
                sc_file = sc_input_file_paths[key]

                # Update and copy objects.xml
                if key == "objects":
                    # Load XML file
                    tree = ET.parse(gm_file)
                    root = tree.getroot()
                    
                    # Update XML File
                    for obj in root.findall("object"):
                        name = obj.find("name").text

                        if name == "LOX_Tank - AA2198":
                            obj.find("primitive/cylinder/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((5.1/gm_total_height) * sc_height)
                            obj.find("mass").text = str((1118/gm_total_mass) * sc_mass)

                        if name == "Structure - AA2198":
                            obj.find("primitive/ring/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((0.5/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((1.5/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((1100/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((2.8/gm_total_height) * sc_height)

                        if name == "He_HPV1 - CFRP":
                            obj.find("primitive/sphere/radius").text = str((0.3/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((75.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((2.85/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((1.0/gm_total_radius) * sc_radius)
                        
                        if name == "He_HPV2 - CFRP":
                            obj.find("primitive/sphere/radius").text = str((0.3/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((75.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((2.85/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((-1.0/gm_total_radius) * sc_radius)

                        if name == "Electronics - CuAl":
                            obj.find("primitive/box/width").text = str((1.0/gm_total_width) * sc_width)
                            obj.find("primitive/box/height").text = str((0.8/gm_total_height) * sc_height)
                            obj.find("primitive/box/length").text = str((0.4/gm_total_length) * sc_length)
                            obj.find("mass").text = str((400.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((2.8/gm_total_height) * sc_height)

                        if name == "RP1_Tank - AA2198":
                            obj.find("primitive/cylinder/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((4.0/gm_total_height) * sc_height)
                            obj.find("mass").text = str((452/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((5.05/gm_total_height) * sc_height)

                        if name == "TF - AlTi":
                            obj.find("primitive/cone/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((0.6/gm_total_height) * sc_height)
                            obj.find("mass").text = str((300.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((7.3/gm_total_height) * sc_height)

                        if name == "MerlinVE - Inconel718":
                            obj.find("primitive/box/width").text = str((1.5/gm_total_width) * sc_width)
                            obj.find("primitive/box/height").text = str((1.5/gm_total_height) * sc_height)
                            obj.find("primitive/box/length").text = str((1.7/gm_total_length) * sc_length)
                            obj.find("mass").text = str((500.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((8.5/gm_total_height) * sc_height)

                        if name == "Nozzle - CFRP":
                            #obj.find("name").text = "VINCInozzle"
                            obj.find("primitive/cone/radius").text = str((1.2/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((4.0/gm_total_height) * sc_height)
                            obj.find("mass").text = str((200.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.0/gm_total_height) * sc_height)

                        if name == "Fluidics - A316":
                            #obj.find("name").text = "Fluidics"
                            obj.find("primitive/ring/radius").text = str((0.8/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((2.0/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((0.5/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((300.0/gm_total_mass) * sc_mass)
                            
                    # Save updated XML to the Scenario file
                    tree.write(sc_file, encoding="utf-8", xml_declaration=True)
                    print(colors.BOLD + colors.GREEN + f"PyDrama: Updated '{sc_file}' based on Generic Model." + colors.END)
                
                # Copy materials.xml and sara.xml
                else:  
                    shutil.copy2(gm_file, sc_file)

        print(colors.BOLD + colors.GREEN + "PyDrama: Scenario Model Update completed successfully." + colors.END)
    except Exception as e:
        handle_error(e, "pydrama", "Error while updating the model data ULPM file.")
    
    # 04 - Run PyDrama Simulation
    try:
        # Setup model and config
        mymodel = sara.get_model(project=scenario_folder_path)
        config = {
            'beginDate': sc_begin_date,
            
            'runMode': "reentry-only",
            'monteCarlo': False,
            
            #'dataPath': xyz,
            
            # Initial State of S/C
            'coordinateSystem': 'geodetic',
            'initialDate': sc_begin_date,
            'element1': sc_altitude,                # Altitude [km]
            'element2': sc_latitude,                # Latitude [°]
            'element3': sc_longitude,               # Longitude [°]
            "element4": sc_velocity,               # Velocity [km/s]
            "element5": sc_flight_path_angle,       # Flight path angle [°]
            "element6": sc_heading_angle,           # Heading angle [°]
            "reflectivityCoefficient": 1.3,
            
            # 'coordinateSystem': 'keplerian',
            # 'element1': 6378 + 480,            # Semi-major axis (r_earth + altitude)
            # 'element2': 0.001,                 # Exxentricity
            # 'element3': 98,                    # Inclination
            # "element4": 120,                   # RAAN
            # "element5": 0,                     # Argument of Perigee
            # "element6": 45                     # True Anomaly
            
            # Environment
            "densityScalingFactor": 1.0,
            "dynamicEnvironment": True,
            "useWind": True,
            "useEnvironmentCSV": False,
            
            "energyThreshold": 15.0,
            
            "plotVisibilityMaps": False,
            "plotObjectTrajectories": False,
        }
            
        print(colors.BOLD + colors.YELLOW + "PyDrama: Running Simulation..." + colors.END)
    
        # Run Simulation
        results = sara.run(config, model=mymodel,  keep_output_files="all", save_output_dirs=output_data_folder)
        print('config',config)
        print('output_data_folder',output_data_folder)
        print('results',results)
        # results = sara.run(model=mymodel, keep_output_files="all", save_output_dirs=output_data_folder)
        
        # Print the full config used in the run
        print(colors.BOLD + colors.YELLOW + "Config of SARA Run:" + colors.END)
        print(results["config"])
        
        print(colors.BOLD + colors.GREEN + "PyDrama: Finished Simulation" + colors.END)
    except Exception as e:
        handle_error(e, "pydrama", "Error while running PyDrama Simulation.")

    # 05 - Rename PyDrama Results Folder
    try:
        # Old folder path
        old_folder_path = os.path.join(output_data_folder, "run_0")
        
        # Initial new folder name
        base_folder_name = "run_"
        index = 1
         
        # Find the next available folder name
        while True:
            # Format with leading zero (01, 02, etc.)
            new_folder_name = f"{base_folder_name}{index:02d}"  
            pydrama_results_folder = os.path.join(output_data_folder, new_folder_name)
            
            if not os.path.exists(pydrama_results_folder):
                break
            
            index += 1

        # Rename the folder
        if os.path.exists(old_folder_path):
            os.rename(old_folder_path, pydrama_results_folder)
        print(colors.BOLD + colors.GREEN + f"PyDrama: Renamed results folder to '{pydrama_results_folder}'" + colors.END)
        # Save the simulation terminal prints to a log file
        # log_file_path = os.path.join(pydrama_results_folder, "simulation.log")
        # with open(log_file_path, "w") as log_file:
        #     log_file.write(simulation_output)
    except Exception as e:
        handle_error(e, "pydrama", "Error while renaming PyDrama results folder.")

    return pydrama_results_folder

#endregion