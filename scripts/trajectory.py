import math
import numpy as np
import pandas as pd
import os
import shutil
import io
import contextlib
import xml.etree.ElementTree as ET
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
    # Druck folgt weiter unten aus den Teilchendichten (p = n * k_B * T)
    
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

    # Pressure and speed of sound from the local NRLMSIS mixture, not dry-air kappa=1.4/R=287.058 (see calculate_atmosphere_data)
    p_atm = total_number_density * boltzmann_constant * t_atm
    molar_mass_mix = sum(densities[s] * molar_masses[s] for s in densities) / total_number_density
    r_specific = universal_gas_constant / molar_mass_mix
    cv_over_r = sum(densities[s] * (1.5 if s in monatomic_species else 2.5)
                    for s in densities) / total_number_density
    kappa_mix = 1.0 + 1.0 / cv_over_r

    # Calculate molar fractions (mol/mol)
    molar_fractions = {species: densities[species] / total_number_density for species in densities}
    
    # Convert molar fractions dictionary to DataFrame
    df_atm_molarfractions = pd.DataFrame([molar_fractions])

    # Calculate speed of sound (m/s)
    c_atm = np.sqrt(kappa_mix * r_specific * t_atm)
    
    return (rho_atm, t_atm, p_atm, c_atm, total_number_density, df_atm_massfractions,
            df_atm_molarfractions, kappa_mix, r_specific)

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

def interpolate_schedule(x, xp, fp, mode='linear'):
    """Value of the table fp(xp) at x, for breakpoints given in either direction."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if xp[0] > xp[-1]:
        xp, fp = xp[::-1], fp[::-1]
    if mode == 'step':
        idx = int(np.searchsorted(xp, x, side='right')) - 1
        return float(fp[min(max(idx, 0), len(fp) - 1)])
    return float(np.interp(x, xp, fp))

def check_aero_schedule():
    """Validate the angle of attack schedule, return True when it is active."""
    if not aero_schedule_basis:
        return False
    if aero_schedule_basis not in ('altitude', 'time'):
        raise ValueError(f"aero_schedule_basis must be None, 'altitude' or 'time', not {aero_schedule_basis!r}.")
    tables = get_aero_tables()
    if len(tables['schedule_x']) != len(tables['schedule_alpha']):
        raise ValueError("The schedule sheet needs one alpha per breakpoint.")
    print(colors.BOLD + colors.GREEN +
          f"Ballistic: angle of attack schedule active over {aero_schedule_basis}, "
          f"alpha {tables['schedule_alpha'].min()}...{tables['schedule_alpha'].max()} deg, "
          f"C_D {tables['c_d'].min()}...{tables['c_d'].max()}, "
          f"L/D {tables['ldr'].min()}...{tables['ldr'].max()}, "
          f"from {file_name_aero}." + colors.END)
    return True

def aero_state(time_s, altitude_m, mass, c_b_default, ldr_default, active, a_ref):
    """Angle of attack, C_D, L/D and ballistic coefficient of the current step.

    a_ref is the fixed area the C_D table is normalised with (scenario 'A_ref').
    """
    if not active:
        return np.nan, np.nan, ldr_default, c_b_default
    tables = get_aero_tables()
    x = altitude_m / 1000 if aero_schedule_basis == 'altitude' else time_s
    alpha = interpolate_schedule(x, tables['schedule_x'], tables['schedule_alpha'], aero_interpolation)
    c_d = interpolate_schedule(alpha, tables['alpha'], tables['c_d'], aero_interpolation)
    ldr = interpolate_schedule(alpha, tables['alpha'], tables['ldr'], aero_interpolation)
    return alpha, c_d, ldr, mass / (c_d * a_ref)


def scenario_reference_areas(scenario, scenario_name, aero_active):
    """Broadside area A_ref and nose-on area A_ref_axial [m^2] from the scenario sheet."""
    a_ref = pd.to_numeric(scenario.get('A_ref', np.nan), errors='coerce')
    a_ref_axial = pd.to_numeric(scenario.get('A_ref_axial', np.nan), errors='coerce')
    if aero_active and not a_ref > 0:
        raise ValueError(f"Scenario '{scenario_name}': A_ref = {scenario.get('A_ref', None)} is not positive, "
                         f"the angle of attack schedule needs it for c_b = mass / (C_D * A_ref).")
    if not a_ref > 0:
        return np.nan, np.nan
    if not a_ref_axial >= 0:
        print(colors.BOLD + colors.RED + f"Scenario '{scenario_name}': A_ref_axial could not be read as a "
              f"number, the reference area follows A_ref * |sin(alpha)| only." + colors.END)
        a_ref_axial = 0.0
    return float(a_ref), float(a_ref_axial)


def step_reference_area(alpha, a_ref, a_ref_axial):
    """Reference area of the step for the NOx air mass: projected at alpha, else constant A_ref."""
    if not a_ref > 0:
        return np.nan
    if np.isfinite(alpha):
        return float(projected_reference_area(alpha, a_ref, a_ref_axial))
    return a_ref

def calculate_heat_flux_convective(c, nose_radius, density, velocity):
    return (c / np.sqrt(nose_radius) * density**0.5 * velocity**3) / 1000

def calculate_heat_flux_radiative(a, b, nose_radius, velocity, density, rho_0):
    return (a * nose_radius * (velocity / b)**8.5 * (density / rho_0)**1.6) / 1000

def calculate_wall_temperature(q_combined, emissivity, sigma):
    return (q_combined * 1000 / (emissivity * sigma))**0.25

def coordinates(inc, lat_1, lon_1, dx):
    inc = np.radians(inc)
    c = ((dx/(R_0*2*np.pi))*360)*np.pi/180

    # Spherical trigonometry, Napier's rule: a = delta lon, b = delta lat

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
    (ballistic_coefficient, nose_radius, velocity, flight_path_angle, surface_emissivity, lift_to_drag_ratio, altitude, date, lat, lon, delta_t, mass, aero_active, a_ref, a_ref_axial) = args
    alpha, c_d, lift_to_drag_ratio, ballistic_coefficient = aero_state(
        0.0, altitude, mass, ballistic_coefficient, lift_to_drag_ratio, aero_active, a_ref)
    v_x, v_y = calculate_velocity_components(velocity, flight_path_angle)
    
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    density, temperature, pressure, soundspeed, total_number_density, df_atm_massfractions, df_atm_molarfractions, kappa_mix, r_mix  = calculate_atmosphere_traj(date, lat, lon, altitude)
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
        'Velocity [m/s]': velocity,
        'Downrange [m]': 0,
        'Downrange [km]': 0,
        'AoA [deg]': alpha,
        'C_D': c_d,
        'c_b': ballistic_coefficient,
        'ReferenceArea [m^2]': step_reference_area(alpha, a_ref, a_ref_axial),
        'L/D': lift_to_drag_ratio,
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
    mass = scenario['mass']
    aero_active = check_aero_schedule()
    # Without these the integration silently produces inf/NaN instead of a trajectory.
    if not aero_active and not ballistic_coefficient > 0:
        raise ValueError(f"Scenario '{scenario_name}': c_b = {ballistic_coefficient} is not positive. "
                         f"Set a ballistic coefficient or activate aero_schedule_basis.")
    if aero_active and not mass > 0:
        raise ValueError(f"Scenario '{scenario_name}': mass = {mass} is not positive, "
                         f"the angle of attack schedule needs it for c_b = mass / (C_D * A).")
    if not nose_radius > 0 or not surface_emissivity > 0:
        print(colors.BOLD + colors.RED + f"Scenario '{scenario_name}': R_n = {nose_radius}, "
              f"eta_sf = {surface_emissivity} - the heat fluxes stay inf or zero." + colors.END)
    # A_ref normalises the C_D table (fixed), the projected area at alpha sets the NOx air mass
    a_ref, a_ref_axial = scenario_reference_areas(scenario, scenario_name, aero_active)
    args = ballistic_coefficient, nose_radius, velocity, flight_path_angle, surface_emissivity, lift_to_drag_ratio, altitude, date, lat, lon, delta_t, mass, aero_active, a_ref, a_ref_axial
    # Initialize results list and parameters for the first step
    results = [initialize_parameters(args)]
    i = 1
    # Stop at ground contact, above h_max and after t_max (guard against a runaway integration)
    while 0 < altitude < h_max and i * delta_t <= t_max:

        prev = results[-1]  # Use the previous step

        # Adaptive sub-stepping: split delta_t until each explicit Euler step changes the
        # velocity by less than ~20%, keeping one output row per nominal delta_t
        v_x, v_y = prev['v_x'], prev['v_y']
        alt_m = prev['Altitude [m]']
        x = prev['Downrange [m]']
        lat, lon = prev['Lat [deg]'], prev['Lon [deg]']
        date = prev['date']
        q_int_conv = prev['q_int_conv']
        q_int_rad = prev['q_int_rad']
        q_int_combined = prev['q_int_combined']

        t_remaining = delta_t
        last = None
        while t_remaining > 1e-9 and alt_m > 0:
            velocity = np.sqrt(v_x**2 + v_y**2)
            flight_path_angle = -np.arctan2(v_y, v_x)

            # Angle of attack schedule: C_D, L/D and c_b = mass / (C_D * A) per altitude or time
            t_now = (i - 1) * delta_t + (delta_t - t_remaining)
            alpha, c_d, lift_to_drag_ratio, ballistic_coefficient = aero_state(
                t_now, alt_m, mass, ballistic_coefficient, lift_to_drag_ratio, aero_active, a_ref)

            density, temperature, pressure, soundspeed, total_number_density, df_atm_massfractions, df_atm_molarfractions, kappa_mix, r_mix = \
                calculate_atmosphere_traj(date, lat, lon, alt_m)

            p_dyn = 0.5 * density * velocity**2
            a_drag = p_dyn / ballistic_coefficient
            n_drag = a_drag / g_0

            # Drag along the velocity vector, not via cos/sin of the flight path angle: near
            # vertical descent a slightly negative v_x flips that angle and reverses the drag
            v_mag = max(velocity, 1e-12)
            a_drag_x = a_drag * (v_x / v_mag)
            a_drag_y = a_drag * (v_y / v_mag)
            a_grav_y = calculate_gravity(alt_m)
            a_orb_y = velocity**2 / (R_0 + alt_m)
            a_lift_x = -a_drag_y * lift_to_drag_ratio
            a_lift_y = a_drag_x * lift_to_drag_ratio
            a_net_x = a_drag_x + a_lift_x
            a_net_y = a_drag_y + a_orb_y + a_lift_y - a_grav_y

            a_mag = np.sqrt(a_net_x**2 + a_net_y**2)
            v_mag = max(velocity, 1e-6)
            dt_stable = 0.2 * v_mag / a_mag if a_mag > 1e-12 else t_remaining
            dt_sub = min(t_remaining, dt_stable, delta_t)
            dt_sub = max(dt_sub, 1e-6)

            dv_drag = a_drag * dt_sub
            dv_drag_x = a_drag_x * dt_sub
            dv_drag_y = a_drag_y * dt_sub
            dv_grav_y = a_grav_y * dt_sub
            dv_orb_y = a_orb_y * dt_sub
            dv_lift_x = a_lift_x * dt_sub
            dv_lift_y = a_lift_y * dt_sub
            dv_net_x = a_net_x * dt_sub
            dv_net_y = a_net_y * dt_sub

            q_conv = (c_bal / np.sqrt(nose_radius) * density**0.5 * velocity**3) / 1000
            q_rad = (a_bal * nose_radius * (velocity / b_bal)**8.5 * (density / rho_0)**1.6) / 1000
            q_combined = q_conv + q_rad
            q_int_conv += q_conv * dt_sub / 1000
            q_int_rad += q_rad * dt_sub / 1000
            q_int_combined += q_combined * dt_sub / 1000

            # Position update uses the velocity at the start of the sub-step
            x += v_x * dt_sub
            lat, lon = coordinates(inc, lat, lon, v_x * dt_sub)
            alt_m = alt_m - v_y * dt_sub
            date = date + timedelta(seconds=dt_sub)

            v_x = v_x - dv_net_x
            v_y = v_y - dv_net_y

            if alt_m < 0:
                alt_m = 0

            t_remaining -= dt_sub
            last = dict(
                alpha=alpha, c_d=c_d, ldr=lift_to_drag_ratio, c_b=ballistic_coefficient,
                velocity=velocity, flight_path_angle=flight_path_angle, density=density,
                p_dyn=p_dyn, a_drag=a_drag, n_drag=n_drag, dv_drag=dv_drag,
                dv_drag_x=dv_drag_x, dv_drag_y=dv_drag_y, dv_grav_y=dv_grav_y, dv_orb_y=dv_orb_y,
                dv_lift_x=dv_lift_x, dv_lift_y=dv_lift_y, dv_net_x=dv_net_x, dv_net_y=dv_net_y,
                pressure=pressure, q_conv=q_conv, q_rad=q_rad, q_combined=q_combined,
            )

        altitude = alt_m
        velocity_end = np.sqrt(v_x**2 + v_y**2)
        flight_path_angle_end = -np.arctan(v_y / v_x) if v_x != 0 else 0.0

        # Update dynamic parameters (reported using the state at the end of this step)
        h = 0.5 * velocity_end**2 / 1e6
        p_amb = last['pressure']
        mach = velocity_end / np.sqrt(kappa * p_amb / last['density'])
        p_tot = p_amb + last['p_dyn']
        T_wall_eq = (last['q_combined'] * 1000 / (surface_emissivity * sigma_bal))**0.25

        # Save results for this step
        results.append({
            'i': i,
            'Time [s]': i * delta_t,
            'date': date,
            'Altitude [m]': altitude,
            'Altitude [km]': altitude / 1000,
            'Lat [deg]': lat,
            'Lon [deg]': lon,
            'Velocity [m/s]': velocity_end,
            'Downrange [m]': x,
            'Downrange [km]': x / 1000,
            'AoA [deg]': last['alpha'],
            'C_D': last['c_d'],
            'c_b': last['c_b'],
            'ReferenceArea [m^2]': step_reference_area(last['alpha'], a_ref, a_ref_axial),
            'L/D': last['ldr'],
            'v_x': v_x,
            'v_y': v_y,
            'FPA (rad)': flight_path_angle_end,
            'FPA (deg)': flight_path_angle_end * 180 / np.pi,
            'rho': last['density'],
            'a_drag': last['a_drag'],
            'n': last['n_drag'],
            'dv_drag': last['dv_drag'],
            'dv_drag_x': last['dv_drag_x'],
            'dv_drag_y': last['dv_drag_y'],
            'dv_grav_y': last['dv_grav_y'],
            'dv_orb_y': last['dv_orb_y'],
            'dv_lift_x': last['dv_lift_x'],
            'dv_lift_y': last['dv_lift_y'],
            'dv_net_x': last['dv_net_x'],
            'dv_net_y': last['dv_net_y'],
            'h': h,
            'p_amb': p_amb,
            'Ma': mach,
            'p_dyn (Pa)': last['p_dyn'],
            'p_dyn (hPa)': last['p_dyn'] / 100,
            'p_tot (Pa)': p_tot,
            'p_tot (hPa)': p_tot / 100,
            'q_conv': last['q_conv'],
            'q_int_conv': q_int_conv,
            'q_rad': last['q_rad'],
            'q_int_rad': q_int_rad,
            'q_combined': last['q_combined'],
            'q_int_combined': q_int_combined,
            'T_rad_equil_comb': T_wall_eq
        })

        # Increment step
        i += 1

    if i * delta_t > t_max:
        print(colors.BOLD + colors.RED +
              f"Ballistic: stopped at t_max = {t_max} s at {altitude/1000:.2f} km - "
              f"the object had not reached the ground yet." + colors.END)

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
    from drama import sara  # only PyDrama needs the DRAMA/SARA package; keeps other trajectory modes usable without it
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
        scenario_folder_path = os.path.join(input_data_folder_path,"scenarios",reentry_vehicle)
        print('scenario_folder_path',scenario_folder_path)
        print(colors.BOLD + colors.GREEN + "PyDrama: Scenario folder path:" + scenario_folder_path + colors.END)
        input_data_folder = os.path.join(scenario_folder_path, "SARA", "REENTRY", "input")
        print('input_data_folder',input_data_folder)
        output_data_folder = os.path.join(scenario_folder_path, "output")
        print('output_data_folder',output_data_folder)
        sc_input_file_paths = {
            #"materials": os.path.join(input_data_folder, "materials.xml"),
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
        # Without raise this would continue with undefined scenario data; main.py catches per-scenario exceptions and moves on
        raise
    
    # 02 - Loading Generic Model Data    
    try:
        # Generic Model Folder & Files
        valid_generic_models = {
            "F9_Fixed": "F9_Fixed",
            "F9_Tumbling": "F9_Tumbling",
            "GM1_Fixed": "GM1_Fixed",
            "GM1_Tumbling": "GM1_Tumbling",
            "GM2_Fixed": "GM2_Fixed",
            "GM2_Tumbling": "GM2_Tumbling"
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
        # Without raise this would continue without a valid Generic Model; main.py catches per-scenario exceptions and moves on
        raise
    
    # 03 - Update Scenario Model Data
    try:
        if generic_model in ("GM1_Tumbling", "GM1_Fixed"):
            gm_total_mass = 5300.0                  # Generic model total mass
            gm_total_height = 11.6                  # Generic model height
            gm_total_length = 5.4                   # Generic model length
            gm_total_width = 5.4                    # Generic model width
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
                        # Only the name part before " - " is evaluated; the material suffix after it may vary freely
                        name = obj.find("name").text.split(" - ")[0].strip()

                        if name == "ITS":
                            obj.find("primitive/ring/radius").text = str((2.7/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((5.8/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((2.65/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((900.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((4.0/gm_total_height) * sc_height)

                        if name == "LH2Tank":
                            obj.find("primitive/sphere/radius").text = str((2.7/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((1087.287974/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((2.7/gm_total_height) * sc_height)

                        if name == "LOXTank":
                            obj.find("primitive/sphere/radius").text = str((1.75/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((412.7120259/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((6.5/gm_total_height) * sc_height)

                        if name == "TF":
                            obj.find("primitive/cone/radius").text = str((2.4/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((1.8/gm_total_height) * sc_height)
                            obj.find("mass").text = str((300.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((7.6/gm_total_height) * sc_height)

                        if name == "VINCIeng":
                            obj.find("primitive/ring/radius").text = str((0.125/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((0.75/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((0.075/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((510.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((9.125/gm_total_height) * sc_height)

                        if name == "VINCInozzle":
                            obj.find("primitive/cone/radius").text = str((0.9/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((2.3/gm_total_height) * sc_height)
                            obj.find("mass").text = str((90.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.7/gm_total_height) * sc_height)

                        if name == "Electronics1":
                            obj.find("primitive/box/width").text = str((0.8/gm_total_width) * sc_width)
                            obj.find("primitive/box/height").text = str((0.8/gm_total_height) * sc_height)
                            obj.find("primitive/box/length").text = str((1.7/gm_total_length) * sc_length)
                            obj.find("mass").text = str((300.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((5.0/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((2.2/gm_total_radius) * sc_radius)

                        if name == "Electronics2":
                            obj.find("primitive/box/width").text = str((0.8/gm_total_width) * sc_width)
                            obj.find("primitive/box/height").text = str((0.8/gm_total_height) * sc_height)
                            obj.find("primitive/box/length").text = str((1.7/gm_total_length) * sc_length)
                            obj.find("mass").text = str((300.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((5.0/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((-2.2/gm_total_radius) * sc_radius)

                        if name == "Fluidics":
                            obj.find("primitive/ring/radius").text = str((0.25/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((1.0/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((0.125/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((420.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((8.7/gm_total_height) * sc_height)

                        if name == "HeHPV1":
                            obj.find("primitive/sphere/radius").text = str((0.635/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((90.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((8.4/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartZ").text = str((1.5/gm_total_radius) * sc_radius)

                        if name == "HeHPV2":
                            obj.find("primitive/sphere/radius").text = str((0.635/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((90.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((8.4/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartZ").text = str((-1.5/gm_total_radius) * sc_radius)

                        if name == "Commonpart":
                            obj.find("primitive/ring/radius").text = str((2.65/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((1.5/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((2.6/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((300.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((6.0/gm_total_height) * sc_height)

                        if name == "LVA":
                            obj.find("primitive/cone/radius").text = str((2.3/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((2.0/gm_total_height) * sc_height)
                            obj.find("mass").text = str((500.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((0.45/gm_total_height) * sc_height)

                    # Break-up altitude from the config; it lives in <breakUpTriggers> of the
                    # <connection-to> elements, not in <object>
                    for trigger in root.iter("breakUpTriggers"):
                        trigger_altitude = trigger.find("altitude")
                        if trigger_altitude is not None:
                            trigger_altitude.text = str(float(fragmentation_altitude))

                    # Save updated XML to the Scenario file
                    tree.write(sc_file, encoding="utf-8", xml_declaration=True)
                    print(colors.BOLD + colors.GREEN + f"PyDrama: Updated '{sc_file}' based on Generic Model." + colors.END)

                # Copy materials.xml and sara.xml
                else:
                    shutil.copy2(gm_file, sc_file)

        if generic_model in ("GM2_Tumbling", "GM2_Fixed"):
            gm_total_mass = 3445.0                  # Generic model total mass
            gm_total_height = 13.8                  # Generic model height
            gm_total_length = 3.66                  # Generic model length
            gm_total_width = 3.66                   # Generic model width
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
                        # Only the name part before " - " is evaluated; the material suffix after it may vary freely
                        name = obj.find("name").text.split(" - ")[0].strip()

                        if name == "ITS":
                            obj.find("primitive/ring/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((0.75/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((1.75/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((585.0/gm_total_mass) * sc_mass)

                        if name == "LOXTank":
                            obj.find("primitive/cylinder/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((4.2/gm_total_height) * sc_height)
                            obj.find("mass").text = str((592.3709657/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((2.1/gm_total_height) * sc_height)

                        if name == "RP1Tank":
                            obj.find("primitive/cylinder/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((4.9/gm_total_height) * sc_height)
                            obj.find("mass").text = str((382.6290343/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((6.65/gm_total_height) * sc_height)

                        if name == "TF":
                            obj.find("primitive/cone/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((1.0/gm_total_height) * sc_height)
                            obj.find("mass").text = str((195.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((9.44/gm_total_height) * sc_height)

                        if name == "Engine":
                            obj.find("primitive/ring/radius").text = str((0.35/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((1.7/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((0.25/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((312.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.7/gm_total_height) * sc_height)

                        if name == "Nozzle":
                            obj.find("primitive/cone/radius").text = str((1.2/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((4.6/gm_total_height) * sc_height)
                            obj.find("mass").text = str((78.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((13.1/gm_total_height) * sc_height)

                        if name == "Fluidics":
                            obj.find("primitive/ring/radius").text = str((0.5/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((1.8/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((0.35/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((273.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.7/gm_total_height) * sc_height)

                        if name == "Electronics1":
                            obj.find("primitive/box/width").text = str((0.3/gm_total_width) * sc_width)
                            obj.find("primitive/box/height").text = str((0.6/gm_total_height) * sc_height)
                            obj.find("primitive/box/length").text = str((0.3/gm_total_length) * sc_length)
                            obj.find("mass").text = str((195.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((-0.15/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((1.5/gm_total_radius) * sc_radius)

                        if name == "Electronics2":
                            obj.find("primitive/box/width").text = str((0.3/gm_total_width) * sc_width)
                            obj.find("primitive/box/height").text = str((0.6/gm_total_height) * sc_height)
                            obj.find("primitive/box/length").text = str((0.3/gm_total_length) * sc_length)
                            obj.find("mass").text = str((195.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((-0.15/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((-1.5/gm_total_radius) * sc_radius)

                        if name == "HeHPV1":
                            obj.find("primitive/cylinder/radius").text = str((0.45/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((1.4/gm_total_height) * sc_height)
                            obj.find("mass").text = str((39.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.0/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartZ").text = str((1.1/gm_total_radius) * sc_radius)

                        if name == "HeHPV2":
                            obj.find("primitive/cylinder/radius").text = str((0.45/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((1.4/gm_total_height) * sc_height)
                            obj.find("mass").text = str((39.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.0/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((-0.9526279/gm_total_radius) * sc_radius)
                            obj.find("relativePosition/cartZ").text = str((-0.55/gm_total_radius) * sc_radius)

                        if name == "HeHPV3":
                            obj.find("primitive/cylinder/radius").text = str((0.45/gm_total_radius) * sc_radius)
                            obj.find("primitive/cylinder/height").text = str((1.4/gm_total_height) * sc_height)
                            obj.find("mass").text = str((39.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((10.0/gm_total_height) * sc_height)
                            obj.find("relativePosition/cartY").text = str((0.9526279/gm_total_radius) * sc_radius)
                            obj.find("relativePosition/cartZ").text = str((-0.55/gm_total_radius) * sc_radius)

                        if name == "LVA":
                            obj.find("primitive/cone/radius").text = str((1.83/gm_total_radius) * sc_radius)
                            obj.find("primitive/cone/height").text = str((1.5/gm_total_height) * sc_height)
                            obj.find("mass").text = str((325.0/gm_total_mass) * sc_mass)
                            obj.find("relativePosition/cartX").text = str((-0.85/gm_total_height) * sc_height)

                        if name == "Commonpart":
                            obj.find("primitive/ring/radius").text = str((1.8/gm_total_radius) * sc_radius)
                            obj.find("primitive/ring/length").text = str((0.7/gm_total_height) * sc_height)
                            obj.find("primitive/ring/innerRadius").text = str((1.75/gm_total_radius) * sc_radius)
                            obj.find("mass").text = str((195.0/gm_total_mass) * sc_mass)

                    # Break-up altitude from the config; it lives in <breakUpTriggers> of the
                    # <connection-to> elements, not in <object>
                    for trigger in root.iter("breakUpTriggers"):
                        trigger_altitude = trigger.find("altitude")
                        if trigger_altitude is not None:
                            trigger_altitude.text = str(float(fragmentation_altitude))

                    # Save updated XML to the Scenario file
                    tree.write(sc_file, encoding="utf-8", xml_declaration=True)
                    print(colors.BOLD + colors.GREEN + f"PyDrama: Updated '{sc_file}' based on Generic Model." + colors.END)

                # Copy materials.xml and sara.xml
                else:
                    shutil.copy2(gm_file, sc_file)
    
        if generic_model in ("F9_Tumbling", "F9_Fixed"):
            # F9 is copied unscaled, but still get their break-up altitude from the
            # config - each with its own parameter, the F9 value comes from an observed re-entry
            break_up_altitude = fragmentation_altitude_f9
            for key, gm_file in gm_input_file_paths.items():
                sc_file = sc_input_file_paths[key]
                shutil.copy2(gm_file, sc_file)

                if key == "objects":
                    # The altitude lives in <breakUpTriggers>, not in <object>
                    tree = ET.parse(sc_file)
                    root = tree.getroot()
                    n_triggers = 0
                    for trigger in root.iter("breakUpTriggers"):
                        trigger_altitude = trigger.find("altitude")
                        if trigger_altitude is not None:
                            trigger_altitude.text = str(float(break_up_altitude))
                            n_triggers += 1
                    tree.write(sc_file, encoding="utf-8", xml_declaration=True)
                    print(colors.BOLD + colors.GREEN + f"PyDrama: Copied '{key}' from Generic Model "
                          f"'{generic_model}' (no scaling), break-up altitude set to "
                          f"{break_up_altitude} m in {n_triggers} triggers." + colors.END)
                else:
                    print(colors.BOLD + colors.GREEN + f"PyDrama: Copied '{key}' from Generic Model '{generic_model}' (no scaling)." + colors.END)

        print(colors.BOLD + colors.GREEN + "PyDrama: Scenario Model Update completed successfully." + colors.END)
    except Exception as e:
        handle_error(e, "pydrama", "Error while updating the generic model data file.")
        # Without raise, SARA would silently start with whatever objects.xml is left in the scenario folder, e.g. the previous scenario's model; main.py catches per-scenario exceptions and moves on
        raise
    
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
            "useWind": False,
            "useEnvironmentCSV": False,
            
            "energyThreshold": 15.0,
            
            "plotVisibilityMaps": False,
            "plotObjectTrajectories": False,
        }

        # Attitude: derived from generic model suffix
        if "_Tumbling" in generic_model:
            config['attitude'] = 'tumbling'
        elif "_Fixed" in generic_model:
            config['attitude'] = 'fixed'
            config['attack']   = 0.0
            config['sideSlip'] = 0.0
            config['bank']     = 0.0

        print(colors.BOLD + colors.YELLOW + "PyDrama: Running Simulation..." + colors.END)

        # Run Simulation — capture stdout to console and log file simultaneously
        log_path = os.path.join(output_data_folder, "sara_log.txt")
        os.makedirs(output_data_folder, exist_ok=True)
        sara_buffer = io.StringIO()
        with contextlib.redirect_stdout(sara_buffer):
            results = sara.run(config, model=mymodel, keep_output_files="all", save_output_dirs=output_data_folder)

        # Echo captured output to console
        captured = sara_buffer.getvalue()
        print(captured, end="")

        # Write log file
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== SARA Run Config ===\n")
            log_file.write(str(config) + "\n\n")
            log_file.write("=== SARA Console Output ===\n")
            log_file.write(captured + "\n")
            log_file.write("=== SARA Results ===\n")
            log_file.write(str(results) + "\n")
        print(colors.BOLD + colors.YELLOW + f"PyDrama: Log saved to {log_path}" + colors.END)

        print(colors.BOLD + colors.GREEN + "PyDrama: Finished Simulation" + colors.END)
    except Exception as e:
        handle_error(e, "pydrama", "Error while running PyDrama Simulation.")
        # Without raise, an aborted run would return an incomplete run_NN folder and be silently processed further; main.py catches per-scenario exceptions and moves on
        raise

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