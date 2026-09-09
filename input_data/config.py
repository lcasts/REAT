### This file includes all configs to execute main.py properly
### Structure: general settings first, then one region per trajectory input source,
### then emissions and paths. traj_inp_data selects which of the source regions applies.

#region: #### General Config ####
#### Imports ####
#-------------------------------------
import numpy as np

#### Choose which Scenarios to process ####
#-------------------------------------
all_scenarios = True                        # When False only the user_defined_scenarios will be processed
user_defined_scenarios = ["2025-082_Falcon_9-402_Stage_2"]  # Based on the scenario name

#### Selection of Trajectory Data ####
#-------------------------------------
traj_inp_data = 'PyDrama'                   # OwnBallistic, Ballistic, OwnDrama, PyDrama, OwnScarab, OwnDebrisk

#### Component Materials ####
#-------------------------------------
material_source = 'name'                    # 'name': from the component name behind '_-_'; 'model': from the simulation file (DRAMA/DEBRISK only)

#### Constants and Variables ####
#-------------------------------------
R = 287.058                 # J/(kg·K)      for p = ρ⋅R⋅T
kappa = 1.4                 # Heat capacity
delta_t = 0.1               # Timestep [s], trajectory integration and emission mass flow
avogadro_number = 6.02214076e23  # Molecules per mole

# Universal constants for the atmospheric state from the NRLMSIS composition
universal_gas_constant = 8.314462618   # J/(mol*K)
boltzmann_constant = 1.380649e-23      # J/K

# Degrees of freedom for the isentropic exponent of the local mixture: monatomic species
# carry cv = 3/2 R, the remaining (diatomic) ones cv = 5/2 R; vibration is neglected.
monatomic_species = {"O", "He", "H", "Ar", "N"}
threshold = 1e-15           # Species below this mass fraction count as zero
molar_masses = {            # Molar masses of the atm species in kg/mol
    'N2': 28.0134e-3,  # kg/mol
    'O2': 31.9988e-3,  # kg/mol
    'O': 16.00e-3,     # kg/mol
    'He': 4.002602e-3, # kg/mol
    'H': 1.00784e-3,   # kg/mol
    'Ar': 39.948e-3,   # kg/mol
    'N': 14.0067e-3,   # kg/mol
    #'aox': 16.00e-3,   # Anomalous oxygen, assume same as O
    'NO': 30.0061e-3   # kg/mol
}
#endregion

#region: #### OwnBallistic / Ballistic Config ####
#### External Simulation Data ####
#-------------------------------------
file_name_ext_trajectory = 'input_data/example_trajectory_ballistic.csv'   # only for traj_inp_data = 'OwnBallistic': external trajectory CSV file

#### Ballistic Trajectory Simulation Parameters ####
#-------------------------------------
t_max = 750                 # Maximum simulation time [s], stops a runaway integration
h_max = 125000              # Upper altitude bound [m], the integration stops above it
#t_span = (0, t_max)        # unused
#numeric_method = 'RK45'    # unused, would need the integrator on scipy.solve_ivp
#max_step = 0.1             # unused, the step size is delta_t

#### Angle of Attack Dependent Aerodynamics ####
#-------------------------------------
aero_schedule_basis = None      # None = constant c_b/ldr from the scenario; 'altitude' [km] or 'time' [s]
aero_interpolation = 'linear'   # 'linear' between the breakpoints, or 'step' to hold each value
aero_reference_area = 400.0     # Reference area [m^2] of the aero table, for c_b = mass / (C_D * A)
aero_schedule_x = [120, 70, 55, 40, 20, 0]              # Breakpoints of the schedule in the chosen basis
aero_schedule_alpha = [60, 60, 60, 55, 30, 5]           # Angle of attack [deg] at each breakpoint
aero_alpha = [0, 10, 20, 30, 45, 60, 90]                # Angle of attack [deg] of the aerodynamic table
aero_cd = [0.90, 0.92, 1.00, 1.15, 1.40, 1.65, 1.85]    # Drag coefficient at each aero_alpha
aero_ldr = [0.00, 0.20, 0.38, 0.50, 0.45, 0.25, 0.00]   # Lift-to-drag ratio at each aero_alpha

#### Constants of the Ballistic Model ####
#-------------------------------------
R_0 = 6.378388e6            # Mean Earth Radius [m] (WGS84)
g_0 = 9.80665               # Mean Earth Gravity [m/s²]
rho_0 = 1.225               # kg/m^3
#p_0 = 1013.25              # unused
a_bal = 617                 # W/m³
b_bal = 1000                # m/s
c_bal = 0.0001705           # Ws^3 kg^(-0.5) m^(-1)
sigma_bal = 5.670374E-08    # W/(m^2 K^4)
#endregion

#region: #### OwnDrama / PyDrama Config ####
#### Break-up ####
#-------------------------------------
fragmentation_altitude = 80000          # [m] break-up altitude for GM1/GM2, written into <breakUpTriggers> of objects.xml
fragmentation_altitude_f9 = 80000       # 78600 [m] same for F9_Tumbling/F9_Fixed, from the observed Falcon 9-435 fragmentation
fragmentation_altitude_ariane = 80000   # [m] same for Ariane_Tumbling/Ariane_Fixed

#### Material Names of the DRAMA Model ####
#-------------------------------------
drama_material_map = {          # <material> in objects.xml -> material_data.xlsx, only for material_source = 'model'
    'drama-AA7075'          : 'AA7075',
    'drama-A316'            : 'A316L',
    'drama-Carbon-Carbon'   : 'CCSiC',
    'drama-Copper'          : 'Cu',
    'drama-HC-CFRP-8ply'    : 'CFRP',
    'drama-Inconel718'      : 'Inconel718',
    'drama-TiAl6v4'         : 'Ti6Al4V',
}
#endregion

#region: #### OwnScarab / OwnDebrisk Config ####
#### Material Names of the DEBRISK Model ####
#-------------------------------------
debrisk_material_map = {        # 'Material' in <run>.csv -> material_data.xlsx, only for material_source = 'model'; None = no counterpart
    'Aluminium DEBRISK'     : 'AA7075',
    'Steel DEBRISK'         : 'A316L',
    'Titanium DEBRISK'      : 'Ti6Al4V',
    'Copper DEBRISK'        : 'Cu',
    'CFRP DEBRISK'          : 'CFRP',
    'Inconel DEBRISK'       : 'Inconel718',
    'SiC DEBRISK'           : 'CCSiC',
    'Beryllium DEBRISK'     : None,
    'Platinum DEBRISK'      : None,
    'Invar DEBRISK'         : None,
    'Zerodur DEBRISK'       : None,
}
# SCARAB reads its materials per fragment from material.lst (SCARAB_MATERIAL_NAME_MAP in scripts/data_processing.py)
#endregion

#region: #### Data Compression Config ####
#### Applies to OwnDrama, PyDrama, OwnScarab and OwnDebrisk ####
#-------------------------------------
compress_data = True                # For faster emission calculation time with NASA CEA
compress_method = "height"          # Choose between "time" or "height" as the compression interval
compress_interval = 1               # Choose interval steps in [s] or [km]
compress_atmosphere = "averages"    # Choose "averages" to average the atmosphere data of the interval
#compress_atmosphere = "latest"     # Or choose "latest" for just the value at the given time/height step
interval_tolerance = 1e-6           # Allow for floating-point precision issues
compress_lat = .5                   # Spatial bin width in latitude [deg]
compress_lon = .5                   # Spatial bin width in longitude [deg]
#endregion

#region: #### Emissions Config ####
#### General Emission Config ####
#-------------------------------------
calculate_emissions = True         # Include emission calculation in the trajectory run
use_emission_factors = True         # Tabulated factors from emission_factors.xlsx
emission_factor_method = "elemental"  # "stoichiometric" or "elemental"
use_nasa_cea = True                 # Material combustion via NASA CEA
use_cantera = False                 # Material combustion via Cantera instead of NASA CEA
calculate_nox = True                # Air shock NOx
nox_method = "nasa_cea"             # nasa_cea, cantera
calculate_black_carbon = False      # for now not implemented
x_ab = 1.2                            # Emission calculation diameter factor for interaction
temp_factor = "roberts_lam"         # roberts_lam, roberts_turb, eckert_lam, eckert_turb, mix_temperature, isentropic, e.g. 0.5
Pr_lam = 0.7                        # Prandtl laminar, for the recovery temperature
Pr_turb = 0.9                       # Prandtl turbulent, for the recovery temperature

#### NASA CEA Config ####
#-------------------------------------
trace = "1.e-15"                        # CEA trace limit
problem_material_combustion = "tp"      # tp, hp
material_quench_temperature = None      # [K] second tp run at this temperature feeds the balance; None = off, e.g. 1500.0
problem_hotgas_calculation = "shock"       # tp/hp: equilibrium at the perfect-gas post-shock state; shock: CEA does the jump itself
# Oxidiser (air) temperature t_air: "cea_stagnation" (physical post-shock equilibrium, dissociation) or "perfect_gas_stagnation" (calorically perfect, much higher); with "tp" this only affects the CEA retry and 7200 K substitutions, not species distribution
hotgas_air_temperature = "cea_stagnation"

#### Cantera Config ####
#-------------------------------------
ct_species_phase_name = 'nox_emission'          # Phase in cea_species.yaml for the air/NOx equilibrium
cantera_material_element_threshold_pct = 0.5    # Elements below this share [%] are dropped from the material

#### Shock Problem Config ####
#### Only for problem_hotgas_calculation = "shock" ####
#-------------------------------------
shock_flags = "inc eq"              # CEA sub-problems: inc/ref incident/reflected, eq/fr equilibrium/frozen
shock_state = "inc_eq"              # Station used for the balance: inc_eq, inc_fz, ref_eq, ref_fz
shock_ions = False                  # Include ionised species (e-, NO+, N+, O+, ...)
shock_debug = False                 # CEA flag 'deb': write every iteration to the .out file
shock_mach_min = 1.05               # Below this free-stream Mach number CEA computes no shock solution
# Free-stream state handed to CEA's shock problem: "u1" (velocity, CEA derives speed of sound from the local NRLMSIS mixture) or "mach1" (Mach number from calculate_atmosphere_data())
shock_input = "u1"
#endregion

#region: #### Folder Path, Data Names & Co. ####
#### Folders ####
#-------------------------------------
input_data_folder_path = 'input_data/'
output_data_folder_path_trajectory = 'output_data/trajectory'
output_data_folder_path_nasa_cea = 'output_data/emissions'
output_data_folder_path_excel = 'output_data/data_raw/excel/'
output_data_folder_path_raw = 'output_data/data_raw/txt/'
nasa_cea_folder_path = r'NASA_CEA/'

#### Tags ####
#-------------------------------------
output_data_trajectory_name = "TRAJ_"
output_data_emissions_name = "EMIS_"
output_data_raw_name = "RAW_"
output_data_compression_name = "COMP_"

#### Filenames ####
#-------------------------------------
nasa_cea_exe = 'FCEA2.exe'
file_name_emission_factors = 'emission_factors.xlsx'
sheet_name_emission_factors = 'Emission_Factors'

file_name_scenarios = 'scenarios.xlsx'
sheet_name_scenarios = 'scenarios'

file_name_material_data = 'material_data.xlsx'
sheet_name_material_data = 'Material_Data'

file_name_ct_species = 'cea_species.yaml'

#### Colors for Terminal Outputs ####
#-------------------------------------
class colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
#endregion
