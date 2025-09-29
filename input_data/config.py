### This file includes all configs to execute main.py properly

#region: #### General Config ####
#### Imports ####
#-------------------------------------
import numpy as np

#### Choose which Scenarios to process ####
#-------------------------------------
all_scenarios = False # When False only the user_defined_scenarios will be processed
user_defined_scenarios = ["Falcon_9"] # Based on the scenario name

##endregion

#region: #### Trajectory Config ####
#### Selection of Trajectory Data ####
#-------------------------------------
traj_inp_data = 'OwnDrama' #OwnBallistic, Ballistic, OwnDrama, PyDrama, OwnScarab, OwnDebrisk
file_name_ext_trajectory = 'input_data/merged_trajectory_aerothermal_data.csv' 

#### Simulation Parameters ####
#-------------------------------------
t_max = 720                 # Maximum simulation time [s]
t_span = (0, t_max)         # Time span of the simulation
numeric_method = 'RK45'     # Numerical calculation method to be used 
                            # Possible numeric solvers: 'RK45', 'RK23', 'DOP853', 'BDF'
max_step = 0.1              # Step size [s]

#### Constants and Variables ####
#-------------------------------------
R_0 = 6.378388e6            # Mean Earth Radius [m] (WGS84)
g_0 = 9.80665               # Mean Earth Gravity [m/s²]
R = 287.058                 # J/(kg·K)      for p = ρ⋅R⋅T
rho_0 = 1.225               # kg/m^3
p_0 = 1013.25               # hPa
a_bal = 617                 # W/m³
b_bal = 1000                # m/s
c_bal = 0.0001705           # Ws^3 kg^(-0.5) m^(-1)
sigma_bal = 5.670374E-08    # W/(m^2 K^4)

#### Initial Values (Trajectory ODE) ####
#-------------------------------------
delta_t = 0.1

#### Data Compression ####
#-------------------------------------
compress_data = True                # For faster emission calculation time with NASA CEA
compress_method = "height"          # Choose between "time" or "height" as the compression interval
compress_interval = 1               # Choose interval steps in [s] or [km]
compress_atmosphere = "averages"    # Choose "averages" to average the atmosphere data of the interval
#compress_atmosphere = "latest"     # Or choose "latest" for just the value at the given time/height step
interval_tolerance = 1e-6           # Allow for floating-point precision issues
compress_lat = 0.1
compress_lon = 0.1

#endregion

#region: #### Emissions Config ####
#### General Emission Config ####
#-------------------------------------
# Choose if you want to include emission calculation to trajectory calculation
calculate_emissions = True
# Choose which emission calculation type to use
use_emission_factors = False
emission_factor_method = "stoichiometric"
use_nasa_cea = True
calculate_nox = False
nox_method = "nasa_cea" # nasa_cea, cantera
calculate_black_carbon = False # for now not implemented

#### Emission Constants ####
#-------------------------------------
# Molar masses of the atm species in kg/mol
molar_masses = {
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

# Avogadro's number
avogadro_number = 6.02214076e23  # molecules per mole

# Gas constant
R = 8.314  # Universal gas constant in J/(mol*K)

# Threshold for species to be considered not zero
threshold = 1e-12

# Heat capacity
kappa = 1.4


#### NASA CEA Config ####
#-------------------------------------
afterburning = True                 # Include afterburning calculation in final emission results
rof_f = 2 #Afterburning diameter factor for interaction
trace = "1.e-10"
problem_afterburning = "tp"     # tp, hp

#endregion

#region: #### Folder Path, Data Names & Co. ####
# Other Config

# Folders
input_data_folder_path = 'input_data/'
output_data_folder_path_trajectory = 'output_data/trajectory'
output_data_folder_path_nasa_cea = 'output_data/emissions'
output_data_folder_path_excel = 'output_data/data_raw/excel/'
output_data_folder_path_raw = 'output_data/data_raw/txt/'
nasa_cea_folder_path = r'NASA_CEA/'

# Tags
output_data_trajectory_name = "TRAJ_"
output_data_emissions_name = "EMIS_"
output_data_raw_name = "RAW_"
output_data_compression_name = "COMP_"

# Filenames
nasa_cea_exe = 'FCEA2.exe'
file_name_emission_factors = 'emission_factors.xlsx'
sheet_name_emission_factors = 'Emission_Factors'

file_name_scenarios = 'scenarios.xlsx'
sheet_name_scenarios = 'scenarios'

file_name_material_data = 'material_data.xlsx'
sheet_name_material_data = 'Material_Data'

#### Colors for Terminal Outputs ####
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