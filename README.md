# **REAT**

### _Re-entry Emission Assessment Tool_

REAT is a Python-based tool licensed under the GNU Affero General Public License v3.0. The licence can be found in LICENCE.md. It is designed to calculate the global emissions of specified re-entering space transportation vehicles. By combining trajectory simulation, atmospheric modeling, emission inventories, and optional integrating NASA's Chemical Equilibrium Analysis (CEA), it provides detailed insights into re-entry vehicle emissions. The program is highly customizable and supports both simplified and advanced emission calculations. It should be noted that the re-entry and emission calculation is currently subject to great uncertainties and therefore the results should be considered with caution.
REAT is a scientific project and if you use it for publications or presentations in science, please support the project by citing the tool as following:
Jan-Steffen Fischer and Sebastian Winterhoff and Stefanos Fasoulas, Re-entry Emission Assessment Tool (REAT). University of Stuttgart. 2026. Available: https://github.com/lcasts/REAT

---

## **Features**

- **Trajectory simulation**: Calculates detailed re-entry trajectories for each space transportation vehicle (either (semi-)ballistic for heat protected objects or with pydrama for destructive re-entries.)
- **Atmospheric modeling**: Integrates environmental data from pymsis module based on NRLMSIS from Emmert et al. 2021 for accurate analysis.
- **Emission analysis**: Computes emissions either using Emission Factors or NASA CEA or Cantera.
- **Customizable scenarios**: Easily specify re-entries, stages, and analysis parameters.

---

## **How to Use**

### 1. **Include NASA CEA Software & Prepare Input Data**

- Include NASA CEA Software:

  - To calculate the emissions in more detail using NASA CEA you need to put all the software files of the Win Gui into the folder `/NASA_CEA`.
  - You can still calculate the trajectory and using emission factors, if you don't have access to the NASA CEA Software.
  - If you do not have access to a running version of NASA CEA you can request one over this link: `https://software.nasa.gov/software/LEW-17687-1`.

- Update all required data files in the `input_data` folder:

  - All Excel Files include an extra README sheet for further informations.
  - `config.py`: Core configuration file to customize analysis.
  - `emission_factors.xlsx` Definition of the emission factors for simplified calculation for a specific method and altitude.
  - `material_data.xlxs` Definition of the structural materials and their composition for emission calculation.
  - `scenarios.xlsx` Definition of the re-entry initial state and vehicle parameters (optional columns `A_ref` and `A_ref_axial` for the broadside and nose-on reference areas of ballistic entries).
  - `aero_data.xlsx` Angle-of-attack schedule and C_D/L/D tables for ballistic entries.
  - `trajectory.csv` Example file which contains the definition of the trajectory for own inputs 

### 2. **Customize Configurations**

- Modify the `config.py` file to define your analysis:
  - **Scenarios**: Specify scenarios to process (`user_defined_scenarios` or all scenarios).
  - **Simulation**: Adjust numerical methods, tolerances, and simulation duration.
  - **Emission settings**: Choose between simplified emission factors or NASA CEA or Cantera (preliminary) for structural emission calculations and NASA CEA or Cantera for NOx emission calculation.
  - **Data compression**: Optimize runtime by compressing atmospheric data for ballistic trajectory calculation.
  - **Scenario file**: `file_name_scenarios` can also be set with the environment variable `REAT_SCENARIO_FILE`.

### 3. **Run the Program**

1. **Set Up the Virtual Environment**

   - Create a virtual environment (if not already created):
     ```bash
     python -m venv .venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

2. **Install Required Python Modules**

   - Install the dependencies listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Install DRAMA Python Module**

   - Install DRAMA python module from an existing DRAMA installation. DRAMA is avaialbale under https://sdup.esoc.esa.int/drama/ for download.
   - Install the python DRAMA module:
     ```bash
     pip install <DRAMA installation directory>/TOOLS/python_package
     ```

4. **Execute the Main Script**

   - Run the program:
     ```bash
     python main.py
     ```

5. **View the Results**
   - Results will be saved in the `output_data` folder:
     - **`data_raw/`**: Contains the raw data files.
     - **`emissions/`**: Contains emissions data for all timesteps in Excel format. With `all_scenarios = False` the file names get the suffix `_xab_<x_ab>` (e.g. `_xab_1_2`).
     - **`trajectory/`**: Contains the trajectory data for all timesteps in Excel format.

### 4. **Use Cases**

`scenarios.xlsx` also holds the use cases of the dissertation (rows starting with `UC_`). Run one by setting `all_scenarios = False`, `user_defined_scenarios = ["<scenario>"]` and the config switches below in `config.py`, then `python main.py`.

| Use case | Scenario | Files | Config switches |
|---|---|---|---|
| Falcon 9-435 upper stage, inventory initial state | `UC_Falcon_9-435_Inventory_pyDRAMA` | DRAMA model `F9_Tumbling` | `traj_inp_data = 'PyDrama'`, `fragmentation_altitude_f9 = 80000` |
| Falcon 9-435 upper stage, optical break-up | `UC_Falcon_9-435_Optical_pyDRAMA` | DRAMA model `F9_Tumbling` | `traj_inp_data = 'PyDrama'`, `fragmentation_altitude_f9 = 83100` |
| Falcon 9-435 upper stage, trajectory fit | `UC_Falcon_9-435_Fit_pyDRAMA` | DRAMA model `F9_Tumbling` | `traj_inp_data = 'PyDrama'`, `fragmentation_altitude_f9 = 83468.7190557269` |
| Starship IFT-13 ship, ballistic | `UC_Starship_IFT-13_Ballistic` | `aero_data.xlsx` | `traj_inp_data = 'Ballistic'` (default aero settings) |
| Starship IFT-13 ship, reconstructed trajectory | `UC_Starship_IFT-13_OwnBallistic` | `Starship_IFT_13_trajectory_reconstructed.csv` | `traj_inp_data = 'OwnBallistic'`, `file_name_ext_trajectory = 'input_data/Starship_IFT_13_trajectory_reconstructed.csv'` |

The dissertation varies `x_ab` from 1.1 to 16; the default is 4.

### 5. **Optional Utilities**

- To clean up the output folder:
  ```bash
  python cleanup_output_data.py
  ```

---

## **Folder Structure**

```
.
├── input_data/                 # Contains all input files
│   ├── config.py
│   ├── cea_species.yaml        # Contains the converted thermo.lib database
│   ├── drama_model_folder\SARA\REENTRY\input
│   │   ├── materials.xml       # Contains material data
│   │   └── objects.xml         # Contains DRAMA model description
│   ├── emission_factors.xlsx
│   ├── material_data.xlsx
│   ├── scenarios.xlsx
│   ├── aero_data.xlsx
│   ├── folder_to_SCARAB_or_DEBRISK_output # Optional
│   ├── example_trajectory_ballistic.csv  # Optional
│   └── Starship_IFT_13_trajectory_reconstructed.csv  # Use case
├── NASA_CEA/                   # NASA CEA tool and related files
├── output_data/                # Folder for storing output results
│   ├── data_raw/               # Raw output files
│   ├── emissions/              # Emission results
│   ├── trajectory/             # Trajectory and atmospheric data
├── scripts/                    # Helper scripts
│   ├── data_processing.py
│   ├── emissions.py
│   └── trajectory.py
├── changelog.txt               # Version history
├── cleanup_output_data.py      # Script to clear output data
├── LICENSE.txt                 # Software License Agreement
├── main.py                     # Main entry point for the tool
├── README.md                   # Main README
└── requirements.txt            # Required Python Modules
```

---

## **Technical Notes**

1. **Trajectory Calculation**:

   - There are multiple options for the trajectories:
      - OwnBallistic: Trajectory calculation using an existing ballistic trajectory file.
         - Using an existing ballistic trajectory file. An example is included in the input_data folder.
      - Ballistic: Trajectory calculation of (semi-)ballistic entry trajectories.
      - PyDrama: Trajectory calculation using the python package of DRAMA.
         - Using an existing DRAMA model with the materials.xml and objects.xml
         - Using a generic model to change mass and dimensions within the scenarios.xlsx file
         - Please make sure to use own materials for emission calculation to indicate the name of the material in the name of the systems each with their material seperated with "_-_" (e.g. RP1_Tank_-_AA2198) 
         - Alternatively the material can be extracted from DRAMA or DEBRISK Inputs. Please match them in the config to a material from the material_data.xlsx 
         - Each scenario gets its own model and result folder `input_data/scenarios/<scenario>`, so several scenarios can use the same vehicle.
         - `heat_flux_scaling` scales the DRAMA heat flux, `reflectivity_coefficient` sets the SARA reflectivity coefficient and `drama_materials = "input_data"` uses an own `input_data/materials.xml` instead of the one of the DRAMA installation.
      - OwnDrama: Trajectory calculation based on existing DRAMA files. Please make sure the folder is named as the vehicle.
         - Please make sure to name the systems each with their material seperated with "_-_" (e.g. RP1_Tank_-_AA2198)
      - OwnScarab: Trajectory calculation based on existing SCARAB files in `scarab_output_folder_path` (relative to the project folder). Please make sure the folder is named as the vehicle and includes all the fragment folders (1.1, 1.2, 2.1, ...) and output files.
         - The projected area is read from column 2 of `arshad.hst`.
         - Break-up continuations of the main body (0.1 -> 1.1 -> 2.1 ...) are merged into one object (column `ObjectID`), with a check that no ablated mass is lost.
      - OwnDebrisk: Trajectory calculation based on existing DEBRISK files. Please make sure the folder is named as the vehicle and includes the .dat output file.

2. **Emission Calculation**:
   - For PyDrama, OwnDrama, OwnScarab and OwnDebrisk runs
      - Emission calculation with either emission factors or NASA CEA TP (if added, can also be changed to HP), defined with either "use_emission_factors" or "use_nasa_cea" set as true.
      - The combustion temperature follows from `temp_factor` (default `roberts_lam`) and the interaction area from `x_ab` (default 4). With HP the column "Temp ab [K]" holds the CEA combustion temperature.
      - Optional wake run (`material_wake_temperature`): the combustion products are brought to a lower temperature in a second CEA TP run, optionally mixed with hot gas (`hotgas_wake`, `wake_rof`). It replaces the former quench run.
   - For OwnBallistic and Ballistic trajectories:
      - NOx Emission calculation with either Cantera HP or NASA CEA Shock or HP (if added), defined with the "nox_method" variable.

## **Acknowledgments**
- The balistic trajectory calculation code is partly based on a lecture from Adam Pagan and Georg Herdrich

- The authors would like to gratefully acknowledge funding by the German Space Agency DLR, funding reference 50RL2180 from the Federal Ministry for Economic Affairs and Climate Action of Germany based on a decision of the German Bundestag.

---

## **Contact**
- If you have questions, remarks or wishes please contact us under jan-steffen.fischer[at]irs.uni-stuttgart.de

---

## Contributors
- **Jan-Steffen Fischer** - [lcasts](https://github.com/lcasts)
- **Sebastian Winterhoff** - [SWinterhoff](https://github.com/SWinterhoff)
- **Felix Dorn**
- **Tinus Breitinger**
- **Stefanos Fasoulas**