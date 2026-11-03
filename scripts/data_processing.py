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
import xml.etree.ElementTree as ET   # objects.xml (DRAMA) / result XML (DEBRISK) for material_source = 'model'
from datetime import datetime, timedelta
from tqdm import tqdm

#region: Angle of Attack Aerodynamics
# Cache so the workbook is read once per run instead of once per integration step
_aero_tables = None


def get_aero_tables(force_reload=False):
    """Angle of attack schedule and aerodynamic coefficients from the aero workbook.

    Returns a dict with
        schedule_x      breakpoints in the basis chosen by aero_schedule_basis
                        ('altitude' -> km, 'time' -> s)
        schedule_alpha  angle of attack [deg] at each breakpoint
        alpha           angle of attack [deg] of the coefficient table
        c_d             drag coefficient at each alpha
        ldr             lift-to-drag ratio at each alpha
    """
    global _aero_tables
    if _aero_tables is not None and not force_reload:
        return _aero_tables

    path = os.path.join(input_data_folder_path, file_name_aero)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Aero workbook '{path}' not found. It holds the angle of attack schedule and "
            f"the C_D / L/D tables; set aero_schedule_basis = None to run without it.")

    schedule = pd.read_excel(path, sheet_name=sheet_name_aero_schedule)
    coefficients = pd.read_excel(path, sheet_name=sheet_name_aero_coefficients)
    schedule.columns = [str(c).strip() for c in schedule.columns]
    coefficients.columns = [str(c).strip() for c in coefficients.columns]

    # The basis column is named after the basis, so a workbook cannot be used with the wrong aero_schedule_basis
    basis_column = 'time_s' if aero_schedule_basis == 'time' else 'altitude_km'
    for frame, column, sheet in ((schedule, basis_column, sheet_name_aero_schedule),
                                 (schedule, 'alpha_deg', sheet_name_aero_schedule),
                                 (coefficients, 'alpha_deg', sheet_name_aero_coefficients),
                                 (coefficients, 'c_d', sheet_name_aero_coefficients),
                                 (coefficients, 'ldr', sheet_name_aero_coefficients)):
        if column not in frame.columns:
            raise ValueError(f"Sheet '{sheet}' of '{path}' has no column '{column}'. "
                             f"Found: {list(frame.columns)}")

    schedule = schedule.dropna(subset=[basis_column, 'alpha_deg'])
    coefficients = coefficients.dropna(subset=['alpha_deg', 'c_d', 'ldr']).sort_values('alpha_deg')

    if schedule.empty or coefficients.empty:
        raise ValueError(f"The aero workbook '{path}' carries no usable rows.")
    if (coefficients['c_d'] <= 0).any():
        raise ValueError(f"'{sheet_name_aero_coefficients}' contains C_D <= 0; "
                         f"c_b = mass / (C_D * A) would diverge.")

    _aero_tables = {
        'schedule_x': schedule[basis_column].to_numpy(dtype=float),
        'schedule_alpha': schedule['alpha_deg'].to_numpy(dtype=float),
        'alpha': coefficients['alpha_deg'].to_numpy(dtype=float),
        'c_d': coefficients['c_d'].to_numpy(dtype=float),
        'ldr': coefficients['ldr'].to_numpy(dtype=float),
    }
    return _aero_tables
#endregion

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

# --- Component materials from objects.xml (DRAMA) or the result CSV/XML (DEBRISK) ---
def map_tool_material(raw_material, mapping, tool):
    """Map a tool-specific material name onto the material database. None if unusable."""
    if raw_material is None:
        return None
    raw = str(raw_material).strip()
    if raw not in mapping:
        print(colors.BOLD + colors.RED +
              f"{tool}: material '{raw}' is not in the material map - component skipped. "
              f"Add it to the map in config.py." + colors.END)
        return None
    mapped = mapping[raw]
    if mapped is None:
        print(colors.BOLD + colors.RED +
              f"{tool}: material '{raw}' has no counterpart in the material database "
              f"- component skipped." + colors.END)
    return mapped


def find_drama_objects_xml(scenario_folder, pydrama_results_folder=None):
    """Locate the objects.xml of a DRAMA run, for the PyDrama and the external layout."""
    candidates = []
    if pydrama_results_folder:
        base = os.path.dirname(os.path.dirname(os.path.abspath(pydrama_results_folder)))
        candidates.append(os.path.join(base, 'SARA', 'REENTRY', 'input', 'objects.xml'))
    base = os.path.abspath(scenario_folder)
    for _ in range(4):
        candidates.append(os.path.join(base, 'input', 'objects.xml'))
        base = os.path.dirname(base)
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def drama_material_by_object(objects_xml_path):
    """{uniqueID: mapped material} from a DRAMA objects.xml; uniqueID is the ObjectID."""
    result = {}
    if not objects_xml_path:
        return result
    try:
        root = ET.parse(objects_xml_path).getroot()
    except Exception as e:
        print(colors.BOLD + colors.RED + f"Could not read {objects_xml_path}: {e}" + colors.END)
        return result
    for obj in root.findall('object'):
        uid = (obj.findtext('uniqueID') or '').strip()
        if uid:
            result[uid] = map_tool_material(obj.findtext('material'), drama_material_map, 'DRAMA')
    return result


def debrisk_material_by_component(scenario_folder_path):
    """{component name: mapped material} for a DEBRISK run, from the result CSV or the XML."""
    result = {}
    csvs = [f for f in os.listdir(scenario_folder_path)
            if f.endswith('.csv') and not f.startswith('CbMach')]
    for name in csvs:
        try:
            with open(os.path.join(scenario_folder_path, name), 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            header_idx = next(i for i, l in enumerate(lines) if l.startswith('Parent id;'))
        except (StopIteration, OSError):
            continue
        header = [h.strip().strip('"') for h in lines[header_idx].rstrip('\n').split(';')]
        try:
            i_name, i_mat = header.index('Name'), header.index('Material')
        except ValueError:
            continue
        for line in lines[header_idx + 1:]:
            cells = [c.strip().strip('"') for c in line.rstrip('\n').split(';')]
            if len(cells) > max(i_name, i_mat) and cells[i_name]:
                result[cells[i_name]] = map_tool_material(cells[i_mat], debrisk_material_map, 'DEBRISK')
        if result:
            return result

    for name in [f for f in os.listdir(scenario_folder_path) if f.endswith('.xml')]:
        try:
            root = ET.parse(os.path.join(scenario_folder_path, name)).getroot()
        except Exception:
            continue
        for obj in root.iter('object'):
            nm = obj.find('name')
            mat = obj.find('.//material')
            if nm is not None and nm.text and mat is not None:
                result[nm.text.strip()] = map_tool_material(mat.text, debrisk_material_map, 'DEBRISK')
        if result:
            print(colors.BOLD + colors.GREEN +
                  f"DEBRISK: materials taken from {name} (no usable CSV found)." + colors.END)
            return result
    return result


# Import Drama Files
def drama_input(df_scenarios,scenario_name, pydrama_results_folder=None):
    reentry_vehicle = df_scenarios.loc[scenario_name, 'reentry_vehicle']
    folder = 'SARA/REENTRY/output'
    scenario_folder = os.path.join(input_data_folder_path, reentry_vehicle, folder)

    if pydrama_results_folder:
        scenario_folder = os.path.join(pydrama_results_folder, "reentry")
        print(colors.BOLD + colors.GREEN + "Using PyDrama results folder: ", scenario_folder + colors.END)
    
    if not os.path.exists(scenario_folder):
        # Without the return the following os.listdir() raises a FileNotFoundError.
        print(colors.BOLD + colors.RED + f"Warning: Scenario folder '{scenario_folder}' not found." + colors.END)
        return pd.DataFrame()
    
    # Find all relevant files in the scenario folder
    aerothermal_files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if 'AeroThermalHistory' in f]
    #print('aerothermal_files',aerothermal_files)
    trajectory_files = [os.path.join(scenario_folder, f) for f in os.listdir(scenario_folder) if 'Trajectory' in f]
    #print('trajectory_files',trajectory_files)
    reentry_time = df_scenarios.loc[scenario_name, 'date']
    result_list = []

    # DRAMA writes one aerothermal file per component before break-up (compound track) and one after
    traj_index = {}
    for traj_file in trajectory_files:
        try:
            traj_meta, _ = extract_metadata_and_data(traj_file)
            traj_filename = os.path.basename(traj_file)
            traj_base_match = re.match(r"(.+)_Trajectory", traj_filename)
            traj_base_name = traj_base_match.group(1) if traj_base_match else traj_filename
            traj_index[traj_base_name] = (traj_file, traj_meta.get('ObjectID'))
        except Exception as e:
            print(f"Error reading trajectory file {traj_file}: {e}")

    # Make the compound trajectories addressable by their (purely numeric) ObjectID
    compound_traj = {oid: f for _, (f, oid) in traj_index.items()
                     if oid is not None and str(oid).isdigit()}

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

            traj_file, phase = None, None
            exact = traj_index.get(aero_base_name)
            if exact is not None and exact[1] == object_id:
                # Own trajectory available -> phase after break-up
                traj_file, phase = exact[0], 'post_breakup'
            else:
                # No own trajectory -> pre-break-up phase, use the compound track
                m = re.search(r"\.(\d+)_[0-9a-fA-F-]{36}$", aero_base_name)
                if m and m.group(1) in compound_traj:
                    traj_file, phase = compound_traj[m.group(1)], 'pre_breakup'

            if traj_file is None:
                print(f"  ! No trajectory file found for '{aero_base_name}' - skipped.")
                continue

            df_processed = process_files(aero_file, traj_file, reentry_time, pydrama_results_folder)
            df_processed['Phase'] = phase
            df_processed['scenario'] = scenario_name  # Add scenario column for traceability
            df_processed['reentry_vehicle'] = reentry_vehicle
            result_list.append(df_processed)
            print(f"Processed Data Frame of Object: {aero_base_name}  [{phase}]")
        except Exception as e:
            print(f"Error processing aerothermal file {aero_file}: {e}")

    # 3) Combine All Results into a Single DataFrame
    if result_list:
        df_trajaerodata = pd.concat(result_list, ignore_index=True)
        # Join both phases into one history, drop the duplicated break-up sample and rebuild 'd_m [kg]'
        df_trajaerodata = (df_trajaerodata
                           .sort_values(['ObjectID', 'Time [s]'])
                           .drop_duplicates(subset=['ObjectID', 'Time [s]'], keep='first')
                           .reset_index(drop=True))
        df_trajaerodata['d_m [kg]'] = (df_trajaerodata.groupby('ObjectID')['Mass [kg]']
                                       .diff().abs())

        if material_source == 'model':
            xml_path = find_drama_objects_xml(scenario_folder, pydrama_results_folder)
            by_id = drama_material_by_object(xml_path)
            if by_id:
                print(colors.BOLD + colors.GREEN +
                      f"Materials taken from {xml_path} (material_source = 'model')." + colors.END)
                df_trajaerodata['Material'] = df_trajaerodata['ObjectID'].map(by_id)
                unresolved = sorted(df_trajaerodata.loc[df_trajaerodata['Material'].isna(),
                                                        'ObjectName'].unique())
                if unresolved:
                    print(colors.BOLD + colors.RED +
                          f"No material from objects.xml for: {unresolved}" + colors.END)
            else:
                print(colors.BOLD + colors.RED +
                      "material_source = 'model', but no usable objects.xml was found - "
                      "keeping the materials from the component names." + colors.END)
        base_name = f"{output_data_trajectory_name}{output_data_raw_name}{scenario_name}"
        _, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        df_trajaerodata.to_csv(file_path, index=False)
        print(colors.BOLD + colors.GREEN + f"Trajectory data written to '{os.path.normpath(file_path)}'" + colors.END)
        return df_trajaerodata
    else:
        df_trajaerodata = pd.DataFrame()
        print("No valid results were generated.")
        return df_trajaerodata

# Parse a fragment's material.lst into {column_number: material_name}, the numbering is fragment-specific
def parse_scarab_material_list(fragment_path):
    material_list_path = os.path.join(fragment_path, 'material.lst')
    material_map = {}
    if not os.path.isfile(material_list_path):
        return material_map
    with open(material_list_path, 'r') as f:
        for line in f:
            match = re.match(r'^\s*(\d+)\s+(\S+)', line)
            if match:
                material_map[match.group(1)] = match.group(2)
    return material_map

# Nearest-time merge on TIME[S] for the 3-decimal rounding of tqm.hst/masmat.hst
def merge_asof_on_time(left, right, tolerance=0.005):
    left = left.sort_values('TIME[S]')
    right = right.sort_values('TIME[S]')
    right_cols = [c for c in right.columns if c != 'TIME[S]']
    merged = pd.merge_asof(left, right, on='TIME[S]', direction='nearest', tolerance=tolerance)
    # merge_asof is a left join; drop the unmatched rows to keep inner-join semantics
    return merged.dropna(subset=right_cols)

def merge_exact_on_time(left, right):
    return pd.merge(left, right, on='TIME[S]')

# Build one fragment's merged dataframe with the given merge function (exact or tolerant)
def build_fragment_df(merge_fn, df_mas, df_tqm, df_trj, df_area, df_masmat):
    df_mastqm = merge_fn(df_mas[['TIME[S]', 'MASS[KG]']],
                         df_tqm[['TIME[S]', 'T_MAX[K]']])
    df_mastqmtrj = merge_fn(df_mastqm[['TIME[S]', 'MASS[KG]', 'T_MAX[K]']],
                            df_trj[['TIME[S]', 'ALT[KM]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'TRK[KM]']])
    df_mastqmtrjarea = merge_fn(df_mastqmtrj[['TIME[S]', 'MASS[KG]', 'T_MAX[K]',
                                              'ALT[KM]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'TRK[KM]']],
                                df_area[['TIME[S]', 'PROJ_AREA[M^2]']])
    df_frag = merge_fn(df_mastqmtrjarea[['TIME[S]', 'MASS[KG]', 'T_MAX[K]',
                                         'ALT[KM]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'TRK[KM]',
                                         'PROJ_AREA[M^2]']],
                       df_masmat)
    return df_frag

# Maps raw SCARAB material.lst names to the material names used in emission_factors.xlsx / material_data.xlsx
SCARAB_MATERIAL_NAME_MAP = {
    'aa7075'            : 'AA7075',
    'cfrp-m56-ud134'    : 'CFRP',
    'aa2195'            : 'AA2195',
    'copper'            : 'Cu',
    'cc-sic-ceramic'    : 'CCSiC',
    'a316'              : 'A316L',
    'inconel718'        : 'Inconel718',
    'aa2219'            : 'AA2219',
    'drama-tial6v4'     : 'Ti6Al4V',
}

# Import Scarab Files
def scarab_input(df_scenarios, scenario_name):
    # locate simulation data in <scarab_output_folder_path>/<reentry_vehicle> (SCARAB result folder as-is)
    reentry_vehicle = df_scenarios.loc[scenario_name, 'reentry_vehicle']
    scenario_folder_path = os.path.join(scarab_output_folder_path, reentry_vehicle)

    if not os.path.exists(scenario_folder_path):
        # Without the return the following read_csv() raises a FileNotFoundError.
        print(colors.BOLD + colors.RED + f"Warning: Scenario folder '{scenario_folder_path}' not found." + colors.END)
        return pd.DataFrame()

    # import list of fragments, clean data
    df_frag_list = pd.read_csv(os.path.join(scenario_folder_path, 'frag.lst'), sep='\\s+', skiprows=1, 
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
    # df_materials = pd.read_csv(os.path.join(scenario_folder_path, '0.1', 'material.lst'), sep='\\s+', names=['material_id', 'material_name'], 
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
            
            # masmat.hst has numeric column IDs that material.lst resolves per fragment
            missing = [os.path.basename(p) for p in (masmat_path, mas_path, trj_path, tqm_path, arshad_path)
                       if not os.path.isfile(p)]
            if missing:
                print(colors.BOLD + colors.RED +
                      f"SCARAB: {missing} missing in '{fragment_path}' - fragment skipped." + colors.END)
                continue

            if os.path.isfile(masmat_path):
                with open(masmat_path, 'r') as f:
                    lines = f.readlines()
                    lines = lines[2:]
                    lines[0] = lines[0].split('#')[-1]
                    df_masmat = pd.read_csv(StringIO(''.join(lines)), sep="\\s+", header=0)
                df_masmat.rename(columns={'TIME_[S]': 'TIME[S]'}, inplace=True)
                df_masmat['TIME[S]'] = df_masmat['TIME[S]'].round(4)

                material_map = parse_scarab_material_list(fragment_path)
                df_masmat.rename(columns=material_map, inplace=True)
                unmapped_cols = [c for c in df_masmat.columns if c != 'TIME[S]' and c.isdigit()]
                if unmapped_cols:
                    print(f"Warning: masmat.hst columns {unmapped_cols} in '{fragment_path}' have no matching entry in material.lst.")
            else:
                print('No masmat.hst file found in ', fragment_path)
            
            # import and wrangle mas data (in mas.hst)
            if os.path.isfile(mas_path):
                df_mas = pd.read_csv(mas_path, skiprows=18, sep="\\s+",
                                     names=['TIME', 'ALT', 'TRK', 'MASS', 'COMX', 'COMY', 'COMZ', 'IXX', 'IYY', 'IZZ', 'IXY', 'IXZ', 'IYZ'])
                df_mas.rename(columns={'TIME': 'TIME[S]'}, inplace=True)
                df_mas.rename(columns={'ALT': 'ALT[KM]'}, inplace=True)
                df_mas.rename(columns={'MASS': 'MASS[KG]'}, inplace=True)
                df_mas['TIME[S]'] = df_mas['TIME[S]'].round(4)
            else:
                print('No mas.hst file found in ', fragment_path)

            # import and wrangle trajectory data (in trj.hst)
            if os.path.isfile(trj_path):
                df_trj = pd.read_csv(trj_path, skiprows=28, sep="\\s+",
                                     names=['TIME[S]', 'H[M]', 'LT[DEG]', 'LG[DEG]', 'V[M/S]', 'GM[DEG]', 'PS[DEG]', 'TRK[KM]'])
                df_trj['H[M]'] = df_trj['H[M]']/1000
                df_trj.rename(columns={'H[M]': 'ALT[KM]'}, inplace=True)
                df_trj['TIME[S]'] = df_trj['TIME[S]'].round(4)
            else:
                print('No trj.hst file found in ', fragment_path)

            # import and wrangle temperature and heat flux data (in tqm.hst)
            if os.path.isfile(tqm_path):
                df_tqm = pd.read_csv(tqm_path, skiprows=3, sep="\\s+",
                                     names=['TIME[S]', 'ALT[KM]', 'DST[KM]', 'T_MAX[K]', 'Q_MAX[W/M2]', 'Q_AVG[W/M2]', 'Q_SUM[W]', 'Q_TOT[J]',
                                            'unknown1', 'unknown2', 'unknown3', 'unknown4'])
                df_tqm = df_tqm[['TIME[S]', 'ALT[KM]', 'DST[KM]', 'T_MAX[K]', 'Q_MAX[W/M2]', 'Q_AVG[W/M2]', 'Q_SUM[W]', 'Q_TOT[J]']]
                df_tqm.rename(columns={'DST[KM]': 'TRK[KM]'}, inplace=True)
                df_tqm['TIME[S]'] = df_tqm['TIME[S]'].round(4)
            else:
                print('No tqm.hst file found in ', fragment_path)

            # import and wrangle data containing information about the projected surface area in the direction of flight (in arshad.hst)
            # Column 2 is the projected area (checked against m/beta of ball.hst), column 3 is not an area
            if os.path.isfile(arshad_path):
                df_area = pd.read_csv(arshad_path, sep="\\s+",
                                     names=['TIME[S]', 'PROJ_AREA[M^2]', 'UNKNOWN'])
                df_area['TIME[S]'] = df_area['TIME[S]'].round(4)
            else:
                print('No arshad.hst file found in ', fragment_path)
            
            
            # Exact time match first, nearest-time match if that loses rows (tqm.hst prints only 3 decimals)
            df_frag = build_fragment_df(merge_exact_on_time, df_mas, df_tqm, df_trj, df_area, df_masmat)
            if len(df_frag) < len(df_mas.drop_duplicates('TIME[S]')):
                df_frag_tolerant = build_fragment_df(merge_asof_on_time, df_mas, df_tqm, df_trj, df_area, df_masmat)
                if len(df_frag_tolerant) > len(df_frag):
                    df_frag = df_frag_tolerant.reset_index(drop=True)
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
    
    # Rename material columns onto the material database; unmapped names are kept and flagged
    known_raw_materials = [c for c in df_fragments.columns if c in SCARAB_MATERIAL_NAME_MAP]
    reserved_cols = {'Time [s]', 'Total Fragment Mass [kg]', 'Temp [K]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]',
                      'Velocity [km/s]', 'Downrange [km]', 'ReferenceArea [m^2]', 'FragmentID', 'd_m [kg]', 'Status', 'Datetime'}
    unmapped_raw_materials = [c for c in df_fragments.columns
                              if c not in reserved_cols and c not in SCARAB_MATERIAL_NAME_MAP and not c.startswith('d_m ')]
    if unmapped_raw_materials:
        print(colors.BOLD + colors.YELLOW +
              f"Warning: materials {unmapped_raw_materials} are not in SCARAB_MATERIAL_NAME_MAP; "
              f"keeping raw names, add them to the map in data_processing.py to align with emission_factors.xlsx." + colors.END)

    material_rename = {raw: f'{SCARAB_MATERIAL_NAME_MAP[raw]} [kg]' for raw in known_raw_materials}
    material_rename.update({raw: f'{raw} [kg]' for raw in unmapped_raw_materials})
    df_fragments.rename(columns=material_rename, inplace=True)
    material_cols = sorted(material_rename.values())

    # compute change in material mass for every material (dm: material mass of the current time step - material mass of the former time step)
    df_fragments = df_fragments.sort_values(by=['FragmentID', 'Time [s]'])

    # The last masmat.hst row of a fragmenting parent is a handoff to the children, not mass loss
    frag_status = df_frag_list.set_index('frag_id')['status']
    is_last_row = df_fragments['Time [s]'] == df_fragments.groupby('FragmentID')['Time [s]'].transform('max')
    is_handoff_row = is_last_row & df_fragments['FragmentID'].map(frag_status).eq('Fragmentation')

    # Birth gap: the parent's last mass minus the children's first masses is ablation booked on the handoff row
    first_mass_by_frag = df_fragments.groupby('FragmentID')['Total Fragment Mass [kg]'].first()
    children_by_parent = df_frag_list.loc[df_frag_list['parent'] != '-'].groupby('parent')['frag_id'].apply(list)
    for parent_id, children in children_by_parent.items():
        if frag_status.get(parent_id) != 'Fragmentation':
            continue
        parent_mask = is_handoff_row & (df_fragments['FragmentID'] == parent_id)
        if not parent_mask.any():
            continue
        children_first_sum = sum(first_mass_by_frag.get(c, 0.0) for c in children)
        gap = df_fragments.loc[parent_mask, 'Total Fragment Mass [kg]'].iloc[0] - children_first_sum
        if gap > 0:
            df_fragments.loc[parent_mask, 'd_m [kg]'] = df_fragments.loc[parent_mask, 'd_m [kg]'].fillna(0) + gap

    # A handoff row has no per-material split, so distribute it by the composition of the previous row
    prev_material = df_fragments.groupby('FragmentID')[material_cols].shift(1)
    prev_material_sum = prev_material.sum(axis=1)
    handoff_total_dm = df_fragments['d_m [kg]']  # already includes the birth-gap addition above

    for col in material_cols:
        dm_col = f'd_m {col}'
        df_fragments[dm_col] = df_fragments.groupby(['FragmentID'])[col].diff().abs()

        frac = (prev_material[col] / prev_material_sum).where(prev_material_sum > 0, 0.0)
        df_fragments.loc[is_handoff_row, dm_col] = (handoff_total_dm * frac).loc[is_handoff_row]

    dm_material_cols = [f'd_m {col}' for col in material_cols]

    # ObjectID: the continuation of a body after a break-up (child with the largest start mass) keeps the parent's ID
    def frag_sort_key(frag_id):
        # Parents (lower break-up level) first, ids that are not 'level.number' last
        try:
            return tuple(int(p) for p in frag_id.split('.'))
        except ValueError:
            return (float('inf'),)

    object_id = {}
    for frag_id in sorted(df_frag_list['frag_id'], key=frag_sort_key):
        object_id.setdefault(frag_id, frag_id)
        children = children_by_parent.get(frag_id, [])
        if children:
            continuation = max(children, key=lambda c: (first_mass_by_frag.get(c, 0.0), c.endswith('.1')))
            object_id[continuation] = object_id[frag_id]
    df_fragments['ObjectID'] = df_fragments['FragmentID'].map(object_id).fillna(df_fragments['FragmentID'])

    dm_check_cols = ['d_m [kg]', *dm_material_cols]
    dm_before = df_fragments[dm_check_cols].fillna(0).sum()

    # The first row of a continuation repeats the parent's handoff time step and is dropped
    first_row = df_fragments['Time [s]'] == df_fragments.groupby('FragmentID')['Time [s]'].transform('min')
    is_continuation = df_fragments['ObjectID'] != df_fragments['FragmentID']
    handoff_time = (df_fragments[is_handoff_row].set_index('FragmentID')['Time [s]'])
    parent_of = df_frag_list.set_index('frag_id')['parent']
    parent_handoff_time = df_fragments['FragmentID'].map(parent_of).map(handoff_time)
    drop_duplicate = first_row & is_continuation & ((df_fragments['Time [s]'] - parent_handoff_time).abs() < 1e-3)

    # Fragments born with zero mass are dropped, their mass is already counted in the parent
    zero_mass_frag = df_fragments.groupby('FragmentID')['Total Fragment Mass [kg]'].transform(lambda s: s.fillna(0).max() <= 0)
    zero_dm_frag = df_fragments.groupby('FragmentID')['d_m [kg]'].transform(lambda s: s.fillna(0).sum() <= 0)
    drop_zero_mass = zero_mass_frag & zero_dm_frag

    n_zero_mass_frags = df_fragments.loc[drop_zero_mass, 'FragmentID'].nunique()
    df_fragments = df_fragments[~(drop_duplicate | drop_zero_mass)].copy()

    # Mass check: only rows without d_m may be dropped
    dm_after = df_fragments[dm_check_cols].fillna(0).sum()
    dm_lost = (dm_before - dm_after).abs()
    if (dm_lost > 1e-6 * max(1.0, dm_before['d_m [kg]'])).any():
        print(colors.BOLD + colors.RED + f"SCARAB: d_m lost when merging fragments into objects:\n{dm_lost[dm_lost > 0]}" + colors.END)
    print(f"SCARAB objects: {df_fragments['FragmentID'].nunique()} fragments -> {df_fragments['ObjectID'].nunique()} objects; "
          f"dropped {int(drop_duplicate.sum())} duplicate break-up rows and {int(drop_zero_mass.sum())} rows of "
          f"{n_zero_mass_frags} zero-mass fragments; "
          f"d_m {dm_before['d_m [kg]']:.3f} kg before, {dm_after['d_m [kg]']:.3f} kg after")

    # reorder dataframe
    df_fragments = df_fragments.sort_values(by=['ObjectID', 'Time [s]', 'FragmentID'])
    df_fragments = df_fragments[[
        "Time [s]", "Altitude [km]", "Lat [deg]", "Lon [deg]", "Velocity [km/s]", "Downrange [km]", "Total Fragment Mass [kg]",
        "Temp [K]", "Datetime", "d_m [kg]", "ObjectID", "FragmentID", "ReferenceArea [m^2]",
        *material_cols, *dm_material_cols
        ]]
    
    # add scenario and reentry_vehicle
    df_fragments["scenario"] = scenario_name
    df_fragments["reentry_vehicle"] = reentry_vehicle
    
    if df_fragments.empty:
        print("No valid results were generated.")
    else:
        base_name = f"{output_data_trajectory_name}{output_data_raw_name}{scenario_name}"
        _, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        df_fragments.to_csv(file_path, index=False)
        print(colors.BOLD + colors.GREEN + f"Trajectory data written to '{os.path.normpath(file_path)}'" + colors.END)
        return df_fragments
    
# Import Debrisk Files
def debrisk_input(df_scenarios, scenario_name):
    # locate simulation data in <input_data_folder_path>/<reentry_vehicle> (DEBRISK folder copied in as-is)
    reentry_vehicle = df_scenarios.loc[scenario_name, 'reentry_vehicle']
    scenario_folder_path = os.path.join(input_data_folder_path, reentry_vehicle)

    if not os.path.exists(scenario_folder_path):
        # Without the return the following os.listdir() raises a FileNotFoundError.
        print(colors.BOLD + colors.RED + f"Warning: Scenario folder '{scenario_folder_path}' not found." + colors.END)
        return pd.DataFrame()

    # The .dat file is named after the DEBRISK run, not the scenario, so find it by extension
    dat_files = [f for f in os.listdir(scenario_folder_path) if f.endswith('.dat')]
    if not dat_files:
        # Without the return the following dat_files[0] raises an IndexError.
        print(colors.BOLD + colors.RED + f"Warning: No .dat file found in '{scenario_folder_path}'." + colors.END)
        return pd.DataFrame()

    with open(os.path.join(scenario_folder_path, dat_files[0]), "r") as f:
        lines = f.readlines()

    # extract column names
    variable_line = next(line for line in lines if line.startswith("VARIABLES"))
    column_names = variable_line.replace("VARIABLES =", "").strip().split()

    # additional columns for the zone and material
    column_names.append("zone")
    column_names.append("material")

    # Materials optionally from the DEBRISK result file instead of the component name
    material_by_component = {}
    if material_source == 'model':
        material_by_component = debrisk_material_by_component(scenario_folder_path)
        if material_by_component:
            print(colors.BOLD + colors.GREEN +
                  f"DEBRISK: {len(material_by_component)} materials read from the result files "
                  f"(material_source = 'model')." + colors.END)
        else:
            print(colors.BOLD + colors.RED +
                  "material_source = 'model', but no materials could be read from the DEBRISK "
                  "result files - falling back to the component names." + colors.END)

    # initializing
    data = []
    current_zone = None
    current_material = None

    # go through all the lines of the file
    for line in lines:
        if line.startswith("ZONE"):
            # Zone name of the .dat file, e.g. 'Engine_-_Inconel718'
            current_full = line.split("T='")[1].split("'")[0]
            # Component and material from the name; without '_-_' the material stays empty
            parts = current_full.split("_-_")
            current_zone = parts[0]
            current_material = parts[1] if len(parts) > 1 else None
            if current_material is None and material_source != 'model':
                print(colors.BOLD + colors.RED +
                      f"DEBRISK: component '{current_full}' carries no '_-_' material suffix." + colors.END)
            if material_source == 'model':
                current_material = material_by_component.get(current_full)
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
        base_name = f"{output_data_trajectory_name}{output_data_raw_name}{scenario_name}"
        _, file_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
        df_trajaerodata.to_csv(file_path, index=False)
        print(colors.BOLD + colors.GREEN + f"Trajectory data written to '{os.path.normpath(file_path)}'" + colors.END)
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
                        # .values: assigning the Series itself raises KeyError: 0 in pandas
                        averaged_row[['p_tot (Pa)', 'rho', 'T_rad_equil_comb']] = interval_avg.values
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
                        # .values: assigning the Series itself raises KeyError: 0 in pandas
                        averaged_row[['p_tot (Pa)', 'rho', 'T_rad_equil_comb']] = interval_avg.values
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
                # float() also reads exponent notation such as 4.28391e-05
                metadata['ReferenceArea'] = float(line.split(':')[-1].split('[')[0])
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

        # d_m is summed per bin, the flight state is averaged and 'Time [s]' stays the bin end
        agg_dict = {
            'Time [s]': 'last',
            'Altitude [km]': 'mean',
            'Lat [deg]': 'mean',
            'Lon [deg]': 'mean',
            'Velocity [km/s]': 'mean',
            'Downrange [km]': 'last',
            'Mass [kg]': 'last',
            'Temp [K]': 'last',
            'Datetime': 'mean',
            'ObjectName': 'last',
            'Part': 'last',
            'Material': 'last',
            'ReferenceArea [m^2]': 'last',
            'Phase': 'last',
            'scenario': 'last',
            'reentry_vehicle': 'last',
            'd_m [kg]': 'sum',
        }
        # Only aggregate what this input source delivers; 'Phase' comes from drama_input() only
        agg_dict = {k: v for k, v in agg_dict.items() if k in group_df.columns}

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
    base_name = f"{output_data_trajectory_name}{output_data_compression_name}{scenario_name}"
    _, output_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
    compressed.to_csv(output_path, index=False)
    print(f"Compressed trajectory saved to: {os.path.normpath(output_path)}")

    return compressed

#Compress trajectory from Scarab to easen calculation:
def compress_trajaerodata_scarab_dataframe(df_trajaerodata, compress_method, compress_interval, compress_lat, compress_lon, scenario_name):

    df_copy = df_trajaerodata.copy()
    df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'])

    print('df_copy:\n', df_copy)
    result_list = []

    # material columns vary between scenarios/fragments, so detect them from the actual data instead of hard-coding
    material_cols = [c for c in df_copy.columns if c.endswith(' [kg]') and not c.startswith('d_m ') and c != 'Total Fragment Mass [kg]']

    # Compress per ObjectID (break-up continuations share it, see scarab_input); older files only have FragmentID
    object_col = 'ObjectID' if 'ObjectID' in df_copy.columns else 'FragmentID'
    dm_check_cols = [c for c in df_copy.columns if c.startswith('d_m ')]   # 'd_m [kg]' and every 'd_m <material> [kg]'
    dm_before = df_copy[dm_check_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum()

    for object_id, group_df in df_copy.groupby(object_col):
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
        grouping_fields = [object_col, bin_field, 'LatBin', 'LonBin']

        # d_m is summed per bin, the flight state is averaged and 'Time [s]' stays the bin end
        agg_dict = {
            'Time [s]': 'last',
            'Altitude [km]': 'mean',
            'Lat [deg]': 'mean',
            'Lon [deg]': 'mean',
            'Velocity [km/s]': 'mean',
            'Downrange [km]': 'last',
            'Total Fragment Mass [kg]': 'last',
            'Temp [K]': 'last',
            'Datetime': 'mean',
            'd_m [kg]': 'sum',
            'FragmentID': 'last',
            'ReferenceArea [m^2]': 'last',
            'scenario': 'last',
            'reentry_vehicle': 'last',
        }
        agg_dict.pop(object_col, None)
        for col in material_cols:
            agg_dict[col] = 'last'
            agg_dict[f'd_m {col}'] = 'sum'

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
        by=[object_col, 'Time [s]', 'Lat [deg]', 'Lon [deg]'],
        ascending=[True, True, True, True]
    )

    # Mass check: d_m is summed per bin and must not change
    dm_after = compressed[dm_check_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
    dm_lost = (dm_before - dm_after).abs()
    if (dm_lost > 1e-6 * max(1.0, dm_before['d_m [kg]'])).any():
        print(colors.BOLD + colors.RED + f"SCARAB compression: d_m changed:\n{dm_lost[dm_lost > 0]}" + colors.END)
    print(f"SCARAB compression: {len(df_copy)} -> {len(compressed)} rows; d_m {dm_before['d_m [kg]']:.3f} kg before, {dm_after['d_m [kg]']:.3f} kg after")

    # Save and return
    base_name = f"{output_data_trajectory_name}{output_data_compression_name}{scenario_name}"
    _, output_path = get_unique_filename(base_name, output_data_folder_path_trajectory, ".csv")
    compressed.to_csv(output_path, index=False)
    print(f"Compressed trajectory saved to: {os.path.normpath(output_path)}")

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
def save_emission_results(emission_results, scenario_name,calculation_method):
    # Saving data to csv file
    try:
        if emission_results is None or not isinstance(emission_results, pd.DataFrame):
            raise ValueError("Emission results are not valid DataFrame")
        
        # Saving data to csv file
        base_name = f"{output_data_emissions_name}{scenario_name}_{calculation_method}"
        if all_scenarios == False:
            # Single-scenario runs are usually x_ab studies, so the file name gets the factor
            x_ab_tag = f"{x_ab:g}".replace(".", "_").replace("-", "m")
            base_name += f"_xab_{x_ab_tag}"
        filename, file_path = get_unique_filename(base_name, output_data_folder_path_nasa_cea, ".csv")
        
        emission_results.to_csv(file_path, index=False)

        normalized_file_path = os.path.normpath(file_path)
        print(colors.BOLD + colors.GREEN + f"Successfully written emission results to '{normalized_file_path}'" + colors.END)
        print(colors.BOLD + colors.BLUE + "Emission Data:" + colors.END)
        print(emission_results.head())
        
    except Exception as e:
        handle_error(e, "save_emission_results", "Error finding required data.")

# Column for the material-combustion temperature (filled in emissions.py; unused by emission-factor paths)
COL_T_AB = "Temp ab [K]"


def aggregate_t_ab(values):
    """Aggregates 'Temp ab [K]' during grouping: averages purely numeric values, otherwise keeps them as text to avoid losing information."""
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return pd.NA
    numbers = []
    for v in vals:
        try:
            numbers.append(float(v))
        except (TypeError, ValueError):
            numbers = None
            break
    if numbers is not None:
        return sum(numbers) / len(numbers)
    return "; ".join(dict.fromkeys(str(v) for v in vals))


def filter_emission_data(df, compress_method, compress_interval, compress_lat, compress_lon):
    required_columns = ['Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 
                        'Velocity [km/s]', 'Datetime', 'scenario', 'reentry_vehicle', 'd_m [kg]']
    species_columns = df.columns[df.columns.get_loc('reentry_vehicle') + 1:].tolist()
    
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
    agg_map = {
        'Velocity [km/s]': 'mean',
        'Downrange [km]': 'mean',
        'Temp [K]': 'mean',
    }
    # Carry the combustion temperature along, if the calculation logged it
    if COL_T_AB in df_copy.columns:
        agg_map[COL_T_AB] = aggregate_t_ab
    agg_map.update({
        'Datetime': 'first',
        'scenario': 'first',
        'reentry_vehicle': 'first',
        'd_m [kg]': 'sum',
        **{col: 'sum' for col in species_columns}
    })

    df_grouped = df_copy.groupby(
        ['Time [s]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]']
    ).agg(agg_map).reset_index()

    df_grouped['total_sum_emissions'] = df_grouped[species_columns].sum(axis=1)

    # Drop rows without emissions
    df_grouped = df_grouped.loc[df_grouped['total_sum_emissions'] > 0]

    cols = list(df_grouped.columns)
    dm_index = cols.index('d_m [kg]')
    cols.insert(dm_index + 1, cols.pop(cols.index('total_sum_emissions')))
    df_grouped = df_grouped[cols]
    
    return df_grouped

def projected_reference_area(alpha_deg, a_side, a_axial=0.0):
    """Reference area a body presents to the flow at angle of attack alpha.

    Models the vehicle as a cylinder: at alpha = 90 deg the flow sees the full
    broadside area a_side, at alpha = 0 deg only the axial (nose-on) area
    a_axial. The absolute values make the result independent of the sign
    convention and of the 180 deg wrap in a measured attitude angle.
    """
    alpha = np.radians(np.asarray(alpha_deg, dtype=float))
    return np.abs(np.sin(alpha)) * a_side + np.abs(np.cos(alpha)) * a_axial


# Every column a ballistic trajectory can carry, all other numeric columns are species
TRAJECTORY_COLUMNS_BAL = frozenset({
    # Time and position
    'Time [s]', 'MET [s]', 'date', 'Datetime', 'scenario', 'reentry_vehicle',
    'Altitude [m]', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]',
    'Downrange [m]', 'Downrange [km]',
    # Velocities and attitude
    'Velocity [m/s]', 'Velocity [km/s]', 'Velocity [k/s]', 'v_x', 'v_y',
    'AoA [deg]', 'C_D', 'c_b', 'L/D', 'FPA (rad)', 'FPA (deg)',
    # Accelerations and delta-v budget
    'a_drag', 'n', 'dv_drag', 'dv_drag_x', 'dv_drag_y', 'dv_grav_y', 'dv_orb_y',
    'dv_lift_x', 'dv_lift_y', 'dv_net_x', 'dv_net_y',
    # Atmosphere and aerothermodynamics
    'rho', 'h', 'p_amb', 'Ma', 'Temp [K]',
    'p_dyn (Pa)', 'p_dyn (hPa)', 'p_tot (Pa)', 'p_tot (hPa)',
    'q_conv', 'q_int_conv', 'q_rad', 'q_int_rad', 'q_combined', 'q_int_combined',
    'T_rad_equil_comb', COL_T_AB,
    # Mass loss and aggregates
    'd_m [kg]', 'Mass [kg]', 'ReferenceArea [m^2]',
    'total_sum_species', 'total_sum_emissions',
})


def species_columns_bal(df):
    """Species columns of a ballistic emission result: everything that is not trajectory data.

    The emission loops build their species frame through an empty DataFrame, so the
    columns can arrive as object dtype even though they hold numbers. A column
    therefore counts as a species when every value it holds converts to a number,
    which still keeps stray text columns out of the sum.
    """
    species = []
    for col in df.columns:
        if col in TRAJECTORY_COLUMNS_BAL:
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().sum() == df[col].notna().sum():
            species.append(col)
    return species


def filter_emission_data_bal(df, compress_method, compress_interval, compress_lat, compress_lon):
    required_columns = ['Time [s]', 'date', 'Altitude [km]', 'Lat [deg]', 'Lon [deg]', 'Velocity [m/s]']
    species_columns = species_columns_bal(df)

    # Check if all required columns are in the DataFrame
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    if not species_columns:
        print(colors.BOLD + colors.RED
              + "No species columns in the emission result - nothing to aggregate." + colors.END)
    
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Species columns can carry object dtype, so convert them before sum()
    for col in species_columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    # Round values based on the specified compression intervals
    if compress_method == "time":
        df_copy['Time [s]'] = (df_copy['Time [s]'] // compress_interval) * compress_interval
    elif compress_method == "height":
        # compress_interval is in km for the height method, like in filter_emission_data()
        df_copy['Altitude [km]'] = (df_copy['Altitude [km]'] // compress_interval) * compress_interval
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
    df_grouped['total_sum_species'] = df_grouped[species_columns].sum(axis=1)
    
    # Merge the 'date' column back into the grouped DataFrame (if 'date' exists)
    if 'date' in df.columns:
        df_grouped = pd.merge(df_grouped, df_copy[['Time [s]', 'date']].drop_duplicates(), on='Time [s]', how='left')
    
    # Keep only the required columns, species columns, and the 'sum' column
    columns_to_keep = required_columns + species_columns + ['total_sum_species']

    df_filtered = df_grouped[columns_to_keep]

    return df_filtered

def smart_round(number):
    # Round to 5 significant digits, the precision of the NASA CEA mass fractions
    number = float(number)
    if not np.isfinite(number) or number == 0.0:
        return number
    return float(f"{number:.4e}")

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

    # Pressure from the mixture itself (p = n * k_B * T), dry-air R deviates above ~90 km
    p_atm = total_number_density * boltzmann_constant * t_atm

    # Calculate molar fractions (mol/mol)
    molar_fractions = {species: densities[species] / total_number_density for species in densities}
    
    # Convert molar fractions dictionary to DataFrame
    df_atm_molarfractions = pd.DataFrame([molar_fractions])

    # Speed of sound from the molar mass and isentropic exponent of the local NRLMSIS mixture
    molar_mass_mix = sum(densities[s] * molar_masses[s] for s in densities) / total_number_density
    r_specific = universal_gas_constant / molar_mass_mix
    cv_over_r = sum(densities[s] * (1.5 if s in monatomic_species else 2.5)
                    for s in densities) / total_number_density
    kappa_mix = 1.0 + 1.0 / cv_over_r
    c_atm = np.sqrt(kappa_mix * r_specific * t_atm)
    
    # kappa_mix and r_specific are passed along so compute_shockwave() uses the local mixture instead of kappa=1.4/R=287.058
    return (rho_atm, t_atm, p_atm, c_atm, df_atm_massfractions, df_atm_molarfractions,
            kappa_mix, r_specific)
    
#endregion