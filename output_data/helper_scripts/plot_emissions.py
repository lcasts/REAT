"""Analyse rocket emission inventories: altitude-binned stacked bars, share pie, Excel summary."""
 
from __future__ import annotations
 
import os
import re
 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
 
OUTPUT_DIR = os.path.join("output_data", "plots")
 
 
# --- Helpers ---------------------------------------------------------------
def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
 
 
def _get_cmap(name: str, n_colors: int):
    """matplotlib.cm.get_cmap was removed in 3.9 -- use the modern registry."""
    n_colors = max(int(n_colors), 1)
    try:
        return matplotlib.colormaps[name].resampled(n_colors)
    except AttributeError:  # matplotlib < 3.6
        return plt.cm.get_cmap(name, n_colors)
 
 
def _species_colors(species) -> dict:
    """One fixed colour per species, so the same species looks the same everywhere."""
    species = list(species)
    cmap = _get_cmap("tab20", len(species))
    return {s: cmap(i) for i, s in enumerate(species)}
 
 
# --- Read CSV file ---------------------------------------------------------
def read_emissions_csv(file_path):
    df = pd.read_csv(file_path)
    emission_start_index = df.columns.get_loc("total_sum_emissions") + 1
    emission_columns = df.columns[emission_start_index:]
    clean_columns = [re.sub(r"\*", "", col).strip() for col in emission_columns]
    df = df.rename(columns=dict(zip(emission_columns, clean_columns)))
    return df, clean_columns
 
 
# --- Aggregate emissions by altitude bins ----------------------------------
def aggregate_by_altitude(df, emissions, bin_size=5, max_altitude=100, minor_threshold=0.005):
    df = df[df["Altitude [km]"] <= max_altitude].copy()  # .copy() -> no SettingWithCopyWarning
 
    edges = range(0, max_altitude + bin_size, bin_size)
    # include_lowest keeps rows sitting exactly on 0 km inside the first bin
    df["Altitude_bin"] = pd.cut(df["Altitude [km]"], bins=edges, include_lowest=True)
 
    binned = df.groupby("Altitude_bin", observed=False)[emissions].sum()
 
    totals = binned.sum()
    if totals.sum() == 0:
        raise ValueError("No emissions found below the altitude cut-off.")
    shares = totals / totals.sum()
 
    major_species = shares[shares >= minor_threshold].sort_values(ascending=False).index
    minor_species = shares[shares < minor_threshold].index
 
    binned_major = binned[list(major_species)].copy()
    if len(minor_species):
        binned_major["Other"] = binned[list(minor_species)].sum(axis=1)
 
    return binned_major
 
 
# --- Plot top-5 emissions per bin ------------------------------------------
def plot_top5_emissions_per_bin(binned_emissions, save_name, top_n=5):
    rows, labels = [], []
    for interval, emissions in binned_emissions.iterrows():
        top = emissions.sort_values(ascending=False).head(top_n)
        top = top[top > 0]
        if top.empty:
            continue  # skip altitude bands with no emissions at all
        rows.append(top)
        labels.append(f"{int(interval.left)}-{int(interval.right)} km")
 
    if not rows:
        print("Nothing to plot for the bar chart.")
        return
 
    table = pd.DataFrame(rows).fillna(0.0)
    table.index = labels
    # largest contributor first, so the stacks read consistently
    table = table[table.sum().sort_values(ascending=False).index]
 
    colors = _species_colors(table.columns)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    left = pd.Series(0.0, index=table.index)
    for species in table.columns:
        ax.barh(table.index, table[species], left=left, color=colors[species], label=species)
        left = left + table[species]
 
    ax.set_xlabel("Emissions [kg]")
    ax.set_ylabel("Altitude band")
    ax.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
 
    out_png = os.path.join(_ensure_output_dir(), f"{save_name}_bar_chart.png")
    fig.savefig(out_png, dpi=300)
    plt.show()
    plt.close(fig)
    return out_png
 
 
# --- Plot pie chart for total emissions ------------------------------------
def plot_emissions_pie(df, emissions, save_name, threshold=0.02):
    totals = df[emissions].sum()
    shares = (totals / totals.sum()).sort_values(ascending=False)
 
    major_species = shares[shares >= threshold].copy()
    other = shares[shares < threshold].sum()
    if other > 0:
        major_species["Other"] = other
 
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        major_species,
        labels=major_species.index,
        autopct="%1.1f%%",
        pctdistance=0.9,
        labeldistance=1.05,
        startangle=140,
        colors=[_species_colors(major_species.index)[s] for s in major_species.index],
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title("Total emission share per species")
    fig.tight_layout()
 
    out_png = os.path.join(_ensure_output_dir(), f"{save_name}_pie_chart.png")
    fig.savefig(out_png, dpi=300)
    plt.show()
    plt.close(fig)
    return out_png
 
 
# --- Save total emissions to Excel -----------------------------------------
def save_emissions_summary(df, emissions, save_name):
    totals = df[emissions].sum().sort_values(ascending=False)
    summary_df = totals.to_frame(name="Total Emissions [kg]")
    summary_df["Share [%]"] = 100 * summary_df["Total Emissions [kg]"] / totals.sum()
    summary_df.loc["TOTAL"] = [totals.sum(), 100.0]
 
    out_xlsx = os.path.join(_ensure_output_dir(), f"{save_name}_emissions_summary.xlsx")
    summary_df.to_excel(out_xlsx)
    return out_xlsx
 
 
# --- Main execution --------------------------------------------------------
def main():
    # os.path.join / raw strings avoid backslash escape warnings on Windows
    file_path = os.path.join(
        "output_data", "emissions", "EMIS_2025-030_Falcon_9-435_Stage_2_CEA.csv"
    )
 
    save_name = os.path.splitext(os.path.basename(file_path))[0]
    df, emission_cols = read_emissions_csv(file_path)
    binned_emissions = aggregate_by_altitude(df, emission_cols)
 
    plot_top5_emissions_per_bin(binned_emissions, save_name)
    plot_emissions_pie(df, emission_cols, save_name)
    save_emissions_summary(df, emission_cols, save_name)
 
 
if __name__ == "__main__":
    main()