#!/usr/bin/env python3
"""
plot_flight.py

Reads a CSV file with flight data and generates four plots:
  1) Altitude [km] vs Time [s]
  2) Altitude [km] vs Downrange [km]
  3) Velocity [m/s] vs Time [s]
  4) Altitude [km] vs Velocity [m/s]

Configure the CSV input and output directory paths below.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === Configuration ===
# Path to your CSV file:
CSV_FILE = Path(r"output_data/trajectory/TRAJ_COMP_Dragon(2).csv")
# Directory where plots will be saved:
OUTPUT_DIR = Path(r"output_data\plots")
# ======================

def make_plot(x, y, xlabel, ylabel, title, fname, outdir):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read data from CSV
    df = pd.read_csv(CSV_FILE)

    # 1) Altitude [km] vs Time [s]
    make_plot(
        df["Time [s]"],
        df["Altitude [km]"],
        xlabel="Time [s]",
        ylabel="Altitude [km]",
        title="Altitude vs Time",
        fname="altitude_vs_time.png",
        outdir=OUTPUT_DIR
    )

    # 2) Altitude [km] vs Downrange [km]
    make_plot(
        df["Downrange [km]"],
        df["Altitude [km]"],
        xlabel="Downrange [km]",
        ylabel="Altitude [km]",
        title="Altitude vs Downrange",
        fname="altitude_vs_downrange.png",
        outdir=OUTPUT_DIR
    )

    # 3) Velocity [m/s] vs Time [s]
    make_plot(
        df["Time [s]"],
        df["Velocity [m/s]"],
        xlabel="Time [s]",
        ylabel="Velocity [m/s]",
        title="Velocity vs Time",
        fname="velocity_vs_time.png",
        outdir=OUTPUT_DIR
    )

    # 4) Altitude [km] vs Velocity [m/s]
    make_plot(
        df["Velocity [m/s]"],
        df["Altitude [km]"],
        xlabel="Velocity [m/s]",
        ylabel="Altitude [km]",
        title="Altitude vs Velocity",
        fname="altitude_vs_velocity.png",
        outdir=OUTPUT_DIR
    )

    print(f"Plots saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
