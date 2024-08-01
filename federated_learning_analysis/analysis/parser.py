import os
import pandas as pd
import json

CONFIG_FILE = 'config.json'
METRICS_FILE = 'metrics.csv'

class Simulation:
    def __init__(self, simulation_id: int, simulation_path):
        self.simulation_id = simulation_id
        # Parse config file
        with open(os.path.join(simulation_path, CONFIG_FILE)) as f:
            self.config = json.load(f)
        # Parse metrics csv file
        with open(os.path.join(simulation_path, METRICS_FILE)) as f:
            df_metrics = pd.read_csv(f)
            self.metrics = self._create_metrics_matrix(df_metrics)

    def _create_metrics_matrix(self, metrics_df):
        metrics_matrix = []
        for _, row in metrics_df.iterrows():
            metrics_matrix.append({})
            for c in metrics_df.columns:
                metrics_matrix[-1][c] = row[c]
        return metrics_matrix


def parse_all_simulations(simulations_path):
    all_simulations = []
    for sim_folder in os.listdir(simulations_path):
        sim_path = os.path.join(simulations_path, sim_folder)
        if os.path.isdir(sim_path):
            try:
                simulation = Simulation(int(sim_folder), sim_path)
                all_simulations.append(simulation)
            except Exception as e:
                print(f'Error parsing simulation {sim_folder}: {e}')
    return all_simulations
