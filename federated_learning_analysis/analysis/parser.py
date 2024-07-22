import os
import pandas as pd

class SimulationMetrics:
    def __init__(self, simulation_id, metrics_df):
        self.simulation_id = simulation_id
        self.metrics = self._create_metrics_matrix(metrics_df)
    
    def _create_metrics_matrix(self, metrics_df):
        metrics_matrix = []
        for _, row in metrics_df.iterrows():
            metrics_matrix.append({})
            for c in metrics_df.columns:
                metrics_matrix[-1][c] = row[c]
                
                    
        return metrics_matrix

def parse_log_file(filepath):
    params = {}
    # TODO: Implement log file parameters parsing
    return params

def parse_metrics_file(filepath):
    return pd.read_csv(filepath)

def parse_all_metrics(simulations_path):
    all_simulations = []
    for sim_folder in os.listdir(simulations_path):
        sim_path = os.path.join(simulations_path, sim_folder)
        if os.path.isdir(sim_path):
            metrics_file = os.path.join(sim_path, 'metrics.csv')
            if os.path.exists(metrics_file):
                try:
                    sim_metrics_df = parse_metrics_file(metrics_file)
                    simulation = SimulationMetrics(simulation_id=sim_folder, metrics_df=sim_metrics_df)
                    if simulation.metrics == []:
                        continue
                    all_simulations.append(simulation)
                except Exception as e:
                    print(f'Error parsing metrics file of simulation {sim_folder}: {e}')
    return all_simulations

if __name__ == '__main__':
    metrics = parse_all_metrics('../input/simulations')
