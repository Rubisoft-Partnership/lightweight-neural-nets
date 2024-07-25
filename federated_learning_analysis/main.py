import os
from analysis.parser import parse_all_simulations
from analysis.plotter import plot_server_model_metrics_over_rounds

def main():
    simulations_path = 'input/'

    # Parse metrics files
    simulations = parse_all_simulations(simulations_path)
    
    for sim in simulations:
        plot_server_model_metrics_over_rounds(sim)


if __name__ == '__main__':
    main()