import os
from analysis.parser import parse_all_metrics, parse_log_file
from analysis.plotter import plot_server_model_accuracy_over_rounds

def main():
    input_path = 'input/'
    log_files_path = os.path.join(input_path, 'logs')
    simulations_path = os.path.join(input_path, 'simulations')

    # Parse log files
    # log_files = [os.path.join(log_files_path, f) for f in os.listdir(log_files_path)]
    # log_params = [parse_log_file(f) for f in log_files]

    # Parse metrics files
    all_metrics = parse_all_metrics(simulations_path)
    
    for metrics in all_metrics:
        plot_server_model_accuracy_over_rounds(metrics)


if __name__ == '__main__':
    main()