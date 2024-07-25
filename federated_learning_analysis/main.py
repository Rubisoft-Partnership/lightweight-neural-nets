import os
from analysis.parser import parse_all_simulations
from analysis.plotter import plot_server_model_metrics_over_rounds
import sys

def main():
    input_dir = 'input/'

    # Read input and output paths from the command line
    if len(sys.argv) <= 3:
        print('Usage: python main.py -i <input_path> -o <output_path>')
        sys.exit(1)
    else:
        for i in range(1, len(sys.argv), 2):
            if sys.argv[i] == '-i':
                input_dir = sys.argv[i + 1]
            elif sys.argv[i] == '-o':
                output_dir = sys.argv[i + 1]
    

    # Parse metrics files
    simulations = parse_all_simulations(input_dir)
    
    for sim in simulations:
        plot_server_model_metrics_over_rounds(sim, output_dir=output_dir)


if __name__ == '__main__':
    main()