from analysis.parser import SimulationMetrics
import matplotlib.pyplot as plt
import os



def plot_server_model_accuracy_over_rounds(metrics: SimulationMetrics):
    output_dir = 'output/'
    accuracies = []
    for row in metrics.metrics:
        if row['client_id'] == -1:
            accuracies.append(row['accuracy'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.ylim(0, 1)
    plt.xticks(range(0, len(accuracies)))
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Server model accuracy over rounds')
    
    sim_dir = output_dir + metrics.simulation_id
    
    os.makedirs(sim_dir, exist_ok=True)
    
    plt.savefig(sim_dir + '/server_model_accuracy_over_rounds.png')
    