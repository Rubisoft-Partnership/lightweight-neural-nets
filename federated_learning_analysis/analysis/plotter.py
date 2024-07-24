from analysis.parser import SimulationMetrics
import matplotlib.pyplot as plt
import os

def save_plot(data, title, ylabel, filename, y_limit=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xticks(range(0, len(data), max(1, len(data) // 15)))
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if y_limit:
        plt.ylim(y_limit)
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory

def plot_server_model_metrics_over_rounds(metrics: SimulationMetrics):
    output_dir = 'output/'
    sim_dir = os.path.join(output_dir, metrics.simulation_id)
    os.makedirs(sim_dir, exist_ok=True)

    # Initialize lists to store the metrics
    accuracies, losses, recalls, precisions, f1s = [], [], [], [], []

    # Collect data from the metrics
    for row in metrics.metrics:
        if row['client_id'] == -1:
            accuracies.append(row['accuracy'])
            losses.append(float(row['loss']))
            recalls.append(row['average_recall'])
            precisions.append(row['average_precision'])
            f1s.append(row['average_f1_score'])

    # Plot each metric
    if accuracies:
        save_plot(accuracies, 'Server model accuracy over rounds', 'Accuracy', os.path.join(sim_dir, 'server_model_accuracy_over_rounds.png'), y_limit=(0, 1))
    if losses:
        save_plot(losses, 'Server model loss over rounds', 'Loss', os.path.join(sim_dir, 'server_model_loss_over_rounds.png'), y_limit=(0, max(losses) + 1))
    if recalls:
        save_plot(recalls, 'Server model recall over rounds', 'Recall', os.path.join(sim_dir, 'server_model_recall_over_rounds.png'), y_limit=(0, 1))
    if precisions:
        save_plot(precisions, 'Server model precision over rounds', 'Precision', os.path.join(sim_dir, 'server_model_precision_over_rounds.png'), y_limit=(0, 1))
    if f1s:
        save_plot(f1s, 'Server model F1 score over rounds', 'F1 score', os.path.join(sim_dir, 'server_model_f1_over_rounds.png'), y_limit=(0, 1))

