from analysis.parser import Simulation
import matplotlib.pyplot as plt
import os

def save_plot(data, title, ylabel, filename, y_limit=None, caption=None):
    plt.figure(figsize=(10, 8))
    plt.plot(data)
    plt.xticks(range(0, len(data), max(1, len(data) // 15)))
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if y_limit:
        plt.ylim(y_limit)
    if caption:
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory

def plot_server_model_metrics_over_rounds(simulation: Simulation, output_dir = 'output/', caption=None):
    sim_dir = os.path.join(output_dir, str(simulation.simulation_id))
    os.makedirs(sim_dir, exist_ok=True)

    # Initialize lists to store the metrics
    accuracies, losses, recalls, precisions, f1s = [], [], [], [], []

    # Collect data from the metrics
    for row in simulation.metrics:
        if row['client_id'] == -1:
            accuracies.append(row['accuracy'])
            losses.append(float(row['loss']))
            recalls.append(row['average_recall'])
            precisions.append(row['average_precision'])
            f1s.append(row['average_f1_score'])
    
    if caption is None:        
        caption = f"Model {simulation.config['model_type']}"
        if simulation.config['model_type'] == 'FF':
            caption += f" with {simulation.config['parameters']['ff']['loss']} loss"
        caption += f", {simulation.config['selected_dataset'].replace('/', '')}"
        caption += f", {simulation.config['orchestration']['num_clients']} clients, {simulation.config['orchestration']['num_rounds']} rounds"
        caption += f", Units: {simulation.config['parameters']['units']}"
        caption += f"\nη: {round(simulation.config['training']['learning_rate'], 3)}, Batch: {simulation.config['training']['batch_size']}, epochs: {simulation.config['training']['epochs']}"
        
       

    # Plot each metric
    if accuracies:
        save_plot(accuracies, 'Server model accuracy over rounds', 'Accuracy', os.path.join(sim_dir, 'server_model_accuracy_over_rounds.png'), y_limit=(0, 1), caption=caption)
    if losses:
        save_plot(losses, 'Server model loss over rounds', 'Loss', os.path.join(sim_dir, 'server_model_loss_over_rounds.png'), y_limit=(0, max(losses) + 1), caption=caption)
    if recalls:
        save_plot(recalls, 'Server model recall over rounds', 'Recall', os.path.join(sim_dir, 'server_model_recall_over_rounds.png'), y_limit=(0, 1), caption=caption)
    if precisions:
        save_plot(precisions, 'Server model precision over rounds', 'Precision', os.path.join(sim_dir, 'server_model_precision_over_rounds.png'), y_limit=(0, 1), caption=caption)
    if f1s:
        save_plot(f1s, 'Server model F1 score over rounds', 'F1 score', os.path.join(sim_dir, 'server_model_f1_over_rounds.png'), y_limit=(0, 1), caption=caption)

def plot_average_global_metrics_over_rounds(simulation: Simulation, output_dir = 'output/', caption=None):
    sim_dir = os.path.join(output_dir, str(simulation.simulation_id))
    os.makedirs(sim_dir, exist_ok=True)

    # Initialize lists to store the metrics
    accuracies, losses, recalls, precisions, f1s = [], [], [], [], []

    # Collect data from the metrics
    for row in simulation.metrics:
        if row['client_id'] == -2:
            accuracies.append(row['accuracy'])
            losses.append(float(row['loss']))
            recalls.append(row['average_recall'])
            precisions.append(row['average_precision'])
            f1s.append(row['average_f1_score'])
    
    if caption is None:        
        caption = f"Model {simulation.config['model_type']}"
        if simulation.config['model_type'] == 'FF':
            caption += f" with {simulation.config['parameters']['ff']['loss']} loss"
        caption += f", {simulation.config['selected_dataset'].replace('/', '')}"
        caption += f", {simulation.config['orchestration']['num_clients']} clients, {simulation.config['orchestration']['num_rounds']} rounds"
        caption += f", Units: {simulation.config['parameters']['units']}"
        caption += f"\nη: {round(simulation.config['training']['learning_rate'], 3)}, Batch: {simulation.config['training']['batch_size']}, epochs: {simulation.config['training']['epochs']}"
        
       
        
    

    # Plot each metric
    if accuracies:
        save_plot(accuracies, 'Global accuracy over rounds', 'Accuracy', os.path.join(sim_dir, 'global_accuracy_over_rounds.png'), y_limit=(0, 1), caption=caption)
    if losses:
        save_plot(losses, 'Global loss over rounds', 'Loss', os.path.join(sim_dir, 'global_loss_over_rounds.png'), y_limit=(0, max(losses) + 1), caption=caption)
    if recalls:
        save_plot(recalls, 'Global recall over rounds', 'Recall', os.path.join(sim_dir, 'global_recall_over_rounds.png'), y_limit=(0, 1), caption=caption)
    if precisions:
        save_plot(precisions, 'Global precision over rounds', 'Precision', os.path.join(sim_dir, 'global_precision_over_rounds.png'), y_limit=(0, 1), caption=caption)
    if f1s:
        save_plot(f1s, 'Global F1 score over rounds', 'F1 score', os.path.join(sim_dir, 'global_f1_over_rounds.png'), y_limit=(0, 1), caption=caption)