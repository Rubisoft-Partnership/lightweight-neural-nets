from analysis.parser import SimulationMetrics
import matplotlib.pyplot as plt
import os


def plot_server_model_metrics_over_rounds(metrics: SimulationMetrics):
    output_dir = 'output/'
    accuracies, losses, recalls, precisions, f1s = [], [], [], [], []
    for row in metrics.metrics:
        if row['client_id'] == -1:
            accuracies.append(row['accuracy'])
            losses.append(float(row['loss']))
            recalls.append(row['average_recall'])
            precisions.append(row['average_precision'])
            f1s.append(row['average_f1_score'])

    plt.figure(figsize=(10, 6))

    sim_dir = output_dir + metrics.simulation_id
    os.makedirs(sim_dir, exist_ok=True)

    plt.plot(accuracies)
    plt.xticks(range(0, len(accuracies)))
    plt.xlabel('Round')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Server model accuracy over rounds')
    plt.savefig(sim_dir + '/server_model_accuracy_over_rounds.png')

    plt.cla()

    plt.plot(losses)
    plt.xticks(range(0, len(losses)))
    plt.xlabel('Round')
    plt.ylim(0, int(1 + max(losses)))
    plt.ylabel('Loss')
    plt.title('Server model loss over rounds')
    plt.savefig(sim_dir + '/server_model_loss_over_rounds.png')

    plt.cla()

    plt.plot(recalls)
    plt.xticks(range(0, len(recalls)))
    plt.xlabel('Round')
    plt.ylim(0, 1)
    plt.ylabel('Recall')
    plt.title('Server model recall over rounds')
    plt.savefig(sim_dir + '/server_model_recall_over_rounds.png')

    plt.cla()

    plt.plot(precisions)
    plt.xticks(range(0, len(precisions)))
    plt.xlabel('Round')
    plt.ylim(0, 1)
    plt.ylabel('Precision')
    plt.title('Server model precision over rounds')
    plt.savefig(sim_dir + '/server_model_precision_over_rounds.png')

    plt.cla()

    plt.plot(f1s)
    plt.xticks(range(0, len(f1s)))
    plt.xlabel('Round')
    plt.ylim(0, 1)
    plt.ylabel('F1 score')
    plt.title('Server model F1 score over rounds')
    plt.savefig(sim_dir + '/server_model_f1_over_rounds.png')
