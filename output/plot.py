import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(csv_path, output_dir, title_prefix):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df['train_acc'], 'g', label='Train Acc')
    plt.plot(df['val_acc'], 'b', label='Val Acc')
    plt.title(f'{title_prefix} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df['train_loss'], 'g', label='Train Loss')
    plt.plot(df['val_loss'], 'b', label='Val Loss')
    plt.title(f'{title_prefix} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=150)
    plt.close()
    print(f'  accuracy.png, loss.png saved to {output_dir}')


def plot_confusion_matrix(csv_path, output_dir, title_prefix):
    labels_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cm_df = pd.read_csv(csv_path, index_col=0)
    cm = cm_df.values

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title_prefix} - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels_name))
    plt.xticks(tick_marks, labels_name, rotation=45)
    plt.yticks(tick_marks, labels_name)
    for i in range(len(labels_name)):
        for j in range(len(labels_name)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'  confusion_matrix.png saved to {output_dir}')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for folder, title in [('baseline', 'Baseline'), ('improve', 'Improved Model')]:
        output_dir = os.path.join(script_dir, folder)
        log_path = os.path.join(output_dir, 'training_log.csv')
        cm_path = os.path.join(output_dir, 'confusion_matrix.csv')

        print(f'[{title}]')
        if os.path.exists(log_path):
            plot_training_curves(log_path, output_dir, title)
        else:
            print(f'  training_log.csv not found, skipping curves')

        if os.path.exists(cm_path):
            plot_confusion_matrix(cm_path, output_dir, title)
        else:
            print(f'  confusion_matrix.csv not found, skipping matrix')
