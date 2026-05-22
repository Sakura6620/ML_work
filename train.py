import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data.fer2013 import get_dataloaders
from models.vgg import Vgg
from utils.hparams import setup_hparams
from utils.setup_network import setup_network
from utils.logger import Logger
from utils.checkpoint import save_checkpoint

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'improve')


def train_step(model, images, labels, criterion, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = criterion(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions


def evaluate_step(model, images, labels, criterion):
    if len(images.shape) == 5:
        bs, ncrops, h, w, c = images.shape
        images_flat = tf.reshape(images, [-1, h, w, c])
        outputs = model(images_flat, training=False)
        outputs = tf.reshape(outputs, [bs, ncrops, -1])
        outputs = tf.reduce_mean(outputs, axis=1)
    else:
        outputs = model(images, training=False)
    loss = criterion(labels, outputs)
    return loss, outputs


def train_epoch(model, train_ds, criterion, optimizer):
    total_loss, correct, n_samples = 0.0, 0, 0
    for images, labels in train_ds:
        bs, ncrops, h, w, c = images.shape
        images = tf.reshape(images, [-1, h, w, c])
        labels = tf.repeat(labels, repeats=ncrops)
        loss, predictions = train_step(model, images, labels, criterion, optimizer)
        total_loss += loss.numpy() * len(labels)
        preds = tf.argmax(predictions, axis=1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
        n_samples += len(labels)
    return 100.0 * correct / n_samples, total_loss / n_samples


def eval_epoch(model, val_ds, criterion):
    total_loss, correct, n_samples = 0.0, 0, 0
    y_pred, y_gt = [], []
    for images, labels in val_ds:
        loss, outputs = evaluate_step(model, images, labels, criterion)
        total_loss += loss.numpy() * len(labels)
        preds = tf.argmax(outputs, axis=1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
        n_samples += len(labels)
        y_pred.extend(preds.numpy().tolist())
        y_gt.extend(labels.numpy().tolist())
    acc = 100.0 * correct / n_samples
    avg_loss = total_loss / n_samples
    return acc, avg_loss, y_pred, y_gt


def save_results(logger, y_pred, y_gt):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame({
        'epoch': list(range(1, len(logger.acc_train) + 1)),
        'train_acc': logger.acc_train,
        'val_acc': logger.acc_val,
        'train_loss': logger.loss_train,
        'val_loss': logger.loss_val,
    })
    df.to_csv(os.path.join(OUTPUT_DIR, 'training_log.csv'), index=False)

    plt.figure()
    plt.plot(logger.acc_train, 'g', label='Train Acc')
    plt.plot(logger.acc_val, 'b', label='Val Acc')
    plt.title('Improved Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(logger.loss_train, 'g', label='Train Loss')
    plt.plot(logger.loss_val, 'b', label='Val Loss')
    plt.title('Improved Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss.png'))
    plt.close()

    labels_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cm = confusion_matrix(y_gt, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Improved Model - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels_name))
    plt.xticks(tick_marks, labels_name, rotation=45)
    plt.yticks(tick_marks, labels_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    cm_df = pd.DataFrame(cm, index=labels_name, columns=labels_name)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'))


def run(model, logger, hps):
    train_ds, val_ds, test_ds = get_dataloaders(bs=hps['bs'], augment=True)

    learning_rate = float(hps['lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    best_acc = 0.0
    lr_patience_counter = 0
    lr_patience = 5
    factor = 0.75
    early_stop_patience = 10
    no_improve_count = 0

    print(f"[Improve] Training {hps['name']} on", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

    for epoch in range(hps['start_epoch'], hps['n_epochs']):
        acc_tr, loss_tr = train_epoch(model, train_ds, criterion, optimizer)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v, y_pred, y_gt = eval_epoch(model, val_ds, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        if acc_v > best_acc + 0.1:
            best_acc = acc_v
            lr_patience_counter = 0
            no_improve_count = 0
            save_checkpoint(model, logger, hps, epoch + 1)
        else:
            no_improve_count += 1
            lr_patience_counter += 1
            if lr_patience_counter >= lr_patience:
                new_lr = optimizer.learning_rate.numpy() * factor
                optimizer.learning_rate.assign(new_lr)
                lr_patience_counter = 0
                print(f"  Reducing LR to {new_lr:.6f}")

        print(f"Epoch {epoch+1:3d}\t\tTrain Acc: {acc_tr:.4f}%\t\tVal Acc: {acc_v:.4f}%")

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}, best val acc: {best_acc:.4f}%")
            break

    _, _, y_pred_test, y_gt_test = eval_epoch(model, test_ds, criterion)
    save_results(logger, y_pred_test, y_gt_test)
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    hps = setup_hparams(sys.argv[1:])
    logger, model = setup_network(hps)
    run(model, logger, hps)
