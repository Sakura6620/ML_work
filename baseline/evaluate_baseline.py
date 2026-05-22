import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fer2013 import get_dataloaders
from models.vgg_baseline import VggBaseline
from utils.hparams import setup_hparams
from utils.checkpoint import restore_checkpoint
from utils.logger import Logger


def evaluate(model, dataset, criterion):
    total_loss, n_samples = 0.0, 0
    correct_count1, correct_count2 = 0, 0
    y_pred, y_gt = [], []

    for images, labels in dataset:
        if len(images.shape) == 5:
            bs, ncrops, h, w, c = images.shape
            images_flat = tf.reshape(images, [-1, h, w, c])
            outputs = model(images_flat, training=False)
            outputs = tf.reshape(outputs, [bs, ncrops, -1])
            outputs = tf.reduce_mean(outputs, axis=1)
        else:
            outputs = model(images, training=False)

        loss = criterion(labels, outputs)
        total_loss += loss.numpy() * len(labels)

        top2 = tf.math.top_k(outputs, k=2).indices
        labels_exp = tf.expand_dims(labels, 1)
        correct_count1 += tf.reduce_sum(
            tf.cast(top2[:, :1] == tf.cast(labels_exp, tf.int32), tf.int32)).numpy()
        correct_count2 += tf.reduce_sum(
            tf.cast(tf.reduce_any(top2 == tf.cast(labels_exp, tf.int32), axis=1), tf.int32)).numpy()

        preds = tf.argmax(outputs, axis=1).numpy()
        y_pred.extend(preds.tolist())
        y_gt.extend(labels.numpy().tolist())
        n_samples += len(labels)

    acc1 = 100.0 * correct_count1 / n_samples
    acc2 = 100.0 * correct_count2 / n_samples
    loss = total_loss / n_samples

    print("--------------------------------------------------------")
    print(f"Top 1 Accuracy: {acc1:.6f}%")
    print(f"Top 2 Accuracy: {acc2:.6f}%")
    print(f"Loss: {loss:.6f}")
    print(f"Precision: {precision_score(y_gt, y_pred, average='micro'):.6f}")
    print(f"Recall: {recall_score(y_gt, y_pred, average='micro'):.6f}")
    print(f"F1 Score: {f1_score(y_gt, y_pred, average='micro'):.6f}")
    print("Confusion Matrix:\n", confusion_matrix(y_gt, y_pred), '\n')


if __name__ == "__main__":
    hps = setup_hparams(sys.argv[1:]) if len(sys.argv) > 1 else setup_hparams(['network=vgg_baseline', 'name=baseline'])

    model = VggBaseline()
    logger = Logger()
    if hps['restore_epoch']:
        restore_checkpoint(model, logger, hps)

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_ds, val_ds, test_ds = get_dataloaders(augment=False)

    print("[Baseline] Train")
    evaluate(model, train_ds, criterion)

    print("[Baseline] Val")
    evaluate(model, val_ds, criterion)

    print("[Baseline] Test")
    evaluate(model, test_ds, criterion)
