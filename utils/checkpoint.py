import os
import json
import numpy as np
import tensorflow as tf


def save_checkpoint(model, logger, hps, epoch):
    path = os.path.join(hps['model_save_dir'], f'epoch_{epoch}')
    model.save_weights(path)

    logs_path = os.path.join(hps['model_save_dir'], f'epoch_{epoch}_logs.json')
    logs = {
        'loss_train': logger.loss_train,
        'loss_val': logger.loss_val,
        'acc_train': logger.acc_train,
        'acc_val': logger.acc_val,
    }
    with open(logs_path, 'w') as f:
        json.dump(logs, f)


def restore_checkpoint(model, logger, hps):
    path = os.path.join(hps['model_save_dir'], f'epoch_{hps["restore_epoch"]}')
    logs_path = os.path.join(hps['model_save_dir'], f'epoch_{hps["restore_epoch"]}_logs.json')

    if os.path.exists(path + '.index'):
        try:
            model.load_weights(path)
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    logs = json.load(f)
                logger.restore_logs((
                    logs['loss_train'], logs['loss_val'],
                    logs['acc_train'], logs['acc_val']))
            print("Network Restored!")
        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            hps['start_epoch'] = 0
    else:
        print("Restore point unavailable. Training from scratch.")
        hps['start_epoch'] = 0
