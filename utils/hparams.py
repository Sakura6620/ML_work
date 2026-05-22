import os

hps = {
    'network': '',
    'name': '',
    'n_epochs': 300,
    'model_save_dir': None,
    'restore_epoch': None,
    'start_epoch': 0,
    'lr': 0.01,
    'save_freq': 20,
    'drop': 0.1,
    'bs': 64,
}

possible_nets = {'vgg', 'vgg_baseline'}


def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    if hps['network'] not in possible_nets:
        raise ValueError("Invalid network.\nPossible ones include:\n - " + '\n - '.join(possible_nets))

    hps['n_epochs'] = int(hps['n_epochs'])
    hps['start_epoch'] = int(hps['start_epoch'])
    hps['save_freq'] = int(hps['save_freq'])
    hps['lr'] = float(hps['lr'])
    hps['drop'] = float(hps['drop'])
    hps['bs'] = int(hps['bs'])

    if hps['restore_epoch']:
        hps['restore_epoch'] = int(hps['restore_epoch'])
        hps['start_epoch'] = int(hps['restore_epoch'])

    if hps['n_epochs'] < 20:
        hps['save_freq'] = min(5, hps['n_epochs'])

    hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])
    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    return hps
