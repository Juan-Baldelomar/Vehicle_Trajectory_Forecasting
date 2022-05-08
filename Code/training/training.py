import numpy as np
import tensorflow as tf

from Code.models.Model_traj import STTransformer
from Code.dataset.dataset import buildDataset
from Code.utils.save_utils import load_pkl_data, save_pkl_data, valid_paths, load_parameters


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {'d_model': self.d_model, 'w_steps': self.warmup_steps}

    def from_config(config, custom_object=None):
        d_model = config['d_model']
        warmup_steps = config['w_steps']
        return CustomSchedule(d_model, warmup_steps)


class HalveSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, n_batches):
        super(HalveSchedule, self).__init__()

        self.lr = lr
        self.n_batches = n_batches

    def __call__(self, step):
        if (step + 1) % (10 * 140) == 0:
            self.lr = self.lr / 2

        return self.lr


def save_optimizer(optimizer, weights_path, config_path):
    save_pkl_data(optimizer.get_weights(), weights_path, 4)
    save_pkl_data(optimizer.get_config(), config_path, 4)


def load_optimizer(weights_path, config_path, model, inputs, stds):
    conf = load_pkl_data(config_path)
    weights = load_pkl_data(weights_path)
    # load custom scheduler
    conf['learning_rate'] = CustomSchedule.from_config(conf['learning_rate']['config'])
    # load optimizer conf and weights
    optimizer = tf.keras.optimizers.Adam.from_config(conf)
    model.train_step(*inputs, stds, [], optimizer)
    optimizer.set_weights(weights)
    return optimizer


def split_params(params):
    # validate all needed params are present
    if params is None:
        raise RuntimeError('[ERR] params is None. Parameters should be loaded from params file')
    if params.get('features_size') is None or params.get('seq_size') is None or params.get('neigh_size') is None:
        raise RuntimeError('[ERR] parameters file should contain basic model params (feat_size, seq_size, neigh_size)')
    if params.get('batch') is None or params.get('epochs') is None:
        raise RuntimeError('[ERR] parameters file should contain basic training params (batch, epochs)')
    if params.get('data_path') is None or params.get('maps_dir') is None:
        raise RuntimeError('[ERR] parameters file should contain basic data params (data_path, maps_dir)')

    # get params values
    batch = params['batch']
    epochs = params['epochs']
    model_params = {
        'features_size': params['features_size'],
        'seq_size': params['seq_size'],
        'neigh_size': params['neigh_size'],
        'sp_dk': params.get('sp_dk', 256),
        'sp_enc_heads': params.get('sp_enc_heads', 4),
        'sp_dec_heads': params.get('sp_dec_heads', 4),
        'tm_dk': params.get('tm_dk', 256),
        'tm_enc_heads': params.get('sp_enc_heads', 4),
        'tm_dec_heads': params.get('sp_dec_heads', 4),
        'sp_num_encoders': params.get('sp_num_encoders', 4),
        'sp_num_decoders': params.get('sp_num_decoders', 4),
        'tm_num_encoders': params.get('tm_num_encoders', 4),
        'tm_num_decoders': params.get('tm_num_decoders', 4)
    }
    preload_params = {
        'preload': params.get('preload', False),
        'model_path': params.get('model_path') ,
        'opt_weights_path': params.get('opt_weights_path'),
        'opt_conf_path': params.get('opt_conf_path')
    }

    data_params = {
        'data_path': params['data_path'],
        'maps_dir': params['maps_dir']
    }

    return model_params, batch, epochs, preload_params, data_params


def load_model_and_opt(model_params, dataset, stds, preload=False, model_path=None, opt_weights_path=None, opt_config_path=None):
    # build model
    model = STTransformer(**model_params)
    # load model if possible
    if preload:
        if model_path is not None and valid_paths(model_path):
            past, future, maps, _ = next(iter(dataset)) 
            model((past, future, maps), False, stds)
            model.set_weights(load_pkl_data(model_path))
        # load optimizer if possible
        if opt_config_path is not None and opt_weights_path is not None \
                and valid_paths(opt_config_path, opt_weights_path):
            optimizer = load_optimizer(opt_weights_path, opt_config_path, model, (past, future, maps), stds)
    else:
        learning_rate = CustomSchedule(256)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.99, beta_2=0.9, epsilon=1e-9)

    return model, optimizer


def save_state(model, optimizer, model_path, opt_weight_path, opt_conf_path):
    if model_path is None:
        model_path = 'Code/weights/best_ModelTraj_weights.pkl'
    if opt_weight_path is None:
        opt_weight_path = 'Code/weights/best_opt_weight.pkl'
    if opt_conf_path is None:
        opt_conf_path = 'Code/config/best_opt_conf.pkl'

    save_pkl_data(model.get_weights(), model_path)
    save_optimizer(optimizer, opt_weight_path, opt_conf_path)


def train(model_params, batch, epochs, data_path, maps_dir, preload=False, model_path=None, opt_weights_path=None, opt_conf_path=None):
    # load dataset
    data = load_pkl_data(data_path)
    dataset, std_x, std_y = buildDataset(data, batch, pre_path=maps_dir)
    stds = tf.constant([[[[std_x, std_y]]]], dtype=tf.float32)
    model, optimizer = load_model_and_opt(model_params, dataset, stds, preload, model_path, opt_weights_path, opt_conf_path)

    worst_loss = np.inf
    for epoch in range(epochs):
        print('epoch: ', epoch)
        losses = []
        for (past, future, maps, _) in dataset:
            losses, loss = model.train_step(past, future, maps, stds, losses, optimizer)
            if np.isnan(loss.numpy()):
                break
        # l_ade = []
        # for batch in dataset:
        #  ade = eval_step(batch)
        #  l_ade.append(ade)
        # print('ade: ', np.mean(np.array(l_ade)))
        avg_loss = tf.reduce_mean(losses)
        if avg_loss.numpy() < worst_loss:
            worst_loss = avg_loss.numpy()
            save_state(model,
                       optimizer,
                       model_path=model_path,
                       opt_weight_path=opt_weights_path,
                       opt_conf_path=opt_conf_path
                       )

        print("avg loss", tf.reduce_mean(losses))


if __name__ == '__main__':
    import sys
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path + '/../..')
    print(os.getcwd())
    params_path = sys.argv[1]
    params = load_parameters(params_path)
    model_params, batch, epochs, preload_params, data_params = split_params(params)
    train(model_params, batch, epochs, **data_params, **preload_params)


