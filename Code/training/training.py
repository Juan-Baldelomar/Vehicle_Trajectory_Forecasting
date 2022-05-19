# scientific libraries
import numpy as np
import tensorflow as tf
#utils
import datetime
import time
# own libraries
from Code.models.Model_traj import STTransformer
from Code.models.AgentFormer import STE_Transformer
from Code.dataset.dataset import buildDataset
from Code.utils.save_utils import load_pkl_data, save_pkl_data, valid_file, valid_path, load_parameters


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


def load_optimizer(weights_path, config_path, model, inputs, strategy, lr=None):
    conf = load_pkl_data(config_path)
    weights = load_pkl_data(weights_path)
    # load custom scheduler
    #conf['learning_rate'] = CustomSchedule.from_config(conf['learning_rate']['config'])
    # load optimizer conf and weights
    if lr is not None:
    	conf['learning_rate'] = lr
    optimizer = tf.keras.optimizers.Adam.from_config(conf)
    # perform train_step to init weights
    model.set_optimizer(optimizer)
    #model.train_step(inputs)
    per_replica_losses = strategy.run(model.train_step, args=(inputs,))
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
    lr = params.get('lr', 0.00001)
    
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
        'model_path': params.get('model_path'),
        'opt_weights_path': params.get('opt_weights_path'),
        'opt_conf_path': params.get('opt_conf_path')
    }

    data_params = {
        'data_path': params['data_path'],
        'maps_dir': params['maps_dir']
    }
    logs_dir = params.get('logs_dir')
    return model_params, batch, epochs, lr, preload_params, data_params, logs_dir


def load_model_and_opt(model_params, lr, dataset, stds, dk, strategy, preload=False, model_path=None, opt_weights_path=None, opt_conf_path=None):
    # build model
    model = STE_Transformer(**model_params)
    learning_rate = CustomSchedule(dk)
    optimizer = tf.keras.optimizers.Adam(0.00001, beta_1=0.99, beta_2=0.9, epsilon=1e-9)
    # load model if desired
    if preload:
        # verify model_path is valid file
        if model_path is not None:
            valid_file(model_path)
            # data to call train_step to init weights
            past, future, maps, _ = next(iter(dataset))
            if opt_conf_path is not None and opt_weights_path is not None:
                # load optimizer
                valid_file(opt_conf_path, opt_weights_path)
                optimizer = load_optimizer(opt_weights_path, opt_conf_path, model, (past, future, maps, stds), strategy, lr)
            else:
                # loading optimizer was not possible, perform model.__call__ to init weights
                model((past, future, maps), False, stds)
            # reload model weights
            model.set_weights(load_pkl_data(model_path))

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


def get_logger(logs_dir):
    if logs_dir is None:
        return None

    valid_path(logs_dir)
    summary_writer = tf.summary.create_file_writer(
        logs_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return summary_writer


def eval_model(model, dataset, stds):
    l_ade = []
    for (past, future, maps, _) in dataset:
        ade = model.eval_step(past, future, maps, stds)
        l_ade.append(ade)
        print('ade: ', ade)
    print('mean ade: ', np.mean(np.array(l_ade)))


@tf.function
def distributed_step(inputs, step_fn):
    per_replica_losses = strategy.run(step_fn, args=(inputs,))
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return loss
    

def train(model, epochs, model_path, opt_weights_path, opt_conf_path, logs_dir=None, strategy=None):
    # avoid creating summary writer
    if epochs == 0:
        return
        
    # loggin writer
    summary_writer = get_logger(logs_dir)
    # start training
    worst_loss = 21.08453
    for epoch in range(epochs):
        print('epoch: ', epoch)
        start = time.time()
        losses = []
        for (past, future, maps, _) in dataset:
            #loss = model.train_step([past, future, maps, stds])
            #per_replica_losses = strategy.run(model.train_step, args=([past, future, maps, stds],))
            #loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            loss = distributed_step([past, future, maps, stds], model.train_step)
            losses.append(loss)
            if np.isnan(loss.numpy()):
                break
        avg_loss = tf.reduce_mean(losses)
        if avg_loss.numpy() < worst_loss:
            worst_loss = avg_loss.numpy()
            save_state(
                model,
                model.optimizer,
                model_path=model_path,
                opt_weight_path=opt_weights_path,
                opt_conf_path=opt_conf_path
            )
        
        end = time.time()
        print('TIME ELAPSED:', datetime.timedelta(seconds=end - start))
        print("avg loss", avg_loss, flush=True)
        # log resutls if desired
        if summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)
                
                

if __name__ == '__main__':
    import sys
    import os
    # change working directory
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path + '/../..')
    print(os.getcwd())
    params_path = sys.argv[1]
    # load parameters
    params = load_parameters(params_path)
    model_params, batch, epochs, lr, preload_params, data_params, logs_dir = split_params(params)
    model_path = preload_params['model_path']
    opt_weights_path = preload_params['opt_weights_path']
    opt_conf_path = preload_params['opt_conf_path']
    dk = model_params['sp_dk']
    # get dataset
    data = load_pkl_data(data_params['data_path'])
    strategy = tf.distribute.MirroredStrategy()
    #strategy = None
    dataset, std_x, std_y = buildDataset(data, batch, pre_path=data_params['maps_dir'], strategy=strategy)
    eval_dataset, _, _ = buildDataset(data, batch, pre_path=data_params['maps_dir'], strategy=None)
    
    with strategy.scope():
        stds = tf.constant([[[[std_x, std_y]]]], dtype=tf.float32)
        # get model
        model, optimizer = load_model_and_opt(model_params, lr, dataset, stds, dk, strategy, **preload_params)
        model.set_optimizer(optimizer)
        # train model
        train(model, epochs, model_path, opt_weights_path, opt_conf_path, logs_dir, strategy)
    
    # eval model
    eval_model(model, eval_dataset, stds)



