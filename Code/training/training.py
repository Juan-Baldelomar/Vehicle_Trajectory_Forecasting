import numpy as np
import tensorflow as tf

from Code.models.Model_traj import STTransformer
from Code.dataset.dataset import buildDataset
from Code.utils.save_utils import load_pkl_data, save_pkl_data


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


def load_optimizer(weights_path, config_path):
    conf = load_pkl_data(config_path)
    weights = load_pkl_data(weights_path)
    # load custom scheduler
    conf['learning_rate'] = CustomSchedule.from_config(conf['learning_rate']['config'])
    # load optimizer conf and weights
    optimizer = tf.keras.optimizers.Adam.from_config(conf)
    optimizer.set_weights(weights)
    return optimizer


def load_model_and_opt(modelpath=None, optimizer_weights_path=None, optimizer_config_path=None):
    # build model
    feat_size, dk = 256, 256
    seq_size = 25
    neigh_size = 5
    n_heads = 4
    model = STTransformer(feat_size,
                          seq_size,
                          neigh_size,
                          sp_dk=dk,
                          sp_enc_heads=n_heads,
                          sp_dec_heads=n_heads,
                          tm_dk=dk,
                          tm_enc_heads=n_heads,
                          tm_dec_heads=n_heads,
                          sp_num_encoders=1,
                          sp_num_decoders=1,
                          tm_num_encoders=2,
                          tm_num_decoders=2
                          )

    # load model if possible
    if modelpath is not None:
        model.set_weights(load_pkl_data(modelpath))
    # load optimizer if possible
    if optimizer_config_path is not None and optimizer_weights_path is not None:
        optimizer = load_optimizer(optimizer_weights_path, optimizer_config_path)
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


def train(filename, BATCH_SIZE=32, epochs=20, model_path=None, opt_weights_path=None, opt_conf_path=None):
    # load dataset
    data = load_pkl_data(filename)
    dataset, std_x, std_y = buildDataset(data, BATCH_SIZE, pre_path='Code/data/maps/shifts/')
    stds = tf.constant([[[[std_x, std_y]]]], dtype=tf.float32)
    model, optimizer = load_model_and_opt(model_path, opt_weights_path, opt_conf_path)

    worst_loss = np.inf
    for epoch in range(epochs):
        print('epoch: ', epoch)
        losses = []
        for (past, future, maps, _) in dataset:
            losses, loss = model.train_step(past, future, maps, stds, losses, optimizer)
            if np.isnan(loss.numpy()):
                break;
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
    print(tf.keras.__version__)
    data_path = sys.argv[1]
    batch = int(sys.argv[2])
    epochs = int(sys.argv[3])
    train(data_path, batch, epochs)


