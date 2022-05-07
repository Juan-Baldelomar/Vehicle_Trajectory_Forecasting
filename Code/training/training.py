import numpy as np
import tensorflow as tf

from Code.models.Model_traj import STTransformer
from Code.dataset.dataset import buildDataset
from Code.utils.save_utils import load_pkl_data


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


class HalveSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, n_batches):
        super(HalveSchedule, self).__init__()

        self.lr = lr
        self.n_batches = n_batches

    def __call__(self, step):
        if (step + 1) % (10 * 140) == 0:
            self.lr = self.lr / 2

        return self.lr


def train(filename, BATCH_SIZE=32, epochs=20, modelpath=None):
    feat_size, dk = 256, 256
    seq_size = 25
    neigh_size = 5
    n_heads = 4
    model = STTransformer(feat_size, seq_size, neigh_size,
                          sp_dk=dk, sp_enc_heads=n_heads, sp_dec_heads=n_heads,
                          tm_dk=dk, tm_enc_heads=n_heads, tm_dec_heads=n_heads,
                          sp_num_encoders=1, sp_num_decoders=1, tm_num_encoders=2, tm_num_decoders=2)
    if modelpath is not None:
        model.load_model()
    data = load_pkl_data(filename)
    dataset, std_x, std_y = buildDataset(data, BATCH_SIZE, pre_path='../data/maps/shifts/')
    stds = tf.constant([[[[std_x, std_y]]]], dtype=tf.float32)
    learning_rate = CustomSchedule(256)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.99, beta_2=0.9, epsilon=1e-9)
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
            model.save_model()

        print("avg loss", tf.reduce_mean(losses))


if __name__ == '__main__':
    import sys
    print(tf.keras.__version__)
    data_path = sys.argv[1]
    batch = int(sys.argv[2])
    epochs = int(sys.argv[3])
    train(data_path, batch, epochs)


