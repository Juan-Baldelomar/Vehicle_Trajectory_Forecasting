import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=500):
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

    @staticmethod
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

    def get_config(self):
        return {'lr': self.lr, 'n_batches': self.n_batches}

    @staticmethod
    def from_config(config, custom_object=None):
        lr = config['lr']
        n_bathces = config['n_batches']
        return HalveSchedule(lr, n_bathces)
