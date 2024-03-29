# scientific libraries
import numpy as np
import tensorflow as tf

# utils
import os
import datetime
import time

# own libraries
# from Code.models.Model_traj import STTransformer
from Code.models.VAE_ModelTraj import STTransformer
# from Code.models.RNN_Transformer import STTransformer
from Code.models.AgentFormer import STE_Transformer
from Code.dataset.dataset import buildDataset
from Code.utils.save_utils import load_pkl_data, save_pkl_data, valid_file, valid_path, load_parameters
from Code.eval.quantitative_eval import ADE, FDE
from Code.eval.qualitative_eval import stamp_traj


def get_eval_filename(filename):
    # get parent directory and basename
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    # new basename
    basename, ext = basename.split('.')
    basename = basename + '_best_eval.' + ext
    return '/'.join([directory, basename])


def split_params(params, model_class):
    # validate all needed params are present
    if params is None:
        raise RuntimeError('[ERR] params is None. Parameters should be loaded from params file')
    if params.get('batch') is None or params.get('epochs') is None:
        raise RuntimeError('[ERR] parameters file should contain basic training params (batch, epochs)')
    if params.get('data_path') is None or params.get('maps_dir') is None:
        raise RuntimeError('[ERR] parameters file should contain basic data params (data_path, maps_dir)')

    # get training params
    training_params = {
        'epochs': params['epochs'],
        'lr': params.get('lr', 0.00001)
    }

    # get model params (should be static method)
    model_params = model_class.get_model_params(params)

    # get optimizer params
    optim_params = {
        'lr': params.get('lr', 1e-5),
        'beta_1': params.get('beta_1', 0.99),
        'beta_2': params.get('beta_2', 0.9),
        'epsilon': params.get('epsilon', 1e-9)
    }

    # get preload weights params
    preload_params = {
        'preload': params.get('preload', False),
        'model_path': params.get('model_path'),
        'opt_weights_path': params.get('opt_weights_path'),
        'opt_conf_path': params.get('opt_conf_path'),
        'transfer_learning': params.get('transfer_learning', False)
    }

    # update preload_params with different save paths if present in conf file, if not use the same as preload paths.
    preload_params.update({
        'save_model_path': params.get('save_model_path', preload_params['model_path']),
        'save_opt_weights_path': params.get('save_opt_weights_path', preload_params['opt_weights_path']),
        'save_opt_conf_path': params.get('save_opt_conf_path', preload_params['opt_conf_path'])
    })

    # paths to store opt and model weights for best eval result
    preload_params.update({
        'best_eval_model_path': get_eval_filename(preload_params['save_model_path']),
        'best_eval_opt_weights_path': get_eval_filename(preload_params['save_opt_weights_path']),
    })

    # get dataset params
    data_params = {
        'data_path': params['data_path'],
        'maps_dir': params['maps_dir']
    }

    # get eval dataset params. If none then use the same as train dataset
    data_params.update({
        'eval_data_path': params.get('eval_data_path', data_params['data_path']),
        'eval_maps_dir': params.get('eval_maps_dir', data_params['maps_dir'])
    })

    # logging path
    logs_dir = params.get('logs_dir')
    return model_params, optim_params, training_params, preload_params, data_params, logs_dir


def load_model_and_opt(preload, model: STE_Transformer, model_path=None, opt_weights_path=None):
    init_loss, init_epoch = np.inf, 0
    if preload:
        # load model weights
        if model_path is not None:
            # verify model_path is valid file
            valid_file(model_path)
            # load model weights
            model_data = load_pkl_data(model_path)
            init_loss = model_data.get('loss', np.inf)
            init_epoch = model_data.get('epoch', 0)
            model.set_weights(model_data['weights'])

        # load optimizer weights
        if opt_weights_path is not None:
            # load optimizer
            valid_file(opt_weights_path)
            weights = load_pkl_data(opt_weights_path)
            model.optimizer.set_weights(weights)

    return init_loss, init_epoch + 1


def init_model_and_opt(model_params, dataset, stds, dk, preload_params, optimizer_params=None):
    # get model and optimizer
    past, future, maps, _ = next(iter(dataset))
    preload = preload_params['preload']
    transfer = preload_params['transfer_learning']
    model = STTransformer(**model_params)
    model.get_optimizer(dk, preload, preload_params['save_opt_conf_path'], preload_params['opt_conf_path'], optimizer_params)

    # init weights of model and optimizer
    strategy.run(model.iterative_train_step, args=([past, future, maps, stds],))

    # preload weights
    init_loss, init_epoch = load_model_and_opt(preload, model,
                                               preload_params['model_path'], preload_params['opt_weights_path'])

    eval_metric = load_pkl_data(preload_params['best_eval_model_path'])['loss'] if preload and not transfer else np.inf
    return model, init_loss, init_epoch, eval_metric


def save_state(model, optimizer, loss, epoch, model_path, opt_weight_path):
    class_name = model.__class__.__name__
    if model_path is None:
        model_path = 'Code/weights/best_' + class_name + '_weights.pkl'
    if opt_weight_path is None:
        opt_weight_path = 'Code/weights/best_opt_' + class_name + 'weight.pkl'

    # validate if paths exist or create them
    directories = list(map(os.path.dirname, [model_path, opt_weight_path]))
    valid_path(*directories)
    # store weights
    save_pkl_data({'weights': model.get_weights(), 'loss': loss, 'epoch': epoch}, model_path)
    save_pkl_data(optimizer.get_weights(), opt_weight_path, 4)
    

def get_logger(logs_dir):
    if logs_dir is None:
        return None

    valid_path(logs_dir)
    summary_writer = tf.summary.create_file_writer(
        logs_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return summary_writer

def eval_model(model, dataset, stds, perform_qualitative_eval=False):
    losses, l_ade, l_fde, l_weights, l_masks, l_ids = [], [], [], [], [], []
    counter = 0
    for (past, future, maps, targets) in dataset:
        batch_size = len(past)
        preds, loss, weights = model.eval_step(past, future, maps)
        l_weights.append(weights.numpy())
        l_masks.append(past[3].numpy())
        l_ids.append(targets[3].numpy())
        # swap neighbors and sequence dimension
        all_targets = tf.transpose(future[0][:, :, :, :2], [0, 2, 1, 3])
        all_preds = tf.transpose(preds[:, :, :, :2], [0, 2, 1, 3])
        # reshape to remove batch
        all_targets = tf.reshape(all_targets, (-1, 26, 2))
        all_preds = tf.reshape(all_preds, (-1, 26, 2))
        ade = ADE(all_targets.numpy(), all_preds.numpy())
        fde = FDE(all_targets.numpy(), all_preds.numpy())
        l_ade.append(ade)
        l_fde.append(fde)
        losses.append(loss)

        if np.random.rand() < 0.2 and perform_qualitative_eval:
            n_element = np.random.choice(batch_size)
            bitmaps = np.transpose(maps[n_element].numpy(), [0, 3, 1, 2])
            map_id = targets[3][n_element].numpy().decode().split('/')[-1]
            mask_tar = tf.squeeze(future[3][n_element]).numpy()
            yaw = targets[2][n_element]
            target_traj = future[0][n_element].numpy()
            pred_traj = preds[n_element].numpy()
            bitmaps = stamp_traj(target_traj, mask_tar, bitmaps, 1.0, yaw)
            bitmaps = stamp_traj(pred_traj, mask_tar, bitmaps, 1.0, yaw, bottom=False)
            name = 'Code/qual_eval/' + map_id[:-4]
            np.savez_compressed(name, bitmaps=bitmaps)
            print('traj plot created:', name, flush=True)
            counter += 1

    l_weights = np.array(l_weights)
    l_masks = np.array(l_masks)
    l_ids = np.array(l_ids) 
    np.savez_compressed('attn_weights.npz', weights=l_weights, masks=l_masks, ids=l_ids)
    mean_ade = np.mean(np.array(l_ade))
    mean_fde = np.mean(np.array(l_fde))
    mean_loss = np.mean(losses)
    print('mean loss: ', mean_loss)
    print('mean ade: ', mean_ade)
    print('mean fde: ', mean_fde)
    return mean_ade, mean_loss


@tf.function
def distributed_step(inputs, step_fn):
    per_replica_losses = strategy.run(step_fn, args=(inputs,))
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return loss


def train(model, epochs, init_loss, init_epoch, eval_metric, model_path, opt_weights_path,
          opt_conf_path, best_model_path, best_opt_path, logs_dir=None):
    # avoid creating summary writer
    if epochs == 0:
        return

    # loggin writer and best loss
    summary_writer = get_logger(logs_dir)
    best_loss = init_loss  # 21.08453
    # start training
    for epoch in range(init_epoch, init_epoch + epochs):
        print('epoch: ', epoch)

        losses = []
        start = time.time()
        for batch_index, (past, future, maps, _) in enumerate(dataset):
            loss = distributed_step([past, future, maps, stds], model.iterative_train_step)
            losses.append(loss)
            if batch_index % 600 == 0:
                print(batch_index, '. batch loss: ', tf.reduce_mean(losses), flush=True)
            if np.isnan(loss.numpy()):
                break

        # test if model loss is lower
        avg_loss = tf.reduce_mean(losses)
        if avg_loss.numpy() < best_loss:
            best_loss = avg_loss.numpy()
            save_state(model, model.optimizer, best_loss, epoch, model_path=model_path,
                       opt_weight_path=opt_weights_path)

        print(' ----------------------------- EVALUATING MODEL -----------------------------')
        ade, eval_loss = eval_model(model, eval_dataset, stds)
        if ade < eval_metric:
            eval_metric = ade
            save_state(model, model.optimizer, eval_metric, epoch, model_path=best_model_path,
                       opt_weight_path=best_opt_path)

        end = time.time()
        print('TIME ELAPSED:', datetime.timedelta(seconds=end - start))
        print("avg loss", avg_loss, flush=True)
        # log resutls if desired
        if summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)
                tf.summary.scalar('eval_loss', eval_loss, step=epoch)


if __name__ == '__main__':
    import sys
    import os

    # change working directory
    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path + '/../..')
    params_path = sys.argv[1]

    # LOAD PARAMETERS
    params = load_parameters(params_path)
    parameters = split_params(params, STTransformer)
    model_params = parameters[0]
    optim_params = parameters[1]
    training_params = parameters[2]
    preload_params = parameters[3]
    data_params = parameters[4]
    logs_dir = parameters[5]

    # path to store weights
    model_path = preload_params['save_model_path']
    opt_weights_path = preload_params['save_opt_weights_path']
    opt_conf_path = preload_params['save_opt_conf_path']
    best_eval_model_path = preload_params['best_eval_model_path']
    best_eval_opt_path = preload_params['best_eval_opt_weights_path']

    # MODEL PARAMS
    dk = model_params['sp_dk']
    batch = model_params['batch']

    # TRAINING PARAMS
    epochs = training_params['epochs']
    lr = training_params['lr']

    # GET DATA
    data = load_pkl_data(data_params['data_path'])
    eval_data = load_pkl_data(data_params['eval_data_path'])

    # GET DATASETS
    strategy = tf.distribute.MirroredStrategy()
    dataset, std_x, std_y = buildDataset(data, batch, pre_path=data_params['maps_dir'], strategy=strategy)
    eval_dataset, _, _ = buildDataset(eval_data, batch, pre_path=data_params['eval_maps_dir'], strategy=None, shuffle=False)

    with strategy.scope():
        stds = tf.constant([[[[std_x, std_y]]]], dtype=tf.float32)
        # get model
        model, init_loss, init_epoch, eval_metric = init_model_and_opt(model_params, dataset, stds, dk, preload_params, optim_params)
        # train model
        train(model, epochs, init_loss, init_epoch, eval_metric, model_path, opt_weights_path,
              opt_conf_path, best_eval_model_path, best_eval_opt_path, logs_dir)

    # eval model
    eval_model(model, eval_dataset, stds, perform_qualitative_eval=True)
