import os
import sys
import mxnet as mx
from config import config
from nn import nn_loader
import argparse


def _get_iterator(file_name, batch_size, data_shape):
    data_iter = mx.image.ImageIter(path_imglist=file_name, batch_size=batch_size, 
                                   path_root=config.main_conf['path_root'], data_shape=data_shape, 
                                   aug_list=[mx.image.ForceResizeAug(data_shape[1:])], shuffle=True)

    return data_iter

def train():
    os.mkdir("nn/vgg16/finetune")
    conf = config.finetune1
    data_shape = (3, 224, 224)
    devs = [mx.gpu(i) for i in range(conf['num_gpus'])]
    batch_size = conf['batch_size'] * conf['num_gpus']
    # Data
    train_set = _get_iterator("data/DB1M_train.lst", batch_size, data_shape)
    val_set = _get_iterator("data/DB1M_val.lst", 64 * conf['num_gpus'], data_shape)
    # Load or Rebuild
    sym, arg_params, aux_params = nn_loader.vgg16_ft(path="vgg16/imagenet/vgg16", epochs=0)

    mod = mx.mod.Module(symbol=sym, context=devs, label_names=['prob_label'])
    mod.bind(data_shapes=[('data', (batch_size, 3, 224, 224))], label_shapes=[('prob_label', (batch_size,))])
    mod.fit(train_set, val_set, num_epoch=conf['num_epochs'], arg_params=arg_params, 
            aux_params=aux_params, eval_metric='acc', allow_missing=True,
            batch_end_callback = mx.callback.Speedometer(batch_size, 10),
            optimizer='sgd', optimizer_params={'learning_rate': conf['learning_rate']},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), 
            epoch_end_callback=mx.callback.do_checkpoint("nn/vgg16/finetune/vgg16"))

    metric = mx.metric.Accuracy()
    print("Validation Accuracy:", mod.score(val_set, metric))
    return mod

def test():
    conf = config.finetune1
    data_shape = (3, 224, 224)
    devs = [mx.gpu(i) for i in range(conf['num_gpus'])]
    # Data
    test_set = _get_iterator("data/DB1M_test.lst", 64 * conf['num_gpus'], data_shape)
    # Load
    sym, arg_params, aux_params = nn_loader.vgg16(path="vgg16/finetune/vgg16", epochs=0)
    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    print("Test Accuracy:", mod.score(test_set, mx.metric.Accuracy()))
    return mod

if __name__ == "__main__":
    func = globals().get(sys.argv[1])
    if func:
        func(*sys.argv[2:])
    else:
        print("Usage: python %s [function] ..." % (sys.argv[0]))

