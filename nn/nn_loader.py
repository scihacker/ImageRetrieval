import os
import mxnet as mx

cwd = os.path.split(__file__)[0]

def vgg16(path="vgg16/imagenet/vgg16", epochs=0):
    full_path = os.path.join(cwd, path)
    sym, arg_params, aux_params = mx.model.load_checkpoint(full_path, epochs)

    return sym, arg_params, aux_params

# Modify vgg16 for finetuning.
def vgg16_ft(path="vgg16/imagenet/vgg16", epochs=0):
    sym, arg_params, aux_params = vgg16(path, epochs)

    all_layers = sym.get_internals()
    net = all_layers['drop7_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=560, name='fc8')
    net = mx.symbol.SoftmaxOutput(data=net, name='prob')

    new_args = dict({k: arg_params[k] for k in arg_params if 'fc8' not in k})

    return sym, new_args, aux_params
