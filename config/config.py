import os
import mxnet as mx
import json

cwd = os.path.split(__file__)[0]

main_conf = json.load(os.path.join(cwd, "config.json"))
finetune1 = main_conf["finetune1"]
