import os
import mxnet as mx
import json

cwd = os.path.split(__file__)[0]

with open(os.path.join(cwd, "config.json"), 'r') as f:
    main_conf = json.load(f)
finetune1 = main_conf["finetune1"]
