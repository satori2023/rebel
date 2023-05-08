from pl_modules import BasePLModule

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

import torch

import omegaconf

import sys

path = sys.argv[1]
suffix = sys.argv[2]

config = AutoConfig.from_pretrained("Babelscape/rebel-large")

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")

model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

conf = omegaconf.OmegaConf.load(path + '/.hydra/config.yaml')

pl_module = BasePLModule(conf, config, tokenizer, model)

model = pl_module.load_from_checkpoint(checkpoint_path = path + '/experiments/default_name/last.ckpt', config = config, tokenizer = tokenizer, model = model)

model.model.save_pretrained('../model/rebel-large'+suffix)

model.tokenizer.save_pretrained('../model/rebel-large'+suffix)
