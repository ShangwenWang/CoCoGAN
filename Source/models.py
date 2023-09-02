import logging
from statistics import mode

import torch
from transformers import (RobertaConfig, RobertaForSequenceClassification,
                          RobertaModel, RobertaTokenizer, T5Config,
                          T5ForConditionalGeneration, T5Tokenizer)

from discriminator import Discriminator
from discriminator_gcb import Discriminator_gcb
from generator import Generator

logger = logging.getLogger(__name__)

def build_or_load_gen_model(args, model_type, model_name_or_path):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]    
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, local_files_only=True)

    if model_type == 'discriminator':
        config = config_class.from_pretrained(
        args.config_name if args.config_name else args.discriminator_name_or_path)
        model = model_class.from_pretrained(
            model_name_or_path, config=config)
        if args.load_discriminator_path is not None:
            model.load_state_dict(torch.load(args.load_discriminator_path, map_location="cpu"))
            logger.info("Load model from {}".format(args.load_discriminator_path))
            print("Load model from {}".format(args.load_discriminator_path))
    elif model_type == 'discriminator_gcb':
        config = config_class.from_pretrained(
        args.config_name if args.config_name else args.discriminator_name_or_path)
        model = model_class.from_pretrained(
            model_name_or_path)
        model = Discriminator_gcb(model)
        model.config = config
        if args.load_discriminator_path is not None:
            model.load_state_dict(torch.load(args.load_discriminator_path, map_location="cpu"))
            logger.info("Load model from {}".format(args.load_discriminator_path))
            print("Load model from {}".format(args.load_discriminator_path))
    elif model_type == 'generator':
        config = config_class.from_pretrained(
        args.config_name if args.config_name else args.generator_name_or_path)
        model = model_class.from_pretrained(
            model_name_or_path)
        if args.load_generator_path is not None:
            model.load_state_dict(torch.load(args.load_generator_path, map_location="cpu"))
            print("Load model from {}".format(args.load_generator_path))
    model.tokenizer = tokenizer
    model.args = args
    model.to(args.device)
    return model, tokenizer

MODEL_CLASSES = {
    'discriminator': (RobertaConfig, Discriminator, RobertaTokenizer),
    'discriminator_gcb': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'generator': (T5Config, Generator, RobertaTokenizer),
}
