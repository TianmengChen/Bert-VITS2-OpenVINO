import sys

import torch
from transformers import DebertaV2Model, DebertaV2Tokenizer

from config import config
import os 
import openvino as ov
import numpy

LOCAL_PATH = "./bert/deberta-v3-large"

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

models = dict()


def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"

    if not os.path.exists("./ov_models/BERT_EN.xml"):
        print("try to convert BERT_EN to OpenVINO IR")
        model=DebertaV2Model.from_pretrained(LOCAL_PATH)
        model.eval()
        model.config.output_hidden_states = True
        model.config.return_dict = False
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            example_input=inputs.data
            ov_model = ov.convert_model(model, example_input=example_input)
            ov.save_model(ov_model, './ov_models/BERT_EN.xml')
        print("convert success, please run pipeline from begin")
        exit()

    else:
        core = ov.Core()
        bert_zh = core.compile_model("./ov_models/BERT_EN.xml")
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_zh(inputs.data)[-3][0]
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = bert_zh(inputs.data)[-3][0]
            style_res_mean = numpy.mean(style_res, axis=0)
            assert len(word2ph) ==  res.shape[0], (text, res.shape[0], len(word2ph))
        word2phone = word2ph
        phone_level_feature = []
        for i in range(len(word2phone)):
            if style_text:
                repeat_feature = (
                    numpy.tile(res[i], (word2phone[i], 1)) * (1 - style_weight)
                    + numpy.tile(style_res_mean, (word2phone[i], 1))* style_weight
                )
            else:
                repeat_feature = numpy.tile(res[i], (word2phone[i], 1))
            phone_level_feature.append(repeat_feature)

        phone_level_feature = numpy.concatenate(phone_level_feature, axis=0)

        return phone_level_feature.T
