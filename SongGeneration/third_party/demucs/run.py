#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : layers.py
@Time    : 2024/4/22 下午2:40
@Author  : waytan
@Contact : waytan@tencent.com
@License : (C)Copyright 2024, Tencent
"""
import os
import json
import time
import logging
import argparse
from datetime import datetime


import torch

from models.apply import BagOfModels
from models.pretrained import get_model_from_yaml


class Separator:
    def __init__(self, dm_model_path, dm_config_path, gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path) -> BagOfModels:
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def run(self, audio_path, output_dir, ext=".flac"):
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []
        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 4:
            drums_path, bass_path, other_path, vocal_path = output_paths
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
        data_dict = {
            "vocal_path": vocal_path,
            "bgm_path": [drums_path, bass_path, other_path]
        }
        return data_dict


def json_io(input_json, output_json, model_dir, dst_dir, gpu_id=0):
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime('%Y-%m-%d-%H:%M')
    logging.basicConfig(filename=os.path.join(dst_dir, f'logger-separate-{os.path.split(input_json)[1]}-{current_datetime_str}.log'), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    sp = Separator(os.path.join(model_dir, "htdemucs.pth"), os.path.join(model_dir, "htdemucs.yaml"), gpu_id=gpu_id)
    with open(input_json, "r") as fp:
        lines = fp.readlines()
    t1 = time.time()
    success_num =  0
    fail_num = 0
    total_num = len(lines)
    sep_items = []
    for line in lines: 
        item = json.loads(line)
        flac_file = item["path"]
        try:
            fix_data = sp.run(flac_file, dst_dir)
        except Exception as e:
            fail_num += 1
            logging.error(f"process-{success_num + fail_num}/{total_num}|success-{success_num}|fail-{fail_num}|{item['idx']} process fail for {str(e)}")
            continue
        
        item["vocal_path"] = fix_data["vocal_path"]
        item["bgm_path"] = fix_data["bgm_path"]
        sep_items.append(item)
        success_num += 1
        logging.debug(f"process-{success_num + fail_num}/{total_num}|success-{success_num}|fail-{fail_num}|{item['idx']} process success")

    with open(output_json, "w", encoding='utf-8') as fw:
        for item in sep_items:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    t2 = time.time()
    logging.debug(f"total cost {round(t2-t1, 3)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", dest="model_dir")
    parser.add_argument("-d", dest="dst_dir")
    parser.add_argument("-j", dest="input_json")
    parser.add_argument("-o", dest="output_json")
    parser.add_argument("-gid", dest="gpu_id", default=0, type=int)
    args = parser.parse_args()

    if not args.dst_dir:
        dst_dir = os.path.join(os.getcwd(), "separate_result")
        os.makedirs(dst_dir, exist_ok=True)
    else:
        dst_dir = os.path.join(args.dst_dir, "separate_result")
        os.makedirs(dst_dir, exist_ok=True)

    json_io(args.input_json, args.output_json, args.model_dir, dst_dir, gpu_id=args.gpu_id)
