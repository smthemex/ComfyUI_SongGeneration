# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import io
import torchaudio

from pathlib import PureWindowsPath
from .node_utils import gc_clear
from .generate import auto_prompt_type,pre_data,infer_stage2,inference_lowram_final
import time
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
SongGeneration_Weigths_Path = os.path.join(folder_paths.models_dir, "SongGeneration")
if not os.path.exists(SongGeneration_Weigths_Path):
    os.makedirs(SongGeneration_Weigths_Path)
folder_paths.add_model_folder_path("SongGeneration", SongGeneration_Weigths_Path)



class SongGeneration_Stage1:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "demucs_pt":  (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".pth")],),
                "auto_prompt_audio_type": (auto_prompt_type,),
                "cpu_offload":  ("BOOLEAN", {"default": True},),             
            },
             "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("SongGeneration_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "SongGeneration"

    def loader_main(self, demucs_pt,auto_prompt_audio_type,cpu_offload,**kwargs):
        audio=kwargs.get("audio", None)
        if audio is not None:
            prompt_audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{time.strftime('%m%d%H%S')}_temp.wav")
            waveform=audio["waveform"].squeeze(0)
            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")
            with open(prompt_audio_path, 'wb') as f:
                f.write(buff.getbuffer())
            use_descriptions=False #不建议同时提供参考音频和描述文本
        else:
            prompt_audio_path=None
            use_descriptions=True


        if demucs_pt == "none":
            raise ValueError("No demucs_pt selected")
        
        dm_model_path=folder_paths.get_full_path("SongGeneration", demucs_pt)
        dm_config_path=os.path.join(current_node_path, "SongGeneration/third_party/demucs/ckpt/htdemucs.yaml")
        Weigths_Path=os.path.join(SongGeneration_Weigths_Path, "ckpt")
        
        item,max_duration,cfg=pre_data(Weigths_Path,dm_model_path,dm_config_path,folder_paths.get_output_directory(),prompt_audio_path,auto_prompt_audio_type)
       

        gc_clear()

        return ({"item": item, "max_duration": max_duration,"use_descriptions": use_descriptions,"cfg":cfg,"Weigths_Path":Weigths_Path},)


class SongGeneration_Stage2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":  ("SongGeneration_MODEL",),
                "lyric": ("STRING", {"multiline": True, "default": "[intro-short] ; [verse] 雪花舞动在无尽的天际.情缘如同雪花般轻轻逝去.希望与真挚.永不磨灭.你的忧虑.随风而逝 ; [chorus] 我怀抱着守护这片梦境.在这世界中寻找爱与虚幻.苦辣酸甜.我们一起品尝.在雪的光芒中.紧紧相拥 ; [inst-short] ; [verse] 雪花再次在风中飘扬.情愿如同雪花般消失无踪.希望与真挚.永不消失.在痛苦与喧嚣中.你找到解脱 ; [chorus] 我环绕着守护这片梦境.在这世界中感受爱与虚假.苦辣酸甜.我们一起分享.在白银的光芒中.我们同在 ; [outro-short]"}),
                "description": ("STRING", {"multiline": True, "default": "female, dark, pop, sad, piano and drums, the bpm is 125"}), #OPTIONAL
   
            },
        }

    RETURN_TYPES = ("SongGeneration_DICT",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "SongGeneration"

    def loader_main(self, model,lyric,description):

        descriptions=description if model.get("use_descriptions") else None

        items=infer_stage2(model.get("item"),model.get("cfg"),model.get("Weigths_Path"),model.get("max_duration"),lyric,descriptions,cfg_coef = 1.5, temp = 0.9,top_k = 50,top_p = 0.0,record_tokens = True,record_window = 50)
        gc_clear()
        return ({"item":items,"cfg":model.get("cfg"),"max_duration":model.get("max_duration"),},)



class SongGeneration_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SongGeneration_DICT",),
            }
            }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sampler_main"
    CATEGORY = "SongGeneration"

    def sampler_main(self, model):

        audio=inference_lowram_final(model.get("cfg"),model.get("max_duration"),model.get("item"),folder_paths.get_output_directory())

        gc_clear()
        return (audio,)



NODE_CLASS_MAPPINGS = {
    "SongGeneration_Stage1": SongGeneration_Stage1,
    "SongGeneration_Stage2": SongGeneration_Stage2,
    "SongGeneration_Sampler": SongGeneration_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongGeneration_Stage1": "SongGeneration_Stage1",
    "SongGeneration_Stage2": "SongGeneration_Stage2",
    "SongGeneration_Sampler": "SongGeneration_Sampler",
}
