# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import io
import torchaudio
from .node_utils import gc_clear
from .generate import auto_prompt_type,infer_stage2,inference_lowram_final,build_model,Separator,song_infer_lowram
import time
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

from .SongGeneration.codeclm.models import builders
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
SongGeneration_Weigths_Path = os.path.join(folder_paths.models_dir, "SongGeneration")
if not os.path.exists(SongGeneration_Weigths_Path):
    os.makedirs(SongGeneration_Weigths_Path)
folder_paths.add_model_folder_path("SongGeneration", SongGeneration_Weigths_Path)

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class SongGeneration_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "infer_model": (["none"] +[i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".pt") or i.endswith(".gguf") ],),
                "gguf": (["none"] + folder_paths.get_filename_list("gguf"), ),
                "version": (["v2","v1"],),
                "use_flash_attn":("BOOLEAN", {"default": True}),
                "offload_audiolm" :("BOOLEAN", {"default": True}),   
            },
        }

    RETURN_TYPES = ("SongGeneration_Audiolm","SongGeneration_Cfg")
    RETURN_NAMES = ("model","cfg")
    FUNCTION = "main"
    CATEGORY = "SongGeneration"

    def main(self, infer_model,gguf,version,use_flash_attn,offload_audiolm):
        pt_path=folder_paths.get_full_path("SongGeneration", infer_model) if infer_model != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        assert pt_path is not None or gguf_path is not None  ,"模型不能为空.need infer model"
        infer_model_path = pt_path if pt_path is not None else gguf_path
        model,cfg=build_model(os.path.join(SongGeneration_Weigths_Path, "ckpt"),infer_model_path,version,use_flash_attn,offload_audiolm)
        return (model,cfg)



class SongGeneration_Stage1:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {              
                "vae": (folder_paths.get_filename_list("vae"),),
                "seperate_model": (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".safetensors") and not "fix" in i.lower()],),  
                "prompt_pt":  (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if "prompt" in i.lower()],),
                "auto_prompt_audio_type": (auto_prompt_type,),
                "model_1rvq": (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".safetensors")],),
                "demucs_pt":  (["none"] + [i for i in folder_paths.get_filename_list("SongGeneration") if i.endswith(".pth")],),
                "lyric": ("STRING", {"multiline": True, "default": "[intro-short] ;\n[verse]\n在那遥远的爵士乡.有位迷人的姑娘.行人经过她的窗.总为她驻足凝望.她的微笑如夜曲悠扬.温暖中带着一丝忧伤 ;\n[chorus]\n她的眼神如星闪亮.带我坠入梦境徜徉.我愿放下手中诗行.只为她轻声歌唱.日日守候在那街角.盼她一次温柔回望.我不过是远方的风.偶然路过她的乐章.却把心跳谱成了音符.在每一个夜晚反复回响;\n[inst-medium];\n[verse]\n我愿化身为那月光.静静陪在她的身旁 ;\n[bridge]\n任那晚风轻拂梦想.心事只对她默默讲.岁月是位沉默的乐手.将青涩吹奏成沧桑.我的歌谣褪了颜色.却还在老地方.为她播放 ;\n[chorus]\n在那遥远的城市一方.时光深处藏着一位好姑娘.夜色披上霓虹衣裳.迷路的人啊.都向着那扇窗凝望.那光芒.是温柔的故乡;\n[outro-medium]"}),
            },
             "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("SongGeneration_Cond",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "main"
    CATEGORY = "SongGeneration"

    def main(self,vae,seperate_model,prompt_pt, auto_prompt_audio_type,model_1rvq,demucs_pt,lyric,**kwargs):

        audio=kwargs.get("audio", None)
        model_sep_path=folder_paths.get_full_path("SongGeneration", seperate_model) if seperate_model != "none" else None
        vae_model=folder_paths.get_full_path("vae", vae)
        prompt_pt_path=folder_paths.get_full_path("SongGeneration", prompt_pt) if prompt_pt != "none" else None
    
        if audio is not None:
            print("Using audio as reference.")
            prompt_audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{time.strftime('%m%d%H%S')}_temp.wav")
            waveform=audio["waveform"].squeeze(0)
            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")
            with open(prompt_audio_path, 'wb') as f:
                f.write(buff.getbuffer())
            use_descriptions=False #不建议同时提供参考音频和描述文本

            dm_model_path=folder_paths.get_full_path("SongGeneration", demucs_pt) if demucs_pt != "none" else None
            assert dm_model_path is not None ,"使用参考音频时，需要选择htdemucs模型， if use audio need htdemucs model"
            separator = Separator(dm_model_path, os.path.join(current_node_path, "SongGeneration/third_party/demucs/ckpt/htdemucs.yaml"))

            model_1rvq_path=folder_paths.get_full_path("SongGeneration", model_1rvq) if model_1rvq != "none" else None
            assert model_1rvq_path is not None ,"使用参考音频时，需要选择model_模型， if use audio need model_odel"
            audio_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAE1rvq_{model_1rvq_path}",os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json'),vae_model,'inference')
     
            seperate_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAESeparate_{model_sep_path}",os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json'),vae_model,'inference')
                
        else:
            prompt_audio_path,use_descriptions,audio_tokenizer,separator,seperate_tokenizer=None,True,None,None,None

        original_item=song_infer_lowram(seperate_tokenizer,separator,audio_tokenizer,prompt_pt_path, folder_paths.get_output_directory(),prompt_audio_path,auto_prompt_audio_type,lyric)

        gc_clear()
        print("Stage1 is done.")
        return ({"item": original_item, "use_descriptions": use_descriptions,"model_sep_path":model_sep_path,"vae_model":vae_model},)

class SongGeneration_Stage2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":  ("SongGeneration_Audiolm",),
                "cfg": ("SongGeneration_Cfg",),
                "cond": ("SongGeneration_Cond",),
                "gen_type": (["mixed","bgm","vocal",],), 
                "description": ("STRING", {"multiline": True, "default": "jazz, piano, brass section, double bass, drum kit, confident"}), #OPTIONAL
                "cfg_coef": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 3.0, "step": 0.1}),
                "temp": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 10000000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "record_tokens": ("BOOLEAN", {"default": True}),
                "record_window": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}), 
                           
            },
        }

    RETURN_TYPES = ("SongGeneration_Cond",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "main"
    CATEGORY = "SongGeneration"

    def main(self, model,cfg,cond,gen_type,description,cfg_coef,temp,top_k,top_p,record_tokens,record_window):

        lyric=cond.get("item")["gt_lyric"]
        cfg.gen_type=gen_type
        if cfg.version == 'v1':
             descriptions=description.lower() if cond.get("use_descriptions",False) else None
        else:
            if gen_type == 'bgm':
                descriptions = '[Musicality-very-high]' + ', ' + '[Pure-Music]' + ', ' + description.lower() if cond.get("use_descriptions",False) else '.'
            else:
                descriptions = description.lower() if cond.get("use_descriptions",False) else '.'
                descriptions = '[Musicality-very-high]' + ', ' + descriptions

        items=infer_stage2(cond.get("item"),model,cfg.max_dur,lyric,descriptions,gen_type,cfg,cfg_coef, temp,top_k,top_p,record_tokens ,record_window,cfg.offload_audiolm )
        gc_clear()
        return ({"items":items,"cfg":cfg,"model_sep_path":cond["model_sep_path"],"vae_model":cond["vae_model"]},)



class SongGeneration_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("SongGeneration_Cond",),
                "save_separate": ("BOOLEAN", {"default": False}),
            }
            }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sampler_main"
    CATEGORY = "SongGeneration"

    def sampler_main(self,cond,save_separate):
        cfg=cond.get("cfg")
        model_sep_path=cond["model_sep_path"]
        vae_model=cond["vae_model"]
        print("start inference final,loading model")
        seperate_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAESeparate_{model_sep_path}",os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json'),vae_model,'inference')
        seperate_tokenizer = seperate_tokenizer.eval().cuda()
        audio=inference_lowram_final(cfg,seperate_tokenizer,cfg.max_dur,cond.get("items"),folder_paths.get_output_directory(),save_separate)
        del seperate_tokenizer
        gc_clear()
        return (audio,)



NODE_CLASS_MAPPINGS = {
    "SongGeneration_Loader":SongGeneration_Loader,
    "SongGeneration_Stage1": SongGeneration_Stage1,
    "SongGeneration_Stage2": SongGeneration_Stage2,
    "SongGeneration_Sampler": SongGeneration_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongGeneration_Loader": "SongGeneration_Loader",
    "SongGeneration_Stage1": "SongGeneration_Stage1",
    "SongGeneration_Stage2": "SongGeneration_Stage2",
    "SongGeneration_Sampler": "SongGeneration_Sampler",
}
