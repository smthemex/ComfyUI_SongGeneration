import sys
import os
import folder_paths
import time
import json
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from .SongGeneration.codeclm.models import builders
from .SongGeneration.codeclm.trainer.codec_song_pl import CodecLM_PL
from .SongGeneration.codeclm.models import CodecLM
from .SongGeneration.third_party.demucs.models.pretrained import get_model_from_yaml

auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

class Separator():
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        else:
            a = torch.cat([a, a], -1)
        return a[:, 0:48000*10]
    
    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio


def pre_data(Weigths_Path,dm_model_path,dm_config_path,save_dir,prompt_audio_path,auto_prompt_audio_type):
    torch.backends.cudnn.enabled = False
    
    # 检查解析器是否已经注册，如果没有则注册
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    if not OmegaConf.has_resolver("concat"):
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    if not OmegaConf.has_resolver("get_fname"):
        OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    
    curent_dir = os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI_SongGeneration/SongGeneration")
    
    if not OmegaConf.has_resolver("load_yaml"):
        OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(os.path.join(curent_dir,x))))
    np.random.seed(int(time.time()))    
    
    cfg_path = os.path.join(Weigths_Path, 'songgeneration_base/config.yaml')
    
    cfg = OmegaConf.load(cfg_path)
    cfg.mode = 'inference'
    cfg.vae_config=f"{Weigths_Path}/vae/stable_audio_1920_vae.json"
    cfg.vae_model=f"{Weigths_Path}/vae/autoencoder_music_1320k.ckpt"

    cfg.audio_tokenizer_checkpoint=f"Flow1dVAE1rvq_{Weigths_Path}/model_1rvq/model_2_fixed.safetensors"
    cfg.audio_tokenizer_checkpoint_sep=f"Flow1dVAESeparate_{Weigths_Path}/model_septoken/model_2.safetensors"
    cfg.conditioners.type_info.QwTextTokenizer.token_path=os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI_SongGeneration/SongGeneration/third_party/Qwen2-7B")
    max_duration = cfg.max_dur

    separator = Separator(dm_model_path, dm_config_path)
    auto_prompt = torch.load(os.path.join(Weigths_Path,'prompt.pt'),weights_only=False)

    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
    else:
        seperate_tokenizer = None
    audio_tokenizer = audio_tokenizer.eval().cuda()
    if seperate_tokenizer is not None:
        seperate_tokenizer = seperate_tokenizer.eval().cuda()

    merge_prompt = [item for sublist in auto_prompt.values() for item in sublist]
    item=song_infer_lowram(seperate_tokenizer,separator,audio_tokenizer,merge_prompt,auto_prompt, save_dir,prompt_audio_path,auto_prompt_audio_type)
    return item,max_duration,cfg


def infer_stage2(item,cfg,Weigths_Path,max_duration,lyric,descriptions,cfg_coef = 1.5, temp = 0.9,top_k = 50,top_p = 0.0,record_tokens = True,record_window = 50):
    ckpt_path = os.path.join(Weigths_Path, 'songgeneration_base/model.pt')
    # Define model or load pretrained model
    model_light = CodecLM_PL(cfg, ckpt_path)
    model_light = model_light.eval()
    model_light.audiolm.cfg = cfg
    model = CodecLM(name = "tmp",
        lm = model_light.audiolm,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = None,
    )
    del model_light
    torch.cuda.empty_cache()
    model.lm = model.lm.cuda().to(torch.float16)
    
    model.set_generation_params(duration=max_duration, extend_stride=5, temperature=temp, cfg_coef=cfg_coef,
                                top_k=top_k, top_p=top_p, record_tokens=record_tokens, record_window=record_window)
    
    items=inference_lowram_step2(model,lyric,descriptions,item)
    model=None
    torch.cuda.empty_cache()
   
    return items




def inference_lowram_step2(model,lyric,descriptions,item):
    
    pmt_wav = item['pmt_wav']
    vocal_wav = item['vocal_wav']
    bgm_wav = item['bgm_wav']
    melody_is_wav = item['melody_is_wav']
        
    generate_inp = {
        'lyrics': [lyric.replace("  ", " ")],
        'descriptions': [descriptions],
        'melody_wavs': pmt_wav,
        'vocal_wavs': vocal_wav,
        'bgm_wavs': bgm_wav,
        'melody_is_wav': melody_is_wav,
    }
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.generate(**generate_inp, return_tokens=True)
    item['tokens'] = tokens
    
    return item



def inference_lowram_final(cfg,max_duration,item,save_dir):
    target_wav_name = f"{save_dir}/song_audios{time.strftime('%m%d%H%S')}.flac"
    seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
    seperate_tokenizer = seperate_tokenizer.eval().cuda()
    model = CodecLM(name = "tmp",
        lm = None,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = seperate_tokenizer,
    )
   
    with torch.no_grad():
        if 'raw_pmt_wav' in item:   
            wav_seperate = model.generate_audio(item['tokens'], item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True)
            del item['raw_pmt_wav']
            del item['raw_vocal_wav']
            del item['raw_bgm_wav']
        else:
            wav_seperate = model.generate_audio(item['tokens'], chunked=True)
    #torchaudio.save(item['wav_path'], wav_seperate[0].cpu().float(), cfg.sample_rate)
    torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), cfg.sample_rate)
    del item['tokens']
    del item['pmt_wav']
    del item['vocal_wav']
    del item['bgm_wav']
    del item['melody_is_wav']

    return {"waveform": wav_seperate[0].cpu().float().unsqueeze(0), "sample_rate": cfg.sample_rate}



def song_infer_lowram(seperate_tokenizer,separator,audio_tokenizer,merge_prompt,auto_prompt, save_dir,prompt_audio_path,auto_prompt_audio_type): #item dict
    item = {}
    target_wav_name = f"{save_dir}/song_audios{time.strftime('%m%d%H%S')}.flac"
    if prompt_audio_path:
            # assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            # assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
            item['raw_pmt_wav'] = pmt_wav
            item['raw_vocal_wav'] = vocal_wav
            item['raw_bgm_wav'] = bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if vocal_wav.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wav = list(vocal_wav)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            if bgm_wav.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wav = list(bgm_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            if type(vocal_wav) == list:
                vocal_wav = torch.stack(vocal_wav, dim=0)
            if type(bgm_wav) == list:
                bgm_wav = torch.stack(bgm_wav, dim=0)
            pmt_wav = pmt_wav.cuda()
            vocal_wav = vocal_wav.cuda()
            bgm_wav = bgm_wav.cuda()
            pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
            vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)
            melody_is_wav = False
    elif auto_prompt_audio_type:
        #assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
        if auto_prompt_audio_type == "Auto": 
            prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
        else:
            prompt_token = auto_prompt[auto_prompt_audio_type][np.random.randint(0, len(auto_prompt[auto_prompt_audio_type]))]
        pmt_wav = prompt_token[:,[0],:]
        vocal_wav = prompt_token[:,[1],:]
        bgm_wav = prompt_token[:,[2],:]
        melody_is_wav = False
    else:
        pmt_wav = None
        vocal_wav = None
        bgm_wav = None
        melody_is_wav = True
    item['pmt_wav'] = pmt_wav
    item['vocal_wav'] = vocal_wav
    item['bgm_wav'] = bgm_wav
    item['melody_is_wav'] = melody_is_wav
    item["idx"] = 0
    item["wav_path"] = target_wav_name
    

    del audio_tokenizer
    del seperate_tokenizer
    del separator

    return item






# def song_infer(model,separator,lyric,merge_prompt,auto_prompt, save_dir,cfg,prompt_audio_path,auto_prompt_audio_type,descriptions = None): #item dict

#     target_wav_name = f"{save_dir}/song_audios{time.strftime('%m%d%H%S')}.flac"

#     if prompt_audio_path:
#         pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
#         melody_is_wav = True
#     elif auto_prompt_audio_type:
#         if auto_prompt_audio_type== "Auto": 
#             prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
#         else:
#             prompt_token = auto_prompt[auto_prompt_audio_type][np.random.randint(0, len(auto_prompt[auto_prompt_audio_type]))] #need check
#         pmt_wav = prompt_token[:,[0],:]
#         vocal_wav = prompt_token[:,[1],:]
#         bgm_wav = prompt_token[:,[2],:]
#         melody_is_wav = False
#     else:
#         pmt_wav = None
#         vocal_wav = None
#         bgm_wav = None
#         melody_is_wav = True
        
#     generate_inp = {
#         'lyrics': [lyric.replace("  ", " ")],
#         'descriptions': [descriptions],
#         'melody_wavs': pmt_wav,
#         'vocal_wavs': vocal_wav,
#         'bgm_wavs': bgm_wav,
#         'melody_is_wav': melody_is_wav,
#     }
#     start_time = time.time()
#     with torch.autocast(device_type="cuda", dtype=torch.float16):
#         tokens = model.generate(**generate_inp, return_tokens=True)
#     mid_time = time.time()
        
#     with torch.no_grad():
#         if melody_is_wav:   
#             wav_seperate = model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav)
#         else:
#             wav_seperate = model.generate_audio(tokens)
#     end_time = time.time()
#     torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), cfg.sample_rate)
#     print(f"process lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}")
    
#     return {"waveform": wav_seperate[0].cpu().float().unsqueeze(0), "sample_rate": cfg.sample_rate}
