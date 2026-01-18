import sys
import os
import folder_paths
import time
import copy
import torch
import torchaudio
import numpy as np
import gc
from omegaconf import OmegaConf
from .SongGeneration.codeclm.models import builders
from .SongGeneration.codeclm.trainer.codec_song_pl import CodecLM_PL
from .SongGeneration.codeclm.models import CodecLM
from .SongGeneration.third_party.demucs.models.pretrained import get_model_from_yaml
current_node_path = os.path.dirname(os.path.abspath(__file__))

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



def build_model(Weigths_Path,infer_model_path):
    torch.backends.cudnn.enabled = False
    curent_dir = os.path.join(current_node_path,"SongGeneration")
    RESOLVERS = {
        "eval": lambda x: eval(x),
        "concat": lambda *x: [xxx for xx in x for xxx in xx],
        "get_fname": lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0],
        "load_yaml": lambda x: list(OmegaConf.load(os.path.join(curent_dir, x)))
    }

    for name, func in RESOLVERS.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, func)
    np.random.seed(int(time.time())) 
    infer_model_type="new" if "new" in infer_model_path.lower() else "large" if "large" in infer_model_path.lower() else "full" if "full" in infer_model_path.lower() else "base"

    cfg_path = os.path.join(current_node_path, f'SongGeneration/conf/{infer_model_type}_config.yaml')
    
    cfg = OmegaConf.load(cfg_path)
    cfg.mode = 'inference'
    cfg.vae_config=f"{Weigths_Path}/vae/stable_audio_1920_vae.json"
    cfg.vae_model=f"{Weigths_Path}/vae/autoencoder_music_1320k.ckpt"
    cfg.audio_tokenizer_checkpoint=f"Flow1dVAE1rvq_{Weigths_Path}/model_1rvq/model_2_fixed.safetensors"
    cfg.audio_tokenizer_checkpoint_sep=f"Flow1dVAESeparate_{Weigths_Path}/model_septoken/model_2.safetensors"
    cfg.conditioners.type_info.QwTextTokenizer.token_path=os.path.join(current_node_path,"SongGeneration/third_party/Qwen2-7B")

    audiolm = builders.get_lm_model(cfg)
    checkpoint = torch.load(infer_model_path, map_location='cpu',weights_only=False)
    audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    
    #### @tuolaku  https://github.com/smthemex/ComfyUI_SongGeneration/issues/37 #####
    # add 1.5 support，test。。。。
    key = "condition_provider.conditioners.type_info.output_proj.weight"
    expected_vocab_size = 151646
    if key in audiolm_state_dict:
        weight = audiolm_state_dict[key]
        if weight.size(0) > expected_vocab_size:
            print(f"[SongGeneration] Trimming {key} from {weight.size(0)} to {expected_vocab_size}")
            audiolm_state_dict[key] = weight[:expected_vocab_size, :]
    #####
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()
    #audiolm = audiolm.cuda().to(torch.float16)
    del audiolm_state_dict,checkpoint
    return audiolm,cfg
    


# def pre_data(Weigths_Path,dm_model_path,dm_config_path,save_dir,prompt_audio_path,auto_prompt_audio_type,infer_model_type,prompt_pt_path):
    # torch.backends.cudnn.enabled = False
    # curent_dir = os.path.join(current_node_path,"SongGeneration")
    # RESOLVERS = {
    #     "eval": lambda x: eval(x),
    #     "concat": lambda *x: [xxx for xx in x for xxx in xx],
    #     "get_fname": lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0],
    #     "load_yaml": lambda x: list(OmegaConf.load(os.path.join(curent_dir, x)))
    # }

    # for name, func in RESOLVERS.items():
    #     if not OmegaConf.has_resolver(name):
    #         OmegaConf.register_new_resolver(name, func)
    # np.random.seed(int(time.time()))    
    
    # cfg_path = os.path.join(current_node_path, f'SongGeneration/conf/{infer_model_type}_config.yaml')
    
    # cfg = OmegaConf.load(cfg_path)
    # cfg.mode = 'inference'

    # cfg.vae_config=f"{Weigths_Path}/vae/stable_audio_1920_vae.json"
    # cfg.vae_model=f"{Weigths_Path}/vae/autoencoder_music_1320k.ckpt"

    # cfg.audio_tokenizer_checkpoint=f"Flow1dVAE1rvq_{Weigths_Path}/model_1rvq/model_2_fixed.safetensors"
    # cfg.audio_tokenizer_checkpoint_sep=f"Flow1dVAESeparate_{Weigths_Path}/model_septoken/model_2.safetensors"
    # cfg.conditioners.type_info.QwTextTokenizer.token_path=os.path.join(current_node_path,"SongGeneration/third_party/Qwen2-7B")
    # max_duration = cfg.max_dur
    # vae_model=f"{Weigths_Path}/vae/autoencoder_music_1320k.ckpt"
    # vae_config=os.path.join(current_node_path, f'SongGeneration/conf/stable_audio_1920_vae.json')
    # auto_prompt = torch.load(prompt_pt_path,weights_only=False)
    # merge_prompt = [x for sublist in auto_prompt.values() for x in sublist]

    # if prompt_audio_path is not None:
    #     separator = Separator(dm_model_path, dm_config_path)
    #     audio_tokenizer = builders.get_audio_tokenizer_model(f"Flow1dVAE1rvq_{Weigths_Path}/model_1rvq/model_2_fixed.safetensors", vae_config,vae_model)
    #     audio_tokenizer = audio_tokenizer.eval().cuda()

    # else:
    #     audio_tokenizer = None
    #     separator = None
    
    # original_item=song_infer_lowram(cfg,separator,audio_tokenizer,merge_prompt,auto_prompt, save_dir,prompt_audio_path,auto_prompt_audio_type,)
    # print("step1 is done.")
    # return copy.deepcopy(original_item),max_duration,cfg


def infer_stage2(item,audiolm,max_duration,lyric,descriptions,cfg_coef = 1.5, temp = 0.9,top_k = 50,top_p = 0.0,record_tokens = True,record_window = 50):
    #ckpt_path = os.path.join(Weigths_Path, 'songgeneration_base/model.pt')
   
    item_copy = {
        'pmt_wav': item['pmt_wav'],  # 这些是引用，但安全因为后续设为None不影响原始
        'vocal_wav': item['vocal_wav'],
        'bgm_wav': item['bgm_wav'],
        'melody_is_wav': item['melody_is_wav'],
        'idx': item['idx'],
        'wav_path': item['wav_path']
        # 不包含 'tokens' 因为它将在 step2 中生成
    }
    # Define model or load pretrained model
    # model_light = CodecLM_PL(cfg, ckpt_path)
    # model_light = model_light.eval()
    # model_light.audiolm.cfg = cfg
    # model = CodecLM(name = "tmp",
    #     lm = model_light.audiolm,
    #     audiotokenizer = None,
    #     max_duration = max_duration,
    #     seperate_tokenizer = None,
    # )
    # del model_light
    # audiolm = builders.get_lm_model(cfg)
    # checkpoint = torch.load(ckpt_path, map_location='cpu')
    # audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    # audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm=audiolm.to(torch.float16).cuda()
    torch.cuda.empty_cache()
    model = CodecLM(name = "tmp",
        lm = audiolm,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = None,
    )
    
    model.set_generation_params(duration=max_duration, extend_stride=5, temperature=temp, 
                                top_k=top_k, top_p=top_p,cfg_coef=cfg_coef, record_tokens=record_tokens, record_window=record_window)
   
    print("model loaded,start inference step2")
    items=inference_lowram_step2(model,lyric,descriptions,item_copy,)
    audiolm = audiolm.cpu()
    del audiolm
    model=None
    gc.collect()
    torch.cuda.empty_cache()
    return items



def inference_lowram_step2(model,lyric,descriptions,item,):

    generate_inp = {
        'lyrics': [lyric.replace("  ", " ")],
        'descriptions': [descriptions],
        'melody_wavs': item['pmt_wav'],
        'vocal_wavs': item['vocal_wav'],
        'bgm_wavs': item['bgm_wav'],
        'melody_is_wav': item['melody_is_wav'],
    }
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.generate(**generate_inp, return_tokens=True)
    item['tokens'] = tokens
    
    return item

def save_with_fallback(path, tensor, sample_rate):

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.detach().cpu().float().contiguous()

    try:
        import torchaudio
        torchaudio.save(path, tensor, sample_rate)
        return
    except Exception as e:
        print(f"[WARN] torchaudio.save failed, using soundfile: {e}")

    try:
        audio = tensor.numpy().astype("float32")
        import soundfile as sf
        sf.write(path, audio.T if audio.shape[0] < audio.shape[1] else audio, sample_rate)
        print(f"[INFO] salvo com soundfile: {path}")
    except Exception as e:
        raise RuntimeError(f"Falhou soundfile.write também. Path={path}. Erro: {e}")


def inference_lowram_final(cfg, seperate_tokenizer, max_duration, item, save_dir, save_separate):
    target_wav_name = f"{save_dir}/song_audios{time.strftime('%m%d%H%S')}.flac"

    model = CodecLM(
        name="tmp",
        lm=None,
        audiotokenizer=None,
        max_duration=max_duration,
        seperate_tokenizer=seperate_tokenizer,
    )

    print("model loaded, start inference final...")

    with torch.no_grad():
        if item["melody_is_wav"]:
            if save_separate:
                wav_vocal = model.generate_audio(item['tokens'], item['pmt_wav'], item['vocal_wav'], item['bgm_wav'], chunked=True, gen_type='vocal')
                wav_bgm   = model.generate_audio(item['tokens'], item['pmt_wav'], item['vocal_wav'], item['bgm_wav'], chunked=True, gen_type='bgm')
                wav_mix   = model.generate_audio(item['tokens'], item['pmt_wav'], item['vocal_wav'], item['bgm_wav'], chunked=True, gen_type='mixed')
            else:
                wav_mix = model.generate_audio(item['tokens'], item['pmt_wav'], item['pmt_wav'], item['bgm_wav'], chunked=True, gen_type=cfg.gen_type)
        else:
            if save_separate:
                wav_vocal = model.generate_audio(item['tokens'], chunked=True, gen_type='vocal')
                wav_bgm   = model.generate_audio(item['tokens'], chunked=True, gen_type='bgm')
                wav_mix   = model.generate_audio(item['tokens'], chunked=True, gen_type='mixed')
            else:
                wav_mix = model.generate_audio(item['tokens'], chunked=True, gen_type=cfg.gen_type)

    def ensure_valid(name, t):
        if t is None:
            raise ValueError(f"{name} retornou None do modelo")
        if not torch.is_tensor(t[0]):
            raise ValueError(f"{name} retorno inválido: {type(t[0])}")
        if t[0].numel() == 0:
            raise ValueError(f"{name} retornou tensor vazio")
        return t[0]

    if save_separate:
        save_with_fallback(f"{save_dir}/vocal_audios{time.strftime('%m%d%H%S')}.flac", ensure_valid("vocal", wav_vocal), cfg.sample_rate)
        save_with_fallback(f"{save_dir}/bgm_audios{time.strftime('%m%d%H%S')}.flac",   ensure_valid("bgm",   wav_bgm),   cfg.sample_rate)

    save_with_fallback(target_wav_name, ensure_valid("mixed", wav_mix), cfg.sample_rate)

    return {
        "waveform": ensure_valid("mixed", wav_mix).unsqueeze(0),
        "sample_rate": cfg.sample_rate
    }


def song_infer_lowram(seperate_tokenizer,separator,audio_tokenizer,prompt_pt_path, save_dir,prompt_audio_path,auto_prompt_audio_type): #item dict
    item = {}
    target_wav_name = f"{save_dir}/song_audios{time.strftime('%m%d%H%S')}.flac"
    melody_is_wav = False
    if prompt_audio_path is not None:
        
        pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
        pmt_wav = pmt_wav.cuda()
        vocal_wav = vocal_wav.cuda()
        bgm_wav = bgm_wav.cuda()

        audio_tokenizer = audio_tokenizer.eval().cuda()
        with torch.no_grad():
            pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
        audio_tokenizer=None
        separator=None
        gc.collect()

        seperate_tokenizer = seperate_tokenizer.eval().cuda()
        with torch.no_grad():
            vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)
        del seperate_tokenizer
        gc.collect()
        
    elif auto_prompt_audio_type:
        print("auto_prompt_audio_type:",auto_prompt_audio_type) 
        assert  prompt_pt_path is not None ,"prompt模型不能为空,need prmmpt  model"
        auto_prompt = torch.load(prompt_pt_path,weights_only=False)
        #assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
        # if auto_prompt_audio_type == 'Auto': 
        #     prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
        # else:
        #     prompt_token = auto_prompt[auto_prompt_audio_type][np.random.randint(0, len(auto_prompt[auto_prompt_audio_type]))]
        prompt_token = auto_prompt[auto_prompt_audio_type][np.random.randint(0, len(auto_prompt[auto_prompt_audio_type]))]
        if torch.cuda.is_available():
            prompt_token = prompt_token.cuda()
        pmt_wav = prompt_token[:,[0],:]
        vocal_wav = prompt_token[:,[1],:]
        bgm_wav = prompt_token[:,[2],:]
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
    

    return item





