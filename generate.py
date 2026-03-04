import sys
import os
import folder_paths
import time
import librosa
import torch
import torchaudio
import numpy as np
import gc
from omegaconf import OmegaConf
from .SongGeneration.codeclm.models import builders
from .SongGeneration.codeclm.trainer.codec_song_pl import CodecLM_PL
from .SongGeneration.codeclm.models import CodecLM
from .SongGeneration.third_party.demucs.models.pretrained import get_model_from_yaml
import re
current_node_path = os.path.dirname(os.path.abspath(__file__))

auto_prompt_type = ['Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack','Auto']
def check_language_by_text(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_count = len(re.findall(chinese_pattern, text))
    english_count = len(re.findall(english_pattern, text))
    chinese_ratio = chinese_count / len(text)
    english_ratio = english_count / len(text)
    if chinese_ratio >= 0.2:
        return "zh"
    elif english_ratio >= 0.5:
        return "en"
    else:
        return "en"
    
def load_audio_by_librosa(f):
    a, fs= librosa.load(f, sr=48000)
    a = torch.tensor(a).unsqueeze(0)
    if (fs != 48000):
        a = torchaudio.functional.resample(a, fs, 48000)
    if a.shape[-1] >= 48000*10:
        a = a[..., :48000*10]
    return a[:, 0:48000*10], 48000

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
        try:
            a, fs = torchaudio.load(f)
        except:
            a, fs = load_audio_by_librosa(f)
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


def load_gguf_checkpoint_vl(gguf_checkpoint_path):
    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
 
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )
        weights = torch.from_numpy(tensor.data.copy())
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
    
    del reader
    gc.collect()
    return parsed_parameters


def build_model(Weigths_Path,infer_model_path,version,use_flash_attn,offload_audiolm):
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

    cfg_path = os.path.join(current_node_path, f'SongGeneration/conf/{infer_model_type}_config.yaml')  if version=="v1" else os.path.join(current_node_path, f'SongGeneration/conf/{infer_model_type}_config_v2.yaml')
    print(cfg_path)
    cfg = OmegaConf.load(cfg_path)
    cfg.mode = 'inference'
    cfg.lm.use_flash_attn_2 = use_flash_attn
    cfg.vae_config=f"{Weigths_Path}/vae/stable_audio_1920_vae.json"
    cfg.vae_model=f"{Weigths_Path}/vae/autoencoder_music_1320k.ckpt"
    cfg.audio_tokenizer_checkpoint=f"Flow1dVAE1rvq_{Weigths_Path}/model_1rvq/model_2_fixed.safetensors"
    cfg.audio_tokenizer_checkpoint_sep=f"Flow1dVAESeparate_{Weigths_Path}/model_septoken/model_2.safetensors"
    cfg.conditioners.type_info.QwTextTokenizer.token_path=os.path.join(current_node_path,"SongGeneration/third_party/Qwen2-7B")
    cfg.version = version
    cfg.offload_audiolm = offload_audiolm
    
    audiolm = builders.get_lm_model(cfg,version,offload_audiolm) 
    if  not infer_model_path.endswith(".gguf"):       
        checkpoint = torch.load(infer_model_path, map_location='cpu',weights_only=False)
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        del checkpoint
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        del audiolm_state_dict
    else:
        from diffusers import  GGUFQuantizationConfig
        from diffusers.quantizers.gguf import GGUFQuantizer
        from diffusers.models.model_loading_utils import load_model_dict_into_meta
        g_config = GGUFQuantizationConfig(compute_dtype=torch.float16)
        hf_quantizer = GGUFQuantizer(quantization_config=g_config)
        hf_quantizer.pre_quantized = True
        model_state_dict=load_gguf_checkpoint_vl(infer_model_path) 
        gc.collect()    
        hf_quantizer._process_model_before_weight_loading(
            audiolm,
            device_map=None,
            state_dict=model_state_dict
            )
        load_model_dict_into_meta(
            audiolm, 
            model_state_dict, 
            hf_quantizer=hf_quantizer,
            device_map=None,
            dtype=torch.float16,
            
        )
        
        hf_quantizer._process_model_after_weight_loading(audiolm)
        del model_state_dict
    gc.collect()
    audiolm.eval().to(torch.float16)

    return audiolm,cfg


def infer_stage2(item,audiolm,max_duration,lyric,descriptions,gen_type,cfg,cfg_coef = 1.5, temp = 0.9,top_k = 50,top_p = 0.0,record_tokens = True,record_window = 50,offload_audiolm = False):
    #ckpt_path = os.path.join(Weigths_Path, 'songgeneration_base/model.pt')
   
    item_copy = {
        'pmt_wav': item['pmt_wav'], 
        'vocal_wav': item['vocal_wav'],
        'bgm_wav': item['bgm_wav'],
        'melody_is_wav': item['melody_is_wav'],
        'idx': item['idx'],
        'wav_path': item['wav_path']
    }
    if offload_audiolm:
        # from .SongGeneration.codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
        # audiolm_offload_param = OffloadParamParse.parse_config(audiolm, cfg.offload.audiolm)
        # audiolm_offload_param.show()
        # offload_profiler = OffloadProfiler(device_index=0, **(audiolm_offload_param.init_param_dict()))
        # offload_profiler.offload_layer(**(audiolm_offload_param.offload_layer_param_dict()))
        # offload_profiler.clean_cache_wrapper(**(audiolm_offload_param.clean_cache_param_dict()))
        audiolm.to_cuda("cuda")
    else:
        audiolm.cuda()
    
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
    items=inference_lowram_step2(model,lyric,descriptions,item_copy,gen_type)
    audiolm = audiolm.cpu()
    del audiolm
    model=None
    gc.collect()
    torch.cuda.empty_cache()
    return items


def inference_lowram_step2(model,lyric,descriptions,item,gen_type):

    generate_inp = {
        'lyrics': [lyric.replace("  ", " ")] if gen_type != 'bgm' else '.',
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


def song_infer_lowram(seperate_tokenizer,separator,audio_tokenizer,prompt_pt_path, save_dir,prompt_audio_path,auto_prompt_audio_type,lyric): #item dict
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
        lang = check_language_by_text(lyric)
        #prompt_token = auto_prompt[auto_prompt_audio_type][np.random.randint(0, len(auto_prompt[auto_prompt_audio_type]))]
        prompt_token = auto_prompt[auto_prompt_audio_type][lang][np.random.randint(0, len(auto_prompt[auto_prompt_audio_type][lang]))]
        del auto_prompt  
        if torch.cuda.is_available():
            prompt_token = prompt_token.cuda()
        pmt_wav = prompt_token[:,[0],:]
        vocal_wav = prompt_token[:,[1],:]
        bgm_wav = prompt_token[:,[2],:]
        del prompt_token
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
    item["gt_lyric"] = lyric
    

    return item





