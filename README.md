# ComfyUI_SongGeneration
 [SongGeneration](https://github.com/tencent-ailab/SongGeneration):High-Quality Song Generation with Multi-Preference Alignment (SOTA),you can try VRAM>12G

# Update
* 支持v2模型，加入transformer2的层卸载以方便12G Vram用户，需要使用新的工作流,加入gguf模型支持，以加快推理速度，修复offload的bug
* support v2 model, gguf,fix offload bugs.

# Previous
* 11/22 修复入参顺序颠倒的错误，修复一个找很久没找到的print，并修复其模块导入问题
* 10/23 同步官方代码，删除fairseq库，已无安装难度； 
* 10/21同步官方代码，精简模型加载，删除hubert模型，优化lm模型加载顺序，避免转移到显存时峰值OOM；  
* 10/18  修改加载流程，支持最新的full ，new，large模型，large模型12GVram可能会OOM，修复高版本transformer 的函数错误/Modify the loading process to support the latest full, new, and large models, and fix function errors in higher versions of transformers   
*  07/29，支持bgm和人声（vocal，目前还是有bgm底噪）单独输出，选择mixed为合成全部，模型加载方式更合理，去掉诸多debug打印，新增save_separate按钮，开启则保存三个音频（bgm，vocal，mixed）；
* Test env（插件测试环境）：window11，python3.11， torch2.6 ，cu124， VR12G,（transformers 4.45.1）


# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SongGeneration.git
```

# 2. Requirements  

* 如果缺失库，打开requirements_orgin.txt文件，看是少了哪个，手动安装；
* If the library is missing, open the ’requirements_orgin.txt‘ file and see which one is missing, then manually install it；  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.0 v2 version ckpt from [SongGeneration-v2-large](https://huggingface.co/lglg666/SongGeneration-v2-large/tree/main)  # v2 模型地址，[new_auto_prompt.pt](https://github.com/tencent-ailab/SongGeneration/blob/main/tools/new_auto_prompt.pt)  
* 3.1.1 download  ckpt  from [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration/tree/main)   国内建议魔搭[AI-ModelScope/SongGeneration](https://www.modelscope.cn/models/AI-ModelScope/SongGeneration/files)    
* 3.1.2  [new base](https://huggingface.co/lglg666/SongGeneration-base-new),[large ](https://huggingface.co/lglg666/SongGeneration-large),[full](https://huggingface.co/lglg666/SongGeneration-base-full)    
* 3.1.3 new prompt,[emb](https://github.com/tencent-ailab/SongGeneration/tree/main/tools)   
* 3.1.4 download htdemucs.pth [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration/tree/main/third_party/demucs/ckpt)
* 3.1.5 gguf [smthem/SongGeneration-v2-large-gguf](https://huggingface.co/smthem/SongGeneration-v2-large-gguf) # 国内用户可以去我云盘拉取
* 文件结构如下,修改了加载流程，原来的结构也能用：
```
--  ComfyUI/models/SongGeneration/ # 24.4G all 整个文件夹的大小
    |-- htdemucs.pth #150M
    |--prompt.pt  # 3M
    |--new_prompt.pt  # 3M
    |--new_auto_prompt.pt  # v2 version V2版本
    |--model_2.safetensors
    |--model_2_fixed.safetensors
    |--new_model.pt  # rename from model.pt #可选
    |--large_model.pt  #  rename from model.pt #可选
    |--large_model_v2.pt  # must to rename from model.pt #必须重命名才能识别 v2版本
    |--large_model_v2_Q8_0.gguf  # or Q6,optional 可选
    |-- ckpt/  
        |--encode-s12k.pt  # 3.68G
--  ComfyUI/models/vae/
    |--autoencoder_music_1320k.ckpt
--  ComfyUI/models/gguf/
    |--large_model_v2_Q8_0.gguf  # or Q6,optional 可选
```
# 4 Example
![](https://github.com/smthemex/ComfyUI_SongGeneration/blob/main/example_workflows/SongGeneration_g.png)
![](https://github.com/smthemex/ComfyUI_SongGeneration/blob/main/example_workflows/SongGeneration.png)

# 5 Citation
```
@article{lei2025levo,
  title={LeVo: High-Quality Song Generation with Multi-Preference Alignment},
  author={Lei, Shun and Xu, Yaoxun and Lin, Zhiwei and Zhang, Huaicheng and Tan, Wei and Chen, Hangting and Yu, Jianwei and Zhang, Yixuan and Yang, Chenyu and Zhu, Haina and Wang, Shuai and Wu, Zhiyong and Yu, Dong},
  journal={arXiv preprint arXiv:2506.07520},
  year={2025}
}
```



