# ComfyUI_SongGeneration
 [SongGeneration](https://github.com/tencent-ailab/SongGeneration):High-Quality Song Generation with Multi-Preference Alignment (SOTA),you can try VRAM>12G

# Tips 
* Test env（插件测试环境）：window11，python3.11， torch2.6 ，cu124， VR12G,（transformers 4.45.1 官方原本限定小于4.44，我干掉了那个警告）

# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SongGeneration.git
```

# 2. Requirements  
* window平台最难装的就是[fairseq](https://github.com/facebookresearch/fairseq)库，python3.11的建议用轮子安装[liyaodev/fairseq](https://github.com/liyaodev/fairseq/releases/tag/v0.12.3.1)；
* 如果缺失库，打开requirements_orgin.txt文件，看是少了哪个，手动安装；
* The most difficult thing to install on the Windows platform is the Fairseq library. It is recommended to install it on wheels for version 3.11 [liyaodev/fairseq](https://github.com/liyaodev/fairseq/releases/tag/v0.12.3.1)；
* If the library is missing, open the ’requirements_orgin.txt‘ file and see which one is missing, then manually install it；  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  ckpt  from [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration/tree/main)   国内建议魔搭[AI-ModelScope/SongGeneration](https://www.modelscope.cn/models/AI-ModelScope/SongGeneration/files)  
* 3.1.2 download htdemucs.pth [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration/tree/main/third_party/demucs/ckpt)
* 文件结构如下：
```
--  ComfyUI/models/SongGeneration/
    |-- htdemucs.pth #150M
    |-- ckpt/  # 24.4G all 整个文件夹的大小
        |--encode-s12k.pt  # 3.68G
        |--prompt.pt  # 3M
        |--model_1rvq/  
            |--all files  # 全部文件 
        |--model_septoken/
            |--all files  # 全部文件
        |--models--lengyue233--content-vec-best/
            |--all files  # 全部文件
        |--songgeneration_base_zh/
            |--all files  # 全部文件
        |--vae/
            |--all files  # 全部文件
```
# 4 Example
![](https://github.com/smthemex/ComfyUI_SongGeneration/blob/main/example_workflows/example.png)

# 5 Citation
```
@article{lei2025levo,
  title={LeVo: High-Quality Song Generation with Multi-Preference Alignment},
  author={Lei, Shun and Xu, Yaoxun and Lin, Zhiwei and Zhang, Huaicheng and Tan, Wei and Chen, Hangting and Yu, Jianwei and Zhang, Yixuan and Yang, Chenyu and Zhu, Haina and Wang, Shuai and Wu, Zhiyong and Yu, Dong},
  journal={arXiv preprint arXiv:2506.07520},
  year={2025}
}
```



