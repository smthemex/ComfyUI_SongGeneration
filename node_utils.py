# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
from PIL import Image
import numpy as np
import cv2
import gc

from comfy.utils import common_upscale,ProgressBar
from huggingface_hub import hf_hub_download

cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def gc_clear():
    gc.collect()
    torch.cuda.empty_cache()

def cv2pil(cv_image):
    """
    将OpenCV图像转换为PIL图像
    :param cv_image: OpenCV图像
    :return: PIL图像
    """
    # 将图像从BGR转换为RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # 使用PIL的Image.fromarray方法将NumPy数组转换为PIL图像
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def add_mask(masks,images):
    if images.max() > 1.0:
        white_value = 255.0
    else:
        white_value = 1.0

    if masks.max() > 1.0:
        masks = masks / 255.0
    
    masks = masks.unsqueeze(-1)

    white_bg = torch.ones_like(images) * white_value
    
    if images.shape[-1] == 4:
        images = images[..., :3]
    
    tensor_image = images * masks + white_bg * (1 - masks)
    return tensor_image



def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pil_list(image,width,height):
    B,_,_,_=image.size()
    if  B==1:
        ref_image_list=[tensor2pil_upscale(image,width,height)]
    else:
        img_list = list(torch.chunk(image, chunks=B))
        ref_image_list = [tensor2pil_upscale(img,width,height) for img in img_list]
    return ref_image_list


def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def tensor2pil_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil


def tensor2cv(tensor_image,RGB2BGR=True):
    if len(tensor_image.shape)==4:#bhwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu().detach()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    if RGB2BGR:
        img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            #print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry

def equalize_lists(list1, list2):
    """
    比较两个列表的长度，如果不一致，则将较短的列表复制以匹配较长列表的长度。
    
    参数:
    list1 (list): 第一个列表
    list2 (list): 第二个列表
    
    返回:
    tuple: 包含两个长度相等的列表的元组
    """
    len1 = len(list1)
    len2 = len(list2)
    
    if len1 == len2:
        pass
    elif len1 < len2:
        print("list1 is shorter than list2, copying list1 to match list2's length.")
        list1.extend(list1 * ((len2 // len1) + 1))  # 复制list1以匹配list2的长度
        list1 = list1[:len2]  # 确保长度一致
    else:
        print("list2 is shorter than list1, copying list2 to match list1's length.")
        list2.extend(list2 * ((len1 // len2) + 1))  # 复制list2以匹配list1的长度
        list2 = list2[:len1]  # 确保长度一致
    
    return list1, list2

def file_exists(directory, filename):
    # 构建文件的完整路径
    file_path = os.path.join(directory, filename)
    # 检查文件是否存在
    return os.path.isfile(file_path)

def download_weights(file_dir,repo_id,subfolder="",pt_name=""):
    if subfolder:
        file_path = os.path.join(file_dir,subfolder, pt_name)
        sub_dir=os.path.join(file_dir,subfolder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(file_path):
            file_path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=pt_name,
                local_dir = file_dir,
            )
        return file_path
    else:
        file_path = os.path.join(file_dir, pt_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir=file_dir,
            )
        return file_path
