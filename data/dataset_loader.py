import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from .simple_tokenizer import SimpleTokenizer
import numpy as np

import data.img_transforms as T

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def transform_mask(mask,height,width):
    transform_train_mask = T.Compose([
        T.Resize((height , width)),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train_mask(mask)


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None,dataset_name=None):
        self.dataset = dataset
        self.transform = transform
        self.dataset_name = dataset_name
        self.tokenizer = SimpleTokenizer()
        self.text_length = 77
        self.truncate = True

        if self.dataset_name == 'ltcc':
            self.rep = '/LTCC_ReID/'
        # elif self.dataset_name == 'vcclothes':
        #     self.rep = '/VC-Clothes/'
        # elif self.dataset_name == 'celeb_light':
        #     self.rep ='/Celeb_light/'
        # elif self.dataset_name == 'celeb':
        #     self.rep ='/Celeb/'
        # elif self.dataset_name == 'deepchange':
        #     self.rep ='/DeepChangeDataset/'
        else:
            self.rep = '/'+self.dataset_name+'/'


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img_path, pid, camid, clothes_id, caption,  person_caption= self.dataset[index]
        mask_path = img_path.replace(self.rep,self.rep[0:-1]+'_parsing/')
        img = read_image(img_path)
        mask = read_image(mask_path)
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64)
        if self.transform is not None:
            img = self.transform(img)
        mask = transform_mask(mask,img.size()[-2],img.size()[-1])
        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        person_tokens = tokenize(person_caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return img, pid, camid, clothes_id,cloth_id_batch, mask,tokens, person_tokens


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id
        else:
            return clip, pid, camid