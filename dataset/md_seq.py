# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the dance dataset. """
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def paired_collate_fn(insts):
    src_seq, tgt_seq, name = list(zip(*insts))
    src_pos = np.array([
        [pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in src_seq])

    src_seq = torch.FloatTensor(src_seq)
    src_pos = torch.LongTensor(src_pos)
    tgt_seq = torch.FloatTensor(tgt_seq)

    return src_seq, src_pos, tgt_seq, name

def text_collate_fn(batch):
    """
    自定义 collate_fn，用于处理变长的文本特征
    
    Args:
        batch: list of tuples, 每个元素是 (music, dance, text_upper, text_lower, text_torso, text_whole, text_simple_tag, text_meta)
               - music, dance: 定长数组
               - text_upper, text_lower, text_torso, text_whole, text_simple_tag: 变长数组 (N, Dim) 或 (0, Dim)
               - text_meta: 元数据列表
    
    Returns:
        tuple: (music_batch, dance_batch, text_upper_batch, text_lower_batch, 
                text_torso_batch, text_whole_batch, text_simple_tag_batch, text_meta_batch)
    """
    # 解包 batch
    musics, dances, texts_upper, texts_lower, texts_torso, texts_whole, texts_simple_tag, texts_meta = zip(*batch)
    
    # 1. 处理 music 和 dance（定长，直接 stack）
    music_batch = torch.FloatTensor(np.stack(musics))
    if dances is not None:
        dance_batch = torch.FloatTensor(np.stack(dances))
    else:
        dance_batch = None # for finedance
    # 2. 处理文本特征（变长，需要 padding）
    # 将 numpy 数组转为 tensor，并进行 padding
    # pad_sequence 需要输入是 list of tensors，输出是 (max_len, batch, dim)
    # 我们需要转置为 (batch, max_len, dim)
    
    def pad_text_batch(text_list):
        """
        将一个 batch 的变长文本特征进行 padding。
        若该文本模态被禁用（所有元素均为 None），直接返回 None。
        Args:
            text_list: list of np.array or None, 每个 array shape 是 (N_i, Dim)
        Returns:
            padded_tensor: shape (batch, max_len, dim)，或 None
        """
        # 判断 None 情况
        none_flags = [x is None for x in text_list]
        if all(none_flags):
            # 该模态全局被禁用 -> 返回 None
            return None
        if any(none_flags):
            # 同一 batch 内出现混合 None/非None —— 属于数据一致性错误
            raise ValueError(
                "[text_collate_fn] 检测到同一 batch 内某文本模态出现混合 None 情况"
                "（部分样本为 None，部分不为 None）。\n"
                "这是全局 ablation 实验，不应存在样本级混用。\n"
                "请检查 MoDaSeq not_use_* flags 是否对所有样本一致生效。"
            )

        # 转换为 tensor list
        tensor_list = [torch.FloatTensor(text) for text in text_list]
        
        # 使用 pad_sequence 进行 padding (默认用 0 填充)
        # pad_sequence 的输出是 (max_len, batch, dim)
        padded = pad_sequence(tensor_list, batch_first=True, padding_value=0.0)
        
        # 返回 (batch, max_len, dim)
        return padded
    
    text_upper_batch = pad_text_batch(texts_upper)
    text_lower_batch = pad_text_batch(texts_lower)
    text_torso_batch = pad_text_batch(texts_torso)
    text_whole_batch = pad_text_batch(texts_whole)
    text_simple_tag_batch = pad_text_batch(texts_simple_tag)
    
    # 3. 处理 text_meta（保持为 Python List）
    text_meta_batch = list(texts_meta)
    
    return (music_batch, dance_batch, text_upper_batch, text_lower_batch, 
            text_torso_batch, text_whole_batch, text_simple_tag_batch, text_meta_batch)




class MoDaSeq(Dataset):
    def __init__(self, musics, dances=None, texts_upper=None, texts_lower=None, texts_torso=None, texts_whole=None, texts_simple_tag=None, texts_meta=None,
                 not_use_upper=False, not_use_lower=False, not_use_torso=False, not_use_whole=False, not_use_simple_tag=False):
        if dances is not None:
            print('Using Dance')
        else:
            print('Dance is None')
        if dances is not None:
            assert (len(musics) == len(dances)), \
                'the number of dances should be equal to the number of musics'
        self.musics = musics
        self.dances = dances
        self.texts_upper = texts_upper
        self.texts_lower = texts_lower
        self.texts_torso = texts_torso
        self.texts_whole = texts_whole
        self.texts_simple_tag = texts_simple_tag
        self.texts_meta = texts_meta
        # not_use_* flags: when True, __getitem__ always returns None for that modality
        self.not_use_upper = not_use_upper
        self.not_use_lower = not_use_lower
        self.not_use_torso = not_use_torso
        self.not_use_whole = not_use_whole
        self.not_use_simple_tag = not_use_simple_tag
        # if clip_names is not None:
        # self.clip_names = clip_names

    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        dance = self.dances[index] if self.dances is not None else None
        text_upper = None if self.not_use_upper else (self.texts_upper[index] if self.texts_upper is not None else None)
        text_lower = None if self.not_use_lower else (self.texts_lower[index] if self.texts_lower is not None else None)
        text_torso = None if self.not_use_torso else (self.texts_torso[index] if self.texts_torso is not None else None)
        text_whole = None if self.not_use_whole else (self.texts_whole[index] if self.texts_whole is not None else None)
        text_simple_tag = None if self.not_use_simple_tag else (self.texts_simple_tag[index] if self.texts_simple_tag is not None else None)
        text_meta  = self.texts_meta[index]  if self.texts_meta  is not None else None
        return self.musics[index], dance, text_upper, text_lower, text_torso, text_whole, text_simple_tag, text_meta

