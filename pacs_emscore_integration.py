import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import pickle
import numpy as np
import json
import glob
import torch
from tqdm import tqdm
from emscore import EMScorer
from emscore.utils import get_idf_dict, compute_correlation_uniquehuman
from pacscore.models import open_clip
from pacscore.models import clip
from pacscore.evaluation import PACScore1, PACScore2, RefPACScore2

_MODELS = {
    "ViT-B/32": "./pacscore/checkpoints/clip_ViT-B-32.pth",
    "open_clip_ViT-L/14": "./pacscore/checkpoints/openClip_ViT-L-14.pth"
}

def normalize_matrix(A):
    assert len(A.shape) == 2
    A_norm = torch.linalg.norm(A, dim=-1, keepdim=True)
    return A / A_norm

def get_feats_dict(feat_dir_path, de_video_ids):
    """从缓存中加载视频特征并归一化处理"""
    print('Loading cached features...')
    feats_dict = {}
    for vid in tqdm(de_video_ids):
        file_path = os.path.join(feat_dir_path, vid + '.pt')
        if os.path.exists(file_path):
            data = torch.load(file_path)
            # 计算并归一化每个视频的全局特征
            global_feature = normalize_matrix(torch.mean(data, dim=0, keepdim=True)).squeeze()
            feats_dict[vid] = global_feature
    print(f"Total number of keys: {len(feats_dict)}")
    return feats_dict

def compute_pac_scores(model, preprocess, video_feats_dict, de_video_ids, candidates, references, device, compute_refpac=False):
    """计算PAC-S和RefPAC-S评分"""
    video_feats = []
    gen_cs = []
    gts_cs = []

    for vid, gts_i, gen_i in zip(video_feats_dict.keys(), references, candidates):
        video_feats.append(video_feats_dict[vid])
        gen_cs.append(gen_i)
        gts_cs.append(gts_i)

    # 将video_feats中的每个Tensor转换为NumPy数组并堆叠为二维数组
    video_feats = np.array([feat.numpy() for feat in video_feats])

    # PAC-S 评分计算
    _, pac_scores, candidate_feats, len_candidates = PACScore1(
        model, video_feats, candidates, device, w=2.0)
    results = {'PAC-S': pac_scores}
    print("PAC-S.shape", results['PAC-S'].shape)

    # RefPAC-S 评分计算（如果需要）
    if compute_refpac:
        _, per_instance_text_text = RefPACScore2(
            model, references, candidate_feats, device, torch.tensor(len_candidates))
        print(type(per_instance_text_text))
        refpac_scores = (pac_scores + per_instance_text_text) / 2
        results['RefPAC-S'] = refpac_scores
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAC-S & EMScore Integration Evaluation')
    parser.add_argument('--storage_path', default='', type=str, help='存储 VATEX-EVAL 数据集的路径')
    parser.add_argument('--use_n_refs', default=1, type=int, help='评估时使用多少个参考描述（1~9）')
    parser.add_argument('--use_feat_cache', default=True, action='store_true', help='是否使用预处理好的视频特征')
    parser.add_argument('--use_idf', action='store_true', default=True)
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', choices=['ViT-B/32', 'open_clip_ViT-L/14'])
    parser.add_argument('--compute_refpac', action='store_true', help='是否计算RefPAC-S')
    
    opt = parser.parse_args()

    # 加载数据集
    samples_list = pickle.load(open(os.path.join(opt.storage_path, 'candidates_list.pkl'), 'rb'))
    gts_list = pickle.load(open(os.path.join(opt.storage_path, 'gts_list.pkl'), 'rb'))
    all_human_scores = pickle.load(open(os.path.join(opt.storage_path, 'human_scores.pkl'), 'rb'))
    all_human_scores = np.transpose(all_human_scores.reshape(3, -1), (1, 0))

    video_ids = pickle.load(open(os.path.join(opt.storage_path, 'video_ids.pkl'), 'rb'))

    cands = []
    refs = []

    # 加载去重视频ID的JSON文件
    json_file_path = "./de_duplicated_video_ids.json"
    with open(json_file_path, 'r') as f:
        de_video_ids = json.load(f)
    
    # 打印前几条视频ID，确认加载成功
    print("First 5 video IDs:", de_video_ids[:5])
    
    for vid, sample, ref in zip(video_ids, samples_list, gts_list):
        cands.append(sample)
        refs.append(ref[:opt.use_n_refs])

    refs = [ref.tolist() for ref in refs]

    print("*************video_ids.shapeshape", len(video_ids))
    print(f"candidates2: {len(refs)}")

    # 打印处理后的 cands_list 和 refs_list 数据
    print("Processed cands list (first two entries):")
    print(cands[:2])

    print("\nProcessed refs list (first two entries):")
    print(refs[:2])

    # 准备视频特征
    if opt.use_feat_cache:
        vid_clip_feats_dir = os.path.join(opt.storage_path, 'VATEX-EVAL_video_feats')
        video_feats_dict = get_feats_dict(vid_clip_feats_dir, de_video_ids)

    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if opt.clip_model.startswith('open_clip'):
        print("Using Open CLIP Model: " + opt.clip_model)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k')
    else:
        print("Using CLIP Model: " + opt.clip_model)
        model, preprocess = clip.load(opt.clip_model, device=device)

    model = model.to(device)
    model = model.float()

    checkpoint = torch.load(_MODELS[opt.clip_model])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 计算 PAC-S 和 RefPAC-S 评分
    pac_scores = compute_pac_scores(model, preprocess, video_feats_dict, de_video_ids, cands, refs, device, compute_refpac=opt.compute_refpac)
    print("Per (Scores for each video):", pac_scores['PAC-S'])
    
    # 计算与人工评分的相关性
    if 'PAC-S' in pac_scores:
        print('PAC-S correlation --------------------------------------')
        kendall, spear = compute_correlation_uniquehuman(pac_scores['PAC-S'], all_human_scores)
        print(f'Kendall: {kendall}, Spearman: {spear}')

    if 'RefPAC-S' in pac_scores:
        print('RefPAC-S correlation --------------------------------------')
        kendall, spear = compute_correlation_uniquehuman(pac_scores['RefPAC-S'], all_human_scores)
        print(f'Kendall: {kendall}, Spearman: {spear}')
