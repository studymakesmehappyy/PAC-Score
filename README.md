# PAC-Score 评估

## Emscore 和 PAC-Score 的主要区别

Emscore 和 PAC-Score 的主要区别在于它们使用的 CLIP 模型不同，计算公式是相同的。

### 计算 PAC-Score 的步骤

要计算 PAC-Score，请修改以下文件：

1. **在 `VATEX-EVAL-demo.py`、`emscore.utils.py` 和 `emscore.scorer.py` 中：**

将以下代码：

`import clip  # 用于 Emscore`

替换为：

`from models import open_clip  # 用于 PAC-Score`  
`from models.clip import clip  # 用于 PAC-Score`

2. **在 `emscore.scorer.py` 文件中：**

取消第 54-58 行的注释。

### 计算 PAC-Score 的命令

修改完成后，运行以下命令来计算 PAC-Score：

`python VATEX-EVAL-demo.py --storage_path $storage_path --use_n_refs 1 --use_feat_cache --use_idf`

### 参数说明

- `--storage_path`：(**必选**) 数据集存储目录的路径（例如 VATEX-EVAL）。  
  示例：`--storage_path ./VATEX-EVAL`
  
- `--use_n_refs`：(**可选**) 用于评估的参考字幕数量（默认：1）。  
  该值可以在 1 到 9 之间，表示 PAC-Score 将根据多个参考字幕与候选字幕进行比较，以评估生成字幕的质量。

- `--use_feat_cache`：(**可选**) 是否使用预计算的视频特征缓存。如果设置了该参数，脚本将使用之前提取的视频特征，而无需重新计算。

- `--use_idf`：(**可选**) 是否使用逆文档频率（IDF）进行权重计算，以更好地衡量候选字幕与参考字幕之间的匹配。
