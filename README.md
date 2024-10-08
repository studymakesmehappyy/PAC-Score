2024.10.8更新
Emscore和PAC-Score的主要区别在于它们使用的CLIP模型不同，计算公式是相同的。

计算PAC-Score，需对以下文件进行修改：
在 VATEX-EVAL-demo.py、emscore.utils.py 和 emscore.scorer.py 中，将：

import clip  # for Emscore

替换为：

from models import open_clip  # for PAC-Score
from models.clip import clip  # for PAC-Score

同时，取消 emscore.scorer.py 文件中第54-58行的注释。

修改完成后，运行以下命令来计算PAC-Score：
python VATEX-EVAL-demo.py --storage_path $storage_path --use_n_refs 1 --use_feat_cache --use_idf

--storage_path: （必选）数据集存储目录的路径（例如 VATEX-EVAL）。 示例：--storage_path ./VATEX-EVAL

--use_n_refs: （可选）用于评估的参考字幕数量（默认：1）。该值可以在 1 到 9 之间。这意味着 PAC-Score 将根据几个参考字幕与候选字幕进行比较，以评估生成字幕的质量。

--use_feat_cache: （可选）是否使用预计算的视频特征缓存。如果设置了该参数，脚本将使用之前提取的视频特征，而无需重新计算。

--use_idf: （可选）是否使用逆文档频率（IDF）进行权重计算，以更好地衡量候选字幕与参考字幕之间的匹配。













——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
PAC-S 与 EMScore 集成
该项目集成了 PAC-Score 和 EMScore，用于评估视频特征与多个参考字幕的相关性。你可以使用 pacs_emscore_integration.py 脚本，通过多个可配置选项来计算视频和文本的相关性。

您可以使用以下命令运行 pacs_emscore_integration.py 脚本：

python pacs_emscore_integration1.py --storage_path ./VATEX-EVAL --use_n_refs 9 --use_feat_cache --use_idf --clip_model ViT-B/32 --compute_refpac

命令行参数：
--storage_path: （必选）数据集存储目录的路径（例如 VATEX-EVAL）。 示例：--storage_path ./VATEX-EVAL

--use_n_refs: （可选）用于评估的参考字幕数量（默认：1）。该值可以在 1 到 9 之间。这意味着 PAC-Score 将根据几个参考字幕与候选字幕进行比较，以评估生成字幕的质量。

--use_feat_cache: （可选）是否使用预计算的视频特征缓存。如果设置了该参数，脚本将使用之前提取的视频特征，而无需重新计算。

--use_idf: （可选）是否使用逆文档频率（IDF）进行权重计算，以更好地衡量候选字幕与参考字幕之间的匹配。

--clip_model: （可选）选择要使用的 CLIP 模型。默认值为 ViT-B/32，可选择 ViT-B/32 或 open_clip_ViT-L/14。

--compute_refpac: （可选）是否计算 RefPAC-S 分数。如果提供此参数，脚本将计算 RefPAC-S 分数，这是结合候选字幕与参考字幕之间的关系进行的额外评估。
