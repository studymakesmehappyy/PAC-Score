## Emscore 和 PAC-Score 的区别

Emscore 和 PAC-Score 的主要区别在于它们使用的 CLIP 模型不同，但计算公式是相同的。

### 计算 PAC-Score，需对以下文件进行修改：

在 `VATEX-EVAL-demo.py`、`emscore.utils.py` 和 `emscore.scorer.py` 中，将：

```python
import clip  # for Emscore

替换为：

```python
from models import open_clip  # for PAC-Score
from models.clip import clip  # for PAC-Score

同时，取消 emscore.scorer.py 文件中第54-58行的注释。

### 运行以下命令来计算 PAC-Score：

python VATEX-EVAL-demo.py --storage_path $storage_path --use_n_refs 1 --use_feat_cache --use_idf

参数说明：
--storage_path（必选）：数据集存储目录的路径（例如 VATEX-EVAL）。示例：--storage_path ./VATEX-EVAL
--use_n_refs（可选）：用于评估的参考字幕数量，默认值为1。该值可以在1到9之间，表示 PAC-Score 将根据多个参考字幕与候选字幕进行比较，以评估生成字幕的质量。
--use_feat_cache（可选）：是否使用预计算的视频特征缓存。如果设置该参数，脚本将使用之前提取的视频特征，而无需重新计算。
--use_idf（可选）：是否
