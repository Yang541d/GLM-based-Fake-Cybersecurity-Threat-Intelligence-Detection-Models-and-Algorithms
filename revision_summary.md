# Revision Summary

本摘要逐项回应我们在此次重构与整理中的检查点（C1-C6），并给出关键代码位置（含文件与起始行号）作为证据，便于审计与复现。

## C1. FCTIG.py 注释与默认参数一致性（学习率等）
- 学习率默认值：在文件头部注释与常量定义中统一为 2e-5，避免“注释 1.5e-4、代码 2e-5”的不一致。
  - 常量位置：<mcfile name="FCTIG.py" path="src/generation/FCTIG.py"></mcfile>，PAPER_LR 定义起始行：<mcsymbol name="PAPER_LR" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="58" type="function"></mcsymbol>
- 其余默认路径改为基于仓库根目录自动解析，提升可移植性：
  - 根路径常量：<mcsymbol name="PROJECT_ROOT" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="49" type="function"></mcsymbol>
  - 生成/训练数据与LoRA保存目录：
    - <mcsymbol name="DEFAULT_OUTPUT_LORA_DIR" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="51" type="function"></mcsymbol>
    - <mcsymbol name="DEFAULT_GENERATED_CSV" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="52" type="function"></mcsymbol>
    - <mcsymbol name="DEFAULT_TRAIN_CSV" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="53" type="function"></mcsymbol>

## C2. Detection 训练脚本保留技术符号、不删样本、不限样本数
- 文本清洗仅移除不在白名单内字符，保留中文、英文、数字与技术符号 . - / : _：
  - <mcfile name="train_bert_textcnn.py" path="src/detection/train_bert_textcnn.py"></mcfile>
  - <mcsymbol name="clean_text_keep_cti" filename="train_bert_textcnn.py" path="src/detection/train_bert_textcnn.py" startline="30" type="function"></mcsymbol>
- 数据集类无任何随机删样或重标逻辑、也未做切片限制：
  - <mcsymbol name="CTISCleanDataset" filename="train_bert_textcnn.py" path="src/detection/train_bert_textcnn.py" startline="41" type="class"></mcsymbol>

## C3. 无商业API调用（离线可运行）
- 生成侧关键模块：
  - 数据集类：<mcsymbol name="CausalTextDataset" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="90" type="class"></mcsymbol>
  - 训练函数：<mcsymbol name="train" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="248" type="function"></mcsymbol>
  - 生成提示：<mcsymbol name="_build_generation_prompt" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="336" type="function"></mcsymbol>
  - 生成流程：<mcsymbol name="generate_csv" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="348" type="function"></mcsymbol>
  - 参数解析：<mcsymbol name="parse_args" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="439" type="function"></mcsymbol>
- 检测侧关键模块：
  - 模型结构：<mcfile name="bert_textcnn.py" path="src/detection/bert_textcnn.py"></mcfile>；
    - <mcsymbol name="BertTextCNN" filename="bert_textcnn.py" path="src/detection/bert_textcnn.py" startline="16" type="class"></mcsymbol>
    - <mcsymbol name="build_model" filename="bert_textcnn.py" path="src/detection/bert_textcnn.py" startline="61" type="function"></mcsymbol>
  - 训练主函数：<mcsymbol name="train_main" filename="train_bert_textcnn.py" path="src/detection/train_bert_textcnn.py" startline="95" type="function"></mcsymbol>

## C4. 新增评估指标输出（precision / recall / F1 / LMI）
- 评估函数：<mcfile name="eval_metrics.py" path="src/eval/eval_metrics.py"></mcfile>
  - <mcsymbol name="compute_metrics" filename="eval_metrics.py" path="src/eval/eval_metrics.py" startline="9" type="function"></mcsymbol>
  - <mcsymbol name="save_metrics_to_csv_json" filename="eval_metrics.py" path="src/eval/eval_metrics.py" startline="43" type="function"></mcsymbol>

## C5. 依赖清单
- 见 <mcfile name="requirements.txt" path="requirements.txt"></mcfile>，已包含 torch / transformers / peft / tqdm / pandas / scikit-learn / numpy。

## C6. README
- 见 <mcfile name="README.md" path="README.md"></mcfile>，涵盖环境准备、数据布局、训练与生成使用示例，以及注意事项。

## 其他增强
- 训练集缺失时，自动从 data/ 下 JSON 转 CSV：
  - <mcsymbol name="prepare_training_csv_if_missing" filename="FCTIG.py" path="src/generation/FCTIG.py" startline="213" type="function"></mcsymbol>
- 主程序优先尝试准备训练CSV，再执行训练/生成流转，提升“一键可跑”体验（位于 FCTIG.py 文件末尾主入口）。

---
如需进一步交叉验证，请根据上述文件与起始行号在代码中定位查看。