# GLM-based Fake CTI Generation and Detection (Ready for GitHub)

This repository contains:
- generation/: Fine-tuning ChatGLM-6B with LoRA and generating fake CTI
- detection/: BERT-TextCNN CTI detection model

## Setup
1) Create a virtual environment (optional)
2) Install dependencies
```
pip install -r requirements.txt
```

## Data layout
Place your datasets under `project_github_ready/data`:
- Real_CTI.csv (for GLM fine-tuning)
- cti_train.json, cti_test.json (for detector training)

If `Real_CTI.csv` is missing, `src/generation/FCTIG.py` will try to build it from JSON files in `data/`.

## Generation
Example:
```
python src/generation/FCTIG.py --do_train=True --do_generate=True --base_model=/path/to/chatglm-6b
```
Outputs:
- LoRA adapters: experiments/checkpoints/FCTIG_lora_ckpt
- Generated CSV: data/Fake_CTI_generated.csv

## Detection
Example:
```
python src/detection/train_bert_textcnn.py
```
Saves best checkpoint to `FCTICM_ckpt/` in current directory.

## Notes
- No commercial APIs are used; all models are open-source (Hugging Face: ChatGLM-6B, BERT-Chinese).
- Text cleaning keeps CTI tokens like IP/URL/hash/CVE-friendly characters (.-/:_).