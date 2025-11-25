# LLaVA-CoT-100k Dataset Source Information

## Official Sources

### GitHub Repository
- **URL**: [https://github.com/PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **Description**: Official repository for LLaVA-CoT project
- **Paper**: "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step" (ICCV 2025)
- **ArXiv**: [arXiv:2411.10440](https://arxiv.org/abs/2411.10440)

### HuggingFace Dataset
- **Dataset URL**: [https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)
- **Description**: 100,000 visual question-answering samples with structured reasoning annotations
- **Format**: JSON file with LLaVA-style conversations containing structured reasoning tags

### HuggingFace Model
- **Model URL**: [https://huggingface.co/Xkev/Llama-3.2V-11B-cot](https://huggingface.co/Xkev/Llama-3.2V-11B-cot)
- **Description**: Pretrained LLaVA-CoT model weights (for reference)

## Dataset Format

The LLaVA-CoT-100k dataset uses structured reasoning tags in the assistant responses:

```
<SUMMARY>Problem understanding and approach</SUMMARY>
<CAPTION>Visual interpretation from the image</CAPTION>
<REASONING>Step-by-step logical reasoning</REASONING>
<CONCLUSION>Final answer/conclusion</CONCLUSION>
```

## Download Instructions

### Automatic Download
The dataset will be automatically downloaded when you run:
```bash
python fine_tune/prepare_llava_cot.py --dataset_root data
```

### Manual Download
1. Visit [HuggingFace Dataset](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)
2. Download the JSON file (may be named `data.json`, `train.json`, or similar)
3. Place it in `data/download/llava-cot-100k/llava_cot_100k.json`

### Using HuggingFace Datasets Library
```python
from datasets import load_dataset

dataset = load_dataset("Xkev/LLaVA-CoT-100k")
# Save to local file if needed
dataset['train'].to_json("data/download/llava-cot-100k/llava_cot_100k.json")
```

## Notes

- The dataset file name on HuggingFace may vary. If automatic download fails, check the HuggingFace dataset page for the actual file name.
- Images may be referenced by paths that need to be resolved relative to other datasets (COCO, GQA, etc.)
- The dataset follows the LLaVA conversation format, making it compatible with existing `FinetuneDataset` class

## Citation

If you use the LLaVA-CoT dataset, please cite:

```bibtex
@InProceedings{Xu_2025_ICCV,
    author    = {Xu, Guowei and Jin, Peng and Wu, Ziang and Li, Hao and Song, Yibing and Sun, Lichao and Yuan, Li},
    title     = {LLaVA-CoT: Let Vision Language Models Reason Step-by-Step},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {2087-2098}
}
```

