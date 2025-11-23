from datasets import load_dataset
import torch
from pathlib import Path
from cobra import load

hf_token = Path("/home/agf64/project/thinking_cobra/.hf_token").read_text().strip()
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')
# In case your GPU does not support bf16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
model_id = "cobra+3b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=dtype)

dataset = load_dataset("lmms-lab/COCO-Caption", split="val")
user_prompt = "Please carefully observe the image and come up with a caption for the image."

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

ref_captions = {}
generated_captions = {}

for idx, example in enumerate(dataset):
    print(idx)

    image_id = idx
    image = example["image"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    caption = example["answer"]
    ref_captions[image_id] = caption
    generated_text = vlm.generate(
        image,
        prompt_text,
        use_cache=True,
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
    )
    generated_captions[idx] = [generated_text]

# from matplotlib import pyplot as plt

# example = dataset[0]
# image = example["image"]
#
# # Display the image
# plt.imshow(image)
# plt.axis("off")
# plt.show()
#
# print("Ground truth captions:")
# print(ref_captions[0])
#
# print("Predicted captions:")
# print(generated_captions[0])
#
# N = 1000
# ref_captions = {}
# generated_captions = {}

# for idx, example in enumerate(dataset.select(range(N))):
#     image_id = idx
#     image = example["image"]
#     caption = example["answer"]
#     ref_captions[image_id] = caption
#     generated_captions[idx] = [caption[1]]
#
# print(ref_captions[999])
# print('\n')
# print(generated_captions[999])

from pycocoevalcap.bleu.bleu import Bleu

bleu_scorer = Bleu(4)  # Calculate BLEU scores up to 4-grams

bleu_scores, example_scores = bleu_scorer.compute_score(ref_captions, generated_captions)

num_captions = len(generated_captions)

with open("bleu_scores_output.txt", "w") as f:
    f.write(f"Number of generated captions: {num_captions}\n\n")
    f.write("BLEU Scores (BLEU-1 to BLEU-4):\n")
    f.write(str(bleu_scores) + "\n\n")  # Write the tuple of scores

    f.write(f"BLEU-1: {bleu_scores[0]:.4f}\n")
    f.write(f"BLEU-2: {bleu_scores[1]:.4f}\n")
    f.write(f"BLEU-3: {bleu_scores[2]:.4f}\n")
    f.write(f"BLEU-4: {bleu_scores[3]:.4f}\n\n")

    # Write paired captions
    for img_id in generated_captions:
        f.write(f"Image ID: {img_id}\n")
        f.write("Generated caption:\n")
        f.write(f"  {generated_captions[img_id][0]}\n")  # Hypothesis (one caption)
        f.write("Reference captions:\n")
        for ref in ref_captions[img_id]:
            f.write(f"  - {ref}\n")  # Multiple references possible
        f.write("\n")

print("BLEU scores and paired captions saved to bleu_scores_output.txt")

