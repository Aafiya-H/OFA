from fairseq import utils,tasks
from fairseq import checkpoint_utils
from tasks.mm_tasks import StoryContinuationTask

import torch
import argparse
from PIL import Image

tasks.register_task("story_continuation",StoryContinuationTask)
use_cuda = torch.cuda.is_available()

models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths('<checkpoint path>')
    # arg_overrides=overrides
)

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()

def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

def main(args):
    src_text = encode_text(args.first_descriptive_text, append_bos=True,
                           append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])

    image = Image.open(args.first_image_path)

    # code to get continued story from image and text

    return continued_story


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_image_path", type=str, help="Path of the first image")
    parser.add_argument("--first_descriptive_text", type=str, help="Descriptive text for the first image")
    args = parser.parse_args()
    continued_story = main(args)