from fairseq import tasks
from tasks.mm_tasks import VisualStorytelling

import argparse
import torch

tasks.register_task("visual_storytelling",VisualStorytelling)
use_cuda = torch.cuda.is_available()
use_fp16 = True if use_cuda else False

def main(args):
  pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("image_path", type=str, help="Path to the image")
  parser.add_argument("text", type=str, text="Image description")
  args = parser.parse_args()
  main(args)