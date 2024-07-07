"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import sys
sys.path.append('../')

from config import settings

import cv2
import argparse
import os
from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.single_gpu)
    logger.configure(args.log_root)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    if args.classifier_path != "":
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

            # probs = F.softmax(logits, dim=-1)
            # uncertainties = -(probs * log_probs).sum()
            # return th.autograd.grad(uncertainties, x_in)[0] * args.classifier_scale

            # probs = F.softmax(logits, dim=-1)
            # probs_sorted, idxs = probs.sort(descending=True)
            # uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
            # return th.autograd.grad(-uncertainties.sum(), x_in)[0] * args.classifier_scale

            # probs = F.softmax(logits, dim=-1)
            # uncertainties = probs.max(1)[0]
            # return th.autograd.grad(-uncertainties.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        # assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    num_samples = 0

    for ele in range(0, len(args.category_num_list)):
        num_samples = num_samples + args.category_num_list[ele]
    lis = []
    for i in range(len(args.category_num_list)):
        lis.extend([i] * args.category_num_list[i])

    while len(all_images) * args.batch_size < num_samples:
        model_kwargs = {}

        classes = th.tensor(lis[len(all_images) * args.batch_size:len(all_images) * args.batch_size+args.batch_size], device=dist_util.dev())
        print(classes)

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (len(classes), 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None if not args.classifier_scale else cond_fn,
            device=dist_util.dev(),
            cfg=args.cfg
        )
        #
        if args.get_image:
            sample = ((sample + 1) / 2).clamp(0, 1)
            tvu.save_image(sample, os.path.join(logger.get_dir(), f"output.png"))
            sample = sample.to(th.uint8)
            break
        else:
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_samples]
    if dist.get_rank() == 0:
        if args.get_images:
            if not os.path.exists(os.path.join(args.dataset_dir, args.dataset_name)):
                os.makedirs(os.path.join(args.dataset_dir, args.dataset_name))
                for i in args.category_name_list:
                    os.makedirs(os.path.join(args.dataset_dir, args.dataset_name, i))
            for i in range(len(arr)):
                im = Image.fromarray(arr[i])
                im.save(os.path.join(args.dataset_dir, args.dataset_name, args.category_name_list[label_arr[i]],
                                     str(i) + ".png"))
        else:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")
    sys.exit(0)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=settings.SAMPLE_BATCH_SIZE,
        use_ddim=True,
        model_path=settings.SAMPLE_MODEL_PATH,
        classifier_path=settings.SAMPLE_CLASSIFIER_PATH,
        classifier_scale=settings.SAMPLE_CLASSIFIER_SCALE,
        log_root=settings.SAMPLE_LOG_ROOT,
        dataset_dir=settings.SAMPLE_DATASET_DIR,
        category_name_list=settings.SAMPLE_CATEGORY_NAME_LIST,
        category_num_list=settings.SAMPLE_CATEGORY_NUM_LIST,
        dataset_name=settings.SAMPLE_DATASET_NAME,
        get_image=False,
        get_images=True,
        single_gpu=False,
        cfg=settings.CFG,)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    import random
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # set random seed
    setup_seed(20)
    main()
