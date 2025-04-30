import argparse
import glob
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from torchvision import transforms

def load_image(path, size):
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    arr = tf(img).numpy().transpose(1,2,0)
    # back to [0..255] uint8 for SSIM/PSNR
    return (arr * 255).astype(np.uint8)

def main():
    # ─── defaults ────────────────────────────────────────────────────────────
    default_folder     = "outputs_optimized"
    default_input_pref = "input_"
    default_real_pref  = "real_"
    default_fake_pref  = "fake_"
    default_size       = 64
    default_win_size   = 3   # must be odd and <= smallest image dim

    p = argparse.ArgumentParser(
        description="Evaluate Pix2Pix (paired) with SSIM & PSNR"
    )
    p.add_argument("--folder",        type=str, default=default_folder,
                   help=f"Folder containing all PNGs (default: {default_folder})")
    p.add_argument("--input_prefix",  type=str, default=default_input_pref,
                   help=f"Prefix for input images (default: {default_input_pref})")
    p.add_argument("--real_prefix",   type=str, default=default_real_pref,
                   help=f"Prefix for ground-truth images (default: {default_real_pref})")
    p.add_argument("--fake_prefix",   type=str, default=default_fake_pref,
                   help=f"Prefix for Pix2Pix outputs (default: {default_fake_pref})")
    p.add_argument("--size",          type=int, default=default_size,
                   help=f"Resize to (size,size) before eval (default: {default_size})")
    p.add_argument("--win_size",      type=int, default=default_win_size,
                   help=f"SSIM window size, odd (default: {default_win_size})")
    args = p.parse_args()

    folder = args.folder.rstrip("/")

    input_paths = sorted(glob.glob(os.path.join(folder, f"{args.input_prefix}*.png")))
    real_paths  = sorted(glob.glob(os.path.join(folder, f"{args.real_prefix}*.png")))
    fake_paths  = sorted(glob.glob(os.path.join(folder, f"{args.fake_prefix}*.png")))

    n = len(real_paths)
    assert n == len(fake_paths) == len(input_paths), (
        f"Count mismatch: inputs={len(input_paths)}, reals={n}, fakes={len(fake_paths)}"
    )

    ssim_scores = []
    psnr_scores = []

    for inp, real, fake in zip(input_paths, real_paths, fake_paths):
        # load only real vs fake
        im_real = load_image(real, args.size)
        im_fake = load_image(fake, args.size)

        # compute SSIM with explicit window and channel_axis
        s = ssim(
            im_real,
            im_fake,
            win_size=args.win_size,
            channel_axis=2
        )
        p = psnr(im_real, im_fake)

        ssim_scores.append(s)
        psnr_scores.append(p)

    print(f"Evaluated {n} paired samples")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Mean PSNR: {np.mean(psnr_scores):.2f} dB")

if __name__ == "__main__":
    main()
