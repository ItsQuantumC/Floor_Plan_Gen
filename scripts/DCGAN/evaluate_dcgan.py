import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import torch
from torch_fidelity import calculate_metrics

def main():
    default_gen  = "outputs/generated"
    default_real = "data/targets"

    parser = argparse.ArgumentParser(
        description="Compute Inception Score (IS) and FID for your GAN outputs."
    )
    parser.add_argument(
        "--gen_folder", type=str, default=default_gen,
        help=f"Generated images folder (default: {default_gen})"
    )
    parser.add_argument(
        "--real_folder", type=str, default=default_real,
        help=f"Real images folder (default: {default_real})"
    )
    args = parser.parse_args()

    # Auto-detect CUDA
    use_cuda = torch.cuda.is_available()
    print(f"Evaluating generated images in: {args.gen_folder}")
    print(f"Against real images in:           {args.real_folder}")
    print(f"Using CUDA? {use_cuda}\n")

    metrics = calculate_metrics(
        input1=args.gen_folder,
        input2=args.real_folder,
        cuda=use_cuda,   # use GPU if available
        isc=True,        # compute Inception Score
        fid=True,        # compute FID
    )

    print(f"Inception Score       : {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
    print(f"Fréchet Inception Dist: {metrics['frechet_inception_distance']:.4f}")

if __name__ == "__main__":
    main()
