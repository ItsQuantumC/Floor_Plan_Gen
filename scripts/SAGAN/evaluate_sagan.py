import argparse
import torch
from pytorch_fid import fid_score

def main():
    default_real_folder = "data/colored_targets_128"
    default_fake_folder = "outputs_sagan"

    p = argparse.ArgumentParser(
        description="Evaluate SA-GAN with FID"
    )
    p.add_argument("--real_folder", type=str, default=default_real_folder,
                   help=f"Folder of real images (default: {default_real_folder})")
    p.add_argument("--fake_folder", type=str, default=default_fake_folder,
                   help=f"Folder of generated images (default: {default_fake_folder})")
    p.add_argument("--batch_size", type=int, default=50,
                   help="Batch size for Inception (default: 50)")
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"),
                   help="Compute device")
    p.add_argument("--dims", type=int, default=2048,
                   help="Dimensionality of Inception features")
    args = p.parse_args()

    print(f"Real folder : {args.real_folder}")
    print(f"Fake folder : {args.fake_folder}")
    print(f"Device      : {args.device}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Feat dims   : {args.dims}\n")

    fid_value = fid_score.calculate_fid_given_paths(
        [args.real_folder, args.fake_folder],
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims,
        num_workers=0
    )

    print(f"FID: {fid_value:.4f}")

if __name__ == "__main__":
    main()
