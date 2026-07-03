import argparse
import os

from net.imaging_model import load_imaging_model, reconstruct_images
from net.multimodal_model import MiniCPMVModel


def parse_args():
    parser = argparse.ArgumentParser(description="Image-free multimodal sensing.")
    parser.add_argument("--feature-path", default="./features/features.mat")
    parser.add_argument("--question", required=True, help="Question for the multimodal model.")
    parser.add_argument("--image-index", type=int, default=0, help="Measurement index in the .mat file.")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--step1-path", default="./weights/UDLSSPI1k_step1.pth")
    parser.add_argument("--step2-path", default="./weights/UDLSSPI1k_step2.pth")
    parser.add_argument("--vlm-path", default="./weights/MiniCPM-V-4.6")
    parser.add_argument("--vlm-dtype", default="auto", help="auto, float16, bfloat16, or float32.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument(
        "--allow-download-at-runtime",
        action="store_true",
        help="Allow transformers to fetch MiniCPM-V if local weights are missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    imaging_model, device = load_imaging_model(
        step1_path=args.step1_path,
        step2_path=args.step2_path,
        device=args.device,
    )
    recon = reconstruct_images(
        imaging_model,
        feature_path=args.feature_path,
        output_dir=args.output_dir,
        device=device,
        image_index=args.image_index,
    )[0]
    print("Reconstructed image saved to %s" % recon["image_path"])

    vlm = MiniCPMVModel(
        model_path=args.vlm_path,
        device=str(device),
        local_files_only=not args.allow_download_at_runtime,
        torch_dtype=args.vlm_dtype,
    )
    answer = vlm.answer(
        recon["image"],
        args.question,
        max_new_tokens=args.max_new_tokens,
        sampling=args.sampling,
    )

    print("\nQuestion: %s" % args.question)
    print("Answer: %s" % answer)


if __name__ == "__main__":
    main()
