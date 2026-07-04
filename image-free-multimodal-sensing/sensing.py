import argparse

from net.encoder import load_encoder, reconstruct_images
from net.decoder import Decoder


def parse_args():
    parser = argparse.ArgumentParser(description="Image-free multimodal sensing.")
    parser.add_argument("--feature-path", default="./features/features.mat")
    parser.add_argument("--question", required=True, help="Question for the multimodal model.")
    parser.add_argument("--image-index", type=int, default=0, help="Measurement index in the .mat file.")
    parser.add_argument("--step1-path", default="./weights/UDLSSPI1k_step1.pth")
    parser.add_argument("--encoder-path", default="./weights/encoder.pth")
    parser.add_argument("--decoder-path", default="./weights/decoder.py")
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

    encoder, device = load_encoder(
        step1_path=args.step1_path,
        encoder_path=args.encoder_path,
        device=args.device,
    )
    recon = reconstruct_images(
        encoder,
        feature_path=args.feature_path,
        device=device,
        image_index=args.image_index,
    )[0]
    print("Reconstructed image is kept in memory for multimodal sensing.")

    decoder = Decoder(
        model_path=args.decoder_path,
        device=str(device),
        local_files_only=not args.allow_download_at_runtime,
        torch_dtype=args.vlm_dtype,
    )
    answer = decoder.answer(
        recon["image"],
        args.question,
        max_new_tokens=args.max_new_tokens,
        sampling=args.sampling,
    )

    print("\nQuestion: %s" % args.question)
    print("Answer: %s" % answer)


if __name__ == "__main__":
    main()
