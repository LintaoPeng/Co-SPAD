import os

from PIL import Image


class MiniCPMVModel(object):
    def __init__(
        self,
        model_path="./weights/MiniCPM-V-4.6",
        device=None,
        local_files_only=True,
        torch_dtype="auto",
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-V inference requires torch and transformers. "
                "Install the dependencies in your runtime environment first."
            ) from exc

        if local_files_only and not os.path.isdir(model_path):
            raise FileNotFoundError(
                "MiniCPM-V weights were not found at '%s'. Run download_minicpm_v.py first."
                % model_path
            )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._resolve_dtype(torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.eval().to(self.device)

    def _resolve_dtype(self, torch_dtype):
        if torch_dtype == "auto":
            if str(self.device).startswith("cuda") and self.torch.cuda.is_available():
                return self.torch.float16
            return self.torch.float32
        if isinstance(torch_dtype, str):
            return getattr(self.torch, torch_dtype)
        return torch_dtype

    def answer(self, image, question, max_new_tokens=512, sampling=False):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        msgs = [{"role": "user", "content": [image, question]}]
        try:
            return self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
            )
        except TypeError:
            msgs = [{"role": "user", "content": question}]
            return self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
            )
