import os

from PIL import Image


class Decoder(object):
    def __init__(
        self,
        model_path="./weights/decoder.py",
        device=None,
        local_files_only=True,
        torch_dtype="auto",
    ):
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForMultimodalLM as AutoModelForImageTextToText
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

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=dtype,
                **model_kwargs
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                **model_kwargs
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

        if hasattr(self.model, "chat"):
            msgs = [{"role": "user", "content": [image, question]}]
            return self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=getattr(self.processor, "tokenizer", None),
                sampling=sampling,
                max_new_tokens=max_new_tokens,
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("MiniCPM-V processor does not expose a tokenizer.")

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            max_slice_nums=1,
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sampling,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
