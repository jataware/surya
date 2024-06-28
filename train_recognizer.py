

from typing import List
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from surya.settings import settings
from surya.input.processing import slice_bboxes_from_image
from surya.postprocessing.math.latex import fix_math, contains_math
from surya.postprocessing.text import truncate_repetitions
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor

from datasets import load_dataset

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def batch_recognition2(images: List, languages: List[List[str]], model, processor):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)

    for l in languages:
        assert len(l) <= settings.RECOGNITION_MAX_LANGS, f"OCR only supports up to {settings.RECOGNITION_MAX_LANGS} languages per image, you passed {l}."

    images     = [image.convert("RGB") for image in images]
    batch_size = len(images)

    dec_config  = model.config.decoder
    layer_count = dec_config.decoder_layers
    kv_heads    = dec_config.kv_heads
    head_dim    = int(dec_config.d_model / dec_config.decoder_attention_heads)
    min_val     = torch.finfo(model.dtype).min

    initial_kv_mask = torch.zeros((batch_size, 1, 1, 1), dtype=model.dtype, device=model.device)
    initial_attn_mask = torch.zeros((batch_size, 1, settings.RECOGNITION_MAX_LANGS + 1, settings.RECOGNITION_MAX_LANGS + 1), dtype=model.dtype, device=model.device)

    batch_langs        = languages
    has_math           = ["_math" in lang for lang in batch_langs]
    processed_batches  = processor(text=[""]*batch_size, images=images, lang=languages)
    batch_pixel_values = processed_batches["pixel_values"]
    batch_langs        = processed_batches["langs"]
    max_lang_len       = max([len(lang) for lang in batch_langs])
    
    for lang_idx in range(len(batch_langs)):
        lang_len = len(batch_langs[lang_idx])
        if lang_len < max_lang_len:
            batch_langs[lang_idx] = [processor.tokenizer.pad_id] * (max_lang_len - lang_len) + batch_langs[lang_idx]

    batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]
    current_batch_size = len(batch_pixel_values)
    
    batch_langs         = torch.tensor(np.stack(batch_langs, axis=0), dtype=torch.long, device=model.device)
    batch_pixel_values  = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=model.dtype, device=model.device)
    batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)

    token_count           = 0
    inference_token_count = batch_decoder_input.shape[-1]
    batch_predictions     = [[] for _ in range(current_batch_size)]

    kv_mask = initial_kv_mask[:current_batch_size]
    kv_mask.fill_(0)

    attention_mask = initial_attn_mask[:current_batch_size, :, :inference_token_count, :inference_token_count]

    decoder_cache = [None] * layer_count

    encoder_outputs = None
    encoder_cache = [None] * layer_count
    all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)

    all_logits = []

    while token_count < settings.RECOGNITION_MAX_TOKENS:
        is_prefill = token_count == 0
        return_dict = model(
            decoder_input_ids=batch_decoder_input,
            decoder_attention_mask=attention_mask,
            decoder_self_kv_cache=None if is_prefill else decoder_cache,
            decoder_cross_kv_cache=None if is_prefill else encoder_cache,
            decoder_past_token_count=token_count,
            decoder_langs=batch_langs,
            pixel_values=batch_pixel_values,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        logits = return_dict["logits"][:current_batch_size]
        
        all_logits.append(logits)
        preds = torch.argmax(logits[:, -1], dim=-1)
        done = (preds == processor.tokenizer.eos_id) | (preds == processor.tokenizer.pad_id)
        all_done = all_done | done
        
        if is_prefill:
            encoder_outputs = (return_dict["encoder_last_hidden_state"],)

        if all_done.all():
            break

        past_key_values = return_dict["past_key_values"]
        token_range = torch.arange(token_count, token_count + inference_token_count, device=model.device)

        for layer_idx, layer in enumerate(past_key_values):
            if is_prefill:
                encoder_cache[layer_idx] = layer[1]

            if is_prefill:
                decoder_cache[layer_idx] = layer[0]
            else:
                decoder_cache[layer_idx] = torch.cat([decoder_cache[layer_idx], layer[0]], dim=3)

        batch_decoder_input = preds.unsqueeze(1)
        kv_mask = torch.cat([kv_mask, torch.zeros((current_batch_size, 1, 1, inference_token_count), dtype=model.dtype, device=model.device)], dim=-1)

        attention_mask = kv_mask

        for j, (pred, status) in enumerate(zip(preds, all_done)):
            if not status:
                batch_predictions[j].append(int(pred))

        token_count += inference_token_count
        inference_token_count = batch_decoder_input.shape[-1]

    full_logits = torch.cat(all_logits, dim=1)
    y_hat = processor.tokenizer.batch_decode(batch_predictions)
    y_hat = [truncate_repetitions(dt) for dt in y_hat]
    y_hat = [fix_math(text) if math and contains_math(text) else text for text, math in zip(y_hat, has_math)]
    return full_logits, y_hat


class OCRDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [(i, j) for i, item in enumerate(tqdm(dataset)) for j in range(len(item['bboxes']))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_idx, bbox_idx = self.indices[idx]
        item  = self.dataset[image_idx]
        image = item['image']
        bbox  = [item['bboxes'][bbox_idx]]
        text  = item['text'][bbox_idx]
        lang  = item['language']
        slice = slice_bboxes_from_image(image, bbox)
        return slice, text, [[lang]]
    
def _collate_fn(batch):
    slices, texts, langs = zip(*batch)
    slices = [slice[0] for slice in slices]
    langs = [lang[0] for lang in langs]
    return slices, texts, langs

def reset_model(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def hook_fn(grad):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("NaN or Inf detected in gradients")
        return torch.zeros_like(grad)

def print_grad_stats(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name}:")
            print(f"  Mean: {grad.mean().item():.5e}")
            print(f"  Std Dev: {grad.std().item():.5e}")
            print(f"  Max: {grad.max().item():.5e}")
            print(f"  Min: {grad.min().item():.5e}")
            print(f"  Norm: {grad.norm().item():.5e}")

def main():
    train      = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch_size = 32
    rec_model, rec_processor = load_model(), load_processor()
    reset_model(rec_model)

    dataset        = load_dataset("vikp/rec_bench")['train']
    subset_size    = 100  # Define the size of the subset
    subset_dataset = dataset.select(range(subset_size))    

    scaler = torch.cuda.amp.GradScaler()


    ds = OCRDataset(subset_dataset)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    
    if train:
        rec_model.train()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, rec_model.parameters()), lr=1e-6)
        for param in rec_model.encoder.parameters():
            param.requires_grad = False
            
        # for name, param in rec_model.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(lambda grad, name=name: hook_fn(grad))
    else:
        rec_model.eval()
        
    results = {'acc': [], 'loss': [], 'words': []}
    for X, y, langs in dl:

        # get forward pass        
        logits, y_hat = batch_recognition2(X, langs, rec_model, rec_processor)
        
        # tokenize ground truth texts
        y_tok = rec_processor.tokenizer(y, langs)["input_ids"]
        
        # align shapes of logits and y_tok
        y_tok_max_len = max([len(yy) for yy in y_tok]) - 1
        max_len = max(logits.size(1), y_tok_max_len)

        if logits.size(1) < max_len:
            logits = F.pad(logits, (0, 0, 0, max_len - logits.size(1)), value=rec_processor.tokenizer.eos_id)
    
        y_tok = [[1] + yy[2:] + [rec_processor.tokenizer.eos_id] * (max_len - len(yy) + 1) for yy in y_tok]
        y_tok = torch.tensor(y_tok, dtype=torch.long, device=rec_model.device)
        mask  = (y_tok != rec_processor.tokenizer.eos_id).float()
                
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), 
            y_tok.reshape(-1), 
            # ignore_index=rec_processor.tokenizer.eos_id
            reduction='sum'

        )
        print('0', loss)
        correct = ((logits.argmax(-1) == y_tok) * mask).sum().item()
        total = mask.sum().item()
        print(correct, total)
        print(y[0], y_hat[0])
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_model.parameters(), max_norm=0.1)
            
            # print_grad_stats(rec_model)

            optimizer.step()

        results['acc'].append((correct/total))
        results['loss'].append(loss.detach().item())
        
        print('Accuracy:', np.mean(results['acc']), 'Loss:', np.mean(results['loss']))

if __name__ == "__main__":
    main()