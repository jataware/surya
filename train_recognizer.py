

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
    # Validate input types and constraints
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)
    for l in languages:
        assert len(l) <= settings.RECOGNITION_MAX_LANGS, f"OCR only supports up to {settings.RECOGNITION_MAX_LANGS} languages per image, you passed {l}."

    # Convert images to RGB and get batch size
    images = [image.convert("RGB") for image in images]
    batch_size = len(images)

    # Get model configuration
    dec_config = model.config.decoder
    layer_count = dec_config.decoder_layers

    # Initialize attention masks
    initial_kv_mask = torch.zeros((batch_size, 1, 1, 1), dtype=model.dtype, device=model.device)
    initial_attn_mask = torch.zeros((batch_size, 1, settings.RECOGNITION_MAX_LANGS + 1, settings.RECOGNITION_MAX_LANGS + 1), dtype=model.dtype, device=model.device)

    # Process input data
    batch_langs = languages
    has_math = ["_math" in lang for lang in batch_langs]
    processed_batches = processor(text=[""]*batch_size, images=images, lang=languages)
    batch_pixel_values = processed_batches["pixel_values"]
    batch_langs = processed_batches["langs"]

    # Pad language tokens
    max_lang_len = max([len(lang) for lang in batch_langs])
    for lang_idx in range(len(batch_langs)):
        lang_len = len(batch_langs[lang_idx])
        if lang_len < max_lang_len:
            batch_langs[lang_idx] = [processor.tokenizer.pad_id] * (max_lang_len - lang_len) + batch_langs[lang_idx]

    # Prepare decoder input
    batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]
    current_batch_size = len(batch_pixel_values)

    # Convert inputs to tensors
    batch_langs = torch.tensor(np.stack(batch_langs, axis=0), dtype=torch.long, device=model.device)
    batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=model.dtype, device=model.device)
    batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)

    # Initialize variables for token generation
    token_count = 0
    inference_token_count = batch_decoder_input.shape[-1]
    batch_predictions = [[] for _ in range(current_batch_size)]

    # Initialize masks and caches
    kv_mask = initial_kv_mask[:current_batch_size]
    kv_mask.fill_(0)
    attention_mask = initial_attn_mask[:current_batch_size, :, :inference_token_count, :inference_token_count]
    decoder_cache = [None] * layer_count
    encoder_outputs = None
    encoder_cache = [None] * layer_count
    all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)

    all_logits = []
    final_logits = None
    # Main token generation loop
    for token_count in range(settings.RECOGNITION_MAX_TOKENS):
        is_prefill = token_count == 0
        
        # Forward pass through the model
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

        # Process model output
        logits = return_dict["logits"][:current_batch_size]
        all_logits.append(logits)
        preds = torch.argmax(logits[:, -1], dim=-1)
        done = (preds == processor.tokenizer.eos_id) | (preds == processor.tokenizer.pad_id)
        all_done = all_done | done

        if is_prefill:
            encoder_outputs = (return_dict["encoder_last_hidden_state"],)

        if all_done.all():
            break

        # Update caches
        past_key_values = return_dict["past_key_values"]
        for layer_idx, layer in enumerate(past_key_values):
            if is_prefill:
                encoder_cache[layer_idx] = layer[1]
                decoder_cache[layer_idx] = layer[0]
            else:
                decoder_cache[layer_idx] = torch.cat([decoder_cache[layer_idx], layer[0]], dim=3)

        # Prepare for next iteration
        batch_decoder_input = preds.unsqueeze(1)
        kv_mask             = torch.cat([kv_mask, torch.zeros((current_batch_size, 1, 1, inference_token_count), dtype=model.dtype, device=model.device)], dim=-1)
        attention_mask      = kv_mask

        # Append predictions
        for j, (pred, status) in enumerate(zip(preds, all_done)):
            if not status:
                batch_predictions[j].append(int(pred))

        token_count += inference_token_count
        inference_token_count = batch_decoder_input.shape[-1]

    # Post-process results
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
    langs  = [lang[0] for lang in langs]
    return slices, texts, langs

def main():

    train       = True
    batch_size  = 16
    subset_size = 100  # Define the size of the subset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rec_model, rec_processor = load_model(), load_processor()
    dataset        = load_dataset("vikp/rec_bench")['train']
    subset_dataset = dataset.select(range(subset_size))    

    ds = OCRDataset(subset_dataset)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    
    if train:
        rec_model.train()
        for param in rec_model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, rec_model.parameters()), lr=1e-6)

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
        
        y_tok = [[1] + yy[2:] + [rec_processor.tokenizer.eos_id] * (max_len - len(yy) + 1) for yy in y_tok]
        y_tok = torch.tensor(y_tok, dtype=torch.long, device=rec_model.device)
        mask  = (y_tok != rec_processor.tokenizer.eos_id).float()

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), 
            y_tok.reshape(-1), 
            ignore_index=rec_processor.tokenizer.eos_id
        )
        correct = ((logits.argmax(-1) == y_tok) * mask).sum().item()
        total = mask.sum().item()
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(rec_model.parameters(), max_norm=0.1)
            optimizer.step()

        results['acc'].append((correct/total))
        results['loss'].append(loss.detach().item())
        print('Accuracy:', np.mean(results['acc']), 'Loss:', np.mean(results['loss']))
        print("=====================================")

if __name__ == "__main__":
    main()