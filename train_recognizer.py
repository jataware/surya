from typing import List
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torch.optim as optim

from surya.settings import settings
from surya.input.processing import slice_bboxes_from_image
from surya.postprocessing.math.latex import fix_math, contains_math
from surya.postprocessing.text import truncate_repetitions
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor

from datasets import load_dataset

def get_batch_size():
    batch_size = settings.RECOGNITION_BATCH_SIZE
    if batch_size is None:
        batch_size = 32
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 64 # 12GB RAM max
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 256
    return batch_size

def compute_loss_from_scores(scores, tokenized_texts, processor, vocab_size):
    logits = torch.stack([s[:, token_id].unsqueeze(1) for s, token_id in zip(scores, tokenized_texts.unbind(dim=1))], dim=1)
    loss = F.cross_entropy(logits.view(-1, vocab_size), tokenized_texts.view(-1), ignore_index=processor.tokenizer.pad_token_id)
    return loss

def batch_recognition(images: List, languages: List[List[str]], model, processor, texts=None, train=False):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)
    batch_size = get_batch_size()

    images = [image.convert("RGB") for image in images]
    loss_fn = torch.nn.CrossEntropyLoss()

    if train:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        model.eval()

    output_text = []
    confidences = []
    losses      = []
    total_acc, total_words = 0, 0
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_langs  = languages[i:i+batch_size]
        has_math     = ["_math" in lang for lang in batch_langs]
        batch_images = images[i:i+batch_size]
        model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)
        
        batch_pixel_values  = model_inputs["pixel_values"]
        batch_langs         = model_inputs["langs"]
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

        batch_langs         = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
        batch_pixel_values  = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

        with torch.set_grad_enabled(train):
            return_dict = model.generate(
                pixel_values      = batch_pixel_values,
                decoder_input_ids = batch_decoder_input,
                decoder_langs     = batch_langs,
                eos_token_id      = processor.tokenizer.eos_id,
                max_new_tokens    = settings.RECOGNITION_MAX_TOKENS,
                output_scores     = True,
                return_dict_in_generate = True
            )
            generated_ids = return_dict["sequences"]

            # Find confidence scores
            scores          = return_dict["scores"] # Scores is a tuple, one per new sequence position.  Each tuple element is bs x vocab_size
            sequence_scores = torch.zeros(generated_ids.shape[0])
            sequence_lens   = torch.where(
                generated_ids > processor.tokenizer.eos_id,
                torch.ones_like(generated_ids),
                torch.zeros_like(generated_ids)
            ).sum(axis=-1).cpu()
            
            prefix_len = generated_ids.shape[1] - len(scores) # Length of passed in tokens (bos, langs)
            for token_idx, score in enumerate(scores):
                probs     = F.softmax(score, dim=-1)
                max_probs = torch.max(probs, dim=-1).values
                max_probs = torch.where(
                    generated_ids[:, token_idx + prefix_len] <= processor.tokenizer.eos_id,
                    torch.zeros_like(max_probs),
                    max_probs
                ).cpu()
                sequence_scores += max_probs
            sequence_scores /= sequence_lens

            if texts is not None:
                
                
                logits = torch.stack(scores, dim=1)  # stack to shape [batch_size, sequence_length, vocab_size]
                batch_texts     = texts[i:i+batch_size]
                tokenized_texts = processor.tokenizer(batch_texts, languages[i:i+batch_size])["input_ids"]
                tokenized_texts = [text + [1] * (logits.shape[1]+2 - len(text)) for text in tokenized_texts]
                tokenized_texts = torch.tensor(tokenized_texts, dtype=torch.long).to(model.device)
                tokenized_texts = tokenized_texts[:, 2:]
                
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokenized_texts.reshape(-1))
                losses.append(loss.item())
                total_acc += (logits.argmax(-1) == tokenized_texts).sum().item()
                total_words += tokenized_texts.numel()                
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
        
                        
        detected_text = processor.tokenizer.batch_decode(generated_ids)
        detected_text = [truncate_repetitions(dt) for dt in detected_text]
        detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
        output_text.extend(detected_text)
        confidences.extend(sequence_scores.tolist())
   
    total_acc = total_acc / total_words

    return {"text": output_text, "confidence": confidences, "loss": sum(losses) / len(losses) if losses else None, 'acc': total_acc}

class OCRDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        bboxes = item['bboxes']
        texts = item['text']
        lang = item['language']
        slices = slice_bboxes_from_image(image, bboxes)
        return slices, texts, [[lang]]*len(slices)

def _collate_fn(batch):
    slices, texts, langs = zip(*batch)
    slices = [item for sublist in slices for item in sublist]
    texts = [item for sublist in texts for item in sublist]
    langs = [item for sublist in langs for item in sublist]
    return slices, texts, langs

rec_model, rec_processor = load_model(), load_processor()
dataset = load_dataset("vikp/rec_bench")['train']

ds = OCRDataset(dataset)
dl = DataLoader(ds, batch_size=10, shuffle=True, collate_fn=_collate_fn) 
for slices, texts, lang in dl:
    out = batch_recognition(slices, lang, rec_model, rec_processor, texts)
    print(out['loss'], out['acc'])
    