"""
gen_macberth.py - MacBERTh wrapper class for extracting contextualized embeddings.
Equivalent to gen_berts.py for LatinBERT, adapted for HuggingFace MacBERTh model.
"""

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MacBERTh:
    """
    Wrapper class for MacBERTh model to extract contextualized embeddings.
    Similar interface to the LatinBERT class from gen_berts.py.
    """

    def __init__(self, model_name="emanjavacas/MacBERTh", layer_mode="last"):
        """
        Initialize MacBERTh model and tokenizer.

        Args:
            model_name: HF model name or local fine-tuned path
            layer_mode:
                last      -> last hidden layer only
                first4    -> average first 4 layers
                last4     -> average last 4 layers
                all       -> average all layers
        """

        print(f"Loading MacBERTh from {model_name}...")
        print(f"Using device: {device}")
        print(f"Layer mode: {layer_mode}")

        self.layer_mode = layer_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = BertMacBERTh(
            model_name=model_name,
            layer_mode=layer_mode
        )

        self.model.to(device)
        self.model.eval()

        print("MacBERTh loaded successfully.")

    def get_batches(self, sentences, max_batch, max_seq_length=510):
        """
        Prepare batches with transform matrices to map subword embeddings to word embeddings.
        
        Args:
            sentences: List of tokenized sentences (each sentence is a list of words with [CLS] and [SEP])
            max_batch: Maximum batch size
            max_seq_length: Maximum sequence length (BERT limit is 512, use 510 for safety)
            
        Returns:
            batched_data, batched_mask, batched_transforms, ordering, truncated_sentences
        """
        all_data = []
        all_masks = []
        all_labels = []
        all_transforms = []
        truncated_sentences = []  # Track truncated versions

        for sentence in sentences:
            tok_ids = []
            input_mask = []
            labels = []
            transform = []

            # Tokenize each word and track subword counts
            all_toks = []
            n = 0
            for idx, word in enumerate(sentence):
                toks = self.tokenizer.tokenize(word)
                if not toks:  # Handle unknown tokens
                    toks = [self.tokenizer.unk_token]
                all_toks.append(toks)
                n += len(toks)
            
            # Truncate if exceeds max_seq_length
            truncated_sent = list(sentence)  # Copy original
            if n > max_seq_length:
                truncated_toks = []
                truncated_sent = []
                current_len = 0
                for word_idx, toks in enumerate(all_toks):
                    if current_len + len(toks) <= max_seq_length:
                        truncated_toks.append(toks)
                        truncated_sent.append(sentence[word_idx])
                        current_len += len(toks)
                    else:
                        break
                all_toks = truncated_toks
                n = current_len
            
            truncated_sentences.append(truncated_sent)

            # Build transform matrix
            cur = 0
            for idx in range(len(all_toks)):
                toks = all_toks[idx]
                ind = list(np.zeros(n))
                for j in range(cur, cur + len(toks)):
                    ind[j] = 1. / len(toks)
                cur += len(toks)
                transform.append(ind)

                tok_ids.extend(self.tokenizer.convert_tokens_to_ids(toks))
                input_mask.extend(np.ones(len(toks)))
                labels.append(1)

            all_data.append(tok_ids)
            all_masks.append(input_mask)
            all_labels.append(labels)
            all_transforms.append(transform)

        lengths = np.array([len(l) for l in all_data])

        # Order sequences from shortest to longest
        ordering = np.argsort(lengths)

        ordered_data = [None for i in range(len(all_data))]
        ordered_masks = [None for i in range(len(all_data))]
        ordered_labels = [None for i in range(len(all_data))]
        ordered_transforms = [None for i in range(len(all_data))]

        for i, ind in enumerate(ordering):
            ordered_data[i] = all_data[ind]
            ordered_masks[i] = all_masks[ind]
            ordered_labels[i] = all_labels[ind]
            ordered_transforms[i] = all_transforms[ind]

        batched_data = []
        batched_mask = []
        batched_labels = []
        batched_transforms = []

        i = 0
        current_batch = max_batch

        while i < len(ordered_data):
            batch_data = ordered_data[i:i + current_batch]
            batch_mask = ordered_masks[i:i + current_batch]
            batch_labels = ordered_labels[i:i + current_batch]
            batch_transforms = ordered_transforms[i:i + current_batch]

            max_len = max([len(sent) for sent in batch_data])
            max_label = max([len(label) for label in batch_labels])

            for j in range(len(batch_data)):
                blen = len(batch_data[j])
                blab = len(batch_labels[j])

                for k in range(blen, max_len):
                    batch_data[j].append(0)
                    batch_mask[j].append(0)
                    for z in range(len(batch_transforms[j])):
                        batch_transforms[j][z].append(0)

                for k in range(blab, max_label):
                    batch_labels[j].append(-100)

                for k in range(len(batch_transforms[j]), max_label):
                    batch_transforms[j].append(np.zeros(max_len))

            batched_data.append(torch.LongTensor(batch_data))
            batched_mask.append(torch.FloatTensor(batch_mask))
            batched_transforms.append(torch.FloatTensor(np.array(batch_transforms)))

            i += current_batch

            # Adjust batch size for longer sequences
            if max_len > 100:
                current_batch = 12
            if max_len > 200:
                current_batch = 6

        return batched_data, batched_mask, batched_transforms, ordering, truncated_sentences

    def get_berts(self, raw_sents):
        """
        Get BERT embeddings for a list of raw sentences.
        
        Args:
            raw_sents: List of raw sentence strings
            
        Returns:
            List of sentences, where each sentence is a list of (token, embedding) tuples
        """
        sents = convert_to_toks(raw_sents)
        batch_size = 16  # Reduced from 32 for GPU memory
        batched_data, batched_mask, batched_transforms, ordering, truncated_sents = self.get_batches(sents, batch_size)

        ordered_preds = []
        for b in range(len(batched_data)):
            size = batched_transforms[b].shape
            b_size = size[0]
            
            berts = self.model.forward(
                batched_data[b], 
                attention_mask=batched_mask[b], 
                transforms=batched_transforms[b]
            )
            berts = berts.detach().cpu()
            
            for row in range(b_size):
                ordered_preds.append([np.array(r) for r in berts[row]])

        # Restore original order
        preds_in_order = [None for _ in range(len(sents))]
        for i, ind in enumerate(ordering):
            preds_in_order[ind] = ordered_preds[i]

        # Build output: list of (token, embedding) tuples per sentence
        # Use truncated_sents which matches the actual embeddings length
        bert_sents = []
        for idx, sentence in enumerate(truncated_sents):
            bert_sent = []
            preds = preds_in_order[idx]
            
            # Only add embeddings for tokens we actually have
            num_preds = len(preds)
            num_tokens = len(sentence)
            
            for t_idx in range(min(num_preds, num_tokens)):
                token = sentence[t_idx]
                pred = preds[t_idx]
                bert_sent.append((token, pred))
            
            bert_sents.append(bert_sent)

        return bert_sents


class BertMacBERTh(nn.Module):
    """
    MacBERTh model wrapper with transform matrix support.
    Equivalent to BertLatin class.
    """

    def __init__(self, model_name="emanjavacas/MacBERTh",
                 layer_mode="last"):

        super(BertMacBERTh, self).__init__()

        self.layer_mode = layer_mode

        self.bert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        transforms = transforms.to(device)

        outputs = self.bert.forward(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states

        if self.layer_mode == "last":
            sequence_outputs = hidden_states[-1]

        elif self.layer_mode == "first4":
            sequence_outputs = torch.stack(
                hidden_states[1:5]
            ).mean(dim=0)

        elif self.layer_mode == "last4":
            sequence_outputs = torch.stack(
                hidden_states[-4:]
            ).mean(dim=0)

        elif self.layer_mode == "all":
            sequence_outputs = torch.stack(
                hidden_states[1:]
            ).mean(dim=0)

        else:
            raise ValueError(
                f"Unknown layer mode: {self.layer_mode}"
            )

        out = torch.matmul(
            transforms,
            sequence_outputs
        )

        return out

def convert_to_toks(sents, max_words=200):
    """
    Convert raw sentences to tokenized format with [CLS] and [SEP] tokens.
    Simple word tokenization for English (split on whitespace).
    
    Args:
        sents: List of raw sentence strings
        max_words: Maximum number of words to keep (to stay within BERT's 512 token limit)
                   Set to 200 to leave room for subword expansion + special tokens
        
    Returns:
        List of tokenized sentences, each with [CLS] and [SEP]
    """
    all_sents = []

    for data in sents:
        text = data.lower()
        
        # Simple word tokenization (split on whitespace)
        tokens = text.split()
        
        # Truncate to max_words to avoid exceeding BERT's token limit
        if len(tokens) > max_words:
            tokens = tokens[:max_words]
        
        filt_toks = []
        filt_toks.append("[CLS]")
        for tok in tokens:
            if tok != "":
                filt_toks.append(tok)
        filt_toks.append("[SEP]")

        all_sents.append(filt_toks)

    return all_sents


# Test code
if __name__ == "__main__":
    bert = MacBERTh(model_name="emanjavacas/MacBERTh")
    
    sents = [
        "the coffee was served hot",
        "sugar prices rose sharply in the market"
    ]
    
    bert_sents = bert.get_berts(sents)
    
    for sent in bert_sents:
        for (token, bert_emb) in sent:
            print(f"{token}\t{' '.join([f'{x:.5f}' for x in bert_emb[:5]])}...")  # Print first 5 dims
        print()
