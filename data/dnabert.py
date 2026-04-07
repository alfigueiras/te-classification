import os
import torch
import os

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class DNABERTEmbedder:
    def __init__(self,model_name="zhihan1996/DNABERT-2-117M"):  
        
        if torch.cuda.is_available():
            self.device="cuda:1"
        elif torch.backends.mps.is_available():
            self.device="mps"
        else:
            self.device="cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        #config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")

        #if getattr(config, "pad_token_id", None) is None:
        #    config.pad_token_id = self.tokenizer.pad_token_id

        print("default device:", torch.get_default_device())
        print("cuda available:", torch.cuda.is_available())

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def chunk_sequence(seq, chunk_size=512, stride=512):
        if not seq:
            return []

        chunks = []
        start = 0
        seq_len = len(seq)

        while start < seq_len:
            chunk = seq[start:start + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)

            if start + chunk_size >= seq_len:
                break

            start += stride

        return chunks

    @torch.no_grad()
    def embed_chunks_batch(self, chunks, batch_size=16, max_length=512):
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            encoded = self.tokenizer(
                batch_chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model(**encoded, return_dict=True)
            last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

            chunk_embeddings = self.mean_pool(
                last_hidden_state,
                encoded["attention_mask"]
            )

            all_embeddings.append(chunk_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def embed_single_sequence(
        self,
        sequence,
        chunk_size=512,
        stride=512,
        batch_size=16,
        max_length=512
    ):
        chunks = self.chunk_sequence(
            sequence,
            chunk_size=chunk_size,
            stride=stride
        )

        if len(chunks) == 0:
            raise ValueError("Received an empty DNA sequence.")

        chunk_embeddings = self.embed_chunks_batch(
            chunks,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # average the chunk embeddings, weighting by chunk length
        chunk_lengths = torch.tensor(
            [len(chunk) for chunk in chunks],
            dtype=torch.float32
        )
        weights = chunk_lengths / chunk_lengths.sum()
        sequence_embedding = (chunk_embeddings * weights.unsqueeze(1)).sum(dim=0)
        return sequence_embedding

    @torch.no_grad()
    def embed_sequences(
        self,
        sequences,
        chunk_size=512,
        stride=512,
        batch_size=16,
        max_length=512,
        disable_tqdm=False
    ):

        all_sequence_embeddings = []

        for seq in tqdm(sequences, disable=disable_tqdm, desc="Embedding sequences"):
            seq_emb = self.embed_single_sequence(
                seq,
                chunk_size=chunk_size,
                stride=stride,
                batch_size=batch_size,
                max_length=max_length
            )
            all_sequence_embeddings.append(seq_emb.cpu())

        return torch.stack(all_sequence_embeddings, dim=0)


def compute_or_load_dnabert_embeddings(
    sequences,
    save_path,
    node_ids=None,
    model_name="zhihan1996/DNABERT-2-117M",
    chunk_size=512,
    stride=512,
    batch_size=16,
    max_length=512,
    force_recompute=False,
    disable_tqdm=False
):
    
    if os.path.exists(save_path) and not force_recompute:
        saved = torch.load(save_path, map_location="cpu")
        return saved["embeddings"], saved

    embedder = DNABERTEmbedder(model_name=model_name)

    embeddings = embedder.embed_sequences(
        sequences=sequences,
        chunk_size=chunk_size,
        stride=stride,
        batch_size=batch_size,
        max_length=max_length,
        disable_tqdm=disable_tqdm
    ).cpu()

    saved = {
        "embeddings": embeddings,
        "node_ids": node_ids,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "stride": stride,
        "batch_size": batch_size,
        "max_length": max_length,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(saved, save_path)

    return embeddings, saved