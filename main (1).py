!pip install sacrebleu
!pip install sentencepiece
!pip install torch
!pip install torchtext
!pip install wandb

import torch, torchtext, sentencepiece, tqdm, wandb, sacrebleu, logging
from torch.utils.data import Dataset, DataLoader
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from typing import Iterable, List
from torchtext.data.utils import get_tokenizer

class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, model_name, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model_name + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(model_name+ f'_model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]


EN_TRAIN_PATH = "/kaggle/input/bhw2-translation/train.de-en.en"
DE_TRAIN_PATH = "/kaggle/input/bhw2-translation/train.de-en.de"
DE_VAL_PATH = "/kaggle/input/bhw2-translation/val.de-en.de"
EN_VAL_PATH = "/kaggle/input/bhw2-translation/val.de-en.en"
DE_TEST_PATH = "/kaggle/input/bhw2-translation/test1.de-en.de"

# EN_TRAIN_PATH = "train.de-en.en"
# DE_TRAIN_PATH = "train.de-en.de"
# DE_VAL_PATH = "val.de-en.de"
# EN_VAL_PATH = "val.de-en.en"
# DE_TEST_PATH = "test1.de-en.de"

SAVE_MODELS_PATH = "./models"
VOCAB_NAME = "5 min freq"
MIN_FREQ = 5
VOCAB_PATH = ""

SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
specials = ['<pad>', '<bos>', '<eos>', '<unk>'] #pad = 0, bos = 1, eos = 2, unk = 3
specials_indexes = [PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX]

TRAIN = 0
VALID = 1
TEST = 2

DATA_RATIO = 1.0
TOKENIZER = "get_tokenizer"
BATCH_SIZE = 64

class TextDataset(Dataset):
    def __init__(self, en_file_path, de_file_path, set_type, ratio=1.0):
        
        self.texts = {}
        self.set_type = set_type
        
        np.random.seed(0)
        
        for ln, file in zip([SRC_LANGUAGE, TGT_LANGUAGE], [de_file_path, en_file_path]):
            if file == None:
                continue
            with open(file) as file:
                self.texts[ln] = file.readlines()
        
        file_indices = np.random.choice(np.arange(len(self.texts[SRC_LANGUAGE])), size=int(ratio * len(self.texts[SRC_LANGUAGE])), replace=False)
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            if set_type == TRAIN:
                self.texts[ln] = pd.Series(self.texts[ln])[file_indices].to_list()
    

    def __len__(self):
        return len(self.texts[SRC_LANGUAGE])


    def __getitem__(self, item):
        if self.set_type != TEST:
            return self.texts[SRC_LANGUAGE][item], self.texts[TGT_LANGUAGE][item]
        return self.texts[SRC_LANGUAGE][item], None
    

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer(None)
token_transform[TGT_LANGUAGE] = get_tokenizer(None)

    
vocab_transform = {}

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))



# src and tgt language text transforms to convert raw strings into tensors indices




def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])
        
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = TextDataset(en_file_path=EN_TRAIN_PATH, de_file_path=DE_TRAIN_PATH, set_type=TRAIN, ratio=DATA_RATIO)
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=MIN_FREQ,
                                                    specials=specials,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
    
# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

train_set = TextDataset(en_file_path=EN_TRAIN_PATH, de_file_path=DE_TRAIN_PATH, set_type=TRAIN, ratio=DATA_RATIO)
val_set = TextDataset(en_file_path=EN_VAL_PATH, de_file_path=DE_VAL_PATH, set_type=VALID, ratio=1.0)
test_set = TextDataset(en_file_path=None, de_file_path=DE_TEST_PATH, set_type=TEST, ratio=1.0)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

import math
from torch import nn
from torch import Tensor
    
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

torch.manual_seed(0)

LR = 1e-4
MAX_LR = 3e-4
BETA1 = 0.9
BETA2 = 0.98
SCHEDULER = "OneCycleLR"
NUM_EPOCHS = 20

MODEL_NAME = "seq2seq_transformer"
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8

FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

config = {
    "architecture": MODEL_NAME,
    "data_ratio": DATA_RATIO,
    "tokenizer": TOKENIZER,
    "optimizer": "Adam",
    "lr": LR,
    "max_lr": MAX_LR,
    "BETA1": BETA1,
    "BETA2": BETA2,
    "scheduler": SCHEDULER,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "min_freq": MIN_FREQ,
    "heads_num": NHEAD,
    "FFN_HID_DIM": FFN_HID_DIM,
    "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
    "NUM_DECODER_LAYERS": NUM_DECODER_LAYERS
}


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=(BETA1, BETA2), eps=1e-9)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)

from tqdm.notebook import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
import numpy as np
from torch.nn.functional import softmax
from sacrebleu import BLEU
import time

def plot_losses(train_losses: List[float], val_losses: List[float], val_bleus):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(val_bleus) + 1), val_bleus, label='val')
    axs[1].set_ylabel('BLEU')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model, data_loader, optimizer, criterion, scheduler):
    
    model.train()

    epoch_loss = 0
    
    for src, tgt in tqdm(data_loader):
        
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        
        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        optimizer.zero_grad()
        
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()

    return epoch_loss / len(list(data_loader))


def validation_epoch(model, data_loader, dataset, criterion):
    
    model.eval()
    sys, refs = [], []
    epoch_loss = 0
    bleu = BLEU(tokenize=None)
    
    with torch.no_grad():
        for src, tgt in tqdm(data_loader):
            
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            epoch_loss += loss.item()
            
        
        for src, tgt in tqdm(dataset):
            translation = translate(model, src)
            sys.append(translation)
            refs.append(tgt)
                
        return epoch_loss / len(list(data_loader)), bleu.corpus_score(sys, [refs]).score

def train(model, model_name, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, config):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    
    wandb.init(project="bhw2-transformer", config=config, name="5 min freq 4 layers batch size 64 full data")
    wandb.watch(model, log="all")
    saver = CheckpointSaver(dirpath=SAVE_MODELS_PATH, decreasing=False)
    
    train_losses, val_losses = [], []
    val_bleus = []
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, train_loader, optimizer, criterion, scheduler
        )
        val_loss, val_bleu = validation_epoch(
            model, val_loader, val_set, criterion
        )

        train_losses += [train_loss]
        val_losses += [val_loss]
        val_bleus += [val_bleu]
        plot_losses(train_losses, val_losses, val_bleus)
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_bleu': val_bleu})
        saver(model, model_name, epoch, val_bleu)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

train(transformer, MODEL_NAME, optimizer, scheduler, train_loader, val_loader, NUM_EPOCHS, config)

preds = []
for src, _ in tqdm(test_set):
    preds.append(translate(transformer, src))

out_file = 'test1.de-en.en'
with open(out_file, 'w') as file:
    for pred in preds:
        file.write(pred + '\n')