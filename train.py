
'''
# To download benchmark dataset
%%capture
!python -m spacy download en
!python -m spacy download de
'''

import torch
import argparse
from Transformer.model import *
from utils import *
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch.optim as optim
import time
import math
import random
import sys
import os

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        source = batch.src
        target = batch.trg
        optimizer.zero_grad()

        output, _ = model(source, target[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        target = target[:,1:].contiguous().view(-1)

        loss = criterion(output,target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def eval(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            source = batch.src
            target = batch.trg
            output, _ = model(source, target[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = target[:,1:].contiguous().view(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="example")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--hdim", type=int, default=256)
    parser.add_argument("--elayers", type=int, default=3)
    parser.add_argument("--dlayers", type=int, default=3)
    parser.add_argument("--eheads", type=int, default=8)
    parser.add_argument("--dheads", type=int, default=8)
    parser.add_argument("--eidim", type=int, default=512)
    parser.add_argument("--didim", type=int, default=512)
    parser.add_argument("--edout", type=float, default=0.1)
    parser.add_argument("--ddout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    RESULT_PATH = "./Saved_file/"
    
    ensure_path(RESULT_PATH + args.save)
    f = open(RESULT_PATH + args.save + "/report.txt", "a")
    f.write(str(vars(args)) + "\n")
    f.close()

    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
    tokenized = spacy_en.tokenizer("I am a graduate student in UNIST")

    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    source = Field(tokenize = tokenize_de, init_token="<bos>", eos_token="<eos>", lower = True, batch_first= True)
    target = Field(tokenize = tokenize_en, init_token="<bos>", eos_token="<eos>", lower = True, batch_first= True)

    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=('.de', ".en"), fields=(source, target))

    source.build_vocab(train_dataset, min_freq=2)
    target.build_vocab(train_dataset, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size=args.batch,
        device=device)
        
    S_PAD_IDX = source.vocab.stoi[source.pad_token]
    T_PAD_IDX = target.vocab.stoi[target.pad_token]

    encoder = Encoder(len(source.vocab), args.hdim, args.elayers , args.eheads, args.eidim, args.edout, device, 50)
    decoder = Decoder(len(target.vocab), args.hdim, args.dlayers, args.dheads, args.didim, args.ddout, device, 50)

    model = Transformer(encoder, decoder, S_PAD_IDX, T_PAD_IDX, device).to(device)

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index = T_PAD_IDX)

    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(args.epoch):
        start_time = time.time() # 시작 시간 기록

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = eval(model, valid_iterator, criterion)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, 'transformer_german_to_english', RESULT_PATH + args.save)
        
        f = open(RESULT_PATH + args.save + "/report.txt", "a")
        f.write(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s' + "\n")
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}' + "\n")
        f.write(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}' + "\n")
        f.close()
