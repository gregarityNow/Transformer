import pandas as pd
import torchtext
from torchtext import data
from .Batch import MyIterator, batch_size_fn
from .Tokenize import tokenize, CamTok
import os
import dill as pickle

from src import pickLoad


def read_data(opt):
    
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def read_data_felix(opt):
    #todo@feh: create df cleaning func ugh
    df = pickLoad("/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/combined_dfFinal.pickle")
    opt.src_data = list(df.defn.values)
    opt.trg_data = list(df.term.values)



def create_fields(opt, camTok):
    TRG = data.Field(lower=True, tokenize=camTok.tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=camTok.tokenize)
    return (SRC, TRG)

def create_dataset(opt, SRC, TRG):

    print("creating dataset and iterator... ")

    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    os.remove('translate_transformer_temp.csv')

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)

    return train_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i

def create_datasetFEH(srcData, targData, SRC, TRG):
    print("creating dataset and iterator... ")
    raw_data = {'src': srcData, 'trg': targData}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    df.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    train_iter = MyIterator(train, batch_size=1500, device="cuda:0",
                            repeat=True, sort_key=lambda x: (len(x.src), len(x.trg)), train=True, shuffle=True)
    # train_iter = MyIterator(train, batch_size=1500, device="cuda:0",repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), train=True, shuffle=True)
    SRC.build_vocab(train)
    TRG.build_vocab(train)
    os.remove('translate_transformer_temp.csv')
    return train_iter

def create_fieldsFEH(tokenizer):
    camTok = CamTok(tokenizer)
    TRG = data.Field(lower=True, tokenize=camTok.tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=camTok.tokenize)
    return (SRC, TRG)