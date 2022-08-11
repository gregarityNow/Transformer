import pandas as pd
import torchtext
from torchtext import data
from .Batch import MyIterator, batch_size_fn
from .Tokenize import tokenize, CamOrLetterTokenizer
import os
import dill as pickle
import torch

def pickLoad(pth):
    with open(pth,"rb") as fpp:
        return pickle.load(fpp)

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



def create_fields(opt, tokenizer):
    TRG = data.Field(lower=True, tokenize=tokenizer.letter_tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=tokenizer.cam_tokenize)
    if opt.load_weights:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.weightSaveLoc}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.weightSaveLoc}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.weightSaveLoc + "/")
            quit()
    return (SRC, TRG)

def writeMods():
    with open("./spam.txt","w") as fp:
        fp.write(str(model))
        fp.write(str(mod))

def create_dataset_spam():
    raw_data = {'src': [line for line in df.defn.values], 'trg': [line for line in df.term.values]}
    dfr = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (dfr['src'].str.count(' ') < 80) & (dfr['trg'].str.count(' ') < 80)
    dfr = dfr.loc[mask]
    dfr.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    train_iter = MyIterator(train, batch_size=1500, device=torch.device('cuda'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=False)
    SRC.build_vocab(train)
    TRG.build_vocab(train)


def create_dataset(opt, SRC, TRG):

    print("creating dataset and iterator... ")

    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    print("preit")
    train_iter = MyIterator(train, batch_size=opt.batchsize, device=torch.device('cuda'),
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    print("postit")
    os.remove('translate_transformer_temp.csv')

    if not opt.load_weights:
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
    print("curated")

    return train_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i


def create_fieldsFEH(tokenizer):
    camTok = CamOrLetterTokenizer(tokenizer)
    TRG = data.Field(lower=True, tokenize=camTok.tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=camTok.tokenize)
    return (SRC, TRG)