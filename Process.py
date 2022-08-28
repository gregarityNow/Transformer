import pandas as pd
import torchtext
from torchtext import data
from .Batch import MyIterator, batch_size_fn, batch_size_fn_valid
from .Tokenize import tokenize, CamOrLetterTokenizer
import os
import dill as pickle
import torch
import numpy as np
import re

import unicodedata
def strip_accents(s):
    '''
    from @oefe on stackoverflow
    :param s:
    :return:
    '''
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
    s = s.replace("œ","oe").replace("æ","ae");
    return s;

def clean_df(df):
    df = df[df.Domain != "Toponymie"];
    df = df[(~df.term.isna()) & (~df.defn.isna())]
    df = df[df.term.apply(lambda x: len(x) > 3)]
    df = df[df.defn.apply(lambda d: len(d) >= 25)]
    df = df[(df.term.str[0] != "-") & (df.term.str[-1] != "-")]
    df = df[(~df.defn.str.contains("{")) & (~df.defn.str.contains("}"))]
    df = df[df.basic_pos!="UNKNOWN"]
    df = df[df.term.apply(lambda term: re.match("^[a-zA-Z\-\s\'’]+$",strip_accents(term)) is not None)]
    df.Domain = df.Domain.apply(lambda d: (d if len(d) > 0 else "None"))
    df = df[df.Domain != "None"];
    return df


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

def read_data_felix(opt, allTerms = False):
    #todo@feh: create df cleaning func ugh
    if allTerms:
        df = pickLoad("/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/wiktionnaire_df_allWords.pickle")
    else:
        df = pickLoad("/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/combined_dfFinal.pickle")
    df = clean_df(df);
    df = df[["term","defn","subset"]];
    if opt.quickie == 1:
        df = df.sample(100);
    elif opt.quickie > 1:
        df = df.sample(min(len(df),opt.quickie));

    #todo@feh: if opt.camemLayer: modelCamem(df.defn)
    # df = df[df.defn.str.len() < np.percentile(df.defn.apply(lambda x: len(x)),3)]
    for subset in ("valid","train"):
        setattr(opt, "src_data_" + subset, list(df[df.subset==subset].defn.values))
        setattr(opt, "trg_data_" + subset, list(df[df.subset==subset].term.values))
    print("working with",df);
    with open("/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/workinWith" + ("_camemLayer" if opt.camemLayer else "") + ".pickle","wb") as fp:
        pickle.dump(df,fp);
    return df



def create_fields(opt, camOrLetterTokenizer):
    TRG = data.Field(lower=True, tokenize=camOrLetterTokenizer.letter_tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=camOrLetterTokenizer.cam_tokenize)
    if opt.load_weights:
        try:
            srcPath = f'{opt.weightSaveLoc}/SRC.pkl'
            trgPath = f'{opt.weightSaveLoc}/TRG.pkl'
            print("loading presaved fields...",srcPath)
            print(os.path.exists(srcPath),srcPath)
            SRC = pickle.load(open(srcPath, 'rb'))
            print(os.path.exists(trgPath), trgPath)
            TRG = pickle.load(open(trgPath, 'rb'))
        except Exception as e:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.weightSaveLoc + "/")
            raise(e)
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
    return train_iter


def create_dataset(opt, SRC, TRG, validBatchSize = -1):

    print("creating dataset and iterator... ")



    datasets = {}
    for subset in ("train","valid"):
        raw_data = {'src' : [line for line in getattr(opt,"src_data_"+subset)], 'trg': [line for line in getattr(opt,"trg_data_"+subset)]}
        df = pd.DataFrame(raw_data, columns=["src", "trg"])
        print("deeyef",df);

        if True or subset == "valid" and validBatchSize == -1:
            validBatchSize = 10#len(df);
            print("validBatchSize",10)

        mask = (df['src'].str.count(' ') < opt.max_len) & (df['trg'].str.count(' ') < opt.max_len)
        df = df.loc[mask]

        df.to_csv("translate_transformer_temp.csv", index=False)

        data_fields = [('src', SRC), ('trg', TRG)]
        dataset = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
        print("preit")
        myIter = MyIterator(dataset, batch_size=(opt.batchsize if subset == "train" else validBatchSize), device=torch.device('cuda'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=(batch_size_fn if subset== "train" else batch_size_fn_valid), train=True, shuffle=True)
        print("postit")
        os.remove('translate_transformer_temp.csv')
        datasets[subset] = {"iter":myIter, "ds":dataset}

    if not opt.load_weights:
        SRC.build_vocab(datasets["train"]["ds"])
        TRG.build_vocab(datasets["train"]["ds"])
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

    opt.train_len = get_len(datasets["train"]["iter"])
    print("curated", datasets["train"]["iter"])

    return datasets["train"]["iter"], datasets["valid"]["iter"]

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i


def create_fieldsFEH(tokenizer):
    camTok = CamOrLetterTokenizer(tokenizer)
    TRG = data.Field(lower=True, tokenize=camTok.tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=camTok.tokenize)
    return (SRC, TRG)