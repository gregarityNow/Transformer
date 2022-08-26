import torch
import torch.nn as nn 
from .Layers import EncoderLayer, DecoderLayer
from .Embed import Embedder, PositionalEncoder
from .Sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        print("embedded",x.shape, src.shape);
        try:
            x = self.pe(x)
        except Exception as e:
            print("flubbation",src)
            raise Exception("wubble du" + str(e))
        for i in range(self.N):
            x = self.layers[i](x, mask)
        norm = self.norm(x)
        print("normus")
        exit();
        return norm

class TransformerCamembertLayer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        #(src_vocab, d_model, N, heads, dropout)
        self.encoder = EncoderCamemLayer(768, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        '''
        What's the deal with this mask? todo@feh
        :param src:
        :param trg:
        :param src_mask:
        :param trg_mask:
        :return:
        '''
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

class EncoderCamemLayer(nn.Module):
    def __init__(self, camemHiddenSize, d_model, N, heads, dropout):
        super().__init__()
        assert camemHiddenSize==d_model
        self.N = N
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, camemInnerLayer, mask):
        print("embedded",camemInnerLayer.shape)
        x = camemInnerLayer
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output




def get_model(opt, src_vocab, trg_vocab, camLayer = False):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    if camLayer:
        model = TransformerCamembertLayer(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    else:
        model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.weightSaveLoc}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.to("cuda:0")
    
    return model
    


class PreTrainedTokTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, encoding, trg, encoding_mask, trg_mask):
        #print("DECODER")
        d_output = self.decoder(trg, encoding, encoding_mask, trg_mask)
        output = self.out(d_output)
        return output


'''
todo@feh add a symbol at beginning (a n'importe-quoi for unknown)
todo@feh: take the last layer of camembert
todo@feh: pre-tune with Wiktionnaire
perplexity w/ NLTK/Markov
'''