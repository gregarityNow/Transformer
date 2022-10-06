import numpy as np
import torch
import torch.nn as nn
from .Layers import EncoderLayer, DecoderLayer
from .Embed import Embedder, PositionalEncoder
from .Sublayers import Norm
from torch.autograd import Variable
import copy
import pickle


def testModel(mod, tok, src, printWeights = False):
    print('wizard of id',id(mod));
    weights = [x[1] for x in mod.named_parameters() if x[0] == "roberta.encoder.layer.11.output.LayerNorm.weight"][0]
    currWeights = mod.named_parameters();
    currWeights = sorted(currWeights, key = lambda x: x[0]);
    try:
        with open("prevWeights.pkl","rb") as fp:
            prevWeights = pickle.load(fp);
        for weightIndex in range(len(currWeights)):
            if not printWeights:
                continue;
            print("weight",torch.norm(currWeights[weightIndex][1]-prevWeights[weightIndex][1]).item(),currWeights[weightIndex][0], currWeights[weightIndex][1].requires_grad,currWeights[weightIndex][1].grad,prevWeights[weightIndex][1].requires_grad,prevWeights[weightIndex][1].grad);
    except Exception as e:
        print(e)
        pass
    with open("prevWeights.pkl", "wb") as fp:
        pickle.dump(currWeights, fp);


    print("testing 123", src, np.all([x.requires_grad for x in mod.parameters()]), np.any([x.requires_grad for x in mod.parameters()]), weights.shape);
    query = "Guillocher un support en métal pour y faire adhérer un émail."
    token_ids = tok.encode(query, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tok.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    with torch.no_grad():
        print("cudington",torch.cuda.is_available());
        if torch.cuda.is_available():
            output = mod(token_ids.cuda())
        else:
            output = mod(token_ids)
        print("testing model, we got", output[1][-1]);

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
    def forward(self, src, mask, dailleVec = None):
        x = self.embed(src)
        # print("embedded",x.shape, src.shape); #embedded torch.Size([2, 56, 768]) torch.Size([2, 56])
        try:
            x = self.pe(x)
        except Exception as e:
            print("flubbation",src)
            raise Exception("wubble du" + str(e))
        for i in range(self.N):
            x = self.layers[i](x, mask)
            print("xmas", x.shape, x);
        norm = self.norm(x)
        # print("normus")
        return norm



class TransformerCamembertLayer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout, camemModel, doDaille = True, camemTok = None):
        super().__init__()
        print("initializing the erweiteren model")
        #(src_vocab, d_model, N, heads, dropout)
        if not camemModel is None:
            camemModel.eval()
            if not camemTok is None:
                testModel(camemModel, camemTok, "init TransformerCamembertLayer");
        self.encoder = EncoderCamemLayer(768, d_model, 1, heads, dropout, camemModel=camemModel, doDaille = doDaille, camemTok = camemTok)
        self.decoder = Decoder(trg_vocab, d_model , N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

        # print("horscht",self.parameters());
        # for p in self.parameters():
        #     print("param",p, p.requires_grad);

    def forward(self, src, trg, src_mask, trg_mask, dailleVec = None):
        '''
        What's the deal with this mask? todo@feh
        :param src:
        :param trg:
        :param src_mask:
        :param trg_mask:
        :return:
        '''
        e_outputs = self.encoder(src, src_mask, dailleVec)
        # print("davor",e_outputs.shape);
        # if self.doDaille:
        #     q = torch.ones(e_outputs.shape[:2]).reshape(list(e_outputs.shape[:2])+[1]).to("cuda");
        #     e_outputs = torch.cat([e_outputs,q], dim=2);
        # print("how do you like me now",e_outputs.shape)
        # exit()
        # print("DECODER", e_outputs.shape, e_outputs.max(), e_outputs.min(), e_outputs)#,self.decoder)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

class EncoderCamemLayer(nn.Module):
    def __init__(self, camemHiddenSize, d_model, N, heads, dropout, camemModel, doDaille, camemTok):
        super().__init__()
        assert camemHiddenSize==d_model
        self.embed = Embedder(7, d_model)
        self.N = N
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        # self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.camemModel = camemModel
        self.doDaille = doDaille
        testModel(camemModel, camemTok, "init EncoderCamemLayer");
        self.camemTok = camemTok

        for x in self.named_parameters():
            if "camemModel" in x[0] or "roberta" in x[0]:
                x[1].requires_grad=False
                x[1].grad = None
                print(x[0])
        for x in self.parameters():
            print("gradius",x.requires_grad);
        # for x in self.named_parameters():
        #     print("camName", x[0], x[1].shape, x[1].requires_grad)


    def forward(self, src, mask, dailleVec = None):

        testModel(self.camemModel, self.camemTok, "fwd EncoderCamemLayer pre applic")
        with torch.no_grad():
            camemOut = self.camemModel(src)[1][-1]
            testModel(self.camemModel, self.camemTok, "fwd EncoderCamemLayer post nograd applic")
        testModel(self.camemModel, self.camemTok, "fwd EncoderCamemLayer post applic")
        print("davos", src, camemOut, camemOut.shape);
        x = camemOut
        # x = Variable(camemOut,requires_grad=False)
        # print("davai", x, x.shape);
        if self.doDaille:
            dailleEmbedded = self.embed(dailleVec).reshape([x.shape[0],1,x.shape[2]]);
            x = torch.cat([x, dailleEmbedded],dim=1);
        # print("xing",x.shape);
        printState = True#np.random.rand() < 0.01
        # for i in range(self.N):
        #     x = self.layers[i](x, mask)
        #     if printState:
        #         print("xmas",x.shape, x);
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        print("nitializing",d_model, vocab_size);
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        norm = self.norm(x)
        return norm

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask, dailleVec = None):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output




def get_model(opt, SRC, trg_vocabLen, camemModel = None, camemTok = None):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    src_vocab_len = len(SRC.vocab)

    if opt.camemLayer:
        print("getting extended model")
        model = TransformerCamembertLayer(trg_vocabLen, opt.d_model, opt.n_layers, opt.heads, opt.dropout, camemModel, doDaille= opt.daillePrediction, camemTok=camemTok)
        for x in model.named_parameters():
            if "camemModel" in x[0] or "roberta" in x[0]:
                x[1].requires_grad=False
                x[1].grad = None
                print(x[0])
    else:
        model = Transformer(src_vocab_len, trg_vocabLen, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    testModel(camemModel, camemTok,"in get_model")
    if opt.load_weights:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.weightSaveLoc}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
    testModel(camemModel, camemTok, "after xavier")

    if opt.device == 0:
        model = model.to("cuda:0")

    testModel(camemModel, camemTok, "after cuda!?")

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