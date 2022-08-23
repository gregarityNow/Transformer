import torch
from .Batch import nopeak_mask
import torch.nn.functional as F
import math
from numpy import inf

def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    if opt.device == 0:
        outputs = outputs.cuda()
    
    trg_mask = nopeak_mask(1)
    
    out = model.out(model.decoder(outputs,
    e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long()
    if opt.device == 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def outputAndLengthToTerm(TRG, output, length):
    return ''.join([TRG.vocab.itos[tok] for tok in output[1:length]])


def beam_search(src, model, SRC, TRG, opt, gold = ""):
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)

    constructionsAndLikelihoods = []
    seenWords = set()

    for i in range(2, opt.max_len):
        trg_mask = nopeak_mask(i)

        out = model.out(model.decoder(outputs[:,:i],e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        #todo@feh: what are the outputs then?

        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol
        alpha = 0.7
        div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
        # print("slen",sentence_lengths)
        likeScores = (log_scores * div).numpy()[0]
        for wordIndex, score in enumerate(likeScores):
            if score == -inf: continue;
            word = outputAndLengthToTerm(TRG, outputs[wordIndex],sentence_lengths[wordIndex])
            if word in seenWords:
                # print("evicted!",word);
                continue;
            constructionsAndLikelihoods.append((word, score));
            seenWords.add(word);
            print("adding",(word, score),sentence_lengths[wordIndex])

    if len(constructionsAndLikelihoods) > 0:
        print("here we are,|"+gold+"|, constructionsAndLikelihoods",constructionsAndLikelihoods);
        bestWord = max(constructionsAndLikelihoods,key=lambda tup: tup[1])[0]
    else:
        print("ain't got nothing",gold);
        bestWord = "N/A"
    return bestWord



