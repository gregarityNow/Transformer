from torch.autograd import Variable
import re
from .Beam import beam_search
import pathlib
import argparse
import time
# from nltk.corpus import wordnet
import torch
from .Models import get_model
from .Process import *
import torch.nn.functional as F
from .Optim import CosineWithRestarts
from .Batch import create_masks
import dill as pickle
# from ..src.basis_funcs import loadTokenizerAndModel
from .Tokenize import CamOrLetterTokenizer

modelDim = 768

def loadTokenizerAndModel(name, loadFinetunedModels = False, modelToo = False):
    import torch
    techName = ""
    if name == "xlmRob":
        techName = "xlm-roberta-base"
    if name == "camem":
        techName = "camembert-base"
    if name == "flaub":
        techName = "flaubert/flaubert_base_cased"
    print("loading",techName)
    proxDict = {"http": "http://webproxy.lab-ia.fr:8080", "https": "http://webproxy.lab-ia.fr:8080"}
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    try:
        tok = AutoTokenizer.from_pretrained(techName)
    except:
        tok = AutoTokenizer.from_pretrained(techName, proxies=proxDict)
    if not modelToo:
        return tok, None
    if loadFinetunedModels:
        rootPath = "/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/models/"+name + "-finetuned-tech/"
        checkpoints = [x for x in os.listdir(rootPath) if os.path.isdir(rootPath + "/" + x) and "checkpoint-" in x]
        checkpoints.sort(key = lambda cp: int(cp.split("-")[1]))
        latestCheckpoint = rootPath + "/" + checkpoints[-1]
        print("loading model from",latestCheckpoint)
        model = AutoModelForMaskedLM.from_pretrained(latestCheckpoint)
    else:
    # if name in tokModDict:
    #     return tokModDict[name]["tok"],tokModDict[name]["model"]
        try:
            model = AutoModelForMaskedLM.from_pretrained(techName,proxies=proxDict)
        except:
            model = AutoModelForMaskedLM.from_pretrained(techName)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # tokModDict[techName] = {}
    # tokModDict[techName]["tok"] = tok
    # tokModDict[techName]["model"] = model
    return tok, model

def getPredsAndLoss(model, src,trg,  trg_input, src_mask, trg_mask, opt, isTrain = True):
    preds = model(src, trg_input, src_mask, trg_mask)
    print("predis", preds.shape);
    ys = trg[:, 1:].contiguous().view(-1)
    if isTrain:
        opt.optimizer.zero_grad()
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
    return preds, loss


def train_model(model, opt):
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    for validBatch in opt.valid:
        srcValid = validBatch.src.transpose(0, 1)
        trgValid = validBatch.trg.transpose(0, 1)
        print("shaka smart",srcValid.shape, trgValid.shape)
        trg_inputValid = trgValid[:, :-1]
        src_maskValid, trg_maskValid = create_masks(srcValid, trg_inputValid, opt)
        break;

    losses = []


    def shouldBreak(myl):
        try:
            percDiff = (myl[-1] - myl[0]) / myl[-1]
            if percDiff > 0.05:
            #loss at end of epoch was significantly greater than beginning of epoch; that's no good at all!
                print("endingLoss",myl[-1]," was greater than beginning loss",myl[0],"percDiff",percDiff)
                return True;
            percDiffMaxMin = (max(myl) - min(myl)) / max(myl)
            if percDiffMaxMin < 0.05:
                #no loss fluctuation at all
                print("no loss fluxuation",max(myl),min(myl),percDiffMaxMin)
                return True
            return False
        except:
            return False

    outPath = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/byChar"
    bestModel = None
    bestLoss = np.inf
    epoch = 0;
    while True:

        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        for i, batch in enumerate(opt.train):
            print("batch",i,epoch,opt.train_len,len(batch))
            # for i, batch in enumerate(train_iter):
            #     if i == 1: break;

            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)

            # print("trainshape",src.shape, trg.shape)
            trg_input = trg[:, :-1]
            # src_mask, trg_mask = create_masks(src, trg_input, None)
            src_mask, trg_mask = create_masks(src, trg_input, opt)

            preds, loss = getPredsAndLoss(model, src,trg, trg_input, src_mask, trg_mask,opt, isTrain = True)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()

            _, validLoss = getPredsAndLoss(model, srcValid,trgValid, trg_inputValid, src_maskValid, trg_maskValid,opt, isTrain = False)
            losses.append({"epoch":epoch + i/opt.train_len,"train_loss":loss.item(),"valid_loss":validLoss.item()})
            print("trainLoss",loss.item(),"walidLoss",validLoss.item());
            
            total_loss += loss.item()
            
            if (i + 1) % opt.printevery == 0 or i == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                print("dumping model weights");
                torch.save(model.state_dict(),  outPath + 'weights/model_weights')
                cptime = time.time()
            if validLoss < bestLoss:
                torch.save(model.state_dict(), outPath + '/weights/model_weights_best')
                print("saving best model woot", outPath + '/weights/model_weights_best')
                cptime = time.time()
                bestLoss = validLoss

        if shouldBreak([loss["train_loss"] for loss in losses if loss["epoch"] > epoch]):
            print("progress has stopped; breaking")
            break;
        else:
            epoch += 1;
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))
    with open(outPath + "/losses.pickle","wb") as fp:
        pickle.dump(losses, fp);


#
def get_synonym(word, SRC):
    return 0;
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, SRC, TRG, gold = ""):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()

    sentence = beam_search(sentence, model, SRC, TRG, opt, gold = gold)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))

def evaluate(opt, model, SRC, TRG, df, suffix):
    from tqdm import tqdm
    tqdm.pandas()
    df = df.reset_index()
    df["byChar_" + suffix] = df.progress_apply(lambda row: translate_sentence(row.defn, model, opt, SRC, TRG, gold = row.term),axis=1)
    return df


def translateMain():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', default=True)
    parser.add_argument("-weightSaveLoc",type=str,default = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/byChar/weights")
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=modelDim)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')

    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_len > 10

    tokenizer, _ = loadTokenizerAndModel("camem")
    camOrLetterTokenizer = CamOrLetterTokenizer(tokenizer)
    SRC, TRG = create_fields(opt, camOrLetterTokenizer)

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    while True:
        opt.text = input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text == "q":
            break
        if opt.text == 'f':
            fpath = input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opt, model, SRC, TRG)
        print('> ' + phrase + '\n')

def mainFelix():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-d_model', type=int, default=modelDim)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument("-weightSaveLoc",type=str,default = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/byChar/weights")
    parser.add_argument('-load_weights', default=False)
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-quickie', type=int, default=1)
    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    df = read_data_felix(opt)
    tokenizer, mod = loadTokenizerAndModel("camem")
    camOrLetterTokenizer = CamOrLetterTokenizer(tokenizer)
    SRC, TRG = create_fields(opt, camOrLetterTokenizer)
    opt.train, opt.valid = create_dataset(opt, SRC, TRG)
    # create_dataset_spam()
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    print("moodely")

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    if opt.load_weights and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    # df = evaluate(opt, model, SRC, TRG, df[df.subset=="valid"],"_preTrain")
    train_model(model, opt)
    saveModel(model, opt, SRC, TRG, df)
    df = evaluate(opt, model, SRC, TRG, df[df.subset == "valid"], "_postTrain")

    dst = opt.weightSaveLoc
    pickle.dump(df, open(f'{dst}/postTune' + ("_quickie" if opt.quickie else "") + '.pkl','wb'));
    print("df is at",f'{dst}/postTune' + ("_quickie" if opt.quickie else "") + '.pkl')



def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def saveModel(model, opt, SRC, TRG, df):

    saved_once = 1 if opt.load_weights or opt.checkpoint > 0 else 0
    print("salvidor ramirez",saved_once, opt.load_weights, opt.checkpoint)


    dst = opt.weightSaveLoc

    pathlib.Path(dst).mkdir(exist_ok=True,parents=True);
    print("saving weights to " + dst + "/...")
    torch.save(model.state_dict(), f'{dst}/model_weights' + ("_quickie" if opt.quickie else ""))
    pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))

    print("weights and field pickles saved to " + dst)


    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    mainFelix()
