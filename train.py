from torch.autograd import Variable
import psutil
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

def loadTokenizerAndModel(name, loadFinetunedModels = False, modelToo = False, hiddenStates = False):
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
        tok = AutoTokenizer.from_pretrained(techName, padding="max_length", max_length=100)
    except:
        tok = AutoTokenizer.from_pretrained(techName, proxies=proxDict, padding="max_length", max_length=100)
    if not modelToo:
        return tok, None
    if loadFinetunedModels:
        rootPath = "/mnt/beegfs/projects/neo_scf_herron/stage/out/dump/models/"+name + "-finetuned-tech/"
        checkpoints = [x for x in os.listdir(rootPath) if os.path.isdir(rootPath + "/" + x) and "checkpoint-" in x]
        checkpoints.sort(key = lambda cp: int(cp.split("-")[1]))
        latestCheckpoint = rootPath + "/" + checkpoints[-1]
        print("loading model from",latestCheckpoint)
        model = AutoModelForMaskedLM.from_pretrained(latestCheckpoint, output_hidden_states=hiddenStates)
    else:
    # if name in tokModDict:
    #     return tokModDict[name]["tok"],tokModDict[name]["model"]
        try:
            model = AutoModelForMaskedLM.from_pretrained(techName,proxies=proxDict, output_hidden_states=hiddenStates)
        except:
            model = AutoModelForMaskedLM.from_pretrained(techName, output_hidden_states=hiddenStates)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # tokModDict[techName] = {}
    # tokModDict[techName]["tok"] = tok
    # tokModDict[techName]["model"] = model
    return tok, model

def getPredsAndLoss(model, src,trg,  trg_input, src_mask, trg_mask, opt, isTrain = True, camemModel = None, camemTok = None):
    # print("norse",type(src),src.shape,trg.shape);
    # with open("norse.pickle","wb") as fp:
    #     pickle.dump(src,fp);

    preds = model(src, trg_input, src_mask, trg_mask)
    ys = trg[:, 1:].contiguous().view(-1)
    if isTrain:
        opt.optimizer.zero_grad()
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
    return preds, loss


def train_model(model, opt, trainDf, validDf, camemMod = None, camemTok = None, numEpochsShouldBreak = 3, bestLoss = np.inf, losses = [], initialEpoch = 0, fineTune = False):
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()


    shouldBroke = 0
    epoch = initialEpoch

    def shouldBreak(myl, epoch):
        print("epoch over, deliberating break",epoch)
        try:
            percDiff = (myl[-1] - myl[0]) / myl[-1]
            if percDiff > 0:
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

    outPath = opt.weightSaveLoc#"/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/byChar"

    def batchToSrcTrg(batch):
        src = camemTok(list(batch.defn), padding="max_length", max_length=100)['input_ids'].to("cuda")
        trg = camemTok(list(batch.term), padding="max_length", max_length=100)['input_ids'].to("cuda")
        return src, trg

    def doValidation():
        totalValidLoss = 0
        totalSamps = 0
        numBatches = max(len(validDf) // opt.batchsize,1)
        startIndex = 0
        for validBatchIndex in range(numBatches):
            if (not fineTune) and np.random.rand() > 0.33:continue;
            batch = trainDf[trainBatchIndex * batchsize:trainBatchIndex * (batchsize + 1)];
            print("batching", batch);
            srcValid, trgValid = batchToSrcTrg(batch);
            trg_inputValid = trgValid[:, :-1]
            src_maskValid, trg_maskValid = create_masks(srcValid, trg_inputValid, opt)
            _, validLoss = getPredsAndLoss(model, srcValid, trgValid, trg_inputValid, src_maskValid, trg_maskValid, opt, isTrain=False, camemModel=camemMod, camemTok=camemTok)
            thisLoss = validLoss.item() * srcValid.shape[0]
            totalValidLoss += thisLoss
            totalSamps += srcValid.shape[0]
        validLoss = totalValidLoss / totalSamps
        print("validated on",totalSamps)
        return validLoss

    while True:

        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            print("each save")
            torch.save(model.state_dict(), outPath + '/model_weights')

        numBatches = max(len(trainDf) // opt.batchsize,1)
        batchsize = opt.batchsize
        trainDf = trainDf.sample(frac=1);
        print("sizes",numBatches, len(trainDf), batchsize)
        for trainBatchIndex in range(numBatches):
            print("batch",epoch,trainBatchIndex,numBatches, batchsize)

            print("inTrain",psutil.virtual_memory())

            # for i, batch in enumerate(train_iter):
            #     if i == 1: break;
            batch = trainDf[trainBatchIndex*batchsize:trainBatchIndex*(batchsize+1)];
            print("batching",batch);
            src, trg = batchToSrcTrg(batch);

            # src = batch.src.transpose(0,1)
            # trg = batch.trg.transpose(0,1)
            print("what we need",type(src), type(trg),src.shape, trg.shape);

            # print("trainshape",src.shape, trg.shape)
            trg_input = trg[:, :-1]
            # src_mask, trg_mask = create_masks(src, trg_input, None)
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            trainTime = time.time()
            preds, loss = getPredsAndLoss(model, src,trg, trg_input, src_mask, trg_mask,opt, isTrain = True, camemModel=camemMod, camemTok=camemTok)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            trainTime = time.time() - trainTime
            print("did training step",trainTime)
            print("beginning walidation")
            walidTime = time.time()
            validLoss = doValidation()
            walidTime = time.time()-walidTime
            print("walid ending",walidTime)
                # print("shaka smart", srcValid.shape, trgValid.shape, thisLoss)


            losses.append({"epoch":epoch + trainBatchIndex/opt.train_len,"train_loss":loss.item(),"valid_loss":validLoss})
            print("trainLoss",loss.item(),"walidLoss",validLoss);
            
            total_loss += loss.item()
            if (trainBatchIndex + 1) % opt.printevery == 0 or trainBatchIndex == 0:
                 p = int(100 * (trainBatchIndex + 1) / opt.train_len)
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
                torch.save(model.state_dict(),  outPath + '/model_weights')
                cptime = time.time()
            if validLoss < bestLoss:
                bestPath = outPath + '/model_weights_best'# + ("_quickie" if opt.quickie else "");
                if fineTune: bestPath += "_fineTune"
                torch.save(model.state_dict(), bestPath)
                print("saving best model woot", bestPath)
                cptime = time.time()
                bestLoss = validLoss
            dumpLosses(losses, opt.weightSaveLoc)

        if shouldBreak([loss["train_loss"] for loss in losses if loss["epoch"] > epoch], epoch):
            shouldBroke += 1
            if shouldBroke == numEpochsShouldBreak or opt.quickie:
                print("progress has stopped; breaking")
                break;
        elif epoch == 7:
            print("already done 8 epochs, that seems to be quite enough")
            break;
        else:
            shouldBroke = 0;
            epoch += 1;
            print("pssh we ain't brekaing!")
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

    return bestLoss, losses, epoch


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

def getBestModel(model, path, fineTune = True):
    bestPath = f'{path}/model_weights_best'
    if fineTune:
        bestPath += "_fineTune"
    model.load_state_dict(torch.load(bestPath))
    print("the model now has (lowers sunglasses) best weights, ooo");

def evaluate(opt, model, SRC, TRG, df, suffix, fineTune = True):
    from tqdm import tqdm
    tqdm.pandas()
    try:
        print("tryna load",f'{opt.weightSaveLoc}/model_weights_best')
        getBestModel(model, opt.weightSaveLoc, fineTune = fineTune)
    except Exception as e:
        print("no best weights available, no worries hoss",e)
        raise(e);

    df = df.reset_index()
    df["byChar_" + suffix] = df.progress_apply(lambda row: translate_sentence(row.defn, model, opt, SRC, TRG, gold = row.term),axis=1)
    return df

def dumpLosses(losses, dst):
    dumpTime = time.time()
    dumpPath = dst + "/../losses.pickle"
    with open(dumpPath, "wb") as fp:
        pickle.dump(losses, fp);
    dumpTime = time.time()-dumpTime
    print("dumped to",dumpPath,dumpTime)

def mainFelixCamemLayer():
    print("shabloimps")
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-d_model', type=int, default=modelDim)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=20)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-load_weights', default=False)
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-quickie', type=int, default=1)
    parser.add_argument('-doTrain', type=int, default=1)
    parser.add_argument('-doEval', type=int, default=1)
    parser.add_argument('-camemLayer',type=int,default=1)
    parser.add_argument("-hack",type=int,default=0)
    parser.add_argument("-startFromCheckpoint",type=int,default=0);
    parser.add_argument("-fullWiktPretune", type=int, default=1);
    parser.add_argument("-daillePrediction", type=int, default=1);
    opt = parser.parse_args()

    if opt.camemLayer:
        runType = "byCharCamemLayer"
    else:
        runType = "byChar"

    if opt.quickie:
        runType += "_quickie";

    opt.weightSaveLoc = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/" + runType + "/weights"
    pathlib.Path(opt.weightSaveLoc).mkdir(exist_ok=True,parents=True)

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    camemTok, camemMod = loadTokenizerAndModel("camem", modelToo=True, hiddenStates=True)
    camOrLetterTokenizer = CamOrLetterTokenizer(camemTok)

    dst = opt.weightSaveLoc
    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))


    if opt.doTrain:
        dfTrain, dfValid = read_data_felix(opt, allTerms=True)
        SRC, TRG = create_fields(opt, camOrLetterTokenizer)
        opt.train, opt.valid = create_dataset(opt, SRC, TRG)

        pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
        pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))

        model = get_model(opt, SRC, len(TRG.vocab), camemModel=(camemMod if opt.camemLayer else None))

        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
        if opt.SGDR == True:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

        print("moodely", model.state_dict().keys())

        #train on all wiktionnaire data
        if opt.fullWiktPretune:
            bestLossInitialTraining, losses, lastEpoch = train_model(model, opt,dfTrain, dfValid, camemMod=camemMod, camemTok=camemTok, numEpochsShouldBreak=2);
        else:
            bestLossInitialTraining, losses, lastEpoch = np.inf, [], 0

        #finetune on neonyms
        dfTrain, dfValid = read_data_felix(opt, allTerms=False)
        opt.train, opt.valid = create_dataset(opt, SRC, TRG, fineTune=True)
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr*0.1, betas=(0.9, 0.98), eps=1e-9)
        if opt.SGDR == True:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
        if opt.startFromCheckpoint:
            getBestModel(model,opt.weightSaveLoc)
        train_model(model, opt, dfTrain, dfValid, camemMod=camemMod, camemTok=camemTok, bestLoss=bestLossInitialTraining, losses = losses, initialEpoch = lastEpoch+1, numEpochsShouldBreak=2, fineTune = True);
        dumpLosses(losses, dst)
    else:
        SRC = pickle.load(open(f'{dst}/SRC.pkl', 'rb'))
        TRG = pickle.load(open(f'{dst}/TRG.pkl', 'rb'))
        print("srcy",dst)
        model = get_model(opt, len(SRC.vocab), len(TRG.vocab), camemModel=camemMod)
        df = read_data_felix(opt, allTerms=False)
    if opt.doEval:
        dfValid = df[df.subset == "valid"]
        dfPreFinetune = evaluate(opt, model, SRC, TRG, dfValid, "_postTrain")
        df = evaluate(opt, model, SRC, TRG, dfValid, "_postFinetune", fineTune = True)

        pickle.dump(df, open(f'{dst}/postTuneCamemLayer.pkl','wb'));
        print("df is at",f'{dst}/postTuneCamemLayer.pkl')



def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def saveModel(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights or opt.checkpoint > 0 else 0
    print("salvidor ramirez",saved_once, opt.load_weights, opt.checkpoint)

    dst = opt.weightSaveLoc

    pathlib.Path(dst).mkdir(exist_ok=True,parents=True);
    print("saving weights to " + dst + "/...")
    torch.save(model.state_dict(), f'{dst}/model_weights')
    pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))

    print("weights and field pickles saved to " + dst)


    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    mainFelix()
