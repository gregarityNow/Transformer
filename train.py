from torch.autograd import Variable
import psutil
from .Beam import beam_search
import pathlib
import argparse
import time
# from nltk.corpus import wordnet
import torch
from .Models import get_model, testModel
from .Process import *
import torch.nn.functional as F
from .Optim import CosineWithRestarts
from .Batch import create_masks
import dill as pickle
# from ..src.basis_funcs import loadTokenizerAndModel
from .Tokenize import CamOrLetterTokenizer

modelDim = 768

testSentence = "Appareil électronique capable, en appliquant des instructions prédéfinies (programme), d’effectuer des traitements automatisés de données et d’interagir avec l’environnement grâce à des périphériques (écran, clavier...)."


def loadTokenizerAndModel(name, loadFinetunedModels = False, modelToo = False, hiddenStates = False, requireGrad = True):
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

    testModel(model, tok, "loading model");
    if not requireGrad:
        for x in model.parameters():
            x.requires_grad = False
            x.grad = None

    return tok, model

def getPredsAndLoss(model, src,trg,  trg_input, src_mask, trg_mask, opt, isTrain = True, dailleVec = None):

    preds = model(src, trg_input, src_mask, trg_mask, dailleVec)
    ys = trg[:, 1:].contiguous().view(-1)
    if isTrain:
        opt.optimizer.zero_grad()
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
    return preds, loss

def getDailleVec(daille_types):
    dailleVec = torch.tensor([dailleEncoder[x] for x in daille_types]).to("cuda");
    return dailleVec

def tensorCamemEncode(inputs, camemTok, maxLen):
    if maxLen == -1:
        camEnc = torch.tensor(camemTok(inputs)['input_ids'])
    else:
        camEnc = torch.tensor(camemTok(inputs, padding="max_length", max_length=maxLen)['input_ids'])
    return camEnc.to("cuda");

def train_model(model, opt, trainDf, validDf, SRC, TRG, camemMod = None, camemTok = None, numEpochsShouldBreak = 3, bestLoss = np.inf, losses = [], initialEpoch = 0,initialBatchNumber = 0, fineTune = False):
    
    print("training model...",trainDf,validDf)
    testModel(camemMod,camemTok,"begninning of train_model")
    opt.optimizer.zero_grad()
    testModel(camemMod, camemTok, "grad school")
    model.train()
    testModel(camemMod, camemTok, "post .train")
    opt.optimizer.zero_grad()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    # np.random.seed(420);
    # miniTestDf = validDf.sample(5, random_state=420)["defn"];

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



    def batchToSrcTrg(batch, TRG, doDaille = True):
        inputs = list(batch.defn)
        src = tensorCamemEncode(inputs, camemTok, batch.camemDefnLen.max())
        trg = torch.tensor(TRG.process(batch.term)).T.to("cuda")
        if doDaille:
            dailleVec = getDailleVec(list(batch.daille_type))
        else:
            dailleVec = None
        # src = torch.ones_like(src);
        # trg = torch.ones_like(trg);
        return src.to("cuda"), trg.to("cuda"), dailleVec

    def getLoss(src, trg, trg_input, dailleVec = None):
        src_maskValid, trg_maskValid = create_masks(src, trg_input, opt)
        _, validLoss = getPredsAndLoss(model, src, trg, trg_input, src_maskValid, trg_maskValid, opt, isTrain=False, dailleVec=dailleVec)
        thisLoss = validLoss.item() * src.shape[0]
        return thisLoss

    meanCamemLen = pd.concat([trainDf, validDf]).camemDefnLen.mean()
    batchSizeStandard = int(opt.batchsize // meanCamemLen)
    numBatchesValid = int(max(len(validDf) // batchSizeStandard, 1))
    numBatchesTrain = int(max(len(trainDf) // batchSizeStandard, 1))

    def doValidation():
        totalValidLoss = 0
        totalSamps = 0
        for validBatch in opt.valid:
            if opt.camemLayer: break;
            if opt.valid_len > 10 and np.random.rand() > 0.33: continue;
            srcValid = validBatch.src.transpose(0, 1)
            trgValid = validBatch.trg.transpose(0, 1)
            trg_inputValid = trgValid[:, :-1]
            thisLoss = getLoss(srcValid, trgValid, trg_inputValid)
            totalValidLoss += thisLoss
            totalSamps += srcValid.shape[0]

        for validBatchIndex in range(numBatchesValid):
            if not opt.camemLayer: break;
            if (not fineTune) and numBatches > 10 and np.random.rand() > 0.33:continue;
            batch = trainDf[validBatchIndex * batchSizeStandard:(validBatchIndex+1) *batchSizeStandard];
            srcValid, trgValid, dailleVec = batchToSrcTrg(batch, TRG, opt.daillePrediction);
            trg_inputValid = trgValid[:, :-1]
            thisLoss = getLoss(srcValid, trgValid, trg_inputValid, dailleVec)
            totalValidLoss += thisLoss
            totalSamps += srcValid.shape[0]
        validLoss = totalValidLoss / totalSamps
        print("validated on",totalSamps)
        return validLoss

    def handleTrain(src, trg, opt, losses, batchIndex, bestLoss, finetune, dailleVec = None):

        # print("testtrans before any losses can occur!", miniTestDf);
        # testModel(camemMod, camemTok, "before minitest application")
        # miniTestDf.apply(lambda sent: print("test translation pre", sent, translate_sentence(sent, model, opt, SRC, TRG, gold="", daille_type=None, camemTok=camemTok)))
        # testModel(camemMod, camemTok, "after minitest application")

        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        trainTime = time.time()
        preds, loss = getPredsAndLoss(model, src, trg, trg_input, src_mask, trg_mask, opt, isTrain=True, dailleVec=dailleVec)
        loss.backward()

        opt.optimizer.step()

        if opt.SGDR == True:
            opt.sched.step()
        trainTime = time.time() - trainTime
        print("trainTime",trainTime)

        validEvery = 2 if finetune else 10
        if batchIndex % validEvery == 0:

            walidTime = time.time()
            validLoss = doValidation()
            walidTime = time.time() - walidTime
            print("walid ending", walidTime)
            # print("shaka smart", srcValid.shape, trgValid.shape, thisLoss)

            trainLoss = loss.item()
            losses.append({"epoch": epoch + trainBatchIndex / opt.train_len, "train_loss": trainLoss, "valid_loss": validLoss})
            print("trainLoss", trainLoss, "walidLoss", validLoss);

            if validLoss < bestLoss:
                bestPath = outPath + '/model_weights_best'   # + ("_quickie" if opt.quickie else "");
                bestPath += opt.outAddendum
                if fineTune: bestPath += "_fineTune"
                torch.save(model.state_dict(), bestPath)
                print("saving best model woot", bestPath)
                # print("horus",len(model.state_dict()))
                # for k in model.state_dict().keys():
                #     v = model.state_dict()[k]
                #     print("key: ",k, v.shape);
                bestLoss = validLoss

            dumpLosses(losses, opt.weightSaveLoc)
        else:
            print("not computing the walidation this time soary")
        # print("commencing testTrans", miniTestDf);
        # miniTestDf.apply(lambda sent: print("test translation",sent,translate_sentence(sent, model, opt, SRC, TRG, gold="", daille_type=None, camemTok=camemTok)))
        # translation = translate_sentence(testSentence, model, opt, SRC, TRG, gold="", daille_type=None, camemTok=camemTok)
        # print("test translation",translation)
        return bestLoss

    while True:
        print("beginning of while loop")
        testModel(camemMod, camemTok, "beginning of while loop")
        if opt.floyd is False:
            print("floyd   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

        # if opt.checkpoint > 0:
        #     print("each save")
        #     torch.save(model.state_dict(), outPath + '/model_weights')

        numBatches = max(len(trainDf) // opt.batchsize,1)
        trainDf = trainDf.sample(frac=1);
        print("sizes",numBatches, len(trainDf), numBatchesTrain)
        for trainBatchIndex in range(numBatchesTrain):
            if not opt.camemLayer: break;
            if initialBatchNumber > 0:
                initialBatchNumber += -1;
                continue;
            testModel(camemMod, camemTok, "inTrainLoop")
            # print("inTrain",psutil.virtual_memory())
            batchCreateTime = time.time()
            batch = trainDf[trainBatchIndex*batchSizeStandard:(trainBatchIndex+1)*batchSizeStandard];
            src, trg, dailleVec = batchToSrcTrg(batch, TRG, opt.daillePrediction);
            batchCreateTime = time.time()-batchCreateTime
            print("batch", epoch, trainBatchIndex, numBatchesTrain, batchSizeStandard, src.shape, "batchCreateTime",batchCreateTime)
            # print("dailleVec",dailleVec)

            bestLoss = handleTrain(src, trg, opt, losses, trainBatchIndex, bestLoss, fineTune, dailleVec)
        numBatches = opt.train_len
        for trainBatchIndex, batch in enumerate(opt.train):
            if opt.camemLayer: break;
            # print("inTrain",psutil.virtual_memory())

            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)
            # print("batchNoCam", epoch, trainBatchIndex, numBatches, len(batch), src.shape)

            bestLoss = handleTrain(src, trg, opt, losses, trainBatchIndex, bestLoss, fineTune)

        if opt.hundoEpochs and epoch % 5 == 0:
            performValidation(opt, model, SRC, TRG, camemTok, currEpoch = epoch)

        if not opt.hundoEpochs and (shouldBreak([loss["train_loss"] for loss in losses if loss["epoch"] > epoch], epoch)):#
            shouldBroke += 1
            if shouldBroke == numEpochsShouldBreak or opt.quickie:
                print("progress has stopped; breaking")
                break;
            else:
                print("ok you get another chance to do better next time")
                epoch += 1;
        elif opt.hundoEpochs and ((fineTune and epoch >= 100) or (not fineTune and epoch >= 10)):
            print("already done a lot of epochs, that seems to be quite enough")
            break;
        elif opt.quickie:
            print("quickie breaking",shouldBroke, epoch);
            break;
        else:
            shouldBroke = 0;
            epoch += 1;
            print("pssh we ain't breaking!")

    lastPath = outPath + '/model_weights_last' + opt.outAddendum
    if fineTune: lastPath += "_fineTune"
    torch.save(model.state_dict(), lastPath)
    print("saving last model", lastPath)

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


def translate_sentence(sentence, model, opt, SRC, TRG, gold = "", daille_type = None, camemTok = None):
    model.eval()
    if not opt.camemLayer:
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
        # print("sentence",sentence);
    else:
        sentence = tensorCamemEncode([sentence], camemTok, -1)

    if not daille_type is None:
        dailleVec = getDailleVec([daille_type,])
    else:
        dailleVec = None
    sentence = beam_search(sentence, model, SRC, TRG, opt, gold = gold, dailleVec = dailleVec)
    # opt.countdown += -1;
    # if opt.countdown == 0:
    #     exit()

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

    if not os.path.exists(bestPath):
        print("no FT weights, reverting to no FT")
        bestPath = f'{path}/model_weights_best'

    sd = torch.load(bestPath)
    model.load_state_dict(sd)
    print("the model now has (lowers sunglasses) best weights, ooo");

def evaluate(opt, model, SRC, TRG, df, suffix, fineTune = True, camemTok = None):
    from tqdm import tqdm
    tqdm.pandas()
    try:
        print("tryna load",f'{opt.weightSaveLoc}/model_weights_best')
        getBestModel(model, opt.weightSaveLoc, fineTune = fineTune)
    except Exception as e:
        print("no best weights available, no worries hoss",e)
        raise(e);

    # opt.countdown = 2
    df = df.reset_index()
    df["byChar_" + suffix] = df.progress_apply(lambda row: translate_sentence(row.defn, model, opt, SRC, TRG, gold = row.term, daille_type = row.daille_type, camemTok=camemTok),axis=1)
    return df

def dumpLosses(losses, dst):
    dumpTime = time.time()
    dumpPath = dst + "/../losses.pickle"
    with open(dumpPath, "wb") as fp:
        pickle.dump(losses, fp);
    dumpTime = time.time()-dumpTime
    print("dumped to",dumpPath,dumpTime)

def fetchLosses(dst):
    print("nst",dst)
    dumpPath = dst + "/losses.pickle"
    try:
        with open(dumpPath, "rb") as fp:
            losses = pickle.load(fp);
    except:
        print("no losses yet hoss")
        return []
    return losses


def performValidation(opt, model, SRC, TRG, camemTok, currEpoch = 0):
    dst = opt.weightSaveLoc + "/.."
    _, dfValid = read_data_felix(opt, allTerms=False)
    if opt.camemLayer:
        opt.src_pad = camemTok.pad_token_id
    else:
        opt.src_pad = SRC.vocab.stoi['<pad>']
    df = evaluate(opt, model, SRC, TRG, dfValid, "_epoch_"+str(currEpoch), fineTune=True, camemTok=camemTok)
    pickle.dump(df, open(dst + "/allRes100_" + str(currEpoch) + "_" + opt.outAddendum + ".pkl", 'wb'));
    print("df is at", f"{dst}/allRes100_" + str(currEpoch) + "_" + opt.outAddendum + ".pkl")


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
    parser.add_argument('-batchsize', type=int, default=1500)
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
    parser.add_argument("-startFromCheckpoint",type=int,default=1);
    parser.add_argument("-fullWiktPretune", type=int, default=0);
    parser.add_argument("-daillePrediction", type=int, default=1);
    parser.add_argument("-revise", type=int, default=1);
    parser.add_argument("-suffix", type=str, default="");
    parser.add_argument("-outAddendum", type=str, default="");
    parser.add_argument("-useNorm",type=int, default = 1)
    parser.add_argument("-numEncoderLayers", type=int, default=1)
    parser.add_argument("-preVal", type=int, default=0)
    parser.add_argument("-hundoEpochs", type=int, default=1)
    opt = parser.parse_args()

    runType = "byChar"
    if opt.camemLayer:
        runType += "_camemLayer"
    # if opt.daillePrediction:
    #     runType += "_daillePrediction";

    if opt.revise:
        runType += "_revise"


    if opt.quickie:
        runType += "_quickie";

    if len(opt.suffix) > 0:
        runType += "_" + opt.suffix

    if opt.hundoEpochs:
        runType += "_hundo"

    opt.weightSaveLoc = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/final/" + runType + "/weights"
    pathlib.Path(opt.weightSaveLoc).mkdir(exist_ok=True,parents=True)

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    camemTok, camemMod = loadTokenizerAndModel("camem", modelToo=True, hiddenStates=True, requireGrad=False)
    camOrLetterTokenizer = CamOrLetterTokenizer(camemTok)
    testModel(camemMod, camemTok, "after loading model");

    dst = opt.weightSaveLoc + "/.."
    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    losses = fetchLosses(dst)

    currEpoch = 0
    minLoss = 10000
    if opt.hundoEpochs:
        if len(losses) > 0:
            currEpoch = int(max([x["epoch"] for x in losses]))
            minLoss = min([x["valid_loss"] for x in losses])

    if opt.hundoEpochs and currEpoch >= 100:
        print("100 epochs done; woot!")
        exit()

    if opt.doTrain or opt.hundoEpochs:
        dfTrain, dfValid = read_data_felix(opt, allTerms=True)
        SRC, TRG = create_fields(opt, camOrLetterTokenizer)
        opt.train, opt.valid = create_dataset(opt, SRC, TRG, camemTok=camemTok)

        pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
        pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))

        model = get_model(opt, SRC, len(TRG.vocab), camemModel=(camemMod if opt.camemLayer else None), camemTok=(camemTok if opt.camemLayer else None))

        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
        #todo@fehMaxim: consider optimizer parameters
        if opt.SGDR == True:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

        # print("moodely", model.state_dict().keys())

        if opt.hundoEpochs and currEpoch < 10:
            getBestModel(model,opt.weightSaveLoc, fineTune=False)
        elif opt.hundoEpochs and currEpoch >= 10:
            getBestModel(model,opt.weightSaveLoc, fineTune=True)

        print("it's a mad mad mad mad world", opt.fullWiktPretune,opt.startFromCheckpoint)

        #train on all wiktionnaire data
        if opt.fullWiktPretune or (opt.hundoEpochs and currEpoch < 10):
            bestLossInitialTraining, losses, lastEpoch = train_model(model, opt,dfTrain, dfValid, SRC, TRG, camemMod=camemMod, camemTok=camemTok, numEpochsShouldBreak=2, losses=losses, initialEpoch=currEpoch, bestLoss=minLoss);
        elif opt.startFromCheckpoint:
            getBestModel(model, opt.weightSaveLoc, fineTune=False)
            print("checky check boiii");
        elif (opt.hundoEpochs and currEpoch >= 10):
            getBestModel(model, opt.weightSaveLoc, fineTune=True)
        else:
            bestLossInitialTraining, losses, lastEpoch = np.inf, [], 0

        testModel(camemMod, camemTok, "encore from checkpoint!?")

        #finetune on neonyms
        dfTrain, dfValid = read_data_felix(opt, allTerms=False)
        opt.train, opt.valid = create_dataset(opt, SRC, TRG, fineTune=True, camemTok=camemTok)
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr*0.1, betas=(0.9, 0.98), eps=1e-9)
        if opt.SGDR == True:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

        train_model(model, opt, dfTrain, dfValid, SRC, TRG, camemMod=camemMod, camemTok=camemTok, losses = losses, initialEpoch = currEpoch, numEpochsShouldBreak=15, fineTune = True);
        dumpLosses(losses, dst)
    # else:
    #     SRC = pickle.load(open(f'{dst}/SRC.pkl', 'rb'))
    #     TRG = pickle.load(open(f'{dst}/TRG.pkl', 'rb'))
    #     print("srcy",dst)
    #     model = get_model(opt, SRC, len(TRG.vocab), camemModel=camemMod, camemTok=camemTok)
    #     dfTrain, dfValid = read_data_felix(opt, allTerms=False)
    #     if opt.camemLayer:
    #         opt.src_pad = camemTok.pad_token_id
    #     else:
    #         opt.src_pad = SRC.vocab.stoi['<pad>']
    # if opt.doEval:
    #     #
    #     df = evaluate(opt, model, SRC, TRG, dfValid, "_postFinetune", fineTune = True, camemTok=camemTok)
    #     pickle.dump(df, open(dst + "/postTuneCamemLayer" + opt.outAddendum + ".pkl",'wb'));
    #     print("df is at",f"{dst}/postTuneCamemLayer" + opt.outAddendum + ".pkl")
    #
    #     getBestModel(model, opt.weightSaveLoc, fineTune=False)
    #     df = evaluate(opt, model, SRC, TRG, dfValid, "_preFinetune", fineTune=False, camemTok=camemTok)
    #     pickle.dump(df, open(dst + "/preTuneCamemLayer" + opt.outAddendum + ".pkl", 'wb'));
    #     print("df is at", dst + "/postTuneCamemLayer" + opt.outAddendum + ".pkl")



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


#check out skim?
#https://direct.mit.edu/coli/article/37/2/309/2099/Unsupervised-Learning-of-Morphology