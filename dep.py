
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
    parser.add_argument("-weightSaveLoc",type=str,default = "/mnt/beegfs/home/herron/neo_scf_herron/stage/out/dump/byChar/weights/")
    parser.add_argument('-load_weights', default=False)
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-quickie', type=int, default=1)
    parser.add_argument('-doTrain', type=int, default=1)
    parser.add_argument('-doEval', type=int, default=1)
    parser.add_argument('-camemLayer',type=int,default=0)
    parser.add_argument("-daillePrediction", type=int, default=1);
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
    dst = opt.weightSaveLoc

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    if opt.load_weights and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    if opt.doTrain:
        train_model(model, opt, numEpochsShouldBreak = 5)
        saveModel(model, opt, SRC, TRG)
    else:
        SRC = pickle.load(open(f'{dst}/SRC.pkl', 'rb'))
        TRG = pickle.load(open(f'{dst}/TRG.pkl', 'rb'))
        model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    if opt.doEval:
        dfValid = df[df.subset == "valid"]
        df = evaluate(opt, model, SRC, TRG, dfValid, "_postTrain")

        pickle.dump(df, open(f'{dst}/postTune' + ("_quickie" if opt.quickie else "") + '.pkl','wb'));
        print("df is at",f'{dst}/postTune' + ("_quickie" if opt.quickie else "") + '.pkl')
