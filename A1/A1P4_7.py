def batching(X,T,Y,batch_size):
    batches=int((len(X)+batch_size-1)/batch_size)
    X_batched,T_batched,Y_batched=[],[],[]
    for i in range(batches-1):
        X_batched.append(X[i*batch_size:(i+1)*batch_size])
        T_batched.append(T[i*batch_size:(i+1)*batch_size])
        Y_batched.append(Y[i*batch_size:(i+1)*batch_size])
    X_batched.append(X[(batches-1)*batch_size:])
    T_batched.append(T[(batches-1)*batch_size:])
    Y_batched.append(Y[(batches-1)*batch_size:])
    return X_batched,T_batched,Y_batched

def train_sgns(textlist, window, embedding_size):
    # Set up a model with Skip-gram with negative sampling (predict context with word)
    # textlist: a list of strings
    # Create Training Data
    # Split the training data
    # instantiate the network & set up the optimizer

    lemmas, v2i, i2v=prepare_texts(textlist)
    vocab_size=len(v2i)
    X, T, Y =tokenize_and_preprocess_text(lemmas,w2i,window)
    datasize=len(X)
    np.random.seed(26)
    np.random.shuffle(X)
    np.random.seed(26)
    np.random.shuffle(T)
    np.random.seed(26)
    np.random.shuffle(Y)
    X_train, X_test = X[:int(datasize*0.8)], X[int(datasize*0.8):]
    T_train, T_test = T[:int(datasize*0.8)], T[int(datasize*0.8):]
    Y_train, Y_test = Y[:int(datasize*0.8)], Y[int(datasize*0.8):]
    batch_size=4
    X_batched,T_batched,Y_batched=batching(X_train,T_train,Y_train,batch_size)



    network=SkipGramNegativeSampling(vocab_size,embedding_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    num_epochs=30


    loss_train=np.zeros(num_epochs)
    loss_test=np.zeros(num_epochs)
    for i in range(int(num_epochs)):
        for j in range(0,len(X_batched)):
            inputX = torch.FloatTensor(np.zeros([len(X_batched[j]),vocab_size]))
            inputT = torch.FloatTensor(np.zeros([len(X_batched[j]),vocab_size]))
            for k in range(len(X_batched[j])):
                inputX[k,X_batched[j][k]]=1
                inputT[k,T_batched[j][k]]=1
            inputX.requires_grad = True
            inputT.requires_grad = True
            target = torch.FloatTensor(Y_batched[j])


            prediction,e = (network(inputX,inputT))
            optimizer.zero_grad()  # zero the gradient buffers
            loss = criterion(prediction, target)

            loss_train[i]+=float(loss)
            loss.backward()
            optimizer.step()  # Does the update

            if j%5500==0:
                print("target is", target.detach().numpy())
                print("E%d, Loss=%f, prediction=" % (i, float(loss)), prediction)


        for j in range(0,len(X_test)):
            inputX = torch.FloatTensor(np.zeros([1,vocab_size]))
            inputT = torch.FloatTensor(np.zeros([1,vocab_size]))
            inputX[0,X_test[j]]=1
            inputT[0,T_test[j]]=1
            inputX.requires_grad = False
            inputT.requires_grad = False
            target = torch.FloatTensor([Y_test[j]])

            prediction,e = network(inputX,inputT)
            loss = criterion(prediction, target)
            loss_test[i]+=float(loss)

    import matplotlib.pyplot as plt
    fig, splot = plt.subplots(1)
    fig.suptitle('Training and Test Loss Overtime')

    domain = np.arange(num_epochs)
    splot.plot(domain, loss_train, 'g',label="train")
    splot.plot(domain, loss_test, 'r',label="test")
    splot.legend()

    return network

net=train_sgns(open("LargerCorpus.txt", "r", encoding='utf-8').read(), 5, 8 )