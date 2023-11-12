def batching(X,Y,batch_size):
    batches=int((len(X)+batch_size-1)/batch_size)
    X_batched,Y_batched=[],[]
    for i in range(batches-1):
        X_batched.append(X[i*batch_size:(i+1)*batch_size])
        Y_batched.append(Y[i*batch_size:(i+1)*batch_size])
    X_batched.append(X[(batches-1)*batch_size:])
    Y_batched.append(Y[(batches-1)*batch_size:])
    return X_batched,Y_batched

def train_word2vec(textlist, window, embedding_size ):
    # Set up a model with Skip-gram (predict context with word)
    # textlist: a list of the strings
    lemmas, v2i, i2v=prepare_texts(textlist)
    print(v2i)
    vocab_size=len(v2i)
    network=Word2vecModel(vocab_size,embedding_size)


    # Create the training data
    X,Y=tokenize_and_preprocess_text(lemmas, v2i, window)
    # Split the training data
    datasize=len(X)
    np.random.seed(26)
    np.random.shuffle(X)
    np.random.seed(26)
    np.random.shuffle(Y)
    X_train, X_test = X[:int(datasize*0.8)], X[int(datasize*0.8):]
    Y_train, Y_test = Y[:int(datasize*0.8)], Y[int(datasize*0.8):]



    X_batched,Y_batched=batching(X_train,Y_train,4)


    # instantiate the network & set up the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=7e-3)
    num_epochs=50
    batch_size=4

    # training loop
    loss_train=np.zeros(num_epochs)
    loss_test=np.zeros(num_epochs)
    for i in range(int(num_epochs)):
        for j in range(0,len(X_batched)):
            input = torch.FloatTensor(np.zeros([len(X_batched[j]),vocab_size]))
            for k in range(len(X_batched[j])):
                input[k,X_batched[j][k]]=1
            input.requires_grad = True
            target = torch.FloatTensor(np.zeros([len(Y_batched[j]),vocab_size]))
            for k in range(len(Y_batched[j])):
                target[k,Y_batched[j][k]]=1



            prediction,e = (network(input))
            optimizer.zero_grad()  # zero the gradient buffers
            loss = criterion(prediction, target)
            loss_train[i]+=float(loss)
            loss.backward()
            optimizer.step()  # Does the update


        for j in range(0,len(X_test)):
            input = torch.FloatTensor(np.zeros([1,vocab_size]))
            input[0,X_test[j]]=1
            input.requires_grad = False
            target = torch.FloatTensor(np.zeros([1,vocab_size]))
            target[0,Y_test[j]]=1

            prediction,e = (network(input))
            loss = criterion(prediction, target)
            loss_test[i]+=float(loss)



    import matplotlib.pyplot as plt
    fig, splot = plt.subplots(1)
    fig.suptitle('Training and Test Loss Overtime')

    domain = np.arange(num_epochs)
    #for i in range(num_epochs-1):
    #loss_train[i+1] = loss_train[i+1] * 0.01 + loss_train[i] * 0.99
    splot.plot(domain, loss_train, 'g',label="train")
    splot.plot(domain, loss_test, 'r',label="test")
    splot.legend()

    return network

net=train_word2vec(open("SmallSimpleCorpus.txt", "r").read(), 5, 2 )