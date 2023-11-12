def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    :return
        X: word
        T: target word
        Y: truth
    """
    X, T, Y = [], [], []

    # Tokenize the input
    length=len(textlist)
    vocab_size=len(w2i)
    tokens=np.zeros(length)

    for i in range(len(textlist)):
        tokens[i]=w2i[textlist[i]]
    # TO DO

    # Loop through each token
    # TO DO
    for i in range(length):
        #print(i)
        for j in range(-int((window-1)/2),int((window+1)/2),1):
            if -1 < i+j < length and j!=0:
                X.append(int(tokens[i]))
                T.append(int(tokens[i+j]))
                Y.append(1)
                X.append(int(tokens[i]))
                T.append(int(np.random.rand()*vocab_size))
                Y.append(-1)


    return X, T, Y