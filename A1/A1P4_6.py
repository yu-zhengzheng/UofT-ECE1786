class SkipGramNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.enX1 = torch.nn.Linear(vocab_size, embedding_size*3) #encoding X
        self.enT1 = torch.nn.Linear(vocab_size, embedding_size*3) #encoding T
        self.enX2 = torch.nn.Linear(embedding_size*3,embedding_size) #encoding X
        self.enT2 = torch.nn.Linear(embedding_size*3,embedding_size) #encoding T
        self.apply(self._init_weights)

    def _init_weights(self, module):
        torch.manual_seed(26)
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(0, 0.1)
            #print(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, X,T):
        # x: torch.tensor of shape (batch_size), context word

        squash=torch.nn.LeakyReLU(0.2)
        binary = torch.nn.Tanh()
        e= squash(self.enX1(X))
        e= squash(self.enX2(e))
        eT= squash(self.enT1(T))
        eT= squash(self.enT2(eT))
        prediction = binary(torch.inner(e,eT)).diagonal()
        return prediction,e