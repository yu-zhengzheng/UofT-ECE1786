import torch
class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(vocab_size, 5)
        self.fc2 = torch.nn.Linear(5, embedding_size)
        self.fc3 = torch.nn.Linear(embedding_size, 5)
        self.fc4 = torch.nn.Linear(5, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        torch.manual_seed(26)
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(0, 0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """
        squash=torch.nn.LeakyReLU(0.2)
        softmax = torch.nn.Softmax(dim=1)
        #squash = torch.nn.Tanh()
        logits= squash(self.fc1(x))
        e = squash(self.fc2(logits))
        logits = squash(self.fc3(e))
        logits = softmax(self.fc4(logits))

        return logits, e