import torch
import torchtext

glove = torchtext.vocab.GloVe(name="6B",dim=50)

def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

def print_closest_cosine_words(vec, n=5):
    """prints n closest words to the input vector using cosine similarities
    :param vec input word vector
    :param n the number of closest words wanted
    """
    dists = torch.cosine_similarity(glove.vectors, vec, dim=1)
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1],reverse=True) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.3f" % difference)


#%%
#output block
print_closest_cosine_words(glove["dog"], n=10)
print("\ncosine\n---------------------------------\neuclidean\n")
print_closest_words(glove["dog"], n=10)

print("\n")

print_closest_cosine_words(glove["computer"], n=10)
print("\ncosine\n---------------------------------\neuclidean\n")
print_closest_words(glove["computer"], n=10)

