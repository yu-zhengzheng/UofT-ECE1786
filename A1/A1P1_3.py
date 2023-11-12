import torch
import torchtext

glove = torchtext.vocab.GloVe(name="6B",dim=50)

def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[0:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)
    #print("\n")

def print_analogous_word(word):
    print_closest_words((glove['mexican'] - glove['mexico']+glove['austrian'] - glove['austria'])/2 + glove[word],0)



print_analogous_word('finland')
print_analogous_word('denmark')
print_analogous_word('iceland')

print_analogous_word('niger')
print_analogous_word('georgia')

print_analogous_word('singapore')
print_analogous_word('ecuador')
print_analogous_word('peru')

print_analogous_word('philippines')
print_analogous_word('south-africa')
#%%
