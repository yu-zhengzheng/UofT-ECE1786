import torch
import torchtext

glove = torchtext.vocab.GloVe(name="6B",dim=50)

def compare_words_to_category(category,word):
    avg=0
    for i in category:
        # print(i,torch.cosine_similarity(glove[i].unsqueeze(0),glove[word].unsqueeze(0),dim=1))
        avg+=torch.cosine_similarity(glove[i].unsqueeze(0),glove[word].unsqueeze(0),dim=1).item()
    avg/=len(category)
    print("avg of cos similarity =","\t%5.3f" % avg)

    avg=0
    for i in category:
        avg+=glove[i]
    avg/=len(category)
    print("cos similarity with avg =","\t%5.3f" % torch.cosine_similarity(avg.unsqueeze(0),glove[word].unsqueeze(0),dim=1).item())

#%%
category=["peach","banana","watermelon","grape"]
compare_words_to_category(category,"monkey")