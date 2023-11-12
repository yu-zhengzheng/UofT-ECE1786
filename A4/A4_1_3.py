from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from treelib import Node, Tree


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "It is important for all countries to try harder to reduce carbon emissions because"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
torch.manual_seed(11)
outputs = model.generate(input_ids, do_sample=False, num_beams=3, max_length=30, temperature =1,top_p=0.99,return_dict_in_generate=True, output_scores=True,repetition_penalty=1.0)
sequences = outputs.sequences # sequences is a tensor of shape (batch_size, sequence_length)
scores = outputs.scores
print("sequences=",type(sequences),len(sequences[0]))
print(tokenizer.batch_decode(sequences, skip_special_tokens=True))


tree = Tree()
node=0
tree.create_node(str(tokenizer.batch_decode(sequences[0:1,13], skip_special_tokens=True)), node)  # root node
parent=node

i=0
for score in scores:
    token_num=sequences[0,i+14]


    # get the top 3 probable words
    tok=torch.argmax(score[0])
    prob=np.round(100*np.exp(max(score[0]).numpy()))
    string=str([tokenizer.decode(tok)])+" "+str(prob)+"%"
    score[0][torch.argmax(score[0])]-=100
    node+=1
    if tok==token_num:
        parent_next=node
    tree.create_node(string, node ,parent=parent)


    tok=torch.argmax(score[0])
    prob=np.round(100*np.exp(max(score[0]).numpy()))
    string=str([tokenizer.decode(tok)])+" "+str(prob)+"%"
    score[0][torch.argmax(score[0])]-=100
    node+=1
    if tok==token_num:
        parent_next=node
    tree.create_node(string, node ,parent=parent)

    tok=torch.argmax(score[0])
    prob=np.round(100*np.exp(max(score[0]).numpy()))
    string=str([tokenizer.decode(tok)])+" "+str(prob)+"%"
    score[0][torch.argmax(score[0])]-=100
    node+=1
    if tok==token_num:
        parent_next=node
    tree.create_node(string, node ,parent=parent)



    # safe guards for the generated word not in top 3
    if parent==parent_next:
        node+=1
        parent_next=node
        tree.create_node(str(tokenizer.batch_decode(sequences[0:1,i+14])), node ,parent=parent)

    parent=parent_next
    i+=1

tree.show()
#%%