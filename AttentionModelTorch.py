import os, sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
#import pandas as pd


print("Done!")

# Loading the data

# Read the Input File
max_data_size = 1527294

#max_data_size = 10000

def randomized_index(data, label):
    rand_index = np.random.permutation(len(data))
    
    data = np.array([data[i] for i in rand_index])
    label = np.array([label[i] for i in rand_index])
    
    return (data, label)

def check_neg_pos(label):
    pos, neg = 0, 0
    for l in label:
        if l == 0:
            neg += 1
        else:
            pos += 1
    print("neg: {}, pos {}".format(neg, pos))


def read_input_file(file_path, label=-1):
    x_data = []
    y_data = []

    file_read = open(file_path, "r")
    for line in file_read:
        data = [int(i) for i in line.strip()]
        x_data.append(data)
        y_data.append(label)
        #print(x_data[-1], y_data[-1])
        if len(x_data) == max_data_size:
            break
    file_read.close()
    print("Sequences Read: ", len(x_data))
    return np.array(x_data), np.array(y_data)


def load_data(data_size=1000, batch_size = 100):
    #root_dir = "/mnt/scratch7/hirak/"
    root_dir = "/gpfs/scratch/hsarkar/attention_mechanism"

    x_data_pos, y_data_pos = read_input_file(os.path.join(root_dir, "Attention4DNASeqRepresentation/dataset/gene_range_start_codon.txt"), 1)

    original_neg_intergenic_data, original_neg_intergenic_label = read_input_file(os.path.join(root_dir, "Attention4DNASeqRepresentation/dataset/intragenic_start_codon.txt"), 0)
    original_neg_coding_data, original_neg_coding_label = read_input_file(os.path.join(root_dir, "Attention4DNASeqRepresentation/dataset/coding_start_codon.txt"), 0)

    x_data_neg = np.concatenate((original_neg_coding_data, original_neg_intergenic_data))
    y_data_neg = np.concatenate((original_neg_coding_label, original_neg_intergenic_label))

    x_data_neg, y_data_neg = randomized_index(x_data_neg, y_data_neg)

    x_data_pos = x_data_pos[:data_size]
    y_data_pos = y_data_pos[:data_size]

    x_data_neg = x_data_neg[:data_size]
    y_data_neg = y_data_neg[:data_size]

    np.random.shuffle(x_data_neg)
    np.random.shuffle(x_data_pos)

    train_index = int((len(x_data_pos) / batch_size) * 0.60 * batch_size)
    # eval_index = train_index + int((len(x_data_pos) / batch_size) * 0.20 * batch_size)
    test_index = train_index + int((len(x_data_pos) / batch_size) * 0.40 * batch_size)

    print("train, test = ", (train_index, test_index))

    #Process Negative Data

    x_train = x_data_neg[0:train_index]
    y_train = y_data_neg[0:train_index]

    x_test = x_data_neg[train_index:test_index]
    y_test = y_data_neg[train_index:test_index]

    # x_test = x_data_neg[eval_index:test_index]
    # y_test = y_data_neg[eval_index:test_index]

    #Process Positive Data

    x_train = np.append(x_train, x_data_pos[0:train_index], axis=0)
    y_train = np.append(y_train, y_data_pos[0:train_index], axis=0)

    x_test = np.append(x_test, x_data_pos[train_index:test_index], axis=0)
    y_test = np.append(y_test, y_data_pos[train_index:test_index], axis=0)

    # x_test = np.append(x_test, x_data_pos[eval_index:test_index], axis=0)
    # y_test = np.append(y_test, y_data_pos[eval_index:test_index], axis=0)

    print("Sanity Check: ", np.sum(y_train), np.sum(y_test)) 

    return (x_train, y_train, x_test, y_test)
    

class AttnDecoderRNN(nn.Module):#corrected batch faster
    #(self, time_steps, embedding_dim, hidden_dim, vocab_size, tagset_size, mini_batch)
    def __init__(self, vocab_size, embedding_dim, \
                 hidden_dim, device, batch_size=100, debug=1, \
                 tagset_size=1, time_steps=101):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.minibatch_size = batch_size
        self.dropout_p = 0.25
        self.tagset_size = tagset_size
        self.hidden = self.init_hidden()
        self.hidden_bi = self.init_hidden(bidirectional=True)
        self.debug = debug
        self.device = device 

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_one = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout_one = nn.Dropout(0.4)
        self.lstm_two = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout_two = nn.Dropout(0.4)
        self.lstm_three = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout_three = nn.Dropout(0.4)

        self.attn_array = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(time_steps)])
        
        self.lstm_four = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional = True)
        self.dropout_seven = nn.Dropout(0.4)
        
        self.lstm_five = nn.LSTM(hidden_dim*2, hidden_dim*2, batch_first=True, bidirectional = True)
        self.dropout_eight = nn.Dropout(0.4)
        

        """
        self.attn_combine = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)
        #embedding_dim*time_steps
        """

        self.hidden2tag_one = nn.Linear(hidden_dim*2*time_steps, 512)
        self.dropout_four = nn.Dropout(0.25)
        self.hidden2tag_two = nn.Linear(512, 128)
        self.dropout_five = nn.Dropout(0.25)
        self.hidden2tag_three = nn.Linear(128, 64)
        self.dropout_six = nn.Dropout(0.25)

        self.output = nn.Linear(64, tagset_size)

    def forward(self, input):
        init_embed = self.embedding(input)
        #embedded = init_embed.permute(1, 0, 2)
        if self.debug == 1:
            print("Embedding Shape: ", init_embed.shape)

        lstm_out, self.hidden_one = self.lstm_one(init_embed, self.hidden)
        lstm_out = self.dropout_one(lstm_out)
        lstm_out, self.hidden_two = self.lstm_two(lstm_out, self.hidden)
        lstm_out = self.dropout_two(lstm_out)
        lstm_out, self.hidden_three = self.lstm_three(lstm_out, self.hidden)
        lstm_out = self.dropout_three(lstm_out)
        #"""
        lstm_permute = lstm_out.permute(1, 0, 2)
        if self.debug == 1:
            print("LSTM Out Shape: ", lstm_permute.shape)

        attention = [self.attn_array[i](lstm_permute[i][:]) for i in range(self.time_steps)]
        attention = torch.stack(attention)
        attention.to(device)
        
        attention = attention.permute(1, 0, 2)
        if self.debug == 1:
            print("Attention Shape: ", attention.shape)


        
        attn_weights = F.softmax(attention, dim=2)
        #attn_weights = attn_weights.view(self.minibatch_size, self.time_steps, 1)
        if self.debug == 1:
            print("Softmax Shape: ", attn_weights.shape)
        """
        attn_weights = torch.stack(
            [attn_weights]*self.embedding_dim, 2).view(
            self.minibatch_size, self.time_steps, -1)
        if self.debug == 1:
            print("Softmax ReShape: ", attn_weights.shape)
        """
        #attn_applied = init_embed
        attn_applied = attn_weights * init_embed
        
        # potential lstm 
        if self.debug == 1:
            print("Attention Applied Shape: ", attn_applied.shape)

        lstm_out, self.hidden_one = self.lstm_four(attn_applied, self.hidden_bi)
        lstm_out = self.dropout_seven(lstm_out)
        if self.debug == 1:
            print("LSTM out: ", lstm_out.shape)
        # lstm_out, self.hidden_two = self.lstm_five(lstm_out, self.hidden_bi)
        # lstm_out = self.dropout_eight(lstm_out)    
        
        #attn_applied = attn_applied.view(self.minibatch_size, self.time_steps, -1)
        #attn_applied = torch.sum(attn_applied, dim=1)
        if self.debug == 1:
            print("Embedding*Attention Shape: ", attn_applied.shape)

        #output = F.relu(attn_applied)
        #"""

        # lstm_out = attn_applied.contiguous().view(self.minibatch_size, -1)
        lstm_out = lstm_out.contiguous().view(self.minibatch_size, -1)
        if self.debug == 1:
            print("LSTM Output Shape: ", lstm_out.shape)


        dense_out = self.hidden2tag_one(lstm_out[:])
        dense_out = F.relu(dense_out[:])
        dense_out = self.dropout_four(dense_out[:])

        dense_out = self.hidden2tag_two(dense_out[:])
        dense_out = F.relu(dense_out[:])
        dense_out = self.dropout_five(dense_out[:])

        dense_out = self.hidden2tag_three(dense_out[:])
        dense_out = F.relu(dense_out[:])
        dense_out = self.dropout_six(dense_out[:])

        tag_space = self.output(dense_out[:])
        #print(tag_space.shape)
        #tag_scores = F.sigmoid(tag_space)
        #tag_scores = F.softmax(tag_space, dim=1)
        #print(tag_scores.shape)
        return tag_space, attn_applied

    def init_hidden(self, bidirectional = False):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if(bidirectional):
            return (torch.zeros(2, self.minibatch_size, self.hidden_dim, device = device),
                    torch.zeros(2, self.minibatch_size, self.hidden_dim, device = device))
        else:
            return (torch.zeros(1, self.minibatch_size, self.hidden_dim, device = device),
                    torch.zeros(1, self.minibatch_size, self.hidden_dim, device = device))


def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        output = torch.clamp(output,min=1e-8,max=1-1e-8) 
        #loss =  pos_weight * (target * torch.log(output)) + \
        #        neg_weight * ((1 - target) * torch.log(1 - output))
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

import torch.nn.functional as F

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    
    return acc


losses = []
accuracies = []
accuracy_test = []
#batch_size = 10

data_size = 10000
batch_size = 1000
x_train, y_train, x_test, y_test = load_data(data_size, batch_size)

#x_train, y_train, x_eval, y_eval, x_test, y_test = load_data(data_size, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = AttnDecoderRNN(5, 16, 16, device, batch_size=batch_size, debug=0)
#model = model.cuda()
model.to(device)

X = torch.from_numpy(np.array(x_train).astype(int))
Y = torch.from_numpy(np.array(y_train).reshape(len(y_train),1).astype(np.int))

X_test = torch.from_numpy(np.array(x_test).astype(int))
Y_test = torch.from_numpy(np.array(y_test).reshape(len(y_test),1).astype(np.int))

X, Y = X.to(device), Y.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)
#print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print("Cuda should be used from here !!")
num_epochs = 200
with open("progress.txt", "w") as fp:
    for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        total_loss = 0
        total_acc = 0

        for index in range(0, len(X), batch_size):
            model.train()
            sentence = X[index : index+batch_size]#.reshape(len(X[0]))
            tags = Y[index : index+batch_size]#.reshape(len(Y[0]))
            sentence.to(device)
            tags.to(device)
            #print(sentence.shape, tags.shape)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            #targets = prepare_sequence(tags, tag_to_ix)
            targets = tags.float().flatten()

            # Step 3. Run our forward pass.
            tag_scores, attn_weight = model(sentence)
            tag_scores = tag_scores.flatten()
            #print(targets.shape, tag_scores.shape)

            #neg_weight = batch_size / (batch_size-np.sum(data_label[index : index+batch_size]))
            #pos_weight = batch_size / np.sum(data_label[index : index+batch_size])
            #weights = torch.FloatTensor([neg_weight, pos_weight])

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(tag_scores, targets)
            #loss = weighted_binary_cross_entropy(tag_scores, targets, weights=weights)
            total_loss += loss.item()

            acc = binary_accuracy(tag_scores, targets)
            total_acc += acc

            loss.backward()
            optimizer.step()

            #print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], correct/x.shape[0]))


        # run forward on this epoch
        tot_test_acc = 0
        for index in range(0, len(X_test), batch_size):
            model.eval()
            sentence = X_test[index : index+batch_size]#.reshape(len(X[0]))
            tags = Y_test[index : index+batch_size]#.reshape(len(Y[0]))
            sentence.to(device)
            tags.to(device)
            #model.hidden = model.init_hidden()
            tag_scores_test, attn_weight_test = model(sentence)
            tot_test_acc += binary_accuracy(tag_scores_test.flatten(), tags.float().flatten())

        accuracy_test.append(tot_test_acc/(len(X_test)/batch_size))
        losses.append(total_loss)
        accuracies.append(total_acc/(len(X)/batch_size))

        #total_loss.backward()
        #opt.step()

        #print(epoch, total_loss)#, total_acc)
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Test Accuracy {:.3f}".format(epoch+1,num_epochs, losses[-1], accuracies[-1], accuracy_test[-1]))
        fp.write("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Test Accuracy {:.3f}\n".format(epoch+1,num_epochs, losses[-1], accuracies[-1], accuracy_test[-1]))

        if(epoch < 10):
            with open("ten_ecpoch.txt", "a") as fp2:
                fp2.write("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Test Accuracy {:.3f}\n".format(epoch+1,num_epochs, losses[-1], accuracies[-1], accuracy_test[-1]))

        #fp.write("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}\n".format(epoch+1,num_epochs, losses[-1], accuracies[-1]))


## Save the model
torch.save(model.state_dict(), "models/attention_model_apr_7.pt")

with open("attention_loss.tsv", "w") as fp:
    for l in losses:
        fp.write("{}\n".format(l))

with open("attention_accuracy.tsv", "w") as fp:
    for a in accuracies:
        fp.write("{}\n".format(a)) 
