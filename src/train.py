import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

def load_file(filepath):

    with open(filepath, 'r') as file:
        content = json.load(file)
    return content

def get_content():

    content = load_file('../json/intents.json')
    tags = []
    all_words = []
    tag_words = []

    for intent in content["intents"]: #find all tags
        tag = intent["tag"]
        tags.append(tag)

        for pattern in intent['patterns']: #find all words (input)
            words = tokenize(pattern)
            all_words.extend(words)
            tag_words.append((words, tag))
    return tags, all_words, tag_words

tags, all_words, tag_words = get_content()

ignore_special_c = ['?', ',', '!', ':', '.'] #word we don't want to treat
new_all_words = [stem(word) for word in all_words if word not in ignore_special_c] #group words
new_all_words = sorted(set(new_all_words)) #sort all words and remove dupplicates words that permise to have all the possible word in a sentence and compare to a specific pattern
tags = sorted(set(tags))

x_train = []
y_train = []
for (pattern_sentence, tag) in tag_words: #we take in tag words a sentence in x and a tag in y

    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag) # contains the probabilities

    label = tags.index(tag) # contain the index pos
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#hyper_parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset(x_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch +1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
