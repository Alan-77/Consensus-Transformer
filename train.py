import os
import time
import math
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from torchsummaryX import summary


# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# set_seed()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_copies(gold, sub_p, del_p, ins_p):
    res = []
    for w in gold:
        r = np.random.random()
        if r < sub_p:
            res.append(np.random.choice(['A','G','C','T']))
        elif r < sub_p + ins_p:
            res.append(np.random.choice(['A','G','C','T']))
            res.append(w)
        elif r > sub_p + ins_p + del_p:
            res.append(w)
    return ''.join(res)


def generate_strands(strand_num, strand_length, error_rate):
    strands = []
    for i in range(strand_num):
        d = np.random.choice(['A','G','C','T'], size=(strand_length))
        s = ''.join(d)
        strands.append(s)
    print('check strands length', len(strands))

    N = 10
    subs = dels = inss = error_rate
    copies = []
    for strand in strands:
        cluster = []
        for x in range(N):
            cluster.append(generate_copies(strand, subs, dels, inss))
        copies.append(cluster)
    print('check copies length', len(copies))
    return copies, strands


def convert_strands(strand):
    out = []
    for char in strand:
        if char == 'A':
            out.append(1)
        if char == 'G':
            out.append(2)
        if char == 'C':
            out.append(3)
        if char == 'T':
            out.append(4)
    return out


def generate_data(num, strand_length, error_rate):
    print('start to generate data', num, strand_length, error_rate)
    x, y = generate_strands(num, strand_length, error_rate)
    cut = int(len(x) * 0.1)
    train_x, train_y = x[:cut*9], y[:cut*9]
    valid_x, valid_y = x[cut*9:], y[cut*9:]
    
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = np.array(valid_x), np.array(valid_y)
    
    raw = {'train_x': train_x, 'train_y': train_y, 'valid_x': valid_x, 'valid_y': valid_y}
    data = {}

    # processing labels
    for split in ['train_y', 'valid_y']:
        strands = []
        for s in raw[split]:
            s = convert_strands(s)
            strands.append(s)
        data[split] = np.array(strands)
        print(split, 'shape', data[split].shape)

    # processing inputs
    for split in ['train_x', 'valid_x']:
        strands = []
        lengths = []
        for c in raw[split]:
            cluster = []
            for s in c:
                s = convert_strands(s)
                lengths.append(len(s))

                if len(s) > strand_length:
                    while len(s) > strand_length:
                        idx = np.random.randint(len(s))
                        del s[idx]
                elif len(s) < strand_length:
                    while len(s) < strand_length:
                        idx = np.random.randint(len(s))
                        r = np.random.choice([1,2,3,4])
                        s.insert(idx, r)

                cluster.append(s)
            strands.append(cluster)
        data[split] = np.array(strands)        
        print(split, 'shape', data[split].shape)

        # check strands length
        l = np.array(lengths)
        print('lengths shape', l.shape)
        print('lengths stat', l.mean(), l.min(), l.max())

    scaler = StandardScaler(mean=data['train_x'].mean(), std=data['train_x'].std())

    for split in ['train_x', 'valid_x']:
        data[split] = scaler.transform(data[split])
        data[split] = torch.from_numpy(data[split].astype(np.float32))

    for split in ['train_y', 'valid_y']:
        data[split] = torch.from_numpy(data[split].astype('int64'))

    train_dataset = TensorDataset(data['train_x'], data['train_y'])
    valid_dataset = TensorDataset(data['valid_x'], data['valid_y'])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    print('generate done')
    return train_loader, valid_loader


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConsensusTransformer(torch.nn.Module):
        def __init__(self, strand_length):
            super().__init__()
            d_model = 128
            d_hid =  512
            nhead = 8
            dropout = 0.1
            nlayers = 6
            
            self.embed = torch.nn.Linear(10, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout, strand_length)
            
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, nlayers)

            self.linear = torch.nn.Sequential(
                torch.nn.Linear(d_model, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 4),
            )
            self.softmax = torch.nn.Softmax(dim=1)
            
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.embed(x)
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            x = self.linear(x)
            x = x.transpose(0, 1)
            x = x.reshape(x.size(0)*x.size(1), x.size(2))
            x = self.softmax(x)
            return x


def get_acc(predictions, labels):
    length = labels.size(1)
    predictions = predictions.argmax(dim=2)
    matches = torch.sum((predictions == labels), dim=1)
    entry_acc = matches.sum()
    strand_acc = (matches == length).sum()
    return strand_acc, entry_acc


# configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

train_num = 5000
strand_length = 120
error_rate = 0.01

model_dir = 'save_error' + str(error_rate) + '/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train_loader, valid_loader = generate_data(train_num, strand_length, error_rate)

model = ConsensusTransformer(strand_length)
# checkpoint = ''
# model.load_state_dict(torch.load(checkpoint)['model_state'])
# summary(model, torch.zeros((64, 10, strand_length)))
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
steps = [300, 400, 450, 500]
print('lr steps', steps)
scheduler = MultiStepLR(optimizer, milestones=steps, gamma=np.sqrt(0.1), verbose=True)
epochs = 550
max_acc = 0
clip = 5

# training loop
for i in range(epochs):
    t1 = int(time.time())
    total = 0
    train_loss = 0
    train_strand_acc = 0
    train_entry_acc = 0
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y -= 1

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total += batch_y.size(0)
        train_loss += (loss.item() * batch_y.size(0))
        strand_acc, entry_acc = get_acc(outputs.view(batch_y.size(0), batch_y.size(1), -1), batch_y)
        train_strand_acc += strand_acc
        train_entry_acc += entry_acc
    mtrain_loss = train_loss / total
    mtrain_strand_acc = train_strand_acc / total
    mtrain_entry_acc = train_entry_acc / total
    t2 = int(time.time())
    
    total = 0
    valid_loss = 0
    valid_strand_acc = 0
    valid_entry_acc = 0
    model.eval()
    for batch_x, batch_y in valid_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y -= 1
        
        with torch.no_grad():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.view(-1))
            
            total += batch_y.size(0)
            valid_loss += (loss.item() * batch_y.size(0))
            strand_acc, entry_acc = get_acc(outputs.view(batch_y.size(0), batch_y.size(1), -1), batch_y)
            valid_strand_acc += strand_acc
            valid_entry_acc += entry_acc
    mvalid_loss = valid_loss / total
    mvalid_strand_acc = valid_strand_acc / total
    mvalid_entry_acc = valid_entry_acc / total
    
    if max_acc < mvalid_strand_acc:
        states = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
        torch.save(states, model_dir + 'epoch_' + str(i) + '_' + str(round(mvalid_strand_acc.item(), 2)) + '.pth')
        max_acc = mvalid_strand_acc

    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train Strand Acc: {:.4f}, Train Entry Acc: {:.4f}, Valid Loss: {:.4f}, Valid Strand Acc: {:.4f}, Valid Entry Acc: {:.4f}, Training Time: {:.0f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_strand_acc, mtrain_entry_acc, mvalid_loss, mvalid_strand_acc, mvalid_entry_acc, (t2 - t1), flush=True))
    
    scheduler.step()
    
    if (i + 1) % 50 == 0 and (i + 1) != epochs:
        train_loader, valid_loader = generate_data(train_num, strand_length, error_rate)
