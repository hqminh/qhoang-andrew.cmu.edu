import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import pandas
import pickle
from utility import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device


class Generator(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim=[7, 3, 20]):
        super(Generator, self).__init__()
        assert len(h_dim) >= 2, 'Please make sure there are at least two hidden layers'
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.i_dim = i_dim
        self.maps = [nn.Linear(self.h_dim[i], self.h_dim[i + 1]).cuda() for i in range(len(self.h_dim) - 1)]
        self.f_in = nn.Linear(i_dim, self.h_dim[0]).cuda()
        self.f_out = [nn.Linear(self.h_dim[len(self.h_dim) - 1], self.o_dim[i]).cuda() for i in range(len(self.o_dim))]

    def forward(self, x):
        x = F.relu(self.f_in(x))
        for i in range(len(self.h_dim) - 1):
            x = F.relu(self.maps[i](x))
        output = []
        for i in range(len(self.o_dim) - 1):
            output.append(F.softmax(self.f_out[i](x)))
        output.append(F.sigmoid(self.f_out[len(self.o_dim) - 1](x)))
        output = torch.cat(output, dim=1)
        return output


class Discriminator(nn.Module):
    def __init__(self, i_dim, h_dim):
        super(Discriminator, self).__init__()
        self.h_dim = h_dim
        self.i_dim = i_dim
        self.maps = [nn.Linear(self.h_dim[i], self.h_dim[i + 1]).cuda() for i in range(len(self.h_dim) - 1)]
        self.f_in = nn.Linear(i_dim, self.h_dim[0]).cuda()
        self.f_out = nn.Linear(self.h_dim[len(self.h_dim) - 1], 1).cuda()

    def forward(self, x):
        x = F.relu(self.f_in(x))
        for i in range(len(self.h_dim) - 1):
            x = F.relu(self.maps[i](x))
        return F.sigmoid(self.f_out(x))  # probability to be positive


class GAN(nn.Module):
    def __init__(self, data, n_sample=100, epoch=100, gan_epoch=100, load_pickle=True):
        super(GAN, self).__init__()
        self.loss_func = torch.nn.BCELoss().to(device)
        self.gan_epoch = gan_epoch
        self.gen = Generator(10, [5, 5]).to(device)
        self.dis = Discriminator(30, [5, 5]).to(device) # 7 + 3 + 20
        self.func = Flow()
        if load_pickle:
            self.data = pickle.load(open('encoded_data.p', 'rb'))
        else:
            self.data = self.func.encode(data)
            pickle.dump(self.data, open('encoded_data.p', 'wb'))
        self.n_sample = n_sample
        self.epoch = epoch

    def generate(self):
        eps = torch.empty(self.n_sample, self.gen.i_dim).normal_(0, 1).to(device)
        fakes = self.gen(eps)
        pos = np.random.choice(self.data.shape[0], self.n_sample)
        auths = self.data[pos, :].to(device)
        return fakes, auths

    def dis_train(self):

        fakes, auths = self.generate()  # n_sample by x_dim
        X_train = torch.cat([fakes, auths], dim=0)
        Y_train = torch.zeros((2 * self.n_sample, 1)).to(device)
        Y_train[self.n_sample:2 * self.n_sample, 0] = 1
        print("Updating Discriminator ...")
        loss_record = []
        optimizer = opt.Adam(self.dis.parameters())
        for i in range(self.epoch):
            self.dis.train()
            optimizer.zero_grad()
            Y_pred = self.dis(X_train)
            loss = self.loss_func(Y_pred, Y_train)
            loss_record.append(loss.item())
            print('Dis Training Iter ' + str(i) + ': ', loss_record[len(loss_record) - 1])
            loss.backward(retain_graph=True)
            optimizer.step()

        return loss_record

    def gen_train(self):
        print("Updating Generator ...")
        loss_record = []
        optimizer = opt.Adam(self.gen.parameters())
        eps = torch.empty(self.n_sample, self.gen.i_dim).normal_(0, 1).to(device)
        for i in range(self.epoch):
            self.gen.train()
            optimizer.zero_grad()
            X_train = self.gen(eps)  # n_sample by x_dim
            Y_train = torch.zeros((X_train.shape[0], 1)).to(device)
            Y_pred = self.dis(X_train).to(device)
            loss = -self.loss_func(Y_pred, Y_train)
            loss_record.append(loss.item())
            print('Gen Training Iter ' + str(i) + ': ', loss_record[len(loss_record) - 1])
            loss.backward(retain_graph=True)
            optimizer.step()
        return loss_record

    def GAN_train(self):
        dis_all = []
        gen_all = []
        for i in range(self.gan_epoch):
            print('GAN Train Iter ' + str(i) + ': ')
            dis_loss = self.dis_train()
            gen_loss = self.gen_train()
            dis_all.append(dis_loss)
            gen_all.append(gen_loss)
        return dis_all, gen_all


if __name__ == '__main__':
    # load data
    #data = Data('CIDDS-001-internal-week4.csv', load_pickle=False)
    #data.save_to_pickle('data.p')
    data = Data('data.p')
    model = GAN(data.data).to(device)
    model.GAN_train()














