import numpy as np
import torch
from datetime import datetime
import pickle
import pandas

sec_per_day = 86400
no_port = 65535
protocol_tokens = ['TCP', 'UDP', 'ICMP']
#flag_tokens = ['isURG', 'isACK', 'isPSH', 'isRES', 'isSYN', 'isFIN']
numeric_field = ['byte', 'packet', 'dport', 'sport', 'duration']

class Data:
    def __init__(self, fname, load_pickle=True):
        if load_pickle:
            self.load_from_pickle(fname)
        else:
            self.data = dict()
            self.load_from_csv(fname)
            self.clean()

    def isIP(self, s):
        tokens = s.split('.')
        if len(tokens) != 4:
            return False

        for t in tokens:
            if not t.isdigit():
                return False

        return True

    def isNumeric(self, n):
        try:
            nn = float(n)
            return True
        except ValueError:
            return False

    def clean(self):
        valid_id = []
        for i in range(len(self.data['date'])):
            if not self.data['protocol'][i].strip() in protocol_tokens:
                print(self.data['protocol'][i])
                continue

            if not self.isNumeric(self.data['dport'][i]):
                print(self.data['dport'][i])
                continue

            if not self.isNumeric(self.data['sport'][i]):
                print(self.data['sport'][i])
                continue

            if not self.isNumeric(self.data['byte'][i]):
                print(self.data['byte'][i])
                continue

            if not self.isNumeric(self.data['packet'][i]):
                print(self.data['packet'][i])
                continue

            if not self.isNumeric(self.data['duration'][i]):
                print(self.data['duration'][i])
                continue

            if not self.isIP(self.data['sIP'][i]):
                print(self.data['sIP'][i])
                continue

            if not self.isIP(self.data['dIP'][i]):
                print(self.data['dIP'][i])
                continue

            valid_id.append(i)

        for field in self.data:
            temp = []
            for j in valid_id:
                if field in numeric_field:
                    temp.append(float(self.data[field][j]))
                else:
                    temp.append(self.data[field][j])
            self.data[field] = temp

    def save_to_pickle(self, fname):
        f = open(fname, 'wb')
        pickle.dump(self.data, f)

    def load_from_pickle(self, fname):
        f = open(fname, 'rb')
        self.data = pickle.load(f)

    def load_from_csv(self, fname):
        d = pandas.read_csv(fname)

        self.data['date'] = list(d['Date first seen'])
        self.data['duration'] = list(d['Duration'])
        self.data['protocol'] = list(d['Proto'])
        self.data['sIP'] = list(d['Src IP Addr'])
        self.data['sport'] = list(d['Src Pt'])
        self.data['dIP'] = list(d['Dst IP Addr'])
        self.data['dport'] = list(d['Dst Pt'])
        self.data['byte'] = list(d['Bytes'])
        self.data['packet'] = list(d['Packets'])
        self.data['flags'] = list(d['Flags'])
        self.clean()


class Flow:
    def __init__(self):
        pass

    def parse_date(self, raw_date):  # input in batch
        n_data = len(raw_date)
        output = torch.zeros(n_data, 8)
        for i in range(n_data):
            dt = datetime.strptime(raw_date[i][:-4], "%Y-%m-%d %H:%M:%S")
            output[i, int(dt.weekday())] = 1
            daytime = dt.hour * 3600 + dt.minute * 60 + dt.second
            output[i, 7] = 1.0 * daytime / sec_per_day
        return output.float()

    def parse_num(self, duration):  # input in batch
        output = torch.tensor(np.array(duration)).view(-1, 1)
        d_min = torch.min(output).item()
        d_max = torch.max(output).item()
        output = (output - d_min) * 1.0 / (d_max - d_min)
        return output.float()

    def parse_token(self, protocol, tokens):  # input in batch
        output = torch.zeros((len(protocol), len(tokens)))
        for i in range(len(protocol)):
            for pos, t in enumerate(tokens):
                if protocol[i].strip() == t:
                    output[i, pos] = 1
        return output.float()

    def parse_flags(self, flag):
        output = torch.zeros((len(flag), 6))
        for t in range(len(flag)):
            for j in range(6):
                if flag[t][j] != '.':
                    output[t, j] = 1
        return output.float()

    def parse_IP(self, IP):  # input in batch
        output = torch.zeros((len(IP), 4))
        for i in range(len(IP)):
            parts = IP[i].split('.')
            for pos, p in enumerate(parts):
                output[i, pos] = float(p) / 255
        return output.float()

    def parse_port(self, port):  # input in batch
        output = torch.tensor(np.array(port)).view(-1, 1)
        output = output / no_port
        return output.float()

    def encode(self, data):  # data is a list of N structs -- each is a data point
        date = self.parse_date(data['date'])
        duration = self.parse_num(data['duration'])
        protocol = self.parse_token(data['protocol'], protocol_tokens)
        sIP = self.parse_IP(data['sIP'])
        sport = self.parse_port(data['sport'])
        dIP = self.parse_IP(data['dIP'])
        dport = self.parse_port(data['dport'])
        byte = self.parse_num(data['byte'])
        packet = self.parse_num(data['packet'])
        flags = self.parse_flags(data['flags'])
        code = torch.cat([date, duration, protocol, sIP, sport, dIP, dport, byte, packet, flags], dim=1)
        #print(code[0])
        return code

    def extract_date(self, code):
        pass

    def extract_protocol(self, code):
        pass

    def extract_other(self, code):
        pass

    def decode(self, code):  # TBD
        output = []
        for i in range(code.shape[0]):
            date = self.extract_date(code[:, 0 : 7])
            protocol = self.extract_protocol(code[:, 8 : 11])
            duration, IP, port, byte, packet, flags = self.extract_other(code[:, 11 :])
            output.append(Data(date, duration, protocol, IP, port, byte, packet, flags))
        return output

    def binarize(self, code, range):
        pos = torch.argmax(code[:, range], dim = 1)
        rev = torch.zeros_like(code[:, range])
        rev[:, pos.numpy()] = 1.0
        return rev

    def revise(self, code):
        output = torch.cat([self.binarize(code, range(0, 7)), self.binarize(code, range(8, 11)),
                            self.binarize(code, range(11, code.shape[1]))], dim = 1)
        return output
