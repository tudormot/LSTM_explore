import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.input_linear  = nn.Linear(input_size,hidden_size, bias = True)

        if activation == "tanh":
            self.nonlinear = nn.Tanh()
        elif activation == "relu":
            self.nonlinear = nn.ReLU()
        else:
            raise Exception('Invalid activation specified!')


        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []

        if h is not None:
            raise Exception('predefined hidden vectors not implemented')
        seq_len = x.shape[0]
        previous = torch.zeros([x.shape[1], self.hidden_size])
        for i in range(seq_len):
            out = self.nonlinear(self.hidden_linear(previous)+self.input_linear(x[i]))
            h_seq.append(out)
            previous = out
        h_seq = torch.stack(h_seq)
        h = out


        return h_seq , h
    
    
class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lin_Uf = nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin_Ui= nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin_Uo = nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin_Uc = nn.Linear(hidden_size, hidden_size, bias=True)

        #these do not need biases as the result of these layers is summed with the layers above.. one bias is enough for the summation
        self.lin_Wf = nn.Linear(input_size, hidden_size, bias=True)
        self.lin_Wi = nn.Linear(input_size, hidden_size, bias=True)
        self.lin_Wo = nn.Linear(input_size, hidden_size, bias=True)
        self.lin_Wc = nn.Linear(input_size, hidden_size, bias=True)
        pass
       
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        batch_size = x.shape[1]

        if h is not None and c is not None and (h.shape[0] != 1 or c.shape[0] != 1):
            raise Exception('a chain of LSTM layers not implemented..')
        if h is not None:
            h_prev = h.squeeze(dim = 0)
        else:
            h_prev = torch.zeros(size=(batch_size,self.hidden_size)).float()

        if c is not None:
            c_prev = h.squeeze(dim = 0)
        else:
            c_prev = torch.zeros(size=(batch_size,self.hidden_size)).float()

        h_seq = []

        for t,x_t in enumerate(x):
            f = F.sigmoid(self.lin_Wf(x_t) + self.lin_Uf(h_prev))
            i =  F.sigmoid(self.lin_Wi(x_t) + self.lin_Ui(h_prev))
            o = F.sigmoid(self.lin_Wo(x_t) + self.lin_Uo(h_prev))
            c = f *c_prev + i * F.tanh(self.lin_Wc(x_t) + self.lin_Uc(h_prev))
            h = o *  F.tanh (c)
            h_seq.append(h)
            h_prev = h
            c_prev = c

        h_seq = torch.stack(h_seq)


        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

        self.net_main = nn.RNN(input_size,hidden_size,batch_first=False)
        self.linear1 = nn.Linear(hidden_size,classes)
       
    def forward(self, x):
        _,x = self.net_main(x)
        del _

        x = self.linear1(x)
        x = F.relu(x)

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()

        self.net_main = nn.LSTM(input_size, hidden_size,batch_first=False)
        self.linear= nn.Linear(hidden_size,classes)
        pass
    
    def forward(self, x):
        _, (x,__) = self.net_main(x)
        del _
        del __
        x = self.linear(x)
        x = F.relu(x)
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
