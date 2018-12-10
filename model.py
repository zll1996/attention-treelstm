import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import Constants
import utils

class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        if self.cudaFlag:
            self.cx = self.cx.cuda()
            self.ox = self.ox.cuda()

    def forward(self, input):
        c = self.cx(input)
        o = torch.sigmoid(self.ox(input))
        h = o * torch.tanh(c)
        return c, h

class BinaryTreeComposer(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})    
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.H = []

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

        if self.cudaFlag:
            self.ilh = self.ilh.cuda()
            self.irh = self.irh.cuda()
            self.lflh = self.lflh.cuda()
            self.lfrh = self.lfrh.cuda()
            self.rflh = self.rflh.cuda()
            self.rfrh = self.rfrh.cuda()
            self.ulh = self.ulh.cuda()

    def forward(self, lc, lh , rc, rh):
        i = torch.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = torch.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = torch.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = torch.tanh(self.ulh(lh) + self.urh(rh))
        c =  i* update + lf*lc + rf*rc
        h = torch.tanh(c)

        # Appending add the hidden state together
        self.H.append(h)

        return c, h


class BinaryTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, criterion):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.criterion = criterion

        self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, training = False):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        loss = Var(torch.zeros(1)) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        if tree.num_children == 0:
            # leaf case
            tree.state = self.leaf_module.forward(embs[tree.idx-1])
        else:
            for idx in range(tree.num_children):
                _, child_loss = self.forward(tree.children[idx], embs, training)
                loss = loss + child_loss
            lc, lh, rc, rh = self.get_child_state(tree)
            tree.state = self.composer.forward(lc, lh, rc, rh)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[1], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss


    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh

###################################################################


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, criterion):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.H = []

        # self.emb = nn.Embedding(vocab_size,in_dim,
        #                         padding_idx=Constants.PAD)
        # torch.manual_seed(123)

        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)

        self.criterion = criterion
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def node_forward(self, inputs, child_c, child_h):
        """

        :param inputs: (1, 300)
        :param child_c: (num_children, 1, mem_dim)
        :param child_h: (num_children, 1, mem_dim)
        :return: (tuple)
        c: (1, mem_dim)
        h: (1, mem_dim)
        """

        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        i = torch.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = torch.tanh(self.ux(inputs)+self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi)+fx for child_hi in child_h], 0)
        f = torch.sigmoid(f)

        # f = F.torch.unsqueeze(f,1) # comment to fix dimension missmatch
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i,u) + F.torch.sum(fc,0)
        h = F.torch.mul(o, torch.tanh(c))

        # Appending add the hidden state together
        self.H.append(h)

        return c, h

    def forward(self, tree, embs, training = False):
        """
        Child sum tree LSTM forward function
        :param tree:
        :param embs: (sentence_length, 1, 300)
        :param training:
        :return:
        """

        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        loss = Var(torch.zeros(1)) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in range(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], embs, training)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx-1], child_c, child_h)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[1], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss

    def get_child_states(self, tree):
        """
        Get c and h of all children
        :param tree:
        :return: (tuple)
        child_c: (num_children, 1, mem_dim)
        child_h: (num_children, 1, mem_dim)
        """
        # add extra singleton dimension in middle...
        # because pytorch needs mini batches... :sad:
        if tree.num_children==0:
            child_c = Var(torch.zeros(1,1,self.mem_dim))
            child_h = Var(torch.zeros(1,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children,1,self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in range(tree.num_children):
                child_c[idx] = tree.children[idx].state[0]
                child_h[idx] = tree.children[idx].state[1]
                # child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, cuda, mem_dim, attention_dim, dropout):
        super(SelfAttentiveEncoder, self).__init__()

        self.att_units = attention_dim
        self.hops = 1
        self.cudaFlag = cuda
        self.pad_idx = 0
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(mem_dim, self.att_units, bias=False)
        self.ws2 = nn.Linear(self.att_units, self.hops, bias=False)

        self.init_weights()

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, outp, inputs=None, penalize=True):

        outp = torch.unsqueeze(outp, 0)  # Expects input of the form [btch, len, nhid]

        compressed_embeddings = outp.view(outp.size(1), -1)  # [btch*len, nhid]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(1, outp.size(1), -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, outp.size(1)))  # [bsz*hop, len]
        alphas = alphas.view(outp.size(0), self.hops, outp.size(1))  # [hop, len]
        M = torch.bmm(alphas, outp)  # [bsz, hop, mem_dim]
        return M, alphas

##############################################################################


# output module
class SentimentModule(nn.Module):
    def __init__(self, cuda, mem_dim, num_classes, dropout1=False):
        super(SentimentModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.dropout = dropout1
        # torch.manual_seed(456)
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if self.cudaFlag:
            self.l1 = self.l1.cuda()

    def forward(self, vec, training=False):
        """
        Sentiment module forward function
        :param vec: (1, mem_dim)
        :param training:
        :return:
        (1, number_of_class)
        """
        if self.dropout:
            out = self.logsoftmax(self.l1(F.dropout(vec, training=training)))
        else:
            out = self.logsoftmax(self.l1(vec))
        return out


class TreeLSTMSentiment(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, num_classes, model_name, attetion_dim, dropout2,
                 attention_flag, criterion):
        super(TreeLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.model_name = model_name
        self.attention_flag = attention_flag
        if self.model_name == 'dependency':
            self.tree_module = ChildSumTreeLSTM(cuda, in_dim, mem_dim, criterion)
        elif self.model_name == 'constituency':
            self.tree_module = BinaryTreeLSTM(cuda, in_dim, mem_dim, criterion)
        self.attention = SelfAttentiveEncoder(cuda, mem_dim, attetion_dim, dropout2)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout1=True)
        self.tree_module.set_output_module(self.output_module)

        self.wf = nn.Parameter(torch.zeros(1, mem_dim, mem_dim).uniform_(-1, 1))

    def forward(self, tree, inputs, training=False):
        """
        TreeLSTMSentiment forward function
        :param tree:
        :param inputs: (sentence_length, 1, 300)
        :param training:
        :return:
        """

        if self.model_name == 'dependency':
            self.tree_module.H = []
        elif self.model_name == 'constituency':
            self.tree_module.composer.H = []
        tree_state, loss = self.tree_module(tree, inputs, training)

        if self.attention_flag:

            if self.model_name == 'dependency':
                H_combined = torch.cat(self.tree_module.H, 0)
            elif self.model_name == 'constituency':
                H_combined = torch.cat(self.tree_module.composer.H, 0)

            M, att = self.attention.forward(H_combined, inputs)
            M = F.relu(torch.bmm(M, self.wf))

            state = torch.squeeze(M)

            output = self.output_module.forward(state, training)
            output = torch.reshape(output, (1, 3))
        else:
            output = tree.output

        return output, loss
