import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_

import numpy as np

import embedding
from .capsule_layers import PrimaryCaps, EmbCaps

class BaseModel(torch.nn.Module):
    def __init__(self,  params):
        super(BaseModel, self).__init__()
        self.params                  = params
        self.num_ent                 = self.params.ent_vocab_size
        self.num_rel                 = self.params.rel_vocab_size
        self.embedding_dim           = self.params.embedding_dim
        self.perm                    = torch.randperm(self.embedding_dim)
        self.rel_embedding_dim       = self.params.rel_embedding_dim

        print('Entities:{} and Relations:{}'.format(self.num_ent,self.num_rel))
        print('Using Embedding Strategy: {}'.format(embedding.get(self.params.embedding_strategy)))
        # self.entity_embedding        = torch.nn.Embedding(self.num_ent, self.embedding_dim, padding_idx=None)
        self.entity_embedding        = embedding.get(self.params.embedding_strategy)(self.params, is_evaluation=False)
        self.relation_embedding      = torch.nn.Embedding(self.num_rel, self.rel_embedding_dim, padding_idx=None)
        self.bceloss                 = torch.nn.BCELoss()
        self.init()
        self.init_feature_width_height(self.params.concat_form)

    def init_feature_width_height(self, form='plain'):
        if form in ['plain', 'alternate', 'reverse_alternate']:
            self.params.count_h = 2
        elif form in ['alternate_rows']:
            self.params.count_h = 4
        elif form in ['channel2']:
            self.params.count_h = 1
        else:
            self.params.count_h = 1

    def concat(self, e1_embed, rel_embed, form='plain'):
        if self.params.permute_dim:
            e1_embed = e1_embed[:, self.perm]
        if form == 'plain': #ConvE k_h=10, k_w=20, k_in_channels=1 => #ConvE k_h=20, k_w=20, k_in_channels=1
            e1_embed                  = e1_embed. view(-1, 1, self.params.k_h, self.params.k_w)
            rel_embed                 = rel_embed.view(-1, 1, self.params.k_h, self.params.k_w)
            stack_inp                 = torch.cat([e1_embed, rel_embed], 2)
        elif form == 'alternate': #ConvE k_h=10, k_w=20, k_in_channels=1 => #ConvE k_h=20, k_w=20, k_in_channels=1
            e1_embed                  = e1_embed. view(-1, 1, self.embedding_dim)
            rel_embed                 = rel_embed.view(-1, 1, self.embedding_dim)
            stack_inp                 = torch.cat([e1_embed, rel_embed], 1)
            stack_inp                 = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.params.k_h, self.params.k_w))
        elif form == 'reverse_alternate': #ConvE k_h=10, k_w=20, k_in_channels=1 => #ConvE k_h=20, k_w=20, k_in_channels=1
            e1_embed                  = e1_embed. view(-1, 1, self.embedding_dim)
            rel_embed                 = torch.flip(rel_embed, [1]).view(-1, 1, self.embedding_dim)
            stack_inp                 = torch.cat([e1_embed, rel_embed], 1)
            stack_inp                 = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.params.k_h, self.params.k_w))
        elif form == 'alternate_rows': #ConvE k_h=10, k_w=20, k_in_channels=1 => #ConvE k_h=40, k_w=20, k_in_channels=1
            e1_embed                  = e1_embed. view(-1, 1, self.params.k_h, self.params.k_w)
            rel_embed                 = rel_embed.view(-1, 1, self.params.k_h, self.params.k_w)
            stack_one                 = torch.cat([e1_embed, rel_embed], dim=2).view(-1, 1, 2*self.params.k_h, self.params.k_w)
            stack_two                 = torch.cat([e1_embed, torch.flip(rel_embed, dims=(2,))], dim=2).view(-1, 1, 2*self.params.k_h, self.params.k_w)
            stack_inp                 = torch.cat([stack_one, stack_two], dim=2)
        elif form == 'channel2': #ConvE k_h=10, k_w=20, k_in_channels=2 => #ConvE k_h=10, k_w=20, k_in_channels=2
            e1_embed                  = e1_embed. view(-1, 1, self.params.k_h, self.params.k_w)
            rel_embed                 = rel_embed.view(-1, 1, self.params.k_h, self.params.k_w)
            stack_inp                 = torch.cat([e1_embed, rel_embed], 1)
        elif form == 'tensor2d': #k_h=10, k_w=20, k_in_channels=200 => #ConvE k_h=10, k_w=20, k_in_channels=200
            e1_embed                  = e1_embed. view(-1, 1, self.params.k_h, self.params.k_w)
            rel_embed                 = rel_embed.view(-1, self.rel_embedding_dim, 1, 1)
            stack_inp                 = e1_embed * rel_embed
        elif form   == 'tensor3d':
            product                   = torch.bmm(e1_embed.unsqueeze(2), e1_embed.unsqueeze(1))
            product                   = torch.bmm(product.view(-1, self.embedding_dim*self.embedding_dim, 1), rel_embed.unsqueeze(1))
            product                   = product.view(-1, 1, self.embedding_dim, self.embedding_dim, self.embedding_dim)
            return product
        elif form == 'concat':
            return torch.cat([e1_embed, rel_embed], dim=1)
        else: raise NotImplementedError
        return stack_inp

    def init(self):
        # xavier_normal_(self.entity_embedding.weight.data)
        self.entity_embedding.initialize()
        xavier_normal_(self.relation_embedding.weight.data)

    def loss(self, pred, true_label=None, subsampling_weight=None):
        if self.params.loss == 'adversial' or self.params.loss == 'logsigmoid':
            #In self-negative sampling, we do not apply back-propagation on the sampling weight
            positive_score = pred[:,0]; negative_score = pred[:,1:]
            if self.params.loss == 'adversial':
                negative_score = (F.softmax(negative_score * self.params.adversarial_temperature, dim = 1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim = 1)
            elif self.params.loss == 'logsigmoid':
                negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
            positive_score = F.logsigmoid(positive_score)

            if self.params.loss_uniform_weight == 'True':
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
            loss = (positive_sample_loss + negative_sample_loss)/2
            return loss
        elif self.params.loss == 'bce':
            return self.bceloss(pred, true_label)
        elif self.params.loss == 'soft_margin':
            positive_score = pred[:, 0].view(-1, 1)
            negative_score = pred[:, 1:]
            negative_score = torch.mean(negative_score, dim=-1, keepdim=True)
            difference     = torch.nn.Softplus()(positive_score - negative_score + self.params.gamma)
            loss           = torch.mean(difference)
            return loss
        elif self.params.loss == 'hard_margin':
            positive_score = pred[:, 0].view(-1, 1)
            negative_score = pred[:, 1:]
            negative_score = torch.mean(negative_score, dim=-1, keepdim=True)
            difference     = torch.nn.ReLU()(positive_score - negative_score + self.params.gamma)
            loss           = torch.mean(difference)
            return loss
        else: raise NotImplementedError

    def score(self, x):
        if self.params.loss == 'bce':
            pred           = torch.sigmoid(x)
        elif self.params.loss in ['soft_margin', 'hard_margin', 'adversial']:
            pred           = x
        else:
            pred           = self.params.gamma - x
        return pred

    def forward(self, h, r, t, negative_entities=None, mode='tail_prediction', strategy='one_to_n', adjacency_matrix=None):
        pass

class FCConvE(BaseModel):
    def __init__(self,  params):
        super(FCConvE, self).__init__(params)
        self.init_params()

        self.dropout1                   = torch.nn.Dropout(self.params.dropout1)
        self.dropout2                   = torch.nn.Dropout2d(self.params.dropout2)
        self.dropout3                   = torch.nn.Dropout(self.params.dropout3)

        self.bn1                        = torch.nn.BatchNorm1d(self.params.fc1)
        self.bn2                        = torch.nn.BatchNorm2d(self.params.k_out_channels_1)
        self.bn3                        = torch.nn.BatchNorm1d(self.embedding_dim)

        self.conv1                      = torch.nn.Conv2d(self.params.k_in_channels, out_channels=self.params.k_out_channels_1,
                                                          kernel_size=(self.params.ker_sz[0], self.params.ker_sz[1]),
                                                          stride=1, padding=0, bias=self.params.bias, groups = self.params.conv_groups)
        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

        self.fc1                         = torch.nn.Linear(self.embedding_dim+self.rel_embedding_dim, self.params.fc1)

        flat_sz_h                       = int(self.params.count_h*self.params.k_h) - self.params.ker_sz[0] + 1
        flat_sz_w                       = self.params.k_w - self.params.ker_sz[1] + 1
        self.flat_sz                    = flat_sz_h*flat_sz_w*self.params.k_out_channels_1
        self.fc                         = torch.nn.Linear(self.flat_sz, self.embedding_dim)

    def init_params(self):
        self.params.count_h              = 1
        self.params.k_in_channels        = self.params.fc1//(self.params.k_h*self.params.k_w)
        if self.params.group_convolution == 'depthwise':
            self.params.k_out_channels_1 = self.params.k_out_channels_1 * self.params.k_in_channels
            self.params.conv_groups      = self.params.k_in_channels
        else:
            self.params.conv_groups      = 1

    def forward(self, h, r, t, entity_embedding, negative_entities=None, mode='tail_prediction', strategy='one_to_n'):
        if entity_embedding == None: entity_embedding = self.entity_embedding

        h_embedding                         = entity_embedding(h)
        r_embedding                         = self.relation_embedding(r)

        x                                   = torch.cat([h_embedding, r_embedding], dim=1)

        x                                   = self.fc1(x)
        x                                   = self.bn1(x)
        x                                   = F.relu(x)
        x                                   = self.dropout1(x)

        x                                   = x.view(-1, self.params.k_in_channels, self.params.k_h, self.params.k_w)

        x                                   = self.conv1(x)
        x                                   = self.bn2(x)
        x                                   = F.relu(x)
        x                                   = self.dropout2(x)
        x                                   = x.view(-1, self.flat_sz)

        x                                   = self.fc(x)
        x                                   = self.bn3(x)
        x                                   = F.relu(x)
        x                                   = self.dropout3(x)

        if strategy == 'one_to_n':
            x                               = torch.mm(x, entity_embedding.weight.transpose(1,0))
            x                               += self.bias.expand_as(x)
        else:
            x                               = torch.mul(x.unsqueeze(1), entity_embedding(negative_entities)).sum(dim=-1)
            x                               += self.bias[negative_entities]

        pred                                = self.score(x)
        return pred

class FCE(BaseModel):
    def __init__(self, params):
        super(FCE, self).__init__(params)
        self.dropout1                    = torch.nn.Dropout(self.params.dropout1)
        self.bn1                         = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc1                         = torch.nn.Linear(self.embedding_dim+self.rel_embedding_dim, self.embedding_dim)

    def forward(self, h, r, t, entity_embedding, negative_entities=None, mode='tail_prediction', strategy='one_to_n'):
        if entity_embedding==None: entity_embedding = self.entity_embedding

        h_embedding     = entity_embedding(h)
        r_embedding     = self.relation_embedding(r)

        # h_embedding     = self.dropout1(h_embedding)
        # r_embedding     = self.dropout1(r_embedding)
        # x               = h_embedding * r_embedding

        x               = torch.cat([h_embedding, r_embedding], dim=1)
        x               = self.fc1(x)
        x               = self.bn1(x)
        x               = F.relu(x)
        x               = self.dropout1(x)

        if strategy == 'one_to_n':
            x                                    = torch.mm(x, entity_embedding.weight.transpose(1,0))
            # x                                  += self.bias.expand_as(x)
        else:
            x                                    = torch.mul(x.unsqueeze(1), entity_embedding(negative_entities)).sum(dim=-1)
            # x                                  += self.bias[negative_entities]

        pred                                     = self.score(x)
        return pred

