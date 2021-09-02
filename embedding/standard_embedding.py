#pylint: disable=E1101
"""
Wrapper for basic torch.nn.Embedding
"""
import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_, xavier_uniform_

class StandardEmbedding(nn.Module):
    """
    Wrapper over basic torch.nn.Embedding
    Provides lookup method
    """
    def __init__(self, params, is_evaluation=False):
        super(StandardEmbedding, self).__init__()
        '''Initialization'''
        self.ent_vocab_size = params.ent_vocab_size #ent_vocab_size n
        self.embedding_dim = params.embedding_dim #embedding_dim  l

        '''Entity Embedding'''
        self.embedding = nn.Embedding(self.ent_vocab_size, self.embedding_dim, padding_idx=None)

    def initialize(self):
        xavier_normal_(self.embedding.weight.data)

    def get_embedding(self):
        return self.embedding.weight

    def lookup_all(self):
        '''Returns the full embedding matrix'''
        return self.embedding

    def lookup(self, indices):
        '''Returns the embeddings for given entity indices'''
        batch_embed = self.embedding(indices)
        return batch_embed

    def forward(self, x):
        return self.lookup(x)
