from .standard_embedding import StandardEmbedding

def get(name):
    if name.lower() in ['standard', 'standardembedding', 'embedding', '1']:
        return StandardEmbedding
    else:
        raise(Exception("Unknown embedding method"))
