import torch

class GloveEmbeddings(torch.nn.Module):
    def __init__(self, words, word2idx, embs):
        super().__init__()
        self.words = words
        self.word2idx = word2idx
        self.embs = embs
        
    def to_id(self, word):
        if word in self.words:
            return self.word2idx[word]
        else:
            return self.word2idx['<unk>']
        
    def to_token(self, id):
        return self.words[id]
        
    def forward(self, idx_list):
        device = idx_list.device
        embs = [self.embs[id].to(device) for id in idx_list]
        return torch.stack(embs)
    
    def make(self, input, device="cpu"):
        curr_words = input.strip().lower().split()
        token_ids = [self.to_id(word) for word in curr_words]
        result = self.forward(torch.tensor(token_ids).to(device))
        
        return result
    
    def make_one(self, input, device="cpu"):
        curr_words = input.strip().lower().split()
        token_id  = self.to_id(curr_words[0])
        result = self.forward(torch.tensor([token_id]).to(device))
        
        return result.reshape(-1)
    