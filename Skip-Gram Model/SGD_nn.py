import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD

device = torch.device("cpu")

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_size)
        self.context_embeddings = nn.Embedding(vocab_size, embed_size)
        print("Center Embeddings initialized:")
        print("Context Embeddings initialized:")


        #Initialize the embeddings using Xavier Initialization
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        print("Center Embeddings initialized by xavier method:")
        print("Context Embeddings initialized by xavier method:")
    
    
    
    def forward(self, word_idxs, embedding_type):
        if embedding_type == "center":
            embeds = self.center_embeddings(word_idxs)
        elif embedding_type == "context":
            embeds = self.context_embeddings(word_idxs)
        
        return embeds




def train_skip_gram(training_sets, embedding_model, learning_rate=0.025, epochs=10, batch_size=256):
    optimizer = SGD(embedding_model.parameters(), lr=learning_rate)
    print("SGD optimizer instance created.")
    
    for epoch in range(epochs):
        print("SDG stars")
        epoch_loss = 0
        
        # Shuffle corpus at the start of each epoch
        np.random.shuffle(training_sets)
        
        # Create batches
        for batch_idx in range(0, len(training_sets), batch_size):
            batch = training_sets[batch_idx:batch_idx + batch_size]
            
            center_idxs = torch.tensor([set[0] for set in batch])
            pos_idxs = torch.tensor([set[1] for set in batch])
            neg_idxs = torch.tensor([set[2] for set in batch])

            center_embed = embedding_model(center_idxs, "center")
            pos_embed = embedding_model(pos_idxs, "context")
            neg_embed = embedding_model(neg_idxs, "context")

            
            
            # Positive samples: log-sigmoid
            pos_score = torch.sum(center_embed * pos_embed, dim=1)
            pos_loss = F.logsigmoid(pos_score)
            
            # Negative samples: log(1 - sigmoid)
            neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
            
            # Compute total loss
            loss = -(pos_loss + neg_loss).mean()
            
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
       



