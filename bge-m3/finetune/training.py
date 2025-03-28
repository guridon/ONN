import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

class MultilingualTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings):
        """
        Args:
            embeddings (dict): {"EtoK": tensor, "KtoE": tensor, "English": tensor, "Korean": tensor}

        Returns:
            torch.Tensor: Triplet Loss
        """
        anchor = (embeddings["English"] + embeddings["Korean"]) / 2
        positive = (embeddings["EtoK"] + embeddings["KtoE"]) / 2
        negative = torch.roll(anchor, shifts=1, dims=0)

        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)

        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
        return loss

def train_model(model, dataloader, optimizer, criterion, epochs=10):
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                
                embeddings = {}
                for key in batch.keys():
                    inputs = {k: v.squeeze(1).to(device) for k, v in batch[key].items()}
                    outputs = model(**inputs)
                    embeddings[key] = outputs.last_hidden_state[:, 0, :]
                
                loss = criterion(embeddings)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                tepoch.set_postfix(loss=loss.item())
                
                wandb.log({"Train Loss": loss.item()})
        
        wandb.log({"Epoch Loss": total_loss / len(dataloader)})

