class MultilingualTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings):
        """
        embeddings: dict {
            'EtoK': tensor,
            'KtoE': tensor,
            'English': tensor,
            'Korean': tensor
        }
        """
        # English/Korean 원본
        anchor = (embeddings['English'] + embeddings['Korean']) / 2
        
        # 코드스위칭 쿼리
        positive = (embeddings['EtoK'] + embeddings['KtoE']) / 2
        negative = torch.roll(anchor, shifts=1, dims=0)
        
        # Triplet Loss 
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
        return loss
