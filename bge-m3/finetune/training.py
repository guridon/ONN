import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from transformers import get_linear_schedule_with_warmup

class MultilingualTripletLoss(nn.Module):
    def __init__(self, margin=0.5, temp=0.05):
        super().__init__()
        self.margin = margin
        self.temp = temp
        
    def mean_pooling(self, outputs, mask):
        token_embeddings = outputs.last_hidden_state
        input_mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (dict): {
                "EtoK": tensor, 
                "KtoE": tensor, 
                "English": tensor, 
                "Korean": tensor
            }
            labels: 배치 내 실제 라벨 정보
        """
        # 앵커: 단일 언어
        anchor_en = embeddings["English"]
        anchor_ko = embeddings["Korean"]
        
        # 포지티브: 코드 스위칭
        pos_etok = embeddings["EtoK"]
        pos_ktoe = embeddings["KtoE"]
        
        # 네거티브 샘플링 (배치 내 하드 네거티브)
        batch_size = anchor_en.size(0)
        neg_idx = torch.randint(0, batch_size, (batch_size,))
        neg_en = anchor_en[neg_idx]
        neg_ko = anchor_ko[neg_idx]
        
        # 대조 손실 계산
        loss_en = F.triplet_margin_loss(
            anchor_en, 
            (pos_etok + pos_ktoe)/2, 
            neg_en, 
            margin=self.margin
        )
        
        loss_ko = F.triplet_margin_loss(
            anchor_ko,
            (pos_etok + pos_ktoe)/2,
            neg_ko,
            margin=self.margin
        )
        
        # 교차 언어 유사도 정규화
        cross_sim = F.cosine_similarity(anchor_en, anchor_ko)
        reg_loss = torch.mean(torch.abs(cross_sim - 0.8))  # 0.8 목표 유사도
        
        return 0.7*(loss_en + loss_ko) + 0.3*reg_loss

def train_model(model, dataloader, optimizer, criterion, epochs=10):
    device = model.device
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps*0.1),
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            embeddings = {}
            for key in ['English', 'Korean', 'EtoK', 'KtoE']:
                input_ids = batch[key]['input_ids'].to(device)
                attention_mask = batch[key]['attention_mask'].to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, 
                                  attention_mask=attention_mask)
                    
                embeddings[key] = criterion.mean_pooling(outputs, attention_mask)
            
            loss = criterion(embeddings, batch.get('labels'))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 그래디언트 클리핑 완화
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({'loss': avg_loss})
            wandb.log({
                'step_loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })
        
        epoch_loss = total_loss / len(dataloader)
        wandb.log({'epoch_loss': epoch_loss})
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

