import torch

def evaluate_model(model, dataloader):
    """
    Args:
        model: Hugging Face 모델 객체
        dataloader: PyTorch DataLoader 객체
    Returns:
        float: 평균 코사인 유사도 값
    """
    device = next(model.parameters()).device 
    
    model.eval()
    
    similarities = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = {}
            
            for key in batch.keys():
                inputs = {k: v.squeeze(1).to(device) for k, v in batch[key].items()}
                outputs = model(**inputs)
                embeddings[key] = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
            
            anchor = (embeddings["English"] + embeddings["Korean"]) / 2  # anchor 임베딩 계산
            positive = (embeddings["EtoK"] + embeddings["KtoE"]) / 2     # Positive 임베딩 계산
            
            similarity = torch.nn.functional.cosine_similarity(anchor, positive).mean().item()
            similarities.append(similarity)
    
    return sum(similarities) / len(similarities)
