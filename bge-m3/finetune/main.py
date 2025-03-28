import os 
import wandb
from dataset_loader import load_code_switch_dataset
from model_loader import load_model
from training import train_model, MultilingualTripletLoss
from evaluation import evaluate_model
import torch

def main():
    # WandB 초기화
    wandb.init(
        project="bge-m3-finetuning",
        entity="fnelwndls",
        name="code-switching-analysis",
        config={
            "architecture": "bge-m3",
            "dataset": "code-switch",
            "learning_rate": 1e-5,
            "batch_size": 8,
            "epochs": 10,
            "margin": 0.7
        }
    )
    # 모델 로드 및 데이터셋 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model("BAAI/bge-m3", device=device)
    dataloader = load_code_switch_dataset("../dataset/code-switch.json", tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    criterion = MultilingualTripletLoss(margin=wandb.config.margin)

    print("학습 시작...")
    
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=wandb.config.epochs
    )

    print("평가 시작...")
    
    similarity_score = evaluate_model(model=model, dataloader=dataloader)
    
    wandb.log({"Validation Similarity Score": similarity_score})
    
    print(f"평균 유사도 점수: {similarity_score:.4f}")
    
    # 모델 저장 및 WandB 아티팩트 기록
    model.save_pretrained("fine_tuned_bge_m3")
    tokenizer.save_pretrained("fine_tuned_bge_m3")
    wandb.save("fine_tuned_bge_m3/*")

if __name__ == "__main__":
    main()
