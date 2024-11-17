from utils.dataset import get_data_loaders
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from models.logistic_regression import DeepLogisticRegressionModel, Small3DCNNClassifier
import time
import wandb
from tqdm import tqdm

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader, processes the dataset,
    and trains a logistic regression model.
    """

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="DeepEmotion", config=cfg_dict)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if cfg.verbose:
        print(f"Device: {device}")
        print("Loading dataloader...")
        
    train_dataloader, val_dataloader = get_data_loaders(cfg)

    input_dim = 132 * 175 * 48
    output_dim = len(cfg.data.emotion_idx)
    model = Small3DCNNClassifier(output_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=cfg.data.weight_decay)

    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        model.train()
        for batch, label in tqdm(train_dataloader):
            
            batch, label = batch.float().to(device), label.float().to(device)

            output = model(batch)

            wandb.log({
                "labels": label.detach().cpu().numpy()
            })
            
            loss = criterion(output, label.argmax(dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predictions = torch.max(output, dim=1)
            true_labels = label.argmax(dim=1)
            
            correct_predictions += (predictions == true_labels).sum().item()
            total_samples += label.size(0)

        end_time = time.time()
        epoch_duration = end_time - start_time
        accuracy = correct_predictions / total_samples
        normalized_loss = total_loss / total_samples

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_batch, val_label in val_dataloader:

                val_batch, val_label = val_batch.float().to(device), val_label.float().to(device)
                
                val_output = model(val_batch)
                _, val_predictions = torch.max(val_output, dim=1)
                val_true_labels = val_label.argmax(dim=1)
                
                val_correct += (val_predictions == val_true_labels).sum().item()
                val_total += val_label.size(0)

        val_accuracy = val_correct / val_total
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {normalized_loss:.4f}, "
        f"Accuracy: {accuracy*100:.2f}%, Time: {epoch_duration:.2f} seconds",
        f"Validation Accuracy: {val_accuracy * 100:.2f}%\n")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": normalized_loss,
            "train_accuracy": accuracy,
            "val_accuracy": val_accuracy
        })
    
if __name__ == "__main__":
    main()
