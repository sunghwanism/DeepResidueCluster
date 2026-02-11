import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from models.ProposedModel.model import ContrastiveGraphModel

def run_training(config, train_loader, val_loader, test_loader, run_wandb):
    """
    Training loop for Proposed Contrastive Model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize Model
    model = ContrastiveGraphModel(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        out_dim=config.out_dim,
        bb_masking_ratio=config.bb_masking_ratio,
        att_mask_ratio=config.att_mask_ratio,
        alpha=config.alpha,
        beta=config.beta
    ).to(device)

    print("Model Initialized:")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Scheduler (Optional, but good practice)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0
        total_con_loss = 0
        total_pred_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_a, batch_b, labels in train_pbar:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            loss, loss_con, loss_pred = model(batch_a, batch_b, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Record
            total_loss += loss.item()
            total_con_loss += loss_con.item()
            total_pred_loss += loss_pred.item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'con': loss_con.item(), 'pred': loss_pred.item()})
            
        avg_loss = total_loss / len(train_loader)
        avg_con_loss = total_con_loss / len(train_loader)
        avg_pred_loss = total_pred_loss / len(train_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} (Con: {avg_con_loss:.4f}, Pred: {avg_pred_loss:.4f})")
        
        if run_wandb:
            wandb.log({
                "train_loss": avg_loss,
                "train_con_loss": avg_con_loss,
                "train_pred_loss": avg_pred_loss,
                "epoch": epoch
            })

        # Validation
        val_loss, val_con, val_pred = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} (Con: {val_con:.4f}, Pred: {val_pred:.4f})")
        
        if run_wandb:
            wandb.log({
                "val_loss": val_loss,
                "val_con_loss": val_con,
                "val_pred_loss": val_pred,
                "epoch": epoch
            })
            
        # Select best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.SAVEPATH}/best_model_pchk.pth")
            print(f"Saved Best Model at Epoch {epoch}")

    print("Training Finished.")
    
    # Test Evaluation
    if test_loader:
        model.load_state_dict(torch.load(f"{config.SAVEPATH}/best_model_pchk.pth"))
        test_loss, test_con, test_pred = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f} (Con: {test_con:.4f}, Pred: {test_pred:.4f})")
        
        if run_wandb:
            wandb.log({
                "test_loss": test_loss,
                "test_con_loss": test_con,
                "test_pred_loss": test_pred
            })


def evaluate(model, loader, device):
    if not loader:
        return 0.0, 0.0, 0.0
        
    model.eval()
    total_loss = 0
    total_con_loss = 0
    total_pred_loss = 0
    
    with torch.no_grad():
        for batch_a, batch_b, labels in loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            labels = labels.to(device)
            
            loss, loss_con, loss_pred = model(batch_a, batch_b, labels)
            
            total_loss += loss.item()
            total_con_loss += loss_con.item()
            total_pred_loss += loss_pred.item()
            
    avg_loss = total_loss / len(loader)
    avg_con_loss = total_con_loss / len(loader)
    avg_pred_loss = total_pred_loss / len(loader)
    
    return avg_loss, avg_con_loss, avg_pred_loss
