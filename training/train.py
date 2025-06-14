import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from models.msmt import MSMT
import torch.optim as optim
import torch

parser = argparse.ArgumentParser(description='MSMT Training Script')
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
parser.add_argument("--data", type=str, default="PEMS08", help="Dataset name")
parser.add_argument("--input_dim", type=int, default=3, help="Input feature dimension")
parser.add_argument("--channels", type=int, default=128, help="Number of hidden channels")
parser.add_argument("--num_nodes", type=int, default=170, help="Number of nodes in the graph")
parser.add_argument("--input_len", type=int, default=12, help="Input sequence length")
parser.add_argument("--output_len", type=int, default=12, help="Output sequence length")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay rate")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--print_every", type=int, default=50, help="Print frequency during training")
parser.add_argument("--save", type=str, default="./experiments/logs/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "-", 
                   help="Save directory path")
parser.add_argument("--es_patience", type=int, default=100, 
                   help="Early stopping patience (epochs without improvement)")
parser.add_argument("--memory_size", type=int, default=4, help="Memory bank size for MSMT")

args = parser.parse_args()


class MSMTTrainer:
    """
    MSMT Training Manager
    
    Handles training, validation, and evaluation of the MSMT model
    with proper data flow and metric computation.
    """
    
    def __init__(self, scaler, input_dim, channels, num_nodes, input_len, output_len, 
                 dropout, learning_rate, weight_decay, device, memory_size=4):
        
        self.model = MSMT(
            device=device, 
            input_dim=input_dim, 
            channels=channels, 
            num_nodes=num_nodes, 
            input_len=input_len, 
            output_len=output_len, 
            dropout=dropout,
            memory_size=memory_size
        )
        self.model.to(device)
        
        # Use Adam optimizer for stable training
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.loss_fn = util.MAE_torch
        self.scaler = scaler
        self.clip = 5  # Gradient clipping
        self.device = device
        
        print(f"MSMT Model Parameters: {self.model.param_num():,}")
        print("Model Architecture:")
        print(self.model)

    def train_step(self, input_data, target_data):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(input_data)
        if isinstance(output, tuple):
            output = output[0]  # Take main prediction if uncertainty is returned

        # Handle output dimensions
        if output.dim() == 4:
            output = output[..., -1]  # Take last time step if 4D

        B, T_out, N = output.shape

        # Inverse transform predictions to original scale
        flat_output = output.reshape(-1, N)
        inv_output = self.scaler.inverse_transform(flat_output)
        predictions = inv_output.reshape(B, T_out, N)

        # Prepare target data
        targets = target_data
        if targets.dim() == 3 and targets.shape[1] == N and targets.shape[2] == T_out:
            targets = targets.permute(0, 2, 1)  # [B, N, T] -> [B, T, N]

        assert predictions.shape == targets.shape, \
            f"Training shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"

        # Compute loss and metrics
        loss = self.loss_fn(predictions, targets, 0.0)
        loss.backward()
        
        # Gradient clipping
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            mape = util.MAPE_torch(predictions, targets, 0.0).item()
            rmse = util.RMSE_torch(predictions, targets, 0.0).item()
            wmape = util.WMAPE_torch(predictions, targets, 0.0).item()

        return loss.item(), mape, rmse, wmape

    def eval_step(self, input_data, target_data):
        """Single evaluation step"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_data)
            
        if isinstance(output, tuple):
            output = output[0]  # Take main prediction

        if output.dim() == 4:
            output = output[..., -1]

        B, T_out, N = output.shape

        # Inverse transform predictions
        flat_output = output.reshape(-1, N)
        inv_output = self.scaler.inverse_transform(flat_output)
        predictions = inv_output.reshape(B, T_out, N)

        # Prepare target data
        targets = target_data
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1).permute(0, 2, 1)
        elif targets.dim() == 3 and targets.shape[1] == N and targets.shape[2] == T_out:
            targets = targets.permute(0, 2, 1)

        assert predictions.shape == targets.shape, \
            f"Evaluation shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"

        # Compute metrics
        loss = self.loss_fn(predictions, targets, 0.0)
        mape = util.MAPE_torch(predictions, targets, 0.0).item()
        rmse = util.RMSE_torch(predictions, targets, 0.0).item()
        wmape = util.WMAPE_torch(predictions, targets, 0.0).item()

        return loss.item(), mape, rmse, wmape


def setup_seed(seed=6666):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def configure_dataset(args):
    """Configure dataset-specific parameters"""
    data_configs = {
        "PEMS08": {"path": "dataset//PEMS08", "num_nodes": 170},
        "PEMS04": {"path": "dataset//PEMS04", "num_nodes": 307},
        "PEMS03": {"path": "dataset//PEMS03", "num_nodes": 358, "epochs": 300, "es_patience": 100},
        "PEMS07": {"path": "dataset/PEMS07", "num_nodes": 883},
        "bike_drop": {"path": "dataset//bike_drop", "num_nodes": 250},
        "bike_pick": {"path": "dataset//bike_pick", "num_nodes": 250},
        "taxi_drop": {"path": "dataset//taxi_drop", "num_nodes": 266},
        "taxi_pick": {"path": "dataset//taxi_pick", "num_nodes": 266}
    }
    
    if args.data in data_configs:
        config = data_configs[args.data]
        args.data = config["path"]
        args.num_nodes = config["num_nodes"]
        
        # Override specific parameters if they exist
        if "input_len" in config:
            args.input_len = config["input_len"]
        if "output_len" in config:
            args.output_len = config["output_len"]
        if "epochs" in config:
            args.epochs = config["epochs"]
        if "es_patience" in config:
            args.es_patience = config["es_patience"]
    
    return args


def evaluate_test_performance(trainer, dataloader, args, scaler):
    """Comprehensive test evaluation with horizon-wise metrics"""
    outputs = []
    
    # Prepare ground truth data
    realy = torch.Tensor(dataloader["y_test"]).to(trainer.device)
    print(f"Original test data shape: {realy.shape}")
    
    # Standardize shape to [S, T, N]
    if realy.dim() == 4:
        if realy.shape[1] == 1:  # [S, 1, N, T]
            realy = realy.squeeze(1).permute(0, 2, 1)
        elif realy.shape[-1] == 1:  # [S, T, N, 1]
            realy = realy.squeeze(-1)
        else:  # [S, C, N, T] with C > 1
            realy = realy[:, 0, :, :].permute(0, 2, 1)
    elif realy.dim() == 3:
        if realy.shape[1] == args.num_nodes:  # [S, N, T]
            realy = realy.permute(0, 2, 1)
    
    print(f"Processed test data shape: {realy.shape}")

    # Generate predictions
    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(trainer.device)
        testx = testx.transpose(1, 3)
        
        with torch.no_grad():
            preds = trainer.model(testx)
            if isinstance(preds, tuple):
                preds = preds[0]
            if preds.dim() == 4:
                preds = preds[..., -1]
        
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    print(f"Predictions shape: {yhat.shape}")

    # Compute horizon-wise metrics
    horizon_metrics = {
        'mae': [], 'mape': [], 'rmse': [], 'wmape': []
    }

    actual_horizons = min(args.output_len, yhat.shape[1], realy.shape[1])
    print(f"Evaluating {actual_horizons} forecasting horizons")
    
    for horizon in range(actual_horizons):
        pred = scaler.inverse_transform(yhat[:, horizon, :])
        real = realy[:, horizon, :]
        
        mae, mape, rmse, wmape = util.metric(pred, real)
        print(f"Horizon {horizon+1:2d}: MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape:.4f}, WMAPE {wmape:.4f}")
        
        horizon_metrics['mae'].append(mae)
        horizon_metrics['mape'].append(mape)
        horizon_metrics['rmse'].append(rmse)
        horizon_metrics['wmape'].append(wmape)

    # Average performance
    avg_metrics = {k: np.mean(v) for k, v in horizon_metrics.items()}
    print(f"\nAverage Performance:")
    print(f"MAE: {avg_metrics['mae']:.4f}, RMSE: {avg_metrics['rmse']:.4f}, "
          f"MAPE: {avg_metrics['mape']:.4f}, WMAPE: {avg_metrics['wmape']:.4f}")
    
    return avg_metrics['mae'], horizon_metrics


def main():
    # Setup
    setup_seed(6666)
    args = configure_dataset(args)
    device = torch.device(args.device)
    
    print("MSMT Training Configuration:")
    print(args)
    
    # Load data
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_test_mae = float('inf')
    epochs_since_improvement = 0
    save_path = args.save + args.data.split('/')[-1] + "/"
    
    # Create save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Training history
    training_history = []
    training_times = []
    validation_times = []
    
    # Initialize trainer
    trainer = MSMTTrainer(
        scaler=scaler,
        input_dim=args.input_dim,
        channels=args.channels,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        memory_size=args.memory_size
    )
    
    # Shape validation
    print("\n🔍 Performing shape validation...")
    x_sample, y_sample = next(dataloader["train_loader"].get_iterator())
    x_sample = torch.Tensor(x_sample).to(device).transpose(1, 3)
    y_sample = torch.Tensor(y_sample).to(device).transpose(1, 3)
    y_sample = y_sample[:, 0, :, :]
    
    out_sample = trainer.model(x_sample)
    if isinstance(out_sample, tuple): 
        out_sample = out_sample[0]
    if out_sample.dim() == 4: 
        out_sample = out_sample[..., -1]
    
    flat = out_sample.reshape(-1, out_sample.shape[-1])
    inv = scaler.inverse_transform(flat)
    pred_sample = inv.reshape(out_sample.shape)
    real_sample = y_sample.permute(0, 2, 1)
    
    print(f"✅ Shape validation passed: pred {pred_sample.shape} == real {real_sample.shape}")
    assert pred_sample.shape == real_sample.shape, "Shape mismatch detected!"
    
    print("\n🚀 Starting MSMT training...")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training phase
        train_metrics = {'loss': [], 'mape': [], 'rmse': [], 'wmape': []}
        
        train_start = time.time()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            
            loss, mape, rmse, wmape = trainer.train_step(trainx, trainy[:, 0, :, :])
            
            train_metrics['loss'].append(loss)
            train_metrics['mape'].append(mape)
            train_metrics['rmse'].append(rmse)
            train_metrics['wmape'].append(wmape)

            if iter % args.print_every == 0:
                print(f"Iter {iter:3d}: Loss {loss:.4f}, RMSE {rmse:.4f}, MAPE {mape:.4f}, WMAPE {wmape:.4f}")
        
        train_time = time.time() - train_start
        training_times.append(train_time)
        
        # Validation phase
        val_metrics = {'loss': [], 'mape': [], 'rmse': [], 'wmape': []}
        
        val_start = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            valx = torch.Tensor(x).to(device).transpose(1, 3)
            valy = torch.Tensor(y).to(device).transpose(1, 3)
            
            loss, mape, rmse, wmape = trainer.eval_step(valx, valy[:, 0, :, :])
            
            val_metrics['loss'].append(loss)
            val_metrics['mape'].append(mape)
            val_metrics['rmse'].append(rmse)
            val_metrics['wmape'].append(wmape)
        
        val_time = time.time() - val_start
        validation_times.append(val_time)
        
        # Compute average metrics
        avg_train = {k: np.mean(v) for k, v in train_metrics.items()}
        avg_val = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Log epoch results
        print(f"\nEpoch {epoch:3d} | Train Time: {train_time:.2f}s | Val Time: {val_time:.2f}s")
        print(f"Train: Loss {avg_train['loss']:.4f}, RMSE {avg_train['rmse']:.4f}, MAPE {avg_train['mape']:.4f}, WMAPE {avg_train['wmape']:.4f}")
        print(f"Val:   Loss {avg_val['loss']:.4f}, RMSE {avg_val['rmse']:.4f}, MAPE {avg_val['mape']:.4f}, WMAPE {avg_val['wmape']:.4f}")
        
        # Save training history
        epoch_data = {**{f'train_{k}': v for k, v in avg_train.items()}, 
                     **{f'val_{k}': v for k, v in avg_val.items()}}
        training_history.append(epoch_data)
        
        # Model checkpointing and early stopping
        if avg_val['loss'] < best_val_loss:
            if epoch < 100:
                # Early epochs: save based on validation loss
                best_val_loss = avg_val['loss']
                torch.save(trainer.model.state_dict(), save_path + "best_model.pth")
                epochs_since_improvement = 0
                print(f"✅ Model saved! New best validation loss: {best_val_loss:.4f}")
                
            else:
                # Later epochs: validate on test set
                test_mae, _ = evaluate_test_performance(trainer, dataloader, args, scaler)
                
                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    best_val_loss = avg_val['loss']
                    torch.save(trainer.model.state_dict(), save_path + "best_model.pth")
                    epochs_since_improvement = 0
                    print(f"✅ Model saved! New best test MAE: {best_test_mae:.4f}")
                else:
                    epochs_since_improvement += 1
                    print(f"❌ No improvement (patience: {epochs_since_improvement}/{args.es_patience})")
        else:
            epochs_since_improvement += 1
            print(f"❌ No improvement (patience: {epochs_since_improvement}/{args.es_patience})")
        
        # Save training progress
        train_df = pd.DataFrame(training_history)
        train_df.round(6).to_csv(f"{save_path}/training_log.csv", index=False)
        
        # Early stopping
        if epochs_since_improvement >= args.es_patience and epoch >= 300:
            print(f"🛑 Early stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation
    print(f"\n📊 Training completed!")
    print(f"Average training time: {np.mean(training_times):.2f}s per epoch")
    print(f"Average validation time: {np.mean(validation_times):.2f}s per epoch")
    
    # Load best model and evaluate
    print(f"\n🏆 Loading best model and performing final evaluation...")
    trainer.model.load_state_dict(torch.load(save_path + "best_model.pth"))
    
    final_test_mae, horizon_metrics = evaluate_test_performance(trainer, dataloader, args, scaler)
    
    # Save final results
    final_results = {
        'avg_mae': np.mean(horizon_metrics['mae']),
        'avg_rmse': np.mean(horizon_metrics['rmse']),
        'avg_mape': np.mean(horizon_metrics['mape']),
        'avg_wmape': np.mean(horizon_metrics['wmape']),
        'horizon_mae': horizon_metrics['mae'],
        'horizon_rmse': horizon_metrics['rmse'],
        'horizon_mape': horizon_metrics['mape'],
        'horizon_wmape': horizon_metrics['wmape']
    }
    
    # Save results
    results_df = pd.DataFrame([final_results])
    results_df.to_csv(f"{save_path}/final_test_results.csv", index=False)
    
    print(f"\n🎉 MSMT training and evaluation completed!")
    print(f"Final Test MAE: {final_test_mae:.4f}")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")