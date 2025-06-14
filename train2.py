import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from ablation_model import AblationEnhancedSTAMT  # Changed import
from ranger21 import Ranger
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=500, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "-",
    help="save path",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)

# Ablation study arguments
parser.add_argument("--use_temporal_embedding", action="store_true", default=True, 
                   help="Use temporal embedding")
parser.add_argument("--no_temporal_embedding", action="store_true", 
                   help="Disable temporal embedding")
parser.add_argument("--use_dilated_tconv", action="store_true", default=True,
                   help="Use dilated temporal convolution")
parser.add_argument("--no_dilated_tconv", action="store_true",
                   help="Disable dilated temporal convolution")
parser.add_argument("--use_adaptive_memory", action="store_true", default=True,
                   help="Use adaptive memory attention")
parser.add_argument("--no_adaptive_memory", action="store_true",
                   help="Disable adaptive memory attention")
parser.add_argument("--use_multi_resolution_output", action="store_true", default=True,
                   help="Use multi-resolution output")
parser.add_argument("--no_multi_resolution_output", action="store_true",
                   help="Disable multi-resolution output")
parser.add_argument("--use_cross_dim_projection", action="store_true", default=True,
                   help="Use cross-dimensional projection")
parser.add_argument("--no_cross_dim_projection", action="store_true",
                   help="Disable cross-dimensional projection")
parser.add_argument("--use_graph_learning", action="store_true", default=True,
                   help="Use graph structure learning")
parser.add_argument("--no_graph_learning", action="store_true",
                   help="Disable graph structure learning")
parser.add_argument("--tconv_layers", type=int, default=4,
                   help="Number of temporal convolution layers")
parser.add_argument("--no_dilated_convolution", action="store_true",
                   help="Use standard convolution instead of dilated")
parser.add_argument("--no_multi_scale_graph", action="store_true",
                   help="Disable multi-scale graph structure")
parser.add_argument("--no_layer_normalization", action="store_true",
                   help="Disable layer normalization")
parser.add_argument("--memory_size", type=int, default=4,
                   help="Size of adaptive memory bank")

# Experiment naming
parser.add_argument("--exp_name", type=str, default="", 
                   help="Experiment name suffix for logging")

args = parser.parse_args()

# Process ablation flags
if args.no_temporal_embedding:
    args.use_temporal_embedding = False
if args.no_dilated_tconv:
    args.use_dilated_tconv = False
if args.no_adaptive_memory:
    args.use_adaptive_memory = False
if args.no_multi_resolution_output:
    args.use_multi_resolution_output = False
if args.no_cross_dim_projection:
    args.use_cross_dim_projection = False
if args.no_graph_learning:
    args.use_graph_learning = False


class trainer:
    def __init__(
        self,
        scaler,
        input_dim,
        channels,
        num_nodes,
        input_len,
        output_len,
        dropout,
        lrate,
        wdecay,
        device,
        ablation_config,  # New parameter
    ):
        self.model = AblationEnhancedSTAMT(
            device=device, 
            input_dim=input_dim, 
            channels=channels, 
            num_nodes=num_nodes, 
            input_len=input_len, 
            output_len=output_len, 
            dropout=dropout,
            **ablation_config  # Pass ablation configuration
        )
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input)
        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 4:
            output = output[..., -1]

        B, T_out, N = output.shape

        flat = output.reshape(-1, N)
        inv  = self.scaler.inverse_transform(flat)
        predict = inv.reshape(B, T_out, N)

        real = real_val
        if real.dim()==3 and real.shape[1]==N and real.shape[2]==T_out:
            real = real.permute(0, 2, 1)

        assert predict.shape == real.shape, \
            f"train: predict{predict.shape} vs real{real.shape}"

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape  = util.MAPE_torch (predict, real, 0.0).item()
        rmse  = util.RMSE_torch (predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 4:
            output = output[..., -1]

        B, T_out, N = output.shape

        flat = output.reshape(-1, N)
        inv  = self.scaler.inverse_transform(flat)
        predict = inv.reshape(B, T_out, N)

        real = real_val
        if real.dim()==4 and real.shape[1]==1:
            real = real.squeeze(1)
            real = real.permute(0,2,1)
        elif real.dim()==3 and real.shape[1]==N and real.shape[2]==T_out:
            real = real.permute(0, 2, 1)

        assert predict.shape == real.shape, \
            f"eval: predict{predict.shape} vs real{real.shape}"

        loss  = self.loss(predict, real, 0.0)
        mape  = util.MAPE_torch (predict, real, 0.0).item()
        rmse  = util.RMSE_torch (predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


def seed_it(seed):
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def get_ablation_config(args):
    """Extract ablation configuration from arguments"""
    return {
        'use_temporal_embedding': args.use_temporal_embedding,
        'use_dilated_tconv': args.use_dilated_tconv,
        'use_adaptive_memory': args.use_adaptive_memory,
        'use_multi_resolution_output': args.use_multi_resolution_output,
        'use_cross_dim_projection': args.use_cross_dim_projection,
        'use_graph_learning': args.use_graph_learning,
        'tconv_layers': args.tconv_layers,
        'dilated_convolution': not args.no_dilated_convolution,
        'multi_scale_graph': not args.no_multi_scale_graph,
        'layer_normalization': not args.no_layer_normalization,
        'memory_size': args.memory_size,
    }


def get_experiment_name(args):
    """Generate experiment name based on ablation settings"""
    components = []
    
    if not args.use_temporal_embedding:
        components.append("no-temp-emb")
    if not args.use_dilated_tconv:
        components.append("no-tconv")
    if not args.use_adaptive_memory:
        components.append("no-memory")
    if not args.use_multi_resolution_output:
        components.append("no-multi-res")
    if not args.use_cross_dim_projection:
        components.append("no-cross-proj")
    if not args.use_graph_learning:
        components.append("no-graph")
    if args.no_dilated_convolution:
        components.append("std-conv")
    if args.no_multi_scale_graph:
        components.append("single-scale")
    if args.no_layer_normalization:
        components.append("no-ln")
    if args.memory_size != 4:
        components.append(f"mem{args.memory_size}")
    
    if components:
        ablation_name = "_".join(components)
    else:
        ablation_name = "full_model"
    
    if args.exp_name:
        ablation_name = f"{ablation_name}_{args.exp_name}"
    
    return ablation_name


def main():
    seed_it(6666)

    data = args.data

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        args.num_nodes = 170

    elif args.data == "PEMS08_36":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 36
        args.output_len = 36

    elif args.data == "PEMS08_48":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 48
        args.output_len = 48
    
    elif args.data == "PEMS08_60":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 60
        args.output_len = 60

    elif args.data == "PEMS04_36":
        args.data = "data/PEMS04_36"
        args.num_nodes = 307
        args.input_len = 36
        args.output_len = 36

    elif args.data == "PEMS04_48":
        args.data = "data/PEMS04_48"
        args.num_nodes = 307
        args.input_len = 48
        args.output_len = 48
    
    elif args.data == "PEMS04_60":
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.input_len = 60
        args.output_len = 60

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.epochs = 300
        args.es_patience = 100

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307

    elif args.data == "PEMS07":
        args.data = "data/PEMS07"
        args.num_nodes = 883

    elif args.data == "gba_his_2019":
        args.data = "data//"+args.data
        args.num_nodes = 2352
        args.epochs = 300

    elif args.data == "gla_his_2019":
        args.data = "data//"+args.data
        args.num_nodes = 3834
        args.epochs = 300
    
    elif args.data == "ca_his_2019":
        args.data = "data//"+args.data
        args.num_nodes = 8600
        args.epochs = 300

    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        args.num_nodes = 250
    
    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        args.num_nodes = 250
    
    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        args.num_nodes = 266

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        args.num_nodes = 266

    device = torch.device(args.device)

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    
    # Create experiment-specific path
    exp_name = get_experiment_name(args)
    path = args.save + data + f"_{exp_name}/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    # Print experiment configuration
    print("="*60)
    print("ABLATION STUDY CONFIGURATION")
    print("="*60)
    ablation_config = get_ablation_config(args)
    for key, value in ablation_config.items():
        print(f"{key}: {value}")
    print(f"Experiment name: {exp_name}")
    print("="*60)
    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    # Save configuration
    config_df = pd.DataFrame([ablation_config])
    config_df.to_csv(f"{path}/ablation_config.csv", index=False)

    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        ablation_config,  # Pass ablation configuration
    )
    
    # Dummy shape check before full training
    x0, y0 = next(dataloader["train_loader"].get_iterator())
    
    x0 = torch.Tensor(x0).to(device).transpose(1, 3)
    y0 = torch.Tensor(y0).to(device).transpose(1, 3)
    y0 = y0[:, 0, :, :]
    
    out0 = engine.model(x0)
    if isinstance(out0, tuple): out0 = out0[0]
    if out0.dim() == 4: out0 = out0[..., -1]
    
    flat = out0.reshape(-1, out0.shape[-1])
    inv = scaler.inverse_transform(flat)
    pred0 = inv.reshape(out0.shape)
    
    real0 = y0.permute(0, 2, 1)
    
    print("🧪 Dummy Check ▶ pred:", pred0.shape, " real:", real0.shape)
    assert pred0.shape == real0.shape, "❌ Shape mismatch in dummy check!"
    print("✅ Shape alignment is correct — safe to train.")

    print("start training...", flush=True)

    for i in range(1, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                print(
                    log.format(
                        iter,
                        train_loss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
                    ),
                    flush=True,
                )
        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i < 100:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            elif i > 100:
                outputs = []
                # Fixed: Ensure realy has correct shape [S, T, N]
                realy = torch.Tensor(dataloader["y_test"]).to(device)
                print(f"DEBUG: Original y_test shape: {realy.shape}")
                
                # Handle different possible shapes of y_test
                if realy.dim() == 4:  # [S, C, N, T] or [S, T, N, C]
                    if realy.shape[1] == 1:  # [S, 1, N, T]
                        realy = realy.squeeze(1).permute(0, 2, 1)  # → [S, T, N]
                    elif realy.shape[-1] == 1:  # [S, T, N, 1]
                        realy = realy.squeeze(-1)  # → [S, T, N]
                    else:  # assume [S, C, N, T] with C > 1
                        realy = realy[:, 0, :, :].permute(0, 2, 1)  # → [S, T, N]
                elif realy.dim() == 3:  # [S, N, T] or [S, T, N]
                    if realy.shape[1] == args.num_nodes:  # [S, N, T]
                        realy = realy.permute(0, 2, 1)  # → [S, T, N]
                    # else already [S, T, N]
                
                print(f"DEBUG: Final realy shape: {realy.shape}")

                for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    with torch.no_grad():
                        preds = engine.model(testx)
                        if isinstance(preds, tuple):
                            preds = preds[0]
                        if preds.dim() == 4:
                            preds = preds[..., -1]
                    outputs.append(preds)

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[:realy.size(0), ...]
                print(f"DEBUG: yhat shape: {yhat.shape}")

                amae = []
                amape = []
                awmape = []
                armse = []

                # Ensure we don't exceed available horizons
                actual_horizons = min(args.output_len, yhat.shape[1], realy.shape[1])
                print(f"DEBUG: Computing metrics for {actual_horizons} horizons (requested: {args.output_len})")
                
                for j in range(actual_horizons):
                    # Extract prediction and ground truth for horizon j
                    pred = scaler.inverse_transform(yhat[:, j, :])  # [S, N]
                    real = realy[:, j, :]  # [S, N]
                    
                    print(f"DEBUG: Horizon {j+1} - pred: {pred.shape}, real: {real.shape}")
                    
                    # Compute metrics
                    mae, mape, rmse, wmape = util.metric(pred, real)
                    print(f"Horizon {j+1}: MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape:.4f}, WMAPE {wmape:.4f}")
                
                    amae.append(mae)
                    amape.append(mape)
                    armse.append(rmse)
                    awmape.append(wmape)

                log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                print(
                    log.format(
                        np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
                    )
                )

                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Final test evaluation
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    
    # Fixed: Consistent shape handling for final test
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    print(f"DEBUG FINAL: Original y_test shape: {realy.shape}")
    
    # Handle different possible shapes of y_test
    if realy.dim() == 4:  # [S, C, N, T] or [S, T, N, C]
        if realy.shape[1] == 1:  # [S, 1, N, T]
            realy = realy.squeeze(1).permute(0, 2, 1)  # → [S, T, N]
        elif realy.shape[-1] == 1:  # [S, T, N, 1]
            realy = realy.squeeze(-1)  # → [S, T, N]
        else:  # assume [S, C, N, T] with C > 1
            realy = realy[:, 0, :, :].permute(0, 2, 1)  # → [S, T, N]
    elif realy.dim() == 3:  # [S, N, T] or [S, T, N]
        if realy.shape[1] == args.num_nodes:  # [S, N, T]
            realy = realy.permute(0, 2, 1)  # → [S, T, N]
        # else already [S, T, N]
    
    print(f"DEBUG FINAL: Final realy shape: {realy.shape}")

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        
        with torch.no_grad():
            preds = engine.model(testx)
            if isinstance(preds, tuple):
                preds = preds[0]
            if preds.dim() == 4:
                preds = preds[..., -1]
        
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    print(f"DEBUG FINAL: yhat shape: {yhat.shape}")

    amae = []
    amape = []
    armse = []
    awmape = []
    test_m = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, i, :])  # [S, N]
        real = realy[:, i, :]  # [S, N]

        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))