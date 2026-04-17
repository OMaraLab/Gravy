import os
import pickle

import dgl
import dgl.data
import dgl.dataloading
import dgl.function as dfn
import torch
import torch.optim as optim
import torch.nn as tnn
import torch.nn.parallel as tnn_parallel
from torch.distributed import init_process_group, destroy_process_group


class graphDataset(dgl.data.DGLDataset):
    def __init__(self, name, path, url=None):
        self.path = path
        super().__init__(name=name, url=url)

    def process(self):
        graphs, self.molIDs = dgl.load_graphs(self.path)
        assert len(graphs) == len(
            self.molIDs["names"]
        ), "Mismatch in number of graphs and molIDs"
        self._num_graphs = len(graphs)
        self.graphs = None
        del graphs

    def __getitem__(self, idx):
        g, _ = dgl.load_graphs(self.path, [int(idx)])
        return g[0]

    def __len__(self):
        return self._num_graphs


class messagePassingLayer(tnn.Module):
    def __init__(
        self,
        in_feats_node,
        in_feats_edge,
        out_feats_node,
        aggregator_type="pool",
        dropout_rate=0.3,
    ):
        super().__init__()
        self.aggregator_type = aggregator_type

        self.W_msg = tnn.Linear(in_feats_node + in_feats_edge, out_feats_node)
        self.W_self = tnn.Linear(in_feats_node, out_feats_node)
        self.W_concat = tnn.Linear(out_feats_node * 2, out_feats_node)
        self.node_dropout = tnn.Dropout(dropout_rate)

        if self.aggregator_type == "mean":
            self.reduce_func = dfn.mean("m", "h_neigh")
        elif self.aggregator_type == "sum":
            self.reduce_func = dfn.sum("m", "h_neigh")
        elif self.aggregator_type == "max":
            self.reduce_func = dfn.max("m", "h_neigh")
        elif self.aggregator_type == "pool":
            self.mp_pool = tnn.Sequential(
                tnn.Linear(out_feats_node, out_feats_node), tnn.ReLU()
            )
            self.reduce_func = dfn.mean("m", "h_neigh")

    def forward(self, graph, node_features, edge_features):
        with graph.local_scope():
            graph.ndata["h"] = node_features
            graph.edata["e"] = edge_features

            def message_func(edges):
                combined_features = torch.cat([edges.src["h"], edges.data["e"]], dim=1)
                msg = self.W_msg(combined_features)
                if self.aggregator_type == "pool":
                    msg_for_pool = self.mp_pool(msg)
                    return {"m": msg_for_pool}
                return {"m": msg}

            graph.update_all(message_func, self.reduce_func)
            h_neigh = graph.ndata["h_neigh"]
            h_self = self.W_self(graph.ndata["h"])
            h_combined = torch.cat([h_self, h_neigh], dim=1)
            output_node_features = self.node_dropout(
                tnn.functional.relu(self.W_concat(h_combined))
            )
            # L2 normalization
            # output_node_features = tnn.functional.normalize(output_node_features, p=2, dim=1)

            return output_node_features


class edgeFeatureSAGEConv(tnn.Module):
    def __init__(
        self,
        in_feats_node,
        in_feats_edge,
        hidden_feats_node,
        hidden_feats_edge,
        num_gnn_layers,
        aggregator_type="pool",
        dropout_rate=0.3,
    ):
        super().__init__()

        self.embedding_node = tnn.Linear(in_feats_node, hidden_feats_node)
        self.embedding_edge = tnn.Linear(in_feats_edge, hidden_feats_edge)

        self.gnn_layers = tnn.ModuleList()
        for layer in range(num_gnn_layers):
            self.gnn_layers.append(
                messagePassingLayer(
                    hidden_feats_node,
                    hidden_feats_edge,
                    hidden_feats_node,
                    aggregator_type,
                    dropout_rate,
                )
            )

        # predictor_input_dim = hidden_feats_node * 2
        predictor_input_dim = hidden_feats_node

        self.edge_predictor_mlp = tnn.Sequential(
            tnn.Linear(predictor_input_dim, predictor_input_dim // 2),
            tnn.ReLU(),
            tnn.Dropout(dropout_rate),
            tnn.Linear(predictor_input_dim // 2, predictor_input_dim // 4),
            tnn.ReLU(),
            tnn.Dropout(dropout_rate),
            tnn.Linear(predictor_input_dim // 4, 1),
        )

    def forward(self, graph, node_features, edge_features):
        with graph.local_scope():
            h = self.embedding_node(node_features)
            e = self.embedding_edge(edge_features)

            for layer in self.gnn_layers:
                h = layer(graph, h, e)
            graph.ndata["h_out"] = h

            def edge_score_func(edges):
                # combined_node_features = torch.cat(
                #     [edges.src["h_out"], edges.dst["h_out"]], dim=1
                # )
                combined_node_features = edges.src["h_out"] * edges.dst["h_out"]
                score = self.edge_predictor_mlp(combined_node_features)
                return {"score": score}

            graph.apply_edges(edge_score_func)

            return graph.edata["score"]


def get_dataloaders(dataset, seed, batch_size, shuffle=True):
    train_set, val_set, test_set = dgl.data.split_dataset(
        dataset, frac_list=[0.8, 0, 0.2], shuffle=shuffle, random_state=seed
    )
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, use_ddp=True, batch_size=batch_size, shuffle=True
    )
    val_loader = dgl.dataloading.GraphDataLoader(val_set, batch_size=batch_size)
    test_loader = dgl.dataloading.GraphDataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def k_fold_split(dataset, k):
    import numpy as np

    indices = np.arange(len(dataset))
    return np.array_split(indices, k)


def init_model(graph, seed, device, load_path=None):
    epoch_start = 0
    torch.manual_seed(seed)
    model = edgeFeatureSAGEConv(
        graph.ndata["h"].shape[1], graph.edata["e"].shape[1], 512, 64, 2, "pool"
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if load_path:
        map_location = {"cuda:0": str(device)}
        checkpoint = torch.load(load_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = int(checkpoint["epoch"])
    if device.type == "cpu":
        model = tnn_parallel.DistributedDataParallel(model)
    else:
        model = tnn_parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    return model, optimizer, epoch_start


def evaluate(model, dataloader, device, percentage_error=False):
    model.eval()
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        batched_graph = batch.to(device)
        batched_score = batched_graph.edata["score"].to(device)
        node_feats = batched_graph.ndata["h"].to(device)
        edge_feats = batched_graph.edata["e"].to(device)
        with torch.no_grad():
            predicted_scores = model(batched_graph, node_feats, edge_feats)
            loss = tnn.functional.l1_loss(predicted_scores[:, 0], batched_score)
            if percentage_error:
                loss = loss / batched_score.abs().mean()
        total_loss += loss.cpu().item()
        num_batches += 1
    return total_loss / num_batches


def save_model(epoch, model, optimizer, loss, dataset_name, best=False, fold=None):
    os.makedirs("checkpoints", exist_ok=True)
    epoch = str(epoch)
    write_path_prefix = f"{dataset_name}_{epoch}"
    if best:
        write_path_prefix = f"{dataset_name}_best"
    if fold:
        write_path_prefix += f"_fold_{fold}"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"checkpoints/{write_path_prefix}.pt",
    )


def main(
    rank,
    world_size,
    dataset,
    seed,
    total_epoch,
    patience,
    save_dataset_name=None,
    save_freq=0,
    load_path=None,
    min_delta=100,
    k=None,
    k_fold_indices=None,
):
    assert bool(k) == bool(
        k_fold_indices
    ), "If k is specified, k_fold_indices must be provided"
    for x in k_fold_indices if k_fold_indices else []:
        assert x < k, "k_fold_indices must be less than k"
    backend = "nccl" if world_size > 1 else "gloo"
    init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    train_loader, val_loader, test_loader = None, None, None
    if not k:
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset, seed, batch_size=128
        )
    else:
        splits = k_fold_split(dataset, k=k)

    for fold in k_fold_indices if k_fold_indices else range(1):
        best_test_loss = float("inf")
        patience_counter = 0
        patience_limit = patience
        model, optimizer, epoch_start = init_model(dataset[0], seed, device, load_path)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        print(f"Starting fold {fold + 1}/{k if k else 1}")
        if k_fold_indices:
            test_set = dgl.data.utils.Subset(dataset, splits[fold])
            test_loader = dgl.dataloading.GraphDataLoader(test_set, batch_size=128)
            train_set = dgl.data.utils.Subset(dataset, [x for xs in splits for x in xs])
            train_loader = dgl.dataloading.GraphDataLoader(
                train_set, use_ddp=True, batch_size=128, shuffle=True
            )

        for epoch in range(total_epoch):
            model.train()
            train_loader.set_epoch(epoch + epoch_start)
            total_loss = 0
            num_batches = 0
            for batch in train_loader:
                batched_graph = batch.to(device)
                batched_score = batched_graph.edata["score"].to(device)
                node_feats = batched_graph.ndata["h"].to(device)
                edge_feats = batched_graph.edata["e"].to(device)

                optimizer.zero_grad()
                predicted_scores = model(batched_graph, node_feats, edge_feats)
                loss = tnn.functional.l1_loss(predicted_scores[:, 0], batched_score)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                num_batches += 1

            if rank == 0:
                print(
                    f"Epoch: {epoch_start + epoch + 1}/{epoch_start + total_epoch}, Training loss: {total_loss / num_batches:.4f}"
                )
                if save_freq and (epoch + 1) % save_freq == 0:
                    save_model(
                        epoch + 1,
                        model,
                        optimizer,
                        total_loss / num_batches,
                        save_dataset_name,
                        fold=f"{fold}_of_{k}" if k else None,
                    )


            # eval

            with torch.no_grad():
                test_loss = torch.tensor(
                    evaluate(model, test_loader, device), device=device
                )
                test_loss_percent = torch.tensor(
                    evaluate(model, test_loader, device, percentage_error=True),
                    device=device,
                )
                if world_size > 1:
                    torch.distributed.reduce(
                        test_loss, dst=0, op=torch.distributed.ReduceOp.AVG
                    )
                    torch.distributed.reduce(
                        test_loss_percent,
                        dst=0,
                        op=torch.distributed.ReduceOp.AVG,
                    )
                scheduler.step(test_loss)

            # early stopping

            stop_flag = torch.zeros(1, dtype=torch.bool).to(device)
            if rank == 0:
                print(
                    f"Epoch: {epoch_start + epoch + 1}/{epoch_start + total_epoch}, Test loss: {test_loss:.4f}/{test_loss_percent*100}%"
                )
                if test_loss < best_test_loss - min_delta:
                    best_test_loss = test_loss
                    save_model(
                        epoch_start + epoch + 1,
                        model,
                        optimizer,
                        best_test_loss,
                        save_dataset_name,
                        best=True,
                        fold=f"{fold}_of_{k}" if k else None,
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        print(
                            f"Early stopping at epoch {epoch_start + epoch + 1}, best test loss: {best_test_loss:.4f}/{test_loss_percent*100}%"
                        )
                        if rank == 0:
                            save_model(
                                epoch_start + epoch + 1,
                                model,
                                optimizer,
                                best_test_loss,
                                save_dataset_name,
                                best=True,
                                fold=f"{fold}_of_{k}" if k else None,
                            )
                        stop_flag[0] = True

            if world_size > 1:
                torch.distributed.broadcast(stop_flag, src=0)
            if stop_flag.item():
                print(f"Process {rank} stopping early")
                break

        if rank == 0:
            save_model(
                "final",
                model,
                optimizer,
                best_test_loss,
                save_dataset_name,
                fold=f"{fold}_of_{k}" if k else None,
            )
        with torch.no_grad():
            train_loss = evaluate(model, train_loader, device)
            # val_loss = evaluate(model, val_loader, device)
            test_loss = evaluate(model, test_loader, device)
            test_loss_percent = evaluate(
                model, test_loader, device, percentage_error=True
            )

        print(
            f"for fold: {fold}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}/{test_loss_percent*100}%"
        )

        torch.distributed.barrier()
