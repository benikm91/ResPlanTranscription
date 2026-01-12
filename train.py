from itertools import takewhile
import os
from typing import Counter
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import NODE_CLASS_MAPPING, EDGE_CLASS_MAPPING, ResPlanDataset, StaticPlanRenderer
from torch.utils.data import DataLoader
import numpy as np
from model import Sequence2SequenceModel
from torchmetrics import MeanMetric
from tqdm import tqdm
import logging
import networkx as nx


log = logging.getLogger(__name__)


def collate_fn(batch, pad_to, padding_value):
    imgs, graphs = zip(*batch)
    imgs = np.stack(imgs, axis=0)
    nodes = []
    edges = []
    for n, e in graphs:
        assert n.shape[0] <= pad_to, f"Too many nodes! {n.shape[0]}"
        assert e.shape[0] <= pad_to, f"Too many edges! {e.shape[0]}"
        n = np.pad(n, (0, pad_to - n.shape[0]), constant_values=padding_value)
        e = np.pad(e, ((0, pad_to - e.shape[0]), (0, 0)), constant_values=padding_value)
        nodes.append(n)
        edges.append(e)
    nodes = np.stack(nodes, axis=0)
    edges = np.stack(edges, axis=0)
    nodes = torch.from_numpy(nodes)
    edges = torch.from_numpy(edges)
    imgs = torch.from_numpy(imgs) / 255.0
    return imgs, (nodes, edges)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    
    num_nodes_classes = len(NODE_CLASS_MAPPING)
    pad_to = cfg.training.pad_to
    padding_value = cfg.training.padding_value

    collate_wrapper = lambda batch: collate_fn(batch, pad_to, padding_value)
    num_workers = os.cpu_count() // 2

    train_dataset = ResPlanDataset(data_path=cfg.data.path, renderer=StaticPlanRenderer(dpi=cfg.data.renderer_dpi), graph_linearization=cfg.data.graph_linearization)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_wrapper, pin_memory=True, num_workers=num_workers, persistent_workers=False)
    test_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_wrapper, pin_memory=True, num_workers=num_workers, persistent_workers=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = Sequence2SequenceModel(
        num_layers=cfg.model.num_layers,
        num_nodes=pad_to,
        num_edges=pad_to,
        num_nodes_classes=num_nodes_classes,
        num_edges_classes=cfg.model.num_edges_classes,
        vit_config=OmegaConf.to_container(cfg.model.vit, resolve=True),
        transformer_config=OmegaConf.to_container(cfg.model.transformer, resolve=True)
    ).to(device)

    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=padding_value).to(device)

    def calc_loss(nodes_pred, edges_type_pred, edges_from_pred, edges_to_pred, nodes, edges):
        nodes_loss = cross_entropy(nodes_pred.view(-1, num_nodes_classes), nodes.view(-1))
        edges_type_loss = cross_entropy(edges_type_pred.view(-1, cfg.model.num_edges_classes), edges[:, :, 0].view(-1))
        edges_from_loss = cross_entropy(edges_from_pred.view(-1, pad_to), edges[:, :, 1].view(-1))
        edges_to_loss = cross_entropy(edges_to_pred.view(-1, pad_to), edges[:, :, 2].view(-1))
        total_loss = nodes_loss + edges_type_loss + edges_from_loss + edges_to_loss
        return total_loss

    def select_nodes(nodes, nodes_pred, eos_id: int):
        nodes = list(takewhile(lambda x: x != eos_id, nodes.tolist()))
        nodes_pred = list(map(lambda x: x.item(), nodes_pred.argmax(dim=-1)))
        nodes_pred = list(takewhile(lambda x: x != eos_id, nodes_pred))
        return nodes, nodes_pred

    def num_nodes_correct(nodes, nodes_pred, eos_id: int) -> bool:
        nodes, nodes_pred = select_nodes(nodes, nodes_pred, eos_id)
        # print("#Nodes", len(nodes_pred), len(nodes))
        return len(nodes_pred) == len(nodes)
    
    def nodes_correct(nodes, nodes_pred, eos_id: int) -> bool:
        nodes, nodes_pred = select_nodes(nodes, nodes_pred, eos_id)
        if len(nodes_pred) != len(nodes):
            return False
        # same number of nodes per type
        nodes_counter = Counter(nodes)
        nodes_pred_counter = Counter(nodes_pred)
        # print(nodes_counter, nodes_pred_counter)
        return nodes_counter == nodes_pred_counter
    
    def select_edes(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id: int):
        edges = list(takewhile(lambda x: x[0] != eos_id, edges.tolist()))
        edges_type_pred = list(map(lambda x: x.item(), edges_type_pred.argmax(dim=-1)))
        edges_type_pred = list(takewhile(lambda x: x != eos_id, edges_type_pred))
        edges_from_pred = list(map(lambda x: x.item(), edges_from_pred.argmax(dim=-1)))
        edges_from_pred = edges_from_pred[:len(edges_type_pred)]
        edges_to_pred = list(map(lambda x: x.item(), edges_to_pred.argmax(dim=-1)))
        edges_to_pred = edges_to_pred[:len(edges_type_pred)]
        return tuple(zip(*edges)), (edges_type_pred, edges_from_pred, edges_to_pred)
    
    def num_edges_correct(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id: int) -> bool:
        (edges_type, edges_from, edges_to), (edges_type_pred, edges_from_pred, edges_to_pred) = select_edes(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id)
        # print("#Edges", len(edges_type_pred), len(edges_type))
        if len(edges_type_pred) != len(edges_type):
            assert len(edges_from_pred) != len(edges_from) and len(edges_to_pred) != len(edges_to)
            return False
        else:
            assert len(edges_from_pred) == len(edges_type_pred) and len(edges_to_pred) == len(edges_type_pred)
            return True

    def edges_type_correct(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id: int) -> bool:
        if not num_edges_correct(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id):
            return False
        (edges_type, edges_from, edges_to), (edges_type_pred, edges_from_pred, edges_to_pred) = select_edes(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id)
        
        edges_type_counter = Counter(edges_type)
        edges_type_pred_counter = Counter(edges_type_pred)
        return edges_type_counter == edges_type_pred_counter

    def create_graph(nodes, edges_type, edges_from, edges_to):
        G = nx.DiGraph()
        for i, node in enumerate(nodes):
            G.add_node(i, type=node)
        for etype, efrom, eto in zip(edges_type, edges_from, edges_to):
            G.add_edge(efrom, eto, type=etype)
        return G

    def transcription_correct(nodes, edges, nodes_pred, edges_type_pred, edges_from_pred, edges_to_pred, eos_id_node: int, eos_id_edge: int) -> bool:
        if not nodes_correct(nodes, nodes_pred, eos_id_node):
            return False
        
        if not edges_type_correct(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id_edge):
            return False
        
        nodes, nodes_pred = select_nodes(nodes, nodes_pred, eos_id_node)
        (edges_type, edges_from, edges_to), (edges_type_pred, edges_from_pred, edges_to_pred) = select_edes(edges, edges_type_pred, edges_from_pred, edges_to_pred, eos_id_edge)

        gt_graph = create_graph(nodes, edges_type, edges_from, edges_to)
        pred_graph = create_graph(nodes_pred, edges_type_pred, edges_from_pred, edges_to_pred)

        return nx.is_isomorphic(gt_graph, pred_graph)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    for epoch in range(cfg.training.num_epochs):
        avg_loss = MeanMetric().to(device)
        for img, (nodes, edges) in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            img = img.to(device=device, dtype=torch.float32)
            nodes = nodes.to(device=device, dtype=torch.long)
            edges = edges.to(device=device, dtype=torch.long)

            nodes_pred, (edges_type_pred, edges_from_pred, edges_to_pred) = model(img)

            loss = calc_loss(nodes_pred, edges_type_pred, edges_from_pred, edges_to_pred, nodes, edges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item())
        log.info(f"Epoch {epoch} done | Avg Loss: {avg_loss.compute()}")

        num_nodes_acc = MeanMetric().to(device)
        nodes_acc = MeanMetric().to(device)
        num_edges_acc = MeanMetric().to(device)
        edges_type_acc = MeanMetric().to(device)
        transcription_acc = MeanMetric().to(device)
        for img, (nodes, edges) in tqdm(test_dataloader, desc=f"Eval Epoch {epoch}"):
            img = img.to(device=device, dtype=torch.float32)
            nodes = nodes.to(device=device, dtype=torch.long)
            edges = edges.to(device=device, dtype=torch.long)

            nodes_pred, (edges_type_pred, edges_from_pred, edges_to_pred) = model(img)

            for i in range(img.size(0)):
                num_nodes_acc.update(float(num_nodes_correct(nodes[i], nodes_pred[i], eos_id=NODE_CLASS_MAPPING['<EOS>'])))
                nodes_acc.update(float(nodes_correct(nodes[i], nodes_pred[i], eos_id=NODE_CLASS_MAPPING['<EOS>'])))
                num_edges_acc.update(float(num_edges_correct(edges[i], edges_type_pred[i], edges_from_pred[i], edges_to_pred[i], eos_id=EDGE_CLASS_MAPPING['<EOS>'])))
                edges_type_acc.update(float(edges_type_correct(edges[i], edges_type_pred[i], edges_from_pred[i], edges_to_pred[i], eos_id=EDGE_CLASS_MAPPING['<EOS>'])))
                transcription_acc.update(float(transcription_correct(
                    nodes[0], edges[0],
                    nodes_pred[0], edges_type_pred[0], edges_from_pred[0], edges_to_pred[0],
                    eos_id_node=NODE_CLASS_MAPPING['<EOS>'],
                    eos_id_edge=EDGE_CLASS_MAPPING['<EOS>'],    
                )))

        log.info(f"""
        Eval Epoch {epoch} done
        ---
        Num Nodes Acc: {num_nodes_acc.compute()}
        Nodes Acc: {nodes_acc.compute()}
        Num Edges Acc: {num_edges_acc.compute()}
        Edges Type Acc: {edges_type_acc.compute()}
        ---
        Trans. Acc: {transcription_acc.compute()}
        ---
        """)

    log.info("DONE")

if __name__ == "__main__":
    main()