from torch import nn
import torch
import math




class ViTPatching(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def positional_encoding_2d(self, d_model, height, width):
        """
        Create the 2D positional encoding for the input image.
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe.unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)  # B, embed_dim, H/patch_size, W/patch_size
        x = x + self.positional_encoding_2d(x.size(1), x.size(2), x.size(3)).to(x.device)
        x = x.flatten(2)  # B, embed_dim, num_patches
        x = x.transpose(1, 2)  # B, num_patches, embed_dim
        return x


class Sequence2SequenceModel(nn.Module):

    """
    seq2seq model treating images as sequences of pixels and graphs as sequences of nodes and edges.
    Input: batch of images of shape (B, C, H, W)
    Output: batch of graphs represented as (nodes, edges) where nodes is of shape (B, 64, C) and edges is of shape (B, 64, 2)

    Encoder is a Vision Transformer (ViT) that encodes the image into a sequence of embeddings.
    Decoder is a Transformer that decodes the sequence of embeddings into a sequence of nodes and edges.
    """

    def __init__(self,
                 num_layers: int,
                 *,
                 num_nodes: int,
                 num_edges: int,
                 num_nodes_classes: int,
                 num_edges_classes: int,
                 vit_config: dict,
                 transformer_config: dict):
        super().__init__()
        self.patching = ViTPatching(**vit_config)

        d_model = transformer_config['d_model']

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_config['nhead'],
                dim_feedforward=transformer_config['dim_feedforward'],
                dropout=transformer_config['dropout'],
                activation=transformer_config['activation']
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=transformer_config['nhead'],
                dim_feedforward=transformer_config['dim_feedforward'],
                dropout=transformer_config['dropout'],
                activation=transformer_config['activation']
            ),
            num_layers=num_layers,
        )
        self.context_embed = nn.Embedding(num_nodes + (num_edges * 3), d_model)
        self.node_head = nn.Linear(d_model, num_nodes_classes)
        self.edges_type_head = nn.Linear(d_model, num_edges_classes)
        self.edge_head = nn.Linear(d_model, num_nodes)

    def forward(self, img):
        x = self.patching(img)
        x = x.transpose(0, 1)  # num_patches, B, embed_dim
        x = self.encoder(x)
        # The target for the decoder is a learned embedding for each possible output token (node or edge part)
        # We repeat it for each item in the batch.
        tgt = self.context_embed.weight.unsqueeze(0).repeat(x.size(1), 1, 1)  # B, num_context, embed_dim
        tgt = tgt.transpose(0, 1)  # num_context, B, embed_dim
        tgt = self.decoder(tgt, x)
        tgt = tgt.transpose(0, 1)  # B, num_context, embed_dim
        nodes_emb = tgt[:, :64, :]  # B, num_nodes, embed_dim
        nodes = self.node_head(nodes_emb)  # B, num_nodes, num_node_classes
        edges_types_emb = tgt[:, 64::3, :]  # B, num_edges, embed_dim
        edges_from_emb = tgt[:, 65::3, :]  # B, num_edges, embed_dim
        edges_to_emb = tgt[:, 66::3, :]  # B, num_edges, embed_dim
        edges_type = self.edges_type_head(edges_types_emb)  # B, num_edges, 2
        edges_from = self.edge_head(edges_from_emb)  # B, num_edges, 64
        edges_to = self.edge_head(edges_to_emb)  # B, num_edges, 64
        return nodes, (edges_type, edges_from, edges_to)

        