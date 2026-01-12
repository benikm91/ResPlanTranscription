from torch.utils import data
import numpy as np
import pickle
from typing import Any
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, GeometryCollection, base, box
)
import geopandas as gpd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


NODE_CLASS_MAPPING = {
    "living": 0,
    "bedroom": 1,
    "bathroom": 2,
    "kitchen": 3,
    "door": 4,
    "window": 5,
    "wall": 6,
    "front_door": 7,
    "balcony": 8,
    "<EOS>": 9
}

EDGE_CLASS_MAPPING = {
    'edge': 0,
    '<EOS>': 1
}


class PlanRenderer:

    def render(
            self,
            plan: dict[str, Any],
    ) -> np.ndarray:
        raise NotImplementedError


class ResPlanDataset(data.Dataset):

    def __init__(self, data_path, renderer: PlanRenderer, graph_linearization: str):
        def normalize_keys(plan: dict[str, Any]) -> None:
            if "balacony" in plan and "balcony" not in plan:
                plan["balcony"] = plan.pop("balacony")
        with open(data_path, 'rb') as f:
            self.plans = pickle.load(f)
        for p in self.plans:
            normalize_keys(p)
        self.renderer = renderer
        self.graph_linearization = graph_linearization

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, idx):
        def encode_node(x: str) -> int:
            res = None
            for key, value in NODE_CLASS_MAPPING.items():
                if x.startswith(key):
                    assert res is None, f"Ambiguous node value: {x}, {res} and {key}"
                    res = value
            assert res is not None, f"Unknown node value: {x}"
            return res
        
        def encode_edge(edge_type: str) -> int:
             match edge_type:
                case '<EOS>':
                    return EDGE_CLASS_MAPPING['<EOS>']
                case _:
                    return EDGE_CLASS_MAPPING['edge']

        plan = self.plans[idx]
        img_np = self.renderer.render(plan)
        img_tensor = np.transpose(img_np, (2, 0, 1))  # C x H x W
        nodes = [encode_node(node) for node in (list(plan["graph"].nodes))]
        nodes_graph = list(plan["graph"].nodes)
        if self.graph_linearization == 'random':
            # shuffle nodes and node_graph (for positions)
            cup = list(zip(nodes, nodes_graph))
            np.random.shuffle(cup)
            nodes, nodes_graph = zip(*cup)
            nodes, nodes_graph = list(nodes), list(nodes_graph)
        else:
            assert self.graph_linearization == 'sorted'
        nodes_pos = {key: value for value, key in enumerate(nodes_graph)}
        nodes = nodes + [encode_node('<EOS>')]
        edges = [(encode_edge(edge[2]['type']), nodes_pos[edge[0]], nodes_pos[edge[1]]) for edge in plan["graph"].edges(data=True)]
        if self.graph_linearization == 'random':
            np.random.shuffle(edges)
        else:
            assert self.graph_linearization == 'sorted'
        edges = edges + [(encode_edge('<EOS>'), -1, -1)]
        nodes_tensor = np.array(list(nodes), dtype=np.int64)
        edges_tensor = np.array(edges, dtype=np.int64)
        return img_tensor, (nodes_tensor, edges_tensor)


class StaticPlanRenderer(PlanRenderer):

    def __init__(self, dpi=100, figsize=(8, 8)) -> None:
        super().__init__()
        self.dpi = dpi
        self.figsize = figsize


    def render(
            self,
            plan: dict[str, Any],
    ) -> np.ndarray:

        def get_geometries(geom_data: Any) -> list[Any]:
            """Safely extract individual geometries from single/multi/collections."""
            if geom_data is None:
                return []
            if isinstance(geom_data, (Polygon, LineString, Point)):
                return [] if geom_data.is_empty else [geom_data]
            if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
                return [g for g in geom_data.geoms if g is not None and not g.is_empty]
            return []

        colors = {
            "living": "#d9d9d9",     # light gray
            "bedroom": "#66c2a5",    # greenish
            "bathroom": "#fc8d62",   # orange
            "kitchen": "#8da0cb",    # blue
            "door": "#e78ac3",       # pink
            "window": "#a6d854",     # lime
            "wall": "#ffd92f",       # yellow
            "front_door": "#a63603", # dark reddish-brown
            "balcony": "#b3b3b3"     # dark gray
        }

        categories = ["living","bedroom","bathroom","kitchen","door","window","wall","front_door","balcony"]

        geoms, color_list = [], []
        for key in categories:
            geom = plan.get(key)
            if geom is None:
                continue
            parts = get_geometries(geom)
            if not parts:
                continue
            geoms.extend(parts)
            color_list.extend([colors.get(key, "#000000")] * len(parts))

        assert len(geoms) > 0, "No geometries to plot."


        assert len(geoms) > 0, "No geometries to plot."

        # --- Use Matplotlib's OO API (process-safe) ---
        # 1. Create Figure and Canvas objects, NOT using plt
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ---

        # Your plotting logic remains the same
        gseries = gpd.GeoSeries(geoms)
        gseries.plot(ax=ax, color=color_list, edgecolor="black", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        
        # 2. Use fig.tight_layout(), NOT plt.tight_layout()
        fig.tight_layout()

        # 3. Convert plot to NumPy array using the canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_np = np.asarray(buf)
        
        # 4. No need for plt.close(fig), fig is a local object
        #    that will be garbage collected.

        return img_np[:, :, :3]  # Discard alpha channel
