import networkx as nx
import matplotlib.pyplot as plt
import os
import math
import constants


def plot_graph(G: nx.classes.Graph, max_largest_components: int = 64, name: str = "Graph", with_labels: bool = False, block: bool = False, font_size: int = 6, plots_dir: str = os.path.join(constants.OUTPUT_DIR, "plots")):
    """Plots a Graph both as a whole and its max_largest_components
    and saves it to plots_dir
    """
    plt.figure(1)
    plt.suptitle(f"{name}")
    nx.drawing.draw_networkx(G, with_labels=with_labels,
                             node_size=30, font_size=font_size)
    figure_file = os.path.join(plots_dir, f"{name}_plot.png")
    plt.savefig(figure_file)

    largest_components = sorted(
        nx.connected_components(G), key=len, reverse=True)[:max_largest_components]

    plt.figure(2)
    plt.suptitle(
        f"{name}: {max_largest_components} largest Connected Components")
    n = len(largest_components)
    root = math.ceil(math.sqrt(len(largest_components)))
    for i, component in enumerate(largest_components):
        plt.subplot(root, root, i+1)
        H = G.subgraph(component)
        nx.drawing.draw_networkx(H, with_labels=with_labels,
                                 node_size=10, font_size=font_size)

    figure_file = os.path.join(
        plots_dir, f"{name}_connected_components_plot.png")
    plt.savefig(figure_file)
    plt.show(block=block)
