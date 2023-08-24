import typing as tp
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class Color:
    """
    Colors enumerator
    """
    BLACK = (0, 0, 0)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)


JOINTS_NAMES = [
    'head',
    'right_upper_shoulder',
    'right_lower_shoulder',
    'right_upper_elbow',
    'right_lower_elbow',
    'right_upper_forearm',
    'right_lower_forearm',
    'right_wrist',
    'left_upper_shoulder',
    'left_lower_shoulder',
    'left_upper_elbow',
    'left_lower_elbow',
    'left_upper_forearm',
    'left_lower_forearm',
    'left_wrist',
    'base'
]

JOINTS_NAMES_ITOP = [
    'Head',
    'Neck',
    'R Shoulder',
    'L Shoulder',
    'R Elbow',
    'L Elbow',
    'R Hand',
    'L Hand',
    'Torso',
    'R Hip',
    'L Hip',
    'R Knee',
    'L Knee',
    'R Foot',
    'L Foot'
]


def random_blend_grid(true_blends, pred_blends):
    """Stacks predicted and ground truth blended images (heatmap+image) by column.

    Parameters
    ----------
    true_blends: np.array
        Ground truth blended images.
    pred_blends: np.array
        Predicted blended images.

    Returns
    -------
    grid: np.array
        Grid of predicted and ground truth blended images.
    """
    grid = []
    for i in range(0, len(true_blends)):
        grid.append(np.concatenate(true_blends[i], axis=2))
        grid.append(np.concatenate(pred_blends[i], axis=2))
    return grid


def to_colormap(heatmap_tensor, cmap='jet', cmap_range=(None, None)):
    """Converts a heatmap into an image assigning to the gaussian values a colormap.

    Parameters
    ----------
    heatmap_tensor: torch.Tensor
        Heatmap as a tensor (NxHxW).
    cmap: str
        Colormap to use for heatmap visualization.
    cmap_range: tuple
        Range of values for the colormap.

    Returns
    -------
    output: np.array
        Array of N images representing the heatmaps.
    """
    if not isinstance(heatmap_tensor, np.ndarray):
        try:
            heatmap_tensor = heatmap_tensor.to('cpu').numpy()
        except RuntimeError:
            heatmap_tensor = heatmap_tensor.detach().to('cpu').numpy()

    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])

    heatmap_tensor = np.max(heatmap_tensor, axis=1)

    output = []
    batch_size = heatmap_tensor.shape[0]
    for b in range(batch_size):
        rgb = cmap.to_rgba(heatmap_tensor[b])[:, :, :-1]
        output.append(rgb)

    output = np.asarray(output).astype(np.float32)
    output = output.transpose(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)

    return output


def get_keypoint_barplot(x_data, y_data, metric):
    """Computes a barplot given X and Y data.

    Parameters
    ----------
    x_data: np.array
        X data.
    y_data: np.array
        Y data.
    metric: str
        Metric type for graph title.

    Returns
    -------
    canvas: plot
        Barplot to visualize with matplotlib

    """
    fig, ax = plt.subplots()
    ax.bar(x=x_data, height=y_data)
    ax.set_ylim((0, 100))
    ax.set_xticks(x_data)
    ax.set_xticklabels([JOINTS_NAMES[idx] for idx in range(len(x_data))],
                       fontdict={'rotation': 'vertical'})
    ax.set_title(f'{metric} for each joint')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.4)
    fig.canvas.draw()
    canvas = np.array(fig.canvas.renderer._renderer)[:, :, :-1]
    plt.close()
    return canvas


chains_ixs = ([14, 13, 12, 11, 10, 9, 8,
               15,
               1, 2, 3, 4, 5, 6, 7],
              # wrist_l, low_forearm_l, up_forearm_l, low_elbow_l,
              # up_elbow_l, low_shoulder_l, up_shoulder_l,
              # base,
              # wrist_r, low_forearm_r, up_forearm_r, low_elbow_r,
              # up_elbow_r, low_shoulder_r, up_shoulder_r
              [0, 15],
              # base, head
              )

human_chains_ixs = (
    [0, 1, 8],
    [6, 4, 2, 1, 3, 5, 7],
    [8, 9, 11, 13],
    [8, 10, 12, 14]
)


def get_chain_dots(dots: np.ndarray, chain_dots_indexes: tp.List[int]) -> np.ndarray:  # chain of dots
    return dots[chain_dots_indexes]


def get_chains(dots: np.ndarray, arms_chain_ixs: tp.List[int], torso_chain_ixs: tp.List[int]):
    return (get_chain_dots(dots, arms_chain_ixs),
            get_chain_dots(dots, torso_chain_ixs))


def get_human_chains(dots: np.ndarray, torso_chain_ixs: tp.List[int], arms_chain_ixs: tp.List[int], rleg_chain_ixs: tp.List[int], lleg_chain_ixs: tp.List[int]):
    return (get_chain_dots(dots, arms_chain_ixs),
            get_chain_dots(dots, torso_chain_ixs),
            get_chain_dots(dots, rleg_chain_ixs),
            get_chain_dots(dots, lleg_chain_ixs))


def subplot_nodes(dots: np.ndarray, ax, c='red'):
    return ax.scatter3D(dots[:, 0], dots[:, 2], dots[:, 1], c=c)


def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax, c='greens'):
    return [ax.plot(chain[:, 0], chain[:, 2], chain[:, 1], c=c) for chain in chains]

def unravel_indices(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord
