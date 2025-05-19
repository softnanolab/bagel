import argparse
import numpy as np
import matplotlib.pyplot as plt

################################## C O N S T A N T S ##################################
MAX_PAE = 31.0


################################## F U N C T I O N S ##################################
def compute_i_pae(pae_matrix: np.ndarray, binder_length: int) -> float:
    """
    Compute the standard interface PAE. Assume binder is the last chain, always.
    """
    target_indices = list(np.arange(pae_matrix.shape[0] - binder_length))
    binder_indices = list(np.arange(binder_length) + (pae_matrix.shape[0] - binder_length))

    pae1_to_2 = pae_matrix[target_indices, :][:, binder_indices].mean()
    pae2_to_1 = pae_matrix[binder_indices, :][:, target_indices].mean()
    return float((pae1_to_2 + pae2_to_1) / 2) / MAX_PAE


def compute_per_res_min_pae(pae_matrix: np.ndarray, binder_length: int) -> np.ndarray:
    """
    Compute the per-residue minimum PAE.
    """
    target_indices = list(np.arange(pae_matrix.shape[0] - binder_length))
    binder_indices = list(np.arange(binder_length) + (pae_matrix.shape[0] - binder_length))

    # For each binder residue, find the minimum PAE from the corresponding target residue
    per_res_min_pae = np.zeros(binder_length)
    for i in range(binder_length):
        min_pae_1 = np.min(pae_matrix[binder_indices[i], target_indices]) / MAX_PAE
        min_pae_2 = np.min(pae_matrix[target_indices, binder_indices[i]]) / MAX_PAE
        per_res_min_pae[i] = float(min(min_pae_1, min_pae_2))
    return per_res_min_pae

################################## P L O T T I N G ##################################
def plot_pae_matrix(pae_matrix: np.ndarray, binder_length: int = 5) -> None:
    """
    Plot the PAE matrix. Binder is the last chain.
    """

    # Normalize PAE matrix to [0, 1]
    pae_matrix = pae_matrix / MAX_PAE * 100
    total_res = pae_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pae_matrix, cmap="RdYlBu_r")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("PAE (Normalised to 100)", rotation=-90, va="bottom")

    # Add labels to each cell
    text_args = {"ha": "center", "va": "center", "color": "w", "fontsize": 12}
    for i in range(pae_matrix.shape[0]):
        for j in range(pae_matrix.shape[1]):
            ax.text(j, i, f"{pae_matrix[i, j]:2.0f}", **text_args)

    # Add vertical line to separate binder from target
    ax.axvline(total_res - binder_length - 0.5, color="black", linewidth=2)
    ax.axhline(total_res - binder_length - 0.5, color="black", linewidth=2)

    # Add x label "Target" to total_res-binder_length
    # Add second x label "Binder" to total_res-binder_length
    # Put it where the x axis would be
    text_args["color"] = "k"
    ax.text((total_res - binder_length) /2 - 0.5, total_res*1.05, "Target", **text_args)
    ax.text(total_res - 0.5 - binder_length / 2, total_res*1.05, "Binder", **text_args)

    text_args["ha"] = "right"
    ax.text(total_res*-0.125, (total_res - binder_length) / 2 - 0.5, "Target", **text_args)
    ax.text(total_res*-0.125, total_res - binder_length / 2 - 0.5, "Binder", **text_args)

    plt.title(f"PAE matrix", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"pae_matrix.png", dpi=300)
    print(f"Saved PAE matrix to pae_matrix.png")


################################## M A I N ##################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pae_matrix", type=str, required=True)
    parser.add_argument("--binder_length", type=int, required=True)
    args = parser.parse_args()

    pae_matrix = np.load(args.pae_matrix) # (1, NRES, NRES)
    pae_matrix = pae_matrix[0] # (NRES, NRES)

    print(pae_matrix.shape)

    i_pae = compute_i_pae(pae_matrix, args.binder_length)
    per_res_min_pae = compute_per_res_min_pae(pae_matrix, args.binder_length)

    plot_pae_matrix(pae_matrix)

    print(f"I-PAE: {i_pae}")
    print(f"Per-residue minimum PAE: {per_res_min_pae * 100}")
