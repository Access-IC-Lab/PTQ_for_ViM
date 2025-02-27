import torch


def kmeans(tensor, k, num_iters=100, dim=-1):
    """
    Example:

    If a = torch.tensor([100, 1, 2, 3, 200, 110, 210]) and self.k = 3

    Then _kmeans(a) = [1, 0, 0, 0, 2, 1, 2]
    """

    indices = torch.randperm(tensor.size(dim))[:k]  # shape: K
    if dim == -1:
        dim = tensor.dim() - 1
    tensor = tensor.abs().mean(torch.arange(tensor.dim())[torch.arange(tensor.dim()) != dim].tolist())
    # tensor = tensor.abs().log2().mean(torch.arange(tensor.dim())[torch.arange(tensor.dim()) != dim].tolist())
    centroids = tensor[indices]

    for _ in range(num_iters):
        distances = torch.abs(tensor.unsqueeze(-1) - centroids.unsqueeze(0))  # D,K
        labels = torch.argmin(distances, dim=-1)  # D
        new_centroids = torch.stack([tensor[labels == i].mean() for i in range(k)])

        if torch.allclose(centroids, new_centroids, rtol=1e-4):
            break

        centroids = new_centroids

    sorted_centroids, sorted_indices = torch.sort(centroids)

    new_labels = torch.zeros_like(labels)
    for new_idx, old_idx in enumerate(sorted_indices):
        new_labels[labels == old_idx] = new_idx

    return new_labels


def geometric_kmeans(tensor, k, num_iters=100, dim=-1):
    """
    Example:

    If a = torch.tensor([100, 1, 2, 3, 200, 110, 210]) and self.k = 3

    Then _kmeans(a) = [1, 0, 0, 0, 2, 1, 2]
    """
    # randomly select k indices from the target dimension
    indices = torch.randperm(tensor.size(dim))[:k]  # shape: K

    if dim == -1:
        dim = tensor.dim() - 1

    # take the mean of the tensor along all dimensions except the target dimension
    tensor = tensor.abs().mean(torch.arange(tensor.dim())[torch.arange(tensor.dim()) != dim].tolist())
    centroids = tensor[indices]

    for _ in range(num_iters):
        distances = tensor.unsqueeze(-1) / centroids.unsqueeze(0)
        distances[distances < 1] = 1 / distances[distances < 1]
        labels = torch.argmin(distances, dim=-1)  # D
        new_centroids = torch.stack([tensor[labels == i].prod().pow(1.0 / tensor[labels == i].numel()) for i in range(k)])

        if torch.allclose(centroids, new_centroids, rtol=1e-4):
            break

        centroids = new_centroids

    sorted_centroids, sorted_indices = torch.sort(centroids)

    new_labels = torch.zeros_like(labels)
    for new_idx, old_idx in enumerate(sorted_indices):
        new_labels[labels == old_idx] = new_idx

    return new_labels


# a = torch.tensor([[0.9, 1.1], [3, 5], [90, 110], [300, 500], [9000, 11000], [30000, 50000]], dtype=torch.float32)
# print(geometric_kmeans(a, k=3, dim=0))
