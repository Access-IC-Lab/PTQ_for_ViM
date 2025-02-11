import torch
import numpy as np

def hosvd_decomposition(tensor):
    """
    Perform HOSVD-based rank-1 approximation on a 3D PyTorch tensor.
    
    Parameters:
        tensor (torch.Tensor): Input tensor of shape (C, L, D).
    
    Returns:
        r_C (torch.Tensor): Mode-1 singular vector of shape (C,).
        r_L (torch.Tensor): Mode-2 singular vector of shape (L,).
        r_D (torch.Tensor): Mode-3 singular vector of shape (D,).
        rank1_tensor (torch.Tensor): Reconstructed rank-1 tensor of the same shape as input.
    """
    # Step 1: Unfold tensor along each mode
    unfold_1 = tensor.reshape(tensor.shape[0], -1)  # Mode-1 unfolding
    unfold_2 = tensor.permute(1, 2, 0).reshape(tensor.shape[1], -1)  # Mode-2 unfolding
    unfold_3 = tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1)  # Mode-3 unfolding

    # Step 2: Perform SVD on each unfolded matrix
    u1, _, _ = torch.linalg.svd(unfold_1, full_matrices=False)  # Mode-1 singular vectors
    u2, _, _ = torch.linalg.svd(unfold_2, full_matrices=False)  # Mode-2 singular vectors
    u3, _, _ = torch.linalg.svd(unfold_3, full_matrices=False)  # Mode-3 singular vectors

    # Step 3: Extract the first singular vector from each mode
    r_C = u1[:, 0]  # Mode-1 singular vector
    r_L = u2[:, 0]  # Mode-2 singular vector
    r_D = u3[:, 0]  # Mode-3 singular vector

    # Step 4: Reconstruct the rank-1 tensor using the outer product
    # rank1_tensor = torch.ger(r_C, torch.ger(r_L, r_D).flatten()).reshape(tensor.shape)

    # return r_C, r_L, r_D, rank1_tensor
    return r_C, r_L, r_D


def cp_decomposition(H, max_iter=100, tol=1e-6):
    C, L, D = H.shape

    # 初始化向量
    s_C = torch.rand(C, dtype=H.dtype, device=H.device, requires_grad=False)
    s_L = torch.rand(L, dtype=H.dtype, device=H.device, requires_grad=False)
    s_D = torch.rand(D, dtype=H.dtype, device=H.device, requires_grad=False)

    for iteration in range(max_iter):
        # 保存前一輪的向量
        s_C_old, s_L_old, s_D_old = s_C.clone(), s_L.clone(), s_D.clone()

        # 更新 s_C
        S_LD = torch.einsum('j,k->jk', s_L, s_D)  # s_L 和 s_D 的外積
        s_C = torch.einsum('ijk,jk->i', H, S_LD)  # 計算縮減到 i 維
        s_C = s_C / torch.norm(s_C)

        # 更新 s_L
        S_CD = torch.einsum('i,k->ik', s_C, s_D)  # s_C 和 s_D 的外積
        s_L = torch.einsum('ijk,ik->j', H, S_CD)  # 計算縮減到 j 維
        s_L = s_L / torch.norm(s_L)

        # 更新 s_D
        S_CL = torch.einsum('i,j->ij', s_C, s_L)  # s_C 和 s_L 的外積
        s_D = torch.einsum('ijk,ij->k', H, S_CL)  # 計算縮減到 k 維
        s_D = s_D / torch.norm(s_D)

        # 檢查收斂
        diff = (
            torch.norm(s_C - s_C_old) +
            torch.norm(s_L - s_L_old) +
            torch.norm(s_D - s_D_old)
        )
        if diff < tol:
            break

    return s_C, s_L, s_D

def ml_decomposition(H, r_C_init, r_L_init, r_D_init, max_iter=300, learning_rate=0.01, verbose=True):
    """
    將三維張量 H 分解為三個一維向量的張量積 H^* = r_C ⊗ r_L ⊗ r_D，
    並最小化 H/H^* 的變異係數（CV）。

    參數:
        H (torch.Tensor): 原始三維張量，形狀為 (C, L, D)。
        max_iter (int): 最大迭代次數。
        learning_rate (float): 學習率。
        verbose (bool): 是否打印每隔一定步數的損失值。

    返回:
        r_C (torch.Tensor): C 維向量。
        r_L (torch.Tensor): L 維向量。
        r_D (torch.Tensor): D 維向量。
        loss_history (list): 每次迭代的損失值歷史。
    """
    torch.set_grad_enabled(True)
    torch.autograd.set_detect_anomaly(True)

    # 確保輸入是一個三維張量
    if len(H.shape) != 3:
        raise ValueError("Input tensor H must be three-dimensional (C, L, D).")

    # 獲取維度
    # C, L, D = H.shape

    # 初始化分解向量
    r_C = torch.tensor(r_C_init, dtype=torch.float32, requires_grad=True)
    r_L = torch.tensor(r_L_init, dtype=torch.float32, requires_grad=True)
    r_D = torch.tensor(r_D_init, dtype=torch.float32, requires_grad=True)
    # r_C = torch.rand(C, requires_grad=True)
    # r_L = torch.rand(L, requires_grad=True)
    # r_D = torch.rand(D, requires_grad=True)

    # 定義優化器
    # optimizer = torch.optim.Adam([r_C, r_L, r_D], lr=learning_rate)
    optimizer = torch.optim.Adam([r_C, r_L, r_D], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # 記錄損失
    # loss_history = []

    # 優化過程
    for epoch in range(max_iter):
        optimizer.zero_grad()

        # 計算 H^*
        H_star = torch.einsum('i,j,k->ijk', r_C, r_L, r_D)

        
        # 計算損失函數 （變異係數 (CV)）
        ratio = H / (H_star + 1e-8)
        mean_ratio = torch.mean(ratio)
        std_ratio = torch.std(ratio)
        loss = std_ratio / mean_ratio  # CV = std / mean

        # # 計算損失函數
        # lambda_ = (H * H_star).sum() / (H_star**2).sum()
        # loss = ((H - lambda_ * H_star)**2).sum() / (H**2).sum()

        # 記錄損失
        # loss_history.append(loss.item())

        # 反向傳播和更新參數
        loss.backward()
        optimizer.step()

        # 更新學習率
        scheduler.step(epoch + 1)

        # 打印損失值
        if verbose and epoch % (max_iter // 10) == 0:
            print(f"Epoch {epoch}/{max_iter}, Loss: {loss.item()}")

    # 返回結果
    # return r_C.detach(), r_L.detach(), r_D.detach(), loss_history
    return r_C.detach(), r_L.detach(), r_D.detach()

def ml_power(H, C_represent, L_represent, D_represent, max_iter=300, learning_rate=0.01, verbose=True):
    
    torch.set_grad_enabled(True)
    torch.autograd.set_detect_anomaly(True)

    # 確保輸入是一個三維張量
    if len(H.shape) != 3:
        raise ValueError("Input tensor H must be three-dimensional (C, L, D).")

    # 獲取維度
    C, L, D = H.shape

    # 初始化分解向量
    C_power = torch.rand(C, requires_grad=True)
    L_power = torch.rand(L, requires_grad=True)
    D_power = torch.rand(D, requires_grad=True)

    # 定義優化器
    # optimizer = torch.optim.Adam([r_C, r_L, r_D], lr=learning_rate)
    optimizer = torch.optim.Adam([C_power, L_power, D_power], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # 記錄損失
    # loss_history = []

    # 優化過程
    for epoch in range(max_iter):
        optimizer.zero_grad()

        # 計算 H^*
        H_star = torch.einsum('i,j,k->ijk', (C_represent ** C_power + 1e-15), (L_represent ** L_power + 1e-15), (D_represent ** D_power + 1e-15))

        
        # 計算損失函數 （變異係數 (CV)）
        ratio = H / (H_star + 1e-8)
        mean_ratio = torch.mean(ratio)
        std_ratio = torch.std(ratio)
        loss = std_ratio / mean_ratio  # CV = std / mean

        # # 計算損失函數
        # lambda_ = (H * H_star).sum() / (H_star**2).sum()
        # loss = ((H - lambda_ * H_star)**2).sum() / (H**2).sum()

        # 記錄損失
        # loss_history.append(loss.item())

        # 反向傳播和更新參數
        loss.backward()
        optimizer.step()

        # 更新學習率
        scheduler.step(epoch + 1)

        # 打印損失值
        if verbose and epoch % (max_iter // 10) == 0:
            print(f"Epoch {epoch}/{max_iter}, Loss: {loss.item()}")

    # 返回結果
    # return r_C.detach(), r_L.detach(), r_D.detach(), loss_history
    return C_power.detach(), L_power.detach(), D_power.detach()


# # Example usage
# if __name__ == "__main__":
#     # Create a random tensor of shape (C, L, D) in PyTorch
#     # tensor = torch.rand(2, 3, 4)  # Replace with your actual 
#     r_C_true = torch.rand(4)
#     r_L_true = torch.rand(5)
#     r_D_true = torch.rand(6)

#     tensor = torch.ger(r_C_true, torch.ger(r_L_true, r_D_true).flatten()).reshape(4, 5, 6)

#     # Perform HOSVD-based rank-1 approximation
#     r_C, r_L, r_D, rank1_tensor = hosvd_rank1_approximation(tensor)

#     # Output the results
#     print("Original tensor:", tensor)
#     print("r_C (Mode-1 singular vector):", r_C)
#     print("r_L (Mode-2 singular vector):", r_L)
#     print("r_D (Mode-3 singular vector):", r_D)
#     print("Rank-1 approximation tensor:", rank1_tensor)
#     print("Diff of tensors:", tensor / rank1_tensor)
#     print("Diff of tensors:", torch.norm(tensor / rank1_tensor))
