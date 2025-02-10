import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np


# def SSMStep(h, deltaA, deltaB_x, C, step):
#     h = deltaB_x[:, :, step] + deltaA[:, :, step] * h
#     y = torch.matmul(h, C[:, :, step].unsqueeze(-1)).squeeze(-1) # B, d_inner

#     # plt.savefig(f"intermediate_layer/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/hidden_state/" + str(layer_idx).rjust(2, '0') + "_channel_max.png")
    
#     return h, y


# def SSM(x, deltaA, deltaB, C, D):
#     """
#     x     : B, d_inner, L          = B, 384, 197
#     deltaA: B, d_inner, L, d_state = B, 384, 197, 16
#     deltaB: B, d_inner, L, d_state = B, 384, 197, 16
#     C     : B, d_state, L          = B, 16, 197
#     D     : d_inner                = 384
#     """

#     L = x.shape[2]

#     deltaB_x = deltaB * x.unsqueeze(-1) # B, d_inner, L, d_state
#     h = 0

#     ys = []
#     for i in range(L):
#         # h = deltaB_x[:, :, i] + deltaA[:, :, i] * h

#         # y = torch.matmul(h, C[:, :, i].unsqueeze(-1)).squeeze(-1) # B, d_inner

#         h, y = SSMStep(h, deltaA, deltaB_x, C, i)

#         ys.append(y)
    
#     y = torch.stack(ys, dim=-1) # B, d_inner, L

#     out = y + x * D.unsqueeze(-1) # B, d_inner, L

#     return out

class SequentialSSM(nn.Module):
    def __init__(
        self,
        layer_idx,
        direction,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.direction = direction

    def ssm_step(self, h, deltaA, deltaB_x, C, step):
        h = deltaB_x[:, :, step] + deltaA[:, :, step] * h
        y = torch.matmul(h, C[:, :, step].unsqueeze(-1)).squeeze(-1) # B, d_inner

        # print(f"max: {h.max()}, min: {h.min()}")

        # h_plot = h.transpose(1, 2).abs().amax(0).cpu().detach().numpy()
        # fig = plt.figure()
        # ax = plt.subplot(projection='3d')
        # plot_x = np.arange(0, 384, 1)
        # plot_y = np.arange(0, 16, 1)
        # plot_x, plot_y = np.meshgrid(plot_x, plot_y)
        # plot_z = h_plot
        # surf = ax.plot_surface(plot_x, plot_y, plot_z, cmap=cm.coolwarm)
        # # ax.set_zlim(0, 10)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # fig.colorbar(surf)
        # # ax.plot(x, y, z)
        # # plt.savefig(f"test/test.png")
        # plt.savefig(f"intermediate_layer/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/hidden_state_flip/" + str(self.layer_idx).rjust(2, '0') + "_" + str(step).rjust(3, '0') + ".png")
        # plt.close()
        
        return h, y

    def ssm(self, x, deltaA, deltaB, C, D):
        """
        x     : B, d_inner, L          = B, 384, 197
        deltaA: B, d_inner, L, d_state = B, 384, 197, 16
        deltaB: B, d_inner, L, d_state = B, 384, 197, 16
        C     : B, d_state, L          = B, 16, 197
        D     : d_inner                = 384
        """

        # print(self.layer_idx)
        # print(deltaA.shape)
        # print(deltaB.shape)

        # if self.layer_idx > 22:
        #     for i in range(197):
        #         deltaA_plot = deltaA[:, :, i].transpose(1, 2).abs().amax(0).cpu().detach().numpy()
        #         fig = plt.figure()
        #         ax = plt.subplot(projection='3d')
        #         plot_x = np.arange(0, 384, 1)
        #         plot_y = np.arange(0, 16, 1)
        #         plot_x, plot_y = np.meshgrid(plot_x, plot_y)
        #         plot_z = deltaA_plot
        #         surf = ax.plot_surface(plot_x, plot_y, plot_z, cmap=cm.coolwarm)
        #         ax.zaxis.set_major_locator(LinearLocator(10))
        #         fig.colorbar(surf)
        #         plt.savefig(f"intermediate_layer/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/deltaA/" + str(self.layer_idx).rjust(2, '0') + "_" + str(i).rjust(3, '0') + ".png")
        #         plt.close()

        #         deltaB_plot = deltaB[:, :, i].transpose(1, 2).abs().amax(0).cpu().detach().numpy()
        #         fig = plt.figure()
        #         ax = plt.subplot(projection='3d')
        #         plot_x = np.arange(0, 384, 1)
        #         plot_y = np.arange(0, 16, 1)
        #         plot_x, plot_y = np.meshgrid(plot_x, plot_y)
        #         plot_z = deltaB_plot
        #         surf = ax.plot_surface(plot_x, plot_y, plot_z, cmap=cm.coolwarm)
        #         ax.zaxis.set_major_locator(LinearLocator(10))
        #         fig.colorbar(surf)
        #         plt.savefig(f"intermediate_layer/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/deltaB/" + str(self.layer_idx).rjust(2, '0') + "_" + str(i).rjust(3, '0') + ".png")
        #         plt.close()

        # print("-----------------")

        L = x.shape[2]

        deltaB_x = deltaB * x.unsqueeze(-1) # B, d_inner, L, d_state
        h = 0
        h_sim = 0

        ys = []
        h_record = torch.zeros(deltaB_x.shape) # --
        h_sim_record = torch.zeros(deltaB_x.shape) # --
        for i in range(L):

            h, y = self.ssm_step(h, deltaA, deltaB_x, C, i)
            h_sim, _ = self.ssm_step(h_sim, deltaA, deltaB_x, C, i) # --
            s = h.abs().max() / 127 + 1e-15
            h_sim = (h_sim / s).round_().clamp_(-128, 127).mul_(s)
            h_record[:, :, i] = h # --
            h_sim_record[:, :, i] = h_sim # --

            ys.append(y)
        
        # print(h_record.shape)
        # print(h_sim_record.shape)
        i = 8
        a = h_record[i].mean((0, 2)).cpu().detach().numpy()
        b = h_sim_record[i].mean((0, 2)).cpu().detach().numpy()

        # plt.figure(figsize=(8, 3))
        # plt.figure()
        fig, ax1 = plt.subplots(figsize=(6, 3.2))
        ax2 = ax1.twinx()
        ax1.plot(a, label = 'h')
        ax1.plot(b, label = 'h_q')
        ax2.plot(np.abs(a - b), color='r', linewidth=2, label = 'diff')
        # plt.plot(h_record[3].mean((0, 2)).cpu().detach().numpy())
        # plt.plot(h_sim_record[3].mean((0, 2)).cpu().detach().numpy())
        ax1.set_xlabel("Recurrent Step")
        ax1.set_ylabel("Value")
        ax2.set_ylabel("Difference")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc = "lower right")
        # ax1.legend(["h", "h_q"], loc="lower right")
        # ax2.legend(["diff"], loc="lower right")
        # plt.legend(["h", "h_q", "diff"])
        plt.tight_layout()
        plt.savefig("figures/ssm.png")
        plt.close()

        exit()

        # --
        # h_record = h_record.abs().mean(0)
        # C_represent = h_record.mean((1, 2))
        # L_represent = h_record.mean((0, 2))
        # D_represent = h_record.mean((0, 1))
        # C_trend = C_represent.std()
        # L_trend = L_represent.std()
        # D_trend = D_represent.std()
        # C_power = C_trend / (C_trend + L_trend + D_trend)
        # L_power = L_trend / (C_trend + L_trend + D_trend)
        # D_power = D_trend / (C_trend + L_trend + D_trend)
        # r_C = (C_represent ** C_power + 1e-15).cpu().detach().numpy()
        # r_L = (L_represent ** L_power + 1e-15).cpu().detach().numpy()
        # r_D = (D_represent ** D_power + 1e-15).cpu().detach().numpy()

        # np.savetxt(f'./small_factor/r_C/{self.layer_idx}_{self.direction}.txt', r_C, fmt='%10.10f')
        # np.savetxt(f'./small_factor/r_L/{self.layer_idx}_{self.direction}.txt', r_L, fmt='%10.10f')
        # np.savetxt(f'./small_factor/r_D/{self.layer_idx}_{self.direction}.txt', r_D, fmt='%10.10f')


        y = torch.stack(ys, dim=-1) # B, d_inner, L

        out = y + x * D.unsqueeze(-1) # B, d_inner, L

        return out

    def forward(self, x, deltaA, deltaB, C, D):
        # print(self.layer_idx)
        return self.ssm(x, deltaA, deltaB, C, D)