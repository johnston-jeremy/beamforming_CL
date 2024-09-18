import torch as th
import numpy as np
import pickle as pkl
from baseline_mmse import compute_mmse_beamformer
class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4096, device=th.device("cuda:0" if th.cuda.is_available() else "cpu")):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.device = device
        self.basis_vectors, _ = th.linalg.qr(th.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=th.float, device=self.device))
        self.subspace_dim = 129
        self.num_env = num_env
        self.episode_length = episode_length
        # with open("./K8N8Samples=100.pkl", 'rb') as f:
        self.test_H = th.randn(100,self.K, self.N, dtype=th.cfloat, device=self.device)

    def reset(self, test=False, test_P = None):
        self.test = False

        if self.subspace_dim <= 2 * self.K * self.N:
            self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors)
        else:
            self.vec_H = th.randn(self.num_env, 2 * self.K * self.N, dtype=th.cfloat, device=self.device)
        if test:
            self.test = True
            self.test_P = test_P
            self.mat_H = self.test_H * np.sqrt(test_P)

        else:
            self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
            self.mat_H[:self.mat_H.shape[0] // 3] *= np.sqrt(10 ** 1)
            self.mat_H[self.mat_H.shape[0] // 3:2 * self.mat_H.shape[0] // 3] *= np.sqrt(10 ** 1.5)
            self.mat_H[2 * self.mat_H.shape[0] // 3:] *= np.sqrt(10 ** 2)
            self.mat_H = self.mat_H / np.sqrt(2)
        self.mat_W, _= compute_mmse_beamformer(self.mat_H,  K=self.K, N=self.N, P=self.P, noise_power=1, device=self.device)
        HW = th.bmm(self.mat_H, self.mat_W.transpose(-1, -2))
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W, self.P, HW)
    def step(self, action ):
        HW = th.bmm(self.mat_H, action.transpose(-1, -2))
        S = th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
        I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
        N = 1
        SINR = S/(I+N)
        self.reward =  th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
        self.mat_W = action.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W, self.P, HW.detach()), self.reward, self.done

    def generate_channel_batch(self, N, K, batch_size, subspace_dim, basis_vectors):
        coordinates = th.randn(batch_size, subspace_dim, 1, device=self.device)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
        vec_channel = th.bmm(basis_vectors_batch, coordinates).reshape(-1, 2 * K * N)
        return vec_channel
    def get_reward(self, H, W):
        HW = th.matmul(H, W.T)
        S = th.abs(HW.diag()) ** 2
        I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(HW.diag()) ** 2
        N = 1
        SINR = S / (I + N)
        reward = th.log2(1 + SINR).sum(dim=-1)
        return reward, HW
