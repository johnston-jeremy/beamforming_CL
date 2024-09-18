import os
import torch as th
from net_mimo import Policy_Net_MIMO
from env_mimo import MIMOEnv
from tqdm import tqdm
import sys
def train_curriculum_learning(policy_net_mimo, optimizer, save_path, device, K=4, N=4, P=10, noise_power=1, num_epochs=400000,
                    num_epochs_per_subspace=1200, num_epochs_to_save_model=10000, episode_length=6):
    env_mimo = MIMOEnv(K=K, N=N, P=P, noise_power=noise_power, device=device, num_env=4096, episode_length=episode_length)
    pbar = tqdm(range(num_epochs))
    sum_rate = th.zeros(100, env_mimo.episode_length, 5)
    test_P = [10 ** 0.5, 10 ** 1, 10 ** 1.5, 10 ** 2, 10 ** 2.5]
    
    
    
    T = 6
    for epoch in pbar:
        state = env_mimo.reset()
        obj_H = 0
        reward_list = []
        
        while(1):
            action = policy_net_mimo(state)
            next_state, reward, done = env_mimo.step(action)
            reward_list.append(reward.mean())
            state = next_state
            if done:
                break
        
        for i in range(T):
            obj_H += reward_list[i]
        
        obj_H *= -1.0
        optimizer.zero_grad()
        obj_H.backward()
        optimizer.step()
        
        
        
        
        if (epoch+1) % num_epochs_to_save_model == 0:
            th.save(policy_net_mimo.state_dict(), save_path + f"{epoch}.pth")
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_mimo.subspace_dim <= 2 * K * N:
            env_mimo.subspace_dim +=1
        if (epoch) % 20 == 0:
            with th.no_grad():
                for i_p in range(5):
                    state = env_mimo.reset(test=True, test_P = test_P[i_p])
                    while(1):
                        action = policy_net_mimo(state)
                        next_state, reward, done = env_mimo.step(action)
                        sum_rate[:, env_mimo.num_steps-1, i_p] = reward.squeeze()
                        state = next_state
                        if done:
                            break

                pbar.set_description(f"id: {epoch}|SNR=5:{sum_rate[:, :, 0].max(dim=1)[0].mean():.6f}|SNR=10:{sum_rate[:, :, 1].max(dim=1)[0].mean():.6f}|SNR15:{sum_rate[:, :, 2].max(dim=1)[0].mean():.6f}|SNR=20: {sum_rate[:, :, 3].max(dim=1)[0].mean():.6f}|SNR=25: {sum_rate[:, :, 4].max(dim=1)[0].mean():.6f}|training_loss: {obj_H.mean().item() / env_mimo.episode_length:.6f}|memory: {th.cuda.memory_allocated():3d}")

def get_cwd(env_name):
    file_list = os.listdir()
    if env_name not in file_list:
        os.mkdir(env_name)
    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/"

if __name__  == "__main__":
    N = 4   # number of antennas
    K = 4   # number of users
    SNR = 20
    P = 10 ** (SNR / 10)
    noise_power = 1
    learning_rate = 5e-5
    cwd = f"RANDOM_H_CL_REINFORCE_N{N}K{K}SNR{SNR}"
    config = {
        'method': 'REINFORCE',
    }
    env_name = f"RANDOM_N{N}K{K}SNR{SNR}_mimo_beamforming"
    save_path = get_cwd(env_name)
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim=1024, K=K, N=N, P=P).to(device)
    optimizer = th.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    try:
        train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path, device=device, P=P, noise_power=noise_power, episode_length=6)
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        print(f"saved at " + save_path)
    except KeyboardInterrupt:
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        print(f"saved at " + save_path)
        exit()


