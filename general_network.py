"""
@author: Eric Armbruster, Borislav Polovnikov, Thomas Huber
"""

import torch
import os
import json 
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib.animation import FuncAnimation
from lava.lib.dl.slayer.neuron import rf  # Resonate and Fire
from SNN_visualizer import SNN_visualizer
from datetime import datetime
from tqdm import tqdm


class RfNeuron(rf.Neuron):
    
    def __init__(self, threshold, period, decay, persistent_state=True, shared_param=False):
        super().__init__(threshold, period, decay, persistent_state=persistent_state, shared_param=shared_param)
        self.debug = True
       
    def passthrough(self, real_in: np.ndarray, imag_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        quantized_real_weight = self.quantize_8bit(real_in)
        quantized_imag_weight = self.quantize_8bit(imag_in)
        
        real_1_out, imag_1_out = self.dynamics((
            quantized_real_weight * real_in,
            quantized_imag_weight * imag_in
        ))

        spike_1 = self.spike(real_1_out, imag_1_out)
        
        return spike_1, real_1_out, imag_1_out

class Network: 
    
    def __init__(self, first_layer: RfNeuron, input_layer_size: int, device: str = 'cpu'):
        self.device = device
        self.layers: List[RfNeuron] = [first_layer]
        self.layer_sizes: List[int] = [input_layer_size]
        self.weights: List[np.ndarray] = []
        
    
    def add_layer(self, layer: RfNeuron, layer_size: int):      
        self.layers.append(layer)
        self.layer_sizes.append(layer_size)
        
        ### Make a dummy passthrough through the layer to instantiate the neuron such that it 
        ### knows its size and decay/frequency/threshold are set
        dummy_real_in = torch.tensor(np.ones((1,layer_size,1))).to(self.device)
        dummy_imag_in = torch.tensor(np.ones((1,layer_size,1))).to(self.device)
        self.layers[-1].passthrough(dummy_real_in, dummy_imag_in)
        
        #self.weights.append(np.random.random((4, self.layer_sizes[-2], self.layer_sizes[-1])))  ##random weights
        self.weights.append(np.ones((4, self.layer_sizes[-2], self.layer_sizes[-1]))) ##all weights set to one
          
    
    def forward(self, real_in: np.ndarray, imag_in: np.ndarray):
        self.layer_data_spikes: List[np.ndarray] = []
        self.layer_data_reals: List[np.ndarray] = []
        self.layer_data_imags: List[np.ndarray] = []
        
        self.layer_data_reals.append(real_in)
        self.layer_data_imags.append(imag_in)
        
        for i, layer in enumerate(self.layers):
            # rf rf calculation
            if i == 0:
                spike_1, real1_out, imag1_out = layer.passthrough(real_in, imag_in)
            else: 
                real_in, imag_in = self.run_forward_helper(i, layer)    
                spike_1, real1_out, imag1_out = layer.passthrough(real_in, imag_in)
            
            # store data for visualization
            self.layer_data_reals.append(real1_out)
            self.layer_data_imags.append(imag1_out)
            self.layer_data_spikes.append(spike_1)
            
        # remove input values again as we don't need them
        self.layer_data_reals.pop(0)
        self.layer_data_imags.pop(0)


    def run_forward_helper(self, i, layer):
        f = self.layers[i-1].frequency
        w = 2*np.pi*f
                
        N = self.layer_sizes[i-1] # number of neurons in previous layer
        M = self.layer_sizes[i] # number of neurons in current layer
        
        a_1 = np.zeros(N)
        a_2 = np.ones(N)*w*self.layers[i-1].decay
        a_3 = np.ones(N)*pow(w,2)/a_2
        a_4 = np.zeros(N)
                
        u1_real = a_2/2
        u1_imag = -a_2*a_4/(2*w)
        u2_real = np.zeros(N)
        u2_imag = a_2 * a_3 /(2*w)
        u3_real =  np.zeros(N)
        u3_imag = u2_imag
        u4_real = a_3 / 2
        u4_imag = -a_3*a_4 / (2*w)
                
        U_real = np.array([np.atleast_2d(u1_real).T, np.atleast_2d(u2_real).T, np.atleast_2d(u3_real).T, np.atleast_2d(u4_real).T]) # [4, N, 1]
        U_imag = np.array([np.atleast_2d(u1_imag).T, np.atleast_2d(u2_imag).T, np.atleast_2d(u3_imag).T, np.atleast_2d(u4_imag).T]) # [4, N, 1]
                
        C_real = np.sum(np.repeat(U_real, repeats=M, axis=2) * self.weights[i-1], axis=0) # [N, M]
        C_imag = np.sum(np.repeat(U_imag, repeats=M, axis=2) * self.weights[i-1], axis=0) # [N, M]
                
        real_out_previous = self.layer_data_reals[-1] # [B, N, t]
        imag_out_previous = self.layer_data_imags[-1] # [B, N, t]
                
        real_in = np.matmul(C_real.transpose(), real_out_previous) - np.matmul(C_imag.transpose(), imag_out_previous) # [B,M,t]
        imag_in = np.matmul(C_real.transpose(), imag_out_previous) + np.matmul(C_imag.transpose(), real_out_previous) # [B,M,t]
        
        
        ### for normalization
        N_time = self.layer_data_reals[i].shape[-1]
        real_in = np.array(real_in) # [B, M, t]
        imag_in = np.array(imag_in) # [B, M, t]
        
        real_max = np.abs(real_in).max(axis=2) # [B,t]
        imag_max = np.abs(imag_in).max(axis=2) # [B,t]
        
        real_in = real_in / np.repeat(real_max[:,:,np.newaxis],repeats=N_time,axis=2)
        imag_in = imag_in / np.repeat(imag_max[:,:,np.newaxis],repeats=N_time,axis=2)
        
        return torch.tensor(real_in).to(self.device), torch.tensor(imag_in).to(self.device)
        
        
    def run_learning_rule(self, b=0.1, beta=np.array([0.1, 0.1, 0.1, 0.1])):
        # N = number of neurons in current layer
        for i, S in enumerate(self.weights): # S = [4,N,M]
            N_time = self.layer_data_reals[i].shape[-1]
            N_batches = self.layer_data_reals[i].shape[0]
            N = N_time*N_batches
            
            real1_out = self.layer_data_reals[i] # [B, N, t]
            real2_out = self.layer_data_reals[i+1] # [B, N, t]
            imag1_out = self.layer_data_imags[i] # [B, N, t]
            imag2_out = self.layer_data_imags[i+1] # [B, N, t]
            
            real1_real2= beta[0]*np.tensordot(real1_out, real2_out, axes=([0,2],[0,2]))/N 
            real1_imag2= beta[1]*np.tensordot(real1_out, imag2_out, axes=([0,2],[0,2]))/N
            imag1_real2= beta[2]*np.tensordot(imag1_out, real2_out, axes=([0,2],[0,2]))/N
            imag1_imag2= beta[3]*np.tensordot(imag1_out, imag2_out, axes=([0,2],[0,2]))/N
            
            S[0] = (1-b)*S[0] + real1_real2        
            S[1] = (1-b)*S[1] + real1_imag2
            S[2] = (1-b)*S[2] + imag1_real2
            S[3] = (1-b)*S[3] + imag1_imag2

def store_hyperparams(network: Network, dir):
    hyperparams = {'b': b, 'beta': beta.tolist(), 'threshold': threshold, 'decay': decay, 'period': period}
    
    with open(f'{dir}/hyperparams.json', 'w') as file:
        json.dump(hyperparams, file)

def store(network: Network, spike_input, dir):
    store_hyperparams(network, dir)
    
    np.save(f'{dir}/spike_input.npy', np.array(spike_input))
    
    for i in range(len(network.layers)):
        layername = f'layer{i}_neurons{network.layer_sizes[i]}_'
        
        np.save(f'{dir}/{layername}_reals.npy',network.layer_data_reals[i])
        np.save(f'{dir}/{layername}_imags.npy',network.layer_data_imags[i])
        np.save(f'{dir}/{layername}_spikes.npy',network.layer_data_spikes[i])
        if i != len(network.layers) - 1:
            np.save(f'{dir}/{layername}_weights.npy', network.weights[i])
    

def plot_time_trace(dir, network, layers=[0], neurons=[0], title_prefix=''):
    fig = plt.figure(figsize=(16, 10), label=title_prefix)
    N_subplots=len(neurons)
    axes=[]
    loc_max=0
    for i, layer in enumerate(layers):
        axes.append(fig.add_subplot(N_subplots,1,i+1))
        local_layer= network.layers[layer]
        period = round(1/local_layer.frequency[neurons[i]],1)
        decay = round(float(local_layer.decay[neurons[i]]),3)
        axes[-1].set_title(f'{title_prefix}: Layer {layer+1}, neuron {neurons[i]+1} with period = {period} and decay = {decay}')
        
        axes[-1].plot(network.layer_data_reals[layer][0,neurons[i],:], label='real')
        axes[-1].plot(network.layer_data_imags[layer][0,neurons[i],:], label='imag')
        axes[-1].plot(network.layer_data_spikes[layer][0,neurons[i],:], label='spikes')
        axes[-1].legend()
        if i!=0:
            axes[-1].sharex(axes[-2])
            axes[-1].sharey(axes[-1])
        loc_max=max(loc_max, np.abs(network.layer_data_reals[layer][0,neurons[i],:]).max(), 
                    np.abs(network.layer_data_imags[layer][0,neurons[i],:]).max(),
                    np.abs(network.layer_data_spikes[layer][0,neurons[i],:]).max())
    for ax in axes:
        ax.set_ylim(-loc_max, loc_max)
    plt.tight_layout()
    hyperparams = {'b': b, 'beta': beta.tolist(), 'threshold': threshold, 'decay': decay, 'period': period}
    
    figname = '_'.join(str(key) + '-' + str(value) for key, value in hyperparams.items()) 
    fig.savefig(f'{dir}/{title_prefix}_{figname}.png')

def get_train_input(device):
    time = torch.FloatTensor(np.arange(200)).to(device)

    # expand to (batch, neuron, time) tensor
    spike_input = torch.autograd.Variable(
        torch.zeros([1, 1, len(time)]), requires_grad=False
    ).to(device)

    # random Input
    # spike_input.data[..., np.random.randint(spike_input.shape[-1], size=5)] = 1

    # periodic Input
    # per_test = 10
    # spike_input.data[..., np.arange(0,spike_input.shape[-1],period)] = 1
    
    ### Input from data
    import os
    os.chdir("pattern_recognition_resonators")
    input_data = np.load('datasets/2.5k-cleanedup.npy')[:200].reshape(1,1,-1)
    spike_input = torch.tensor(input_data).to(device)
    
    s = torch.tensor([1, 0, 0, 0])
    o = torch.tensor([0, 0, 0, 0])
    space = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0]*2)

    S = torch.concatenate([s, s, s])
    O = torch.concatenate([o, o, o])

    SOS = torch.concatenate([S, O, S, space])
    spike_input = SOS.repeat(6)
    
    spike_input = spike_input[None, None, :]
    spike_input = torch.autograd.Variable(spike_input).to(device)

    # spike_input = torch.autograd.Variable(
    #     torch.zeros([1, 1, len(time)]), requires_grad=False
    # ).to(device)
    # period1 = 10
    
    # spike_input.data[..., np.arange(0, spike_input.shape[-1], period1)] = 1
    # spike_input.data[..., np.arange(5, spike_input.shape[-1], period1)] = 1
    # spike_input.data[..., np.arange(8, spike_input.shape[-1], period1)] = 1
    
    return spike_input


def get_test_input(device):
    time = torch.FloatTensor(np.arange(200)).to(device)

    test_input = torch.autograd.Variable(
        torch.zeros([1, 1, len(time)]), requires_grad=False
    ).to(device)
    period1 = 10
    
    test_input.data[..., np.arange(0, test_input.shape[-1], period1)] = 1
    
    return test_input
    

def setup_network_arch():
    first_layer = RfNeuron(threshold, [4, 50], decay, persistent_state=True, shared_param=False).to(device) 
    second_layer = RfNeuron(threshold, [4, 55], decay, persistent_state=True, shared_param=False).to(device)
    third_layer = RfNeuron(threshold, [8, 50], decay, persistent_state=True, shared_param=False).to(device) 
    fourth_layer = RfNeuron(threshold, [5,50], decay, persistent_state=True, shared_param=False).to(device) 
    
    our_SNN = Network(first_layer, input_layer_size=5)
    our_SNN.add_layer(second_layer, layer_size=15)
    our_SNN.add_layer(third_layer, layer_size=10)
    our_SNN.add_layer(fourth_layer, layer_size=2)
    
    return our_SNN

def run_learning_network(network: Network, spike_input: np.ndarray, iterations: int = 10):
    # Learn
    for _ in tqdm(range(iterations)):
        network.forward(np.repeat(spike_input,repeats=network.layer_sizes[0], axis=1), np.repeat(spike_input,repeats=network.layer_sizes[0], axis=1))
        network.run_learning_rule(b=b, beta=beta)
        
    network.forward(np.repeat(spike_input,repeats=network.layer_sizes[0], axis=1), np.repeat(spike_input,repeats=network.layer_sizes[0], axis=1))

def run_test_network(network: Network, spike_input: np.ndarray):
    network.forward(np.repeat(spike_input,repeats=network.layer_sizes[0], axis=1), np.repeat(spike_input,repeats=network.layer_sizes[0], axis=1))


def visualize(dir, network: Network, spike_input: np.ndarray, layers=[0,0,0,0,0], neurons=[0,1,2,3,4], title=''):
    #####
    # Plot the output of selected neurons in selected layers
    #####
    
    plot_time_trace(dir, network, layers=layers, neurons=neurons, title_prefix = title)
    
    #####
    # Visualize the spiking animation
    #####    
    out = SNN_visualizer(*[tensor[0,...] for tensor in network.layer_data_spikes], input_spikes=spike_input[0], title=title)
    
    def animateLive(frame):
        out.update(frame)
        out.time_slider.set_val(int(frame))
        
    animationLive=FuncAnimation(out.fig, animateLive, frames=spike_input.shape[-1], interval=40)
    print("Storing gifs...")
    animationLive.save(f'{dir}/{title}_SNN_output.gif', fps=30, writer='ffmpeg') #saving gifs takes a lot of time
    print("Storing gifs done")
    plt.show()

if __name__ == '__main__':
    # Parameter setup
    threshold = 1
    decay = 0.0672
    period = 10
    b = 0.1
    beta = np.array([0.1, 0.1, 0.1, 0.1])
    
    device = torch.device('cpu')
    train_input = get_train_input(device)
    our_SNN = setup_network_arch()
    
    run_learning_network(our_SNN, train_input, 2000)
    
    ### create directory to save data from the current run
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir = f'output/{now}'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    store(our_SNN, train_input, dir)
    
    visualize(dir, our_SNN, train_input, layers=[3,3], neurons=[0,1], title='Train')
    
    test_input = get_test_input(device)
    run_test_network(our_SNN, test_input)

    visualize(dir, our_SNN, test_input, layers=[3,3], neurons=[0,1],  title='Test')
