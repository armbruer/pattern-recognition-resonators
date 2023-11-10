# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:17:21 2023

@author: Borislav Polovnikov
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


class neuron:
    def __init__(self, axis, color_low = 'beige', color_high = 'coral', position = [0,0], radius = 0.5):
        
        self.axis=axis
        self.color_low = color_low
        self.color_high = color_high
        self.position = position
        self.radius = radius
        self.circle_graphic = patches.Circle(self.position,self.radius, fc = self.color_low) #start with a neuron in the low-state
        self.axis.add_patch(self.circle_graphic)
        
        
        
    def set_high(self):
        self.circle_graphic.set_facecolor(self.color_high)
        self.circle_graphic.set_radius(self.radius*1.1)
    
    def set_low(self):
        self.circle_graphic.set_facecolor(self.color_low)
        self.circle_graphic.set_radius(self.radius)
        
class SNN_visualizer:
    '''
    input: List of arrays [#N_l, t] where N_l is the number of neurons in the layer l, and t is the time-dimension.
    The first element of that list corresponds to the spike-outputs of the first layer, the second element to the spike-output of the second layer, etc.
    '''
    def __init__(self, *layers_data_input, input_spikes=None, title=''):
        self.total_layer_number = len(layers_data_input)
        self.neurons_per_layer = [layer.shape[0] for layer in layers_data_input]
        self.spacing = 0.8/max(self.neurons_per_layer)
        self.layers_data_input = layers_data_input
        self.input_spikes = input_spikes
        
        #####
        ## Initialize a matplotlib figure and axis instance
        #####
        self.fig=plt.figure(figsize=(9,9), label=title+' network')   ###plotting environment for live animations
        self.ax=self.fig.add_axes([0,0,1,1],frame_on=False)
        self.ax.set_xticks([]), self.ax.set_yticks([])
        
        ### add time slider for both control and visualization
        self.fig.subplots_adjust(bottom=0.2)
        axis_color = 'lightgoldenrodyellow'
    	
        self.time_slider_ax  = self.fig.add_axes([0.17, 0.01, 0.65, 0.02], facecolor=axis_color) 
        self.time_slider = Slider(self.time_slider_ax, 'Time', 0, self.layers_data_input[0].shape[-1]-1, valinit=0,valfmt='%0.0f')
        self.time_slider.on_changed(self.update)
        
        
        #### create a list of layers where each layer is itself a list of neurons in that layer
        self.layers = []
        for ind_layer, N_l in enumerate(self.neurons_per_layer):
            self.layers.append( [neuron(self.ax, position = [self.get_layer_position(ind_layer), self.get_neuron_height(ind_layer, ind_neuron)], radius =self.spacing/3) for ind_neuron in range(N_l)] )
        
        
        #### if provided, plot the time trace of the input signal
        self.input_x = np.linspace(0.13,0.05,20)
        if self.input_spikes is not None:
            input_y = self.input_spikes[0,:20]
            self.input_lines =[]
            for i in range(self.neurons_per_layer[0]):
                self.input_lines.append(self.ax.plot(self.input_x, input_y*self.spacing/2 + self.get_neuron_height(0, i), color='blue', zorder=-2)[0])
            
        
        
        #### plot the connecting lines between all neurons in neighboring layers
        self.line_layers = []
        for i, layer in enumerate(self.layers[:-1]):
            neurons_outgoing = []
            for neuron1 in layer:
                lines=[]
                for neuron2 in self.layers[i+1]:
                    lines.append(self.plot_connecting_line(neuron1, neuron2))
                neurons_outgoing.append(lines)
            self.line_layers.append(neurons_outgoing)
        
        
        #####
        ## initialize the colors as the first entry in the time-domains
        #####
        for ind_layer, layer in enumerate(self.layers):
            for ind_neuron, neur in enumerate(layer):
                
                if self.layers_data_input[ind_layer][ind_neuron,0] == 1:
                    neur.set_high()
                else:
                    neur.set_low()

    
    def get_layer_position(self, layer_number):
        return 0.15 + 0.7/(self.total_layer_number-1)*layer_number
    
    def get_neuron_height(self, layer_number, neuron_number):
        # the biggest layer defines the spacing between individual neurons, and all the layers with a smaller number of neurons
        # are centered around the center horizontla line        
        return 0.5 + (self.spacing*self.neurons_per_layer[layer_number])/2 - self.spacing*neuron_number
    
    def plot_connecting_line(self, neuron1, neuron2):
        # given two neuron instances plot and return a line connecting them
        line = self.ax.plot([neuron1.position[0], neuron2.position[0]], [neuron1.position[1], neuron2.position[1]], color = 'lightgrey', alpha=0.04, lw=2, zorder=-1)
        return line[0] #return the matplotlib line patch for dynamic color update
    
    def update(self, val_ind):
        
        for ind_layer, layer in enumerate(self.layers):
            for ind_neuron, neur in enumerate(layer):
                if ind_layer == 0 and self.input_spikes is not None:
                    input_y = self.input_spikes[0,int(val_ind):int(val_ind)+20]
                    length_y = len(input_y)
                    if length_y < 20:
                        input_y = np.hstack((np.zeros(20-length_y), input_y))
                    self.input_lines[ind_neuron].set_ydata(input_y*self.spacing/2 + self.get_neuron_height(0, ind_neuron))
                    
                        
                # first, update the line-boldness depending on the previous state of the firing neuron
                try:
                    if self.layers_data_input[ind_layer][ind_neuron,int(val_ind)-1] == 1:
                        for line in self.line_layers[ind_layer][ind_neuron]:
                            line.set_alpha(0.25)
                    else:
                        for line in self.line_layers[ind_layer][ind_neuron]:
                            line.set_alpha(0.04)
                except:
                    pass
                # second, update the state (high/low) of the neurons depending on whether they spike or not
                if self.layers_data_input[ind_layer][ind_neuron,int(val_ind)] == 1:
                    neur.set_high()
                else:
                    neur.set_low()
        plt.draw()

if __name__ == '__main__':
    out = SNN_visualizer(*[   np.random.randint(0,2, size=(2,100)),np.random.randint(0,2, size=(6,100)), np.random.randint(0,2, size=(10,100)), np.random.randint(0,2, size=(3,100))], input_spikes=np.random.randint(0,2,size=(1,100)) )
    
    def animateLive(frame):
        out.update(frame)
        out.time_slider.set_val(int(frame))

    #animationLive=FuncAnimation(out.fig, animateLive, frames=100, interval=10)
    plt.show()