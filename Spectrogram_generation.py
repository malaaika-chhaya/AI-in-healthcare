# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:27:22 2023

@author: Lenovo
"""
#%%Imports
from mne.time_frequency import tfr_morlet
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import scipy as sp
from scipy.io import loadmat
from scipy import fft
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import butter, lfilter,filtfilt
import os
import os.path as op
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
#import h5py
%matplotlib qt
#%%Loading data
#list=['A01T','A02T', 'A03T','A04T','A05T','A06T','A07T','A08T','A09T',]
for k in range(10):
    raw=mne.io.read_raw_gdf('C:/Users/Lenovo/Downloads/BCICIV_2a/Train/A0'+str(k+1)+'T.gdf', preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
    #raw.plot()
    #drop channels
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    #events
    events=mne.events_from_annotations(raw)
    #event dictionary
    event_dict={
        'reject':1,
        'eye move':2,
        'eye open':3,
        'eye close':4,
        'new run':5,
        'new trial':6,
        'class 1':7,
        'class 2':8,
        'class 3':9,
        'class 4': 10,
        }
    #plot events
    #fig=mne.viz.plot_events(events[0],event_id=event_dict, sfreq=raw.info['sfreq'],first_samp=raw.first_samp)
    
    #%% Filter raw data between 8-30 Hz 
    rawfilt=raw.copy()
    #rawfilt.plot_psd()
    rawfilt.filter(l_freq=8, h_freq=30, fir_design='firwin')
    #rawfilt.plot_psd()
    #rawfilt.plot()
    
    
    
    #%%
    epochs = mne.Epochs(rawfilt, events[0], event_id=[7,8],tmin= -0.1, tmax=0.7)
    epochs.get_data().shape #(144, 22, 201)
    #epochs.plot(scalings='auto')
    #plt.suptitle('Epoched Time series')
    label=epochs.events[:,-1]
    len(label) #144
    #%% Evoked
    evoked_1=epochs['7'].average()
    evoked_2=epochs['8'].average()
    #evoked_3=epoch['9'].average()
    #evoked_4=epoch['10'].average()
    
    dicts={'left':evoked_1, 'right':evoked_2}
    #mne.viz.plot_compare_evokeds(dicts)
    
    #%%TFR
    for i in range(4):
        freqs = np.logspace(*np.log10([1, 50]), num=10)
        n_cycles = freqs / 2.
        power = mne.time_frequency.tfr_stockwell(epochs['7'], return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        fig=power.plot([i], title='Left Hand', baseline=(-0.2, -0.05))
        ax = plt.gca()
        ax.set_ylim([1, 50])
    
    #%%Morlet
    #Left Hand
    fig, axs = plt.subplots(nrows=22, ncols=1, figsize=(10, 50))
    for i in range(22):
        freqs = np.logspace(*np.log10([1, 50]), num=10)
        n_cycles = freqs / 2.
        power = mne.time_frequency.tfr_morlet(epochs['7'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        power.plot([i], baseline=(-0.2, -0.05), axes=axs[i], colorbar=False)
        for ax in axs:
            ax.set_axis_off()
    plt.subplots_adjust(hspace=0)
    plt.savefig('C:/Users/Lenovo/Downloads/evalset/morlet/' + str(k+1) + '/7.png', bbox_inches='tight')
    
    #Right Hand
    fig1, axs1 = plt.subplots(nrows=22, ncols=1, figsize=(10, 50))
    for j in range(22):
        freqs = np.logspace(*np.log10([1, 50]), num=10)
        n_cycles = freqs / 2.
        power = mne.time_frequency.tfr_morlet(epochs['8'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        power.plot([j], baseline=(-0.2, -0.05), axes=axs1[j], colorbar=False)
        for ax in axs1:
            ax.set_axis_off()
    plt.subplots_adjust(hspace=0)
    plt.savefig('C:/Users/Lenovo/Downloads/evalset/morlet/' + str(k+1) + '/8.png', bbox_inches='tight')
    #%%S Transform
    fig, axs = plt.subplots(nrows=22, ncols=1, figsize=(10, 50))
    for i in range(22):
        power = mne.time_frequency.tfr_stockwell(epochs['7'], return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        power.plot([i], baseline=(-0.2, -0.05), axes=axs[i], colorbar=False)
        ax = plt.gca()
        ax.set_ylim([1, 50])
        for ax in axs:
            ax.set_axis_off()
    plt.subplots_adjust(hspace=0)
    plt.savefig('C:/Users/Lenovo/Downloads/evalset/S_transform/' + str(k+1) + '/7.png', bbox_inches='tight')

    
    #Right Hand
    fig1, axs1 = plt.subplots(nrows=22, ncols=1, figsize=(10, 50))
    for j in range(22):
        power = mne.time_frequency.tfr_stockwell(epochs['8'], return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        power.plot([j], baseline=(-0.2, -0.05), axes=axs1[j], colorbar=False)
        ax = plt.gca()
        ax.set_ylim([1, 50])
        for ax in axs1:
            ax.set_axis_off()
    plt.subplots_adjust(hspace=0)
    plt.savefig('C:/Users/Lenovo/Downloads/evalset/S_transform/' + str(k+1) + '/8.png', bbox_inches='tight')
    #%%Multitapers
    #Left Hand
    fig, axs = plt.subplots(nrows=22, ncols=1, figsize=(10, 50))
    for i in range(22):
        freqs = np.logspace(*np.log10([1, 50]), num=10)
        n_cycles = freqs / 2.
        power = mne.time_frequency.tfr_multitaper(epochs['7'], freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0, use_fft=True, return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        power.plot([i], baseline=(-0.2, -0.05), axes=axs[i], colorbar=False)
        for ax in axs:
            ax.set_axis_off()
    plt.subplots_adjust(hspace=0)
    plt.savefig('C:/Users/Lenovo/Downloads/evalset/multitaper/' + str(k+1) + '/7.png', bbox_inches='tight')
    
    #Right Hand
    fig1, axs1 = plt.subplots(nrows=22, ncols=1, figsize=(10, 50))
    for j in range(22):
        freqs = np.logspace(*np.log10([1, 50]), num=10)
        n_cycles = freqs / 2.
        power = mne.time_frequency.tfr_multitaper(epochs['8'], freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0, use_fft=True, return_itc=False, n_jobs=1)
        # Plot the time-frequency analysis
        power.plot([j], baseline=(-0.2, -0.05), axes=axs1[j], colorbar=False)
        for ax in axs1:
            ax.set_axis_off()
    plt.subplots_adjust(hspace=0)
    plt.savefig('C:/Users/Lenovo/Downloads/evalset/multitaper/' + str(k+1) + '/8.png', bbox_inches='tight')
