import numpy as np
from scipy.signal import butter
import torch
from torchaudio.functional import filtfilt
import numpy as np
from matplotlib import pyplot as plt
# from torch import nn
# from .decompose import *

trend_token_dim = 8
season_token_dim = 4
residual_token_dim = 8


def obtain_moving_avg(signal,window_size=7,mode='valid',axis=-1):
    
    # Initialize an array to hold the moving average
    moving_avg = np.empty(signal.shape)
    moving_squared_mean = np.empty(signal.shape)

    # Compute the cumulative average for the initial part
    for i in range(window_size-1):
        moving_avg[...,i] = np.mean(signal[...,:i+1],axis=axis)
        moving_squared_mean[...,i] = np.mean(signal[...,:i+1],axis=axis)**2
        
    # Compute the moving average for the remaining part
    moving_avg[...,window_size-1:] = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size)/window_size, mode=mode), axis=axis, arr=signal)

    # Compute rolling squared mean using convolution
    data_squared = signal ** 2
    moving_squared_mean[...,window_size-1:] = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size)/window_size, mode=mode), axis=axis, arr=data_squared)
    
    # Calculate rolling standard deviation
    moving_average_squared = moving_avg ** 2
    moving_std = np.sqrt(moving_squared_mean - moving_average_squared)

    return moving_avg, moving_std

# Butterworth low-pass filter design
def butter_lowpass(cutoff, fs, order=5,device='cpu'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    b = torch.tensor(b, dtype=torch.float32,device=device)
    a = torch.tensor(a, dtype=torch.float32,device=device)

    # Ensure a[0] is 1
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    return b, a

def lowpass_filter(data, cutoff, fs, order=5,device='cpu'):
    b, a = butter_lowpass(cutoff, fs, order=order,device=device)
    y = filtfilt(data,a,b,clamp=False)
    return y

def decompose_trend(signal,fs=1.0,filter_cutoff=0.05,filter_order=3,double_smooth=False,windows_size=7,**kwargs):
    if double_smooth:
        moving_avg, moving_std = obtain_moving_avg(signal,window_size=windows_size,mode='valid')
        trend_component = lowpass_filter(moving_avg, filter_cutoff, fs, filter_order)
    else:
        trend_component = lowpass_filter(signal, filter_cutoff, fs, filter_order)
    rest = signal-trend_component
    return trend_component,rest

def ignore_single_step_change(trend_idx,axis=-1):
    # remove single change in head and tail first
    trend_idx[...,0] = trend_idx[...,1]
    trend_idx[...,-1] = trend_idx[...,-2]
    
    while True:
        trend_idx_cp = trend_idx.astype(int)
        single_step_idx = np.diff(np.diff(trend_idx_cp,axis=axis),axis=axis)
        single_step_idx = np.insert(single_step_idx,0,0,axis=axis)
        single_step_idx = np.insert(single_step_idx,-1,0,axis=axis)
        if (np.abs(single_step_idx)>1).sum()==0:
            break
        trend_idx[np.abs(single_step_idx)>1] = ~trend_idx[np.abs(single_step_idx)>1]
    return trend_idx


def get_segment_idx(trend_idx, axis=-1):
    """Identifies the indices where segments start and end in a trend index array.

    Args:
        trend_idx (np.ndarray): Array indicating the trend indices.
        axis (int, optional): Axis along which to operate. Defaults to -1.

    Returns:
        np.ndarray: Indices where segments start and end.
    """
    
    trend_idx = ignore_single_step_change(trend_idx,axis=axis)
    seg_idx_a = np.diff(trend_idx,axis=axis)
    
    start = seg_idx_a & trend_idx[...,1:]
    start = np.insert(start,0,0,axis=axis)
    start[...,0] = trend_idx[...,0]

    end = seg_idx_a & trend_idx[...,:-1]
    end = np.insert(end,-1,0,axis=axis)
    end[...,-1] = trend_idx[...,-1]
    
    return start,end


def find_trend_segments(array, diff_thd=0.001, axis=-1):
    """
    Find trend segments in a 2D array along the specified axis using matrix operations.
    
    Parameters:
    array (np.ndarray): 2D array to find trend segments in.
    axis (int): Axis along which to find trend segments (0 for rows, 1 for columns).
    
    Returns:
    list: A list of trend segments for each row/column.
    """
    assert axis==-1, "only support axis=-1"
    # Calculate differences along the specified axis
    diffs = np.diff(array, axis=axis)
    
    # Identify where changes occur
    increase_idx = diffs > diff_thd
    increase_idx = np.insert(increase_idx,0,False,axis=axis)

    
    decrease_idx = diffs < -diff_thd
    decrease_idx = np.insert(decrease_idx,0,False,axis=axis)

    
    plateau_idx = np.abs(diffs) <= diff_thd
    plateau_idx = np.insert(plateau_idx,0,False,axis=axis)

    
    # Initialize result list
    segments = {}    
    # Get segment indices
    segments["increase"] = get_segment_idx(increase_idx, axis=axis)
    segments["decrease"] = get_segment_idx(decrease_idx, axis=axis)
    segments["plateau"] = get_segment_idx(plateau_idx, axis=axis)
            
    return segments



def obtain_main_seasonal_components(signal,significant_indices,fft_result,axis=-1):
    # Loop over each frequency component
    component_fft = np.zeros_like(signal, dtype=complex)
    N = signal.shape[axis]
    half_n = N // 2
    for n in range(half_n):    
        
        # Assign the current frequency component
        component_fft[...,n] = fft_result[...,n]
        
        # Handle the symmetric component (for real signals)
        if n != 0 and n != N // 2:  # Don't double the DC or Nyquist component
            component_fft[...,-n] = np.conj(fft_result[...,n])
            component_fft[...,n][~significant_indices[...,n]]=0.
            component_fft[...,-n][~significant_indices[...,n]]=0.
            
    # Inverse FFT to get the time-domain signal
    seasonal_component = np.fft.ifft(component_fft).real
    
    return seasonal_component

def seasonal_decompose(signal,energy_threshold=0.1,max_freq_num=8,axis=-1):
    fft_result = np.fft.fft(signal)
    N = signal.shape[axis]
    T = 1./N
    freqs = np.fft.fftfreq(N, T)
    magnitudes = np.abs(fft_result)

    # Since the FFT result is symmetrical, we take the first half
    half_n = N // 2
    magnitudes = magnitudes[...,:half_n]
    freqs = freqs[...,:half_n]
    freqs = np.tile(freqs, (*magnitudes.shape[:-1],1))
    
    # Compute the squared magnitudes (which represent energy at each frequency)
    squared_magnitudes = magnitudes ** 2
    
    # Calculate the total energy
    total_energy = np.sum(squared_magnitudes,axis=-1)[...,np.newaxis]
    energy_contrib = squared_magnitudes/total_energy
    # Find significant indices
    significant_indices = energy_contrib >=energy_threshold
    empty_ids = significant_indices.sum(axis=axis)==0
    if empty_ids.sum()>0:
        print("exist empty significant indices")
        ## select the top k largest energy contribution
        new_thd = np.sort(energy_contrib,axis=axis)[empty_ids,-max_freq_num][...,np.newaxis]
        significant_indices[empty_ids] = energy_contrib[empty_ids]>=new_thd
    
    exceed_ids = significant_indices.sum(axis=axis)>max_freq_num
    print('max freq num',max_freq_num,significant_indices.sum(axis=axis).max())
    if exceed_ids.sum()>0:
        print("exist exceed significant indices")
        ## select the top k largest energy contribution
        new_thd = np.sort(energy_contrib,axis=axis)[exceed_ids,-max_freq_num][...,np.newaxis]
        significant_indices[exceed_ids] = energy_contrib[exceed_ids]>=new_thd
    
    seasonal_components = obtain_main_seasonal_components(signal,significant_indices,fft_result,axis=axis)
    
    return freqs,fft_result[...,:half_n],seasonal_components,significant_indices,energy_contrib

def trend_length(segments_idx,mark_start=True):
    seg_index_value = transform_seg_ids_value(segments_idx)
    seg_len = np.zeros_like(segments_idx[0]).astype(int)
    if mark_start:
        seg_len[segments_idx[0]] = seg_index_value[1] - seg_index_value[0] + 1
    else:
        seg_len[segments_idx[1]] = seg_index_value[1] - seg_index_value[0] + 1
    return seg_len

def trend_num(segments_start_idx,axis=-1):
    return segments_start_idx.sum(axis=axis)

def transform_seg_ids_value(trend_seg_idx):
    ids = np.arange(trend_seg_idx[0].shape[-1])
    ids_idx = np.tile(ids, (*trend_seg_idx[0].shape[:-1],1))
    return ids_idx[trend_seg_idx[0]],ids_idx[trend_seg_idx[1]]

def trend_magnitude(segments_idx,signal,mark_start=True):
    mags = np.zeros_like(segments_idx[0]).astype(float)
    if mark_start:
        mags[segments_idx[0]] = np.abs(signal[segments_idx[1]] - signal[segments_idx[0]])
    else:
        mags[segments_idx[1]] = np.abs(signal[segments_idx[1]] - signal[segments_idx[0]])
    return mags

def generate_trend_tokens(trend_signal,raw_signal, diff_thd=0.001,max_token_num=20):
    axis=-1
    segments = find_trend_segments(trend_signal, diff_thd=diff_thd, axis=axis)
    trend_tokens = {}
    d_time = raw_signal.shape[-1]
    n_features = trend_signal.shape[1]
    # feature_names = feature_names if feature_names is not None else np.arange(trend_signal.shape[1])
    signal_diff = np.diff(raw_signal,axis=axis)
    signal_diff = np.insert(signal_diff,0,0.,axis=axis)
    signal_diff2 = np.diff(signal_diff,axis=axis) 
    signal_diff2 = np.insert(signal_diff2,0,0.,axis=axis)
    signal_diff2[:,:,1]=0.
    trend_type_code = {"increase":0,"decrease":1,"plateau":2}
    
    trend_info = {}
    # max_tr_num = 0.
    for trend_type in segments.keys():      
        seg_len = trend_length(segments[trend_type])
        seg_mags = trend_magnitude(segments[trend_type],trend_signal)
        # seg_vals = raw_signal[segments[trend_type][0]]
        start_idx,end_idx = segments[trend_type][0],segments[trend_type][1]
        tr_num = trend_num(segments[trend_type][0],axis=axis)
        # max_tr_num += tr_num
        
        mask = np.tile(np.arange(start_idx.shape[-1]), (*start_idx.shape[:-1],1))
        start_time = start_idx*mask
        end_time = end_idx*mask
        trend_info[trend_type] = [start_idx,end_idx,start_time,end_time,seg_len,seg_mags]
        
    trend_token_seqs = np.zeros([trend_signal.shape[0],n_features,max_token_num,trend_signal.shape[-1]])    
    for i in range(trend_signal.shape[0]):
        trend_tokens[i] = trend_tokens.get(i,{})
        for f in range(n_features):
            trend_tokens[i][f] = trend_tokens[i].get(f,[])
            
            for trend_type in trend_info.keys():
                start_idx,end_idx,start_time,end_time,seg_len,seg_mags = trend_info[trend_type]
                
                stime = start_time[i,f][start_idx[i,f]]
                etime = end_time[i,f][end_idx[i,f]]
                slen = seg_len[i,f][start_idx[i,f]]
                smag = seg_mags[i,f][start_idx[i,f]]
                sval = raw_signal[i,f][start_idx[i,f]]
                tokens = [np.array([stime[e]/d_time,trend_type_code[trend_type],slen[e]/d_time,sval[e],signal_diff[i,f,stime[e]:etime[e]+1].mean(),signal_diff2[i,f,stime[e]:etime[e]+1].mean()]) for e in range(start_idx[i,f].sum())]
                for e in range(start_idx[i,f].sum()):
                    trend_token_seqs[i,f,e,stime[e]:etime[e]] = trend_signal[i,f,stime[e]:etime[e]] 
                # print('trend num',start_idx[i,f].sum(),stime,etime)
                # print([(signal_diff[i,f,stime[e]:etime[e]+1],signal_diff2[i,f,stime[e]:etime[e]+1]) for e in range(start_idx[i,f].sum())])
                trend_tokens[i][f]+=tokens
                
            if len(trend_tokens[i][f])>max_token_num:
                # remove shotest trend if exceed max token num
                trend_tokens[i][f].sort(key=lambda x:x[2])[::-1]
                trend_tokens[i][f] = trend_tokens[i][f][:max_token_num]
            trend_tokens[i][f].sort(key=lambda x:x[0])
    
    return trend_tokens, trend_token_seqs


def generate_trend_residual_tokens(trend_signal,residual,raw_signal,diff_thd=0.001,outlier_threshold=2,max_token_num=20):
    axis=-1
    segments = find_trend_segments(trend_signal, diff_thd=diff_thd, axis=axis)
    trend_tokens = {}
    res_tokens = {}
    print('check mean',trend_signal.mean(axis=-1).max(),residual.mean(axis=-1).max(),raw_signal.mean(axis=-1).max())
    
    d_time = raw_signal.shape[-1]
    n_features = trend_signal.shape[1]
    # feature_names = feature_names if feature_names is not None else np.arange(trend_signal.shape[1])
    signal_diff = np.diff(raw_signal,axis=axis)
    signal_diff = np.insert(signal_diff,0,0.,axis=axis)
    signal_diff2 = np.diff(signal_diff,axis=axis) 
    signal_diff2 = np.insert(signal_diff2,0,0.,axis=axis)
    signal_diff2[:,:,1]=0.
    trend_type_code = {"increase":0,"decrease":1,"plateau":2}   
    trend_info = {}
    
    residual_mean = np.mean(residual,axis=axis)
    residual_std = np.std(residual,axis=axis)
        
    # res_outliers = np.abs(residual-residual_mean[...,np.newaxis]) > (outlier_threshold * residual_std[...,np.newaxis])
    # res_outliers_num = res_outliers.sum(axis=axis)
    # residual_info = {}
    
    # max_tr_num = 0.
    for trend_type in segments.keys():      
        seg_len = trend_length(segments[trend_type])
        seg_mags = trend_magnitude(segments[trend_type],trend_signal)
        # seg_vals = raw_signal[segments[trend_type][0]]
        start_idx,end_idx = segments[trend_type][0],segments[trend_type][1]
        # tr_num = trend_num(segments[trend_type][0],axis=axis)
        # max_tr_num += tr_num
        
        mask = np.tile(np.arange(start_idx.shape[-1]), (*start_idx.shape[:-1],1))
        start_time = start_idx*mask
        end_time = end_idx*mask
        trend_info[trend_type] = [start_idx,end_idx,start_time,end_time,seg_len,seg_mags]
        
    trend_token_seqs = {}  
    trend_token_time = np.zeros([trend_signal.shape[0],n_features,max_token_num,trend_signal.shape[-1]])
    # res_token_seqs = {}
    # res_token_time = np.zeros_like(trend_token_time)
    for i in range(trend_signal.shape[0]):
        trend_tokens[i] = trend_tokens.get(i,{})
        # res_tokens[i] = res_tokens.get(i,{})
        trend_token_seqs[i] = trend_token_seqs.get(i,{})
        # res_token_seqs[i] = res_token_seqs.get(i,{})
        for f in range(n_features):
            trend_tokens[i][f] = trend_tokens[i].get(f,[])
            # res_tokens[i][f] = res_tokens[i].get(f,[])
            trend_token_seqs[i][f] = trend_token_seqs[i].get(f,[])
            # res_token_seqs[i][f] = res_token_seqs[i].get(f,[])
            for trend_type in trend_info.keys():
                start_idx,end_idx,start_time,end_time,seg_len,seg_mags = trend_info[trend_type]
                
                stime = start_time[i,f][start_idx[i,f]]
                etime = end_time[i,f][end_idx[i,f]]
                slen = seg_len[i,f][start_idx[i,f]]
                # smag = seg_mags[i,f][start_idx[i,f]]
                sval = raw_signal[i,f][start_idx[i,f]]
                # print('check',trend_type)
                
                # tokens = [np.array([stime[e]/d_time,trend_type_code[trend_type],slen[e]/d_time,sval[e],signal_diff[i,f,stime[e]:etime[e]+1].mean(),signal_diff2[i,f,stime[e]:etime[e]+1].mean()]) for e in range(start_idx[i,f].sum())]                              
                # trend_tokens[i][f]+=tokens
                enum = start_idx[i,f].sum()
                tokens = enum*[0]
                # rtokens = enum*[0]
                for e in range(enum):
                    
                     
                    
                    # print('check',e,stime[e],etime[e],slen[e])
                    rseq = np.zeros(trend_signal.shape[-1]) 
                    rseq[stime[e]:etime[e]+1] = residual[i,f,stime[e]:etime[e]+1]
                    # res_token_seqs[i][f].append(rseq)
                    res_seq_outliers = np.abs(rseq-residual_mean[i,f]) > (outlier_threshold * residual_std[i,f]) 
                    # rtokens[e] = np.array([stime[e]/d_time,etime[e]/d_time,np.mean(rseq),np.std(rseq),np.mean(np.abs(rseq)),res_outliers.sum(),np.mean(np.abs(rseq[res_outliers]-residual_mean[i,f])),np.mean(np.abs(rseq[res_outliers]))])
                    # rtokens[e][np.isnan(rtokens[e])]=0
                    
                    seq = np.zeros(trend_signal.shape[-1])
                    seq[stime[e]:etime[e]+1] = trend_signal[i,f,stime[e]:etime[e]+1]
                    trend_token_seqs[i][f].append(seq+rseq)
                    
                    tokens[e] = np.array([stime[e]/d_time,trend_type_code[trend_type],slen[e]/d_time,sval[e],signal_diff[i,f,stime[e]:etime[e]+1].mean(),signal_diff2[i,f,stime[e]:etime[e]+1].mean(),res_seq_outliers.sum(),np.mean(np.abs(rseq[res_seq_outliers]-residual_mean[i,f]))])                              
                    tokens[e][np.isnan(tokens[e])]=0
                    
                trend_tokens[i][f]+=tokens    
                # res_tokens[i][f]+=rtokens  
                # print('trend num',start_idx[i,f].sum(),stime,etime)
                # print([(signal_diff[i,f,stime[e]:etime[e]+1],signal_diff2[i,f,stime[e]:etime[e]+1]) for e in range(start_idx[i,f].sum())])
                
               
            if len(trend_tokens[i][f])>max_token_num:
                # remove shotest trend if exceed max token num
                sorted_ids = np.argsort([tk[2] for tk in trend_tokens[i][f]])[::-1]
                trend_tokens[i][f] = [trend_tokens[i][f][e] for e in sorted_ids]
                # res_tokens[i][f] = [res_tokens[i][f][e] for e in sorted_ids]
                trend_token_seqs[i][f] = [trend_token_seqs[i][f][e] for e in sorted_ids]
                # res_token_seqs[i][f] = [res_token_seqs[i][f][e] for e in sorted_ids]
                
                trend_tokens[i][f] = trend_tokens[i][f][:max_token_num]
                # res_tokens[i][f] = res_tokens[i][f][:max_token_num]
                trend_token_seqs[i][f] = trend_token_seqs[i][f][:max_token_num]
                # res_token_seqs[i][f] = res_token_seqs[i][f][:max_token_num]
                
            ## sort tokens by start time
            sorted_ids = np.argsort([tk[0] for tk in trend_tokens[i][f]])
            trend_tokens[i][f] = [trend_tokens[i][f][e] for e in sorted_ids]
            # res_tokens[i][f] = [res_tokens[i][f][e] for e in sorted_ids]
            trend_token_seqs[i][f] = [trend_token_seqs[i][f][e] for e in sorted_ids]
            # res_token_seqs[i][f] = [res_token_seqs[i][f][e] for e in sorted_ids]
                        
            trend_token_time[i,f,:len(trend_token_seqs[i][f])] = np.vstack(trend_token_seqs[i][f])
            # res_token_time[i,f,:len(res_token_seqs[i][f])] = np.vstack(res_token_seqs[i][f]) 
    
    return trend_tokens, trend_token_time


def transform_tokens_to_embedding_matrix(tokens,max_event_num,feature_num,embedding_dim,feature_embeddings=None,feature_names=None):
    sample_num = len(tokens)
    # print('check tokens',sample_num,feature_num,max_event_num,embedding_dim)
    # New dimension after concatenating one-hot encoding
    # new_embedding_dim = embedding_dim + feature_num if feature_embeddings is None else len(feature_embeddings[0])
    # pad_feature_dim = False
    # if new_embedding_dim%2!=0:
    #     new_embedding_dim += 1
    #     pad_feature_dim = True
    # feature_embeddings = feature_embeddings if feature_embeddings is not None else np.eye(feature_num+pad_feature_dim)    
    # print('check',sample_num,feature_num,max_event_num)
    event_embeddings = np.zeros((sample_num,feature_num,max_event_num,embedding_dim))
    for i in range(sample_num):
        for f in range(feature_num):
            fn = feature_names[f] if feature_names is not None else f
            if len(tokens[i][fn]) == 0:
                continue
            sub_tokens = np.vstack(tokens[i][fn])
            # print('check sub tokens',sub_tokens.shape)
            # extended_embeddings = np.hstack([np.tile(feature_embeddings[f], (sub_tokens.shape[0], 1)),sub_tokens])
            event_embeddings[i,f,:sub_tokens.shape[0],:] = sub_tokens    

    return event_embeddings


def generate_seasonal_tokens(seasonal_residual,raw_signal,energy_threshold=0.1,max_freq_num=8,feature_names=None):
    axis=-1
    freqs,magnitudes,seasonal_components,significant_indices,energy_contrib = seasonal_decompose(seasonal_residual,energy_threshold=energy_threshold,max_freq_num=max_freq_num,axis=axis)
    seasonal_tokens = {}
    feature_names = feature_names if feature_names is not None else np.arange(magnitudes.shape[1])
    select_freqs_num = significant_indices.sum(axis=axis)
    # print('select freqs num',select_freqs_num.max())
    T = seasonal_residual.shape[-1]
    seasonal_token_seqs = np.zeros([seasonal_residual.shape[0],magnitudes.shape[1],max_freq_num,T])
    for i in range(magnitudes.shape[0]):
        seasonal_tokens[i] = seasonal_tokens.get(i,{})
        for f, fn in enumerate(feature_names):  
            if_freq = freqs[i,f][significant_indices[i,f]].astype(int)         
            if_periods = 1./if_freq # propotional period
            if_periods[if_periods==np.inf] = 0
            if_real_mags = magnitudes[i,f][significant_indices[i,f]].real
            if_imag_mags = magnitudes[i,f][significant_indices[i,f]].imag
            if_energy_contrib = energy_contrib[i,f][significant_indices[i,f]]
            seasonal_tokens[i][fn] = np.array([np.array([if_periods[e],if_real_mags[e],if_imag_mags[e],if_energy_contrib[e]]) for e in range(select_freqs_num[i,f])])
            
            token_fft = np.zeros([select_freqs_num[i,f],T], dtype=complex)
            token_fft[np.arange(select_freqs_num[i,f]),if_freq] = if_real_mags+1j*if_imag_mags
            token_fft[np.arange(select_freqs_num[i,f]),-if_freq] = if_real_mags-1j*if_imag_mags
            seasonal_token_seqs[i,f,:select_freqs_num[i,f],:] = np.fft.ifft(token_fft).real   
                         
    return seasonal_tokens,seasonal_components,significant_indices,freqs,magnitudes,seasonal_token_seqs 



def generate_residual_outlier_tokens(residual,outlier_threshold=2,feature_names=None): 
    axis=-1       
    residual_mean = np.mean(residual,axis=axis)
    residual_std = np.std(residual,axis=axis)
        
    res_outliers = np.abs(residual-residual_mean[...,np.newaxis]) > (outlier_threshold * residual_std[...,np.newaxis])
    res_outliers_num = res_outliers.sum(axis=axis)
    max_outliers_num = int(res_outliers_num.max())
    
    mask = np.tile(np.arange(residual.shape[-1]), (*residual.shape[:-1],1))
    res_outliers_time = res_outliers*mask
    
    residual_outlier_tokens = {}
    feature_names = feature_names if feature_names is not None else np.arange(residual.shape[1])
    for i in range(residual.shape[0]):
        residual_outlier_tokens[i] = residual_outlier_tokens.get(i,{})
        for f, fn in enumerate(feature_names):
            if res_outliers_num[i,f].sum() == 0:
                #print(i,fn,"no outliers")
                residual_outlier_tokens[i][fn]=[]
                continue
            f_outliers = [ np.array((r,residual[i,f,r])) for r in (res_outliers_time[i,f][res_outliers[i,f]])]
            residual_outlier_tokens[i][fn] = np.vstack(f_outliers)
    return residual_outlier_tokens,max_outliers_num,res_outliers,residual_mean,residual_std


class TimeTokenizer:
    global_max_tr_num = 0
    def __init__(self,d_time,feature_num,feature_names=None,decompose_args=None):
        self.d_time = d_time
        self.feature_num = feature_num
        self.feature_names = feature_names
        self.decompose_args = decompose_args

        print('global max tr num init',TimeTokenizer.global_max_tr_num)
        
    def decompose(self,x):
        trend_component,seasonal_residual = decompose_trend(x,**self.decompose_args)
        freqs,fft_result,seasonal_components,significant_indices,energy_contrib = seasonal_decompose(seasonal_residual,**self.decompose_args)
        residual = seasonal_residual - seasonal_components.sum(axis=-1)
        return trend_component,freqs,fft_result,seasonal_components,significant_indices,residual
    
    def generate_token_embeddings(self,x):
        max_tr_num = self.decompose_args['max_trend_num']
        max_freq_num = self.decompose_args['max_freq_num']
        
        trend_component,seasonal_residual = decompose_trend(x,**self.decompose_args)
        # trend_tokens,trend_token_seqs = generate_trend_tokens(trend_component,x,max_token_num=max_tr_num)
        
        # ## set global max tr num, in case of different max tr num in training and testing
        # if TimeTokenizer.global_max_tr_num == 0:
        #     TimeTokenizer.global_max_tr_num = max_tr_num
        # else:
        #     max_tr_num = TimeTokenizer.global_max_tr_num
        # print('max tr num',max_tr_num,'global max tr num',TimeTokenizer.global_max_tr_num)
            
        seasonal_tokens,seasonal_components,significant_indices,freqs,magnitudes,season_token_seqs = generate_seasonal_tokens(seasonal_residual,x,energy_threshold=0.1,max_freq_num=max_freq_num)
        residual = x - trend_component - seasonal_components
        # residual_outlier_tokens,max_outliers_num,res_outliers_idx,residual_mean,residual_std = generate_residual_outlier_tokens(residual,outlier_threshold=2)
        trend_tokens,trend_token_seqs = generate_trend_residual_tokens(trend_component,residual,x,outlier_threshold=self.decompose_args['outlier_threshold'],max_token_num=max_tr_num)

        trend_embeddings = transform_tokens_to_embedding_matrix(trend_tokens,max_tr_num,self.feature_num,embedding_dim=trend_token_dim,feature_names=self.feature_names)
        seasonal_embeddings = transform_tokens_to_embedding_matrix(seasonal_tokens,max_freq_num,self.feature_num,embedding_dim=season_token_dim,feature_names=self.feature_names)
        # res_embeddings = transform_tokens_to_embedding_matrix(res_tokens,max_tr_num,self.feature_num,embedding_dim=residual_token_dim,feature_names=self.feature_names)
        # plt.figure(figsize=(6,3))
        # plt.plot(seasonal_components[0,0])
        # # Hide x and y axis ticks
        # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # plt.savefig('season.png')
        # plt.close()
        # plt.figure(figsize=(6,3))
        # plt.plot(res_token_seqs[0,13,0])
        # # Hide x and y axis ticks
        # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # plt.savefig('residual_token.png')
        # plt.close()
        # residual_outlier_embeddings = transform_tokens_to_embedding_matrix(residual_outlier_tokens,max_outliers_num,self.feature_num,self.d_time,feature_names=self.feature_names)

        return trend_embeddings,trend_token_seqs,seasonal_embeddings,season_token_seqs
    
    
        
                  
    
