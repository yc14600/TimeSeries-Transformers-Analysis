import torch
from scipy.signal import butter
from torchaudio.functional import filtfilt

trend_token_dim = 8
season_token_dim = 4
residual_token_dim = 8

def obtain_main_seasonal_components(signal,significant_indices,fft_result,axis=-1):

    top_fft_coeffs = torch.where(significant_indices, fft_result, torch.zeros_like(fft_result))
    seasonal_component = torch.fft.ifft(top_fft_coeffs, n=signal.shape[-1]).real
    
    return seasonal_component

def seasonal_decompose(signal,energy_threshold=0.1,max_freq_num=8,axis=-1,device='cpu'):
    fft_result = torch.fft.fft(signal, dim=axis)
    N = signal.shape[axis]
    T = 1./N
    freqs = torch.fft.fftfreq(N, T).to(device)
    magnitudes = torch.abs(fft_result)

    freqs = freqs.repeat(*magnitudes.shape[:-1],1)
    
    # Compute the squared magnitudes (which represent energy at each frequency)
    squared_magnitudes = magnitudes ** 2
    
    # Calculate the total energy
    total_energy = torch.sum(squared_magnitudes,dim=-1,keepdim=True)
    energy_contrib = squared_magnitudes/total_energy
    # Find significant indices
    significant_indices = energy_contrib >=energy_threshold
    empty_ids = significant_indices.sum(dim=axis)==0
    if empty_ids.sum()>0:
        ## select the top k largest energy contribution
        new_thd = torch.sort(energy_contrib,dim=axis)[0][empty_ids][...,-max_freq_num].unsqueeze(-1)
        significant_indices[empty_ids] = energy_contrib[empty_ids]>=new_thd
    
    exceed_ids = significant_indices.sum(axis=axis)>max_freq_num
    if exceed_ids.sum()>0:
        ## select the top k largest energy contribution
        new_thd = torch.sort(energy_contrib,dim=axis)[0][exceed_ids][...,-max_freq_num].unsqueeze(-1)
        significant_indices[exceed_ids] = energy_contrib[exceed_ids]>new_thd
    
    seasonal_components = obtain_main_seasonal_components(signal,significant_indices,fft_result,axis=axis)
    
    return freqs,fft_result,seasonal_components,significant_indices,energy_contrib



def generate_seasonal_tokens(seasonal_residual,energy_threshold=0.1,max_freq_num=8,feature_names=None,device='cpu'):
    axis=-1
    freqs,magnitudes,seasonal_components,significant_indices,energy_contrib = seasonal_decompose(seasonal_residual,energy_threshold=energy_threshold,max_freq_num=max_freq_num,axis=axis)
    # feature_names = feature_names if feature_names is not None else torch.arange(magnitudes.shape[1])

    select_freqs_num = significant_indices.sum(dim=axis)
    # print('select freqs num',select_freqs_num.max())
    T = seasonal_residual.shape[-1]
    # seasonal_token_seqs = torch.zeros([seasonal_residual.shape[0],magnitudes.shape[1],max_freq_num,T],device=device)
    # print(significant_indices.shape)
    batch_indices,row_indices,select_freqs = torch.where(significant_indices)
    token_idx = torch.cat([torch.arange(count.item()) for count in select_freqs_num.view(-1)])

    token_fft = torch.zeros(seasonal_residual.shape[0],seasonal_residual.shape[1],max_freq_num,T, dtype=torch.cfloat, device=device)
    token_emd = torch.zeros(seasonal_residual.shape[0],seasonal_residual.shape[1],max_freq_num,season_token_dim,device=device)
    
    token_emd[batch_indices,row_indices,token_idx,0] = 1./select_freqs
    token_emd[batch_indices,row_indices,token_idx,1] = magnitudes[batch_indices,row_indices,select_freqs].real
    token_emd[batch_indices,row_indices,token_idx,2] = magnitudes[batch_indices,row_indices,select_freqs].imag
    token_emd[batch_indices,row_indices,token_idx,3] = energy_contrib[batch_indices,row_indices,select_freqs]
    ## fill in nan
    token_emd[torch.isnan(token_emd)] = 0
    ## fill in inf
    token_emd[torch.isinf(token_emd)] = 0
    
    token_fft[batch_indices,row_indices,token_idx,select_freqs] = magnitudes[batch_indices,row_indices,select_freqs].real+1j*magnitudes[batch_indices,row_indices,select_freqs].imag
    token_fft[batch_indices,row_indices,token_idx,-select_freqs] = magnitudes[batch_indices,row_indices,select_freqs].real-1j*magnitudes[batch_indices,row_indices,select_freqs].imag
    seasonal_token_seqs = torch.fft.ifft(token_fft).real
    
    return token_emd,seasonal_components,seasonal_token_seqs 



def get_segment_idx(trend_idx, axis=-1):
    """Identifies the indices where segments start and end in a trend index array.

    Args:
        trend_idx (np.ndarray): Array indicating the trend indices.
        axis (int, optional): Axis along which to operate. Defaults to -1.

    Returns:
        np.ndarray: Indices where segments start and end.
    """
    
    seg_idx_a = torch.diff(trend_idx,axis=axis)
    
    start = seg_idx_a & trend_idx[...,1:]
    start = torch.cat([torch.zeros_like(start[...,0]).unsqueeze(-1),start],dim=-1) #np.insert(start,0,0,axis=axis)
    start[...,0] = trend_idx[...,0]

    end = seg_idx_a & trend_idx[...,:-1]
    end = torch.cat([end,torch.zeros_like(end[...,0]).unsqueeze(-1),],dim=-1) #np.insert(end,-1,0,axis=axis)
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
    diffs = torch.diff(array, dim=axis)
    
    # Identify where changes occur
    increase_idx = diffs > diff_thd
    increase_idx = torch.cat([torch.zeros_like(increase_idx[...,0]).unsqueeze(-1).to(bool),increase_idx],dim=-1) #np.insert(increase_idx,0,False,axis=axis)

    
    decrease_idx = diffs < -diff_thd
    decrease_idx = torch.cat([torch.zeros_like(decrease_idx[...,0]).unsqueeze(-1).to(bool),decrease_idx],dim=-1)   #np.insert(decrease_idx,0,False,axis=axis)

    
    plateau_idx = torch.abs(diffs) <= diff_thd
    plateau_idx = torch.cat([torch.zeros_like(plateau_idx[...,0]).unsqueeze(-1).to(bool),plateau_idx],dim=-1)  #np.insert(plateau_idx,0,False,axis=axis)

    
    # Initialize result list
    segments = {}    
    # Get segment indices
    segments["increase"] = get_segment_idx(increase_idx, axis=axis)
    segments["decrease"] = get_segment_idx(decrease_idx, axis=axis)
    segments["plateau"] = get_segment_idx(plateau_idx, axis=axis)
            
    return segments

def transform_seg_ids_value(trend_seg_idx,device='cpu'):
    ids = torch.arange(trend_seg_idx[0].shape[-1],device=device)
    ids_idx = ids.repeat(*trend_seg_idx[0].shape[:-1],1)
    return ids_idx[trend_seg_idx[0]],ids_idx[trend_seg_idx[1]]


def trend_length(segments_idx,mark_start=True,device='cpu'):
    seg_index_value = transform_seg_ids_value(segments_idx,device=device)
    seg_len = torch.zeros_like(segments_idx[0],device=device).to(int)
    if mark_start:
        seg_len[segments_idx[0]] = seg_index_value[1] - seg_index_value[0] + 1
    else:
        seg_len[segments_idx[1]] = seg_index_value[1] - seg_index_value[0] + 1
    return seg_len

def trend_magnitude(segments_idx,signal,mark_start=True,device='cpu'):
    mags = torch.zeros_like(segments_idx[0],device=device).to(torch.float32)
    if mark_start:
        mags[segments_idx[0]] = signal[segments_idx[1]] - signal[segments_idx[0]]
    else:
        mags[segments_idx[1]] = signal[segments_idx[1]] - signal[segments_idx[0]]
    return mags


def get_segs_indices(start_points_matrix,end_points_matrix):
    # Get the indices of the start and end points for all rows
    start_indices = torch.where(start_points_matrix)[-1]
    end_indices = torch.where(end_points_matrix)[-1]
    # print(torch.where(start_points_matrix))

    # Get batch and row indices corresponding to the start and end points
    batch_indices = torch.where(start_points_matrix)[0]
    row_indices = torch.where(start_points_matrix)[1]
    
    # For each batch and row, we need to calculate how many segments it has
    row_segment_counts = start_points_matrix.sum(dim=-1)

    # Create a cumulative count of the segment indices to track which segment we're filling
    segment_idx = torch.cat([torch.arange(count.item()) for count in row_segment_counts.view(-1)])

    return start_indices,end_indices,batch_indices,row_indices,segment_idx


def get_padded_segment_masks_matrix(start_points_matrix, end_points_matrix, signal_length,num_max_segs=16):

    # Get the batch size, number of rows, and signal length
    batch_size, num_rows, _ = start_points_matrix.shape
    start_indices,end_indices,batch_indices,row_indices,segment_idx = get_segs_indices(start_points_matrix,end_points_matrix)

    # Create an array for signal indices
    signal_range = torch.arange(signal_length).unsqueeze(0).unsqueeze(0)

    # Initialize a mask matrix to store the padded masks (shape: [batch_size, num_rows, num_max_segs, signal_length])
    mask_matrix = torch.zeros((batch_size, num_rows, num_max_segs, signal_length), dtype=torch.float32)

    # Create a mask for all segments across batches and rows using broadcasting
    mask = (signal_range >= start_indices.unsqueeze(1)) & (signal_range <= end_indices.unsqueeze(1))

    # Convert the boolean mask to float before assignment
    mask = mask.float()

    # Assign the created mask to the correct batches, rows, and segment indices
    mask_matrix[batch_indices, row_indices, segment_idx] = mask

    return mask_matrix

    
def generate_trend_residual_tokens(trend_signal,residual,raw_signal,diff_thd=0.001,outlier_threshold=2,max_token_num=20,device='cpu'):
    axis=-1
    segments = find_trend_segments(trend_signal, diff_thd=diff_thd, axis=axis)

    batch_size, num_rows, d_time = raw_signal.shape
    # n_features = trend_signal.shape[1]
    # feature_names = feature_names if feature_names is not None else np.arange(trend_signal.shape[1])
    residual_mean = residual.mean(dim=-1)
    residual_std = residual.std(dim=-1)
    
    merged_start_id = torch.zeros_like(segments["increase"][0],device=device).to(bool)
    merged_end_id = torch.zeros_like(segments["increase"][1],device=device).to(bool)
    merged_seg_len_s = torch.zeros_like(segments["increase"][0],device=device)
    merged_seg_len_e = torch.zeros_like(segments["increase"][0],device=device)
    merged_seg_mags = torch.zeros_like(segments["increase"][0],device=device)
        
    for trend_type in segments.keys():      
        seg_len_s = trend_length(segments[trend_type],mark_start=True,device=device)
        seg_len_e = trend_length(segments[trend_type],mark_start=False,device=device)
        seg_mags = trend_magnitude(segments[trend_type],trend_signal,device=device)
        start_idx,end_idx = segments[trend_type][0],segments[trend_type][1]
        merged_start_id = start_idx | merged_start_id
        merged_end_id = end_idx | merged_end_id
        merged_seg_len_s = seg_len_s + merged_seg_len_s
        merged_seg_len_e = seg_len_e + merged_seg_len_e
        merged_seg_mags = seg_mags + merged_seg_mags
        
            
    if torch.any(merged_start_id.sum(dim=-1)>max_token_num):
        
        min_len = int(d_time / max_token_num)
        merged_start_id[merged_seg_len_s<min_len] = 0 
        merged_end_id[merged_seg_len_e<min_len] = 0
        merged_seg_len_s[merged_seg_len_s<min_len] = 0
        merged_seg_len_e[merged_seg_len_e<min_len] = 0
        merged_seg_mags[merged_seg_len_s<min_len] = 0
        
    # Create an array for signal indices
    signal_range = torch.arange(d_time,device=device).unsqueeze(0).unsqueeze(0)
    start_indices,end_indices,batch_indices,row_indices,segment_idx = get_segs_indices(merged_start_id,merged_end_id)

    # Initialize a mask matrix to store the padded masks (shape: [batch_size, num_rows, num_max_segs, signal_length])
    mask_matrix = torch.zeros((batch_size, num_rows, max_token_num, d_time), dtype=torch.float32,device=device)
    # Convert the boolean mask to float before assignment
    mask = (signal_range >= start_indices.unsqueeze(1)) & (signal_range <= end_indices.unsqueeze(1))
    mask = mask.float()
    # Assign the created mask to the correct batches, rows, and segment indices
    mask_matrix[batch_indices, row_indices, segment_idx] = mask
    # segs_mask = get_padded_segment_masks_matrix(merged_start_id, merged_end_id, d_time,num_max_segs=max_token_num)
    trend_token_time = (trend_signal+residual).unsqueeze(2).repeat(1,1,max_token_num,1)*mask_matrix

    signal_diff = torch.diff(raw_signal,dim=-1)
    signal_diff =  torch.cat([torch.zeros_like(signal_diff[...,0],device=device).unsqueeze(-1),signal_diff],dim=-1) #np.insert(signal_diff,0,0.,axis=axis)
    signal_diff2 = torch.diff(signal_diff,dim=-1) 
    signal_diff2 = torch.cat([torch.zeros_like(signal_diff2[...,0],device=device).unsqueeze(-1),signal_diff2],dim=-1)#np.insert(signal_diff2,0,0.,axis=axis)
    signal_diff2[:,:,1]=0.
    
    token_matrix = torch.zeros((trend_signal.shape[0],trend_signal.shape[1],max_token_num,trend_token_dim), dtype=torch.float32, device=device)
    ## start time
    token_matrix[batch_indices,row_indices,segment_idx,0] = start_indices/d_time
    ## end time
    token_matrix[batch_indices,row_indices,segment_idx,1] = end_indices/d_time
    ## start value
    token_matrix[batch_indices,row_indices,segment_idx,2] = trend_signal[batch_indices,row_indices,start_indices]
    ## value change
    token_matrix[batch_indices,row_indices,segment_idx,3] = trend_signal[batch_indices,row_indices,end_indices]-trend_signal[batch_indices,row_indices,start_indices]

    signal_diff_segs = signal_diff.unsqueeze(2).repeat(1,1,max_token_num,1)*mask_matrix
    signal_diff2_segs = signal_diff2.unsqueeze(2).repeat(1,1,max_token_num,1)*mask_matrix
    ## mean value change
    token_matrix[batch_indices,row_indices,segment_idx,4] = signal_diff_segs[batch_indices,row_indices,segment_idx].sum(dim=-1)/mask_matrix[batch_indices,row_indices,segment_idx].sum(dim=-1)
    ## mean value change rate
    token_matrix[batch_indices,row_indices,segment_idx,5] = signal_diff2_segs[batch_indices,row_indices,segment_idx].sum(dim=-1)/mask_matrix[batch_indices,row_indices,segment_idx].sum(dim=-1)
    
    residual_segs = residual.unsqueeze(2).repeat(1,1,max_token_num,1)*mask_matrix
    residual_outliers = torch.abs(residual_segs-residual_mean.unsqueeze(-1).unsqueeze(-1)) > (outlier_threshold * residual_std.unsqueeze(-1).unsqueeze(-1))
    residual_outliers_std = torch.abs(residual_segs-residual_mean.unsqueeze(-1).unsqueeze(-1))*residual_outliers
    ## outlier ratio
    token_matrix[batch_indices,row_indices,segment_idx,6] = residual_outliers[batch_indices,row_indices,segment_idx].sum(dim=-1)/mask_matrix[batch_indices,row_indices,segment_idx].sum(dim=-1)
    ## outlier strength
    token_matrix[batch_indices,row_indices,segment_idx,7] = (residual_outliers_std.sum(dim=-1)/residual_outliers.sum(dim=-1))[batch_indices,row_indices,segment_idx]
    ## fill in nan with 0
    token_matrix[torch.isnan(token_matrix)] = 0
    ## fill in inf with 0   
    token_matrix[torch.isinf(token_matrix)] = 0
    
    return token_matrix, trend_token_time



# def transform_tokens_to_embedding_matrix(tokens,max_event_num,feature_num,embedding_dim,feature_names=None):
#     sample_num = len(tokens)

#     event_embeddings = torch.zeros((sample_num,feature_num,max_event_num,embedding_dim))
#     for i in range(sample_num):
#         for f in range(feature_num):
#             fn = feature_names[f] if feature_names is not None else f
#             if len(tokens[i][fn]) == 0:
#                 continue
#             sub_tokens = torch.vstack(tokens[i][fn])
#             event_embeddings[i,f,:sub_tokens.shape[0],:] = sub_tokens    

#     return event_embeddings

# Butterworth low-pass filter design
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    b = torch.tensor(b, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.float32)

    # Ensure a[0] is 1
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    return b, a



class TimeTokenizer:
    
    trend_token_dim = 8
    season_token_dim = 4
    
    def __init__(self,d_time,feature_num,feature_names=None,decompose_args=None):
        self.d_time = d_time
        self.feature_num = feature_num
        self.feature_names = feature_names
        self.decompose_args = decompose_args
        cutoff = self.decompose_args['filter_cutoff']
        fs = self.decompose_args['fs']
        order = self.decompose_args['filter_order']
        self.b, self.a = butter_lowpass(cutoff, fs, order=order)
    
    def lowpass_filter(self,data):   
        y = filtfilt(data,self.a,self.b,clamp=False)
        return y
    
    def generate_token_embeddings(self,x,device='cpu'):
        max_tr_num = self.decompose_args['max_trend_num']
        max_freq_num = self.decompose_args['max_freq_num']
        
        trend_x = self.lowpass_filter(x)
        seasonal_residual = x - trend_x       
           
        seasonal_tokens,seasonal_components,season_token_seqs = generate_seasonal_tokens(seasonal_residual,energy_threshold=0.1,max_freq_num=max_freq_num,device=device)
        residual = x - trend_x - seasonal_components
        trend_tokens,trend_token_seqs = generate_trend_residual_tokens(trend_x,residual,x,outlier_threshold=self.decompose_args['outlier_threshold'],max_token_num=max_tr_num,device=device)

        # trend_embeddings = transform_tokens_to_embedding_matrix(trend_tokens,max_tr_num,self.feature_num,embedding_dim=self.trend_token_dim,feature_names=self.feature_names)
        # seasonal_embeddings = transform_tokens_to_embedding_matrix(seasonal_tokens,max_freq_num,self.feature_num,embedding_dim=self.season_token_dim,feature_names=self.feature_names)
        
        return trend_tokens,trend_x,seasonal_tokens,seasonal_components,residual
    
    def seq_decompose(self,x):
        trend_x = self.lowpass_filter(x)
        seasonal_residual = x - trend_x
        freqs,magnitudes,seasonal_components,significant_indices,energy_contrib = seasonal_decompose(seasonal_residual,energy_threshold=0.1,max_freq_num=self.decompose_args['max_freq_num'],axis=-1)
        residual = x - trend_x - seasonal_components
        return trend_x,seasonal_components,residual
    






