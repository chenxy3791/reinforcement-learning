import numpy as np

def smooth(a,winsize):
    """
    Smoothing with edge processing.
    Input:
        a:原始数据，NumPy 1-D array containing the data to be smoothed,必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
        winsize: smoothing window size needs, which must be odd number,as in the original MATLAB implementation
    Output:
        
    """
    out0 = np.convolve(a,np.ones(winsize,dtype=int),'valid')/winsize
    r = np.arange(1,winsize-1,2)
    start = np.cumsum(a[:winsize-1])[::2]/r
    stop = (np.cumsum(a[:-winsize:-1])[::2]/r)[::-1]
    return np.concatenate(( start , out0, stop ))
    
def movingaverage1(data, window_size):
    """
    扔掉卷积输出的前(window_size-1)个数据,输入与输出等长.
    The most naive method with two drawbacks:
    (1) Processing latency = window_size-1
    (2) The tail result seem a little bizzare due to lack of edge processing
    """
    window = np.ones(int(window_size))/float(window_size)
    maver = np.convolve(data, window, 'full') 
    return maver[(window_size-1):]    