 
# 这是快速傅里叶变换与快速傅里叶逆变换的python实现代码
# 作者：Foursheeps
 
import cmath
import sys
import time
import matplotlib as plt
from matplotlib.pyplot import plot, show, subplot, title, figure, stem
from scipy.fftpack import fft,ifft
 
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
 
def myfft(arr,le):
 
    len1 = len(arr)
    if le != len1:
        arr.extend([0]*(le-len1))
 
    N = len(arr)
    if N == 1:
        return arr
    W = cmath.exp(-(cmath.pi)*2j/N)
    arr1 = arr[0:N:2]
    arr2 = arr[1:N:2]
    ye = myfft(arr1,N//2)
    yo = myfft(arr2,N//2)
    y = [.0]*N
    for i in range(N//2):
        y[i] = ye[i] + yo[i] * W**i
        y[i+N//2] = ye[i] - yo[i] * W**i
    return y
 
def myifft(arr):
 
    N= len(arr)
    if N == 1:
        return arr
    W = cmath.exp((cmath.pi)*2j/N)
    arr1 = arr[0:N:2]
    arr2 = arr[1:N:2]
    ye = myifft(arr1)
    yo = myifft(arr2)
    y = [.0]*N
    for i in range(N//2):
        y[i] = (ye[i] + yo[i] * W**i)/2
        y[i+N//2] = (ye[i] - yo[i] * W**i)/2
    return y
 
def main():
    arr =[1,-2,3,0,-3,2,-1]

 
    start1 = time.time()
    xk1 = myfft(arr,1024)
    end1 = time.time()
 
    start2 = time.time()
    xn1 =myifft(xk1)
    end2 = time.time()
 
    start3 = time.time()
    xk2 = fft(arr,1024)
    end3 = time.time()
 
    start4 = time.time()
    xn2 = ifft(xk2)
    end4 = time.time()
 
    print("myfft用时",":",end1-start1,"   ","myifft用时",":",end2-start2)
    print("fft用时",":",end3-start3,"   ","ifft用时",":",end4-start4)
 
    figure('fft')
    subplot(411),stem(arr),title('原序列')
    subplot(412),stem(xk1),title('myfft结果')
    subplot(413),stem(xk2),title('fft结果')
    subplot(414),stem(xk1-xk2),title('快速傅里叶变换两者误差')
 
    figure('ifft')
    subplot(311),stem(arr),title('原序列')
    subplot(312),stem(xn1),title('myifft结果')
    subplot(313),stem(xn2),title('ifft结果')
    show()
 
 
if __name__ == '__main__':
    sys.exit(main())
