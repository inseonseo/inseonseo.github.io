

---
title: AutoEncoder
date: 2024-06-10 
categories: [ML, AutoEncoder]
tags: [modeling, stock]     # TAG names should always be lowercase
description: sotck project
---

## AutoEncoder


```python
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import datetime
import time
import requests as req
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
!pip install pandas_datareader
```


```python
!pip install yfinance
```

### 개요
- 목적 : 10일 수익률을 입력값으로 넣었을 때 비슷한 패턴 구현할 수 있는 알고리즘 만들기
    - window 길이 : 10
    - 예측값 : 10개 로그 수익률 입력으로 넣었을 때 얻는 10개 로그 수익률
- 입력 데이터에 대해 최소 - 최대 정규화 진행

## 1. Raw Data


```python
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

nasdaq=pdr.get_data_yahoo('^IXIC', '1999-12-31')
kosdaq = pdr.get_data_yahoo('^KQ11', '1999-12-31')
```

    yfinance: pandas_datareader support is deprecated & semi-broken so will be removed in a future verison. Just use yfinance.


    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed



```python
nasdaq.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>pct_change</th>
      <th>log_ret</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1999-12-31</th>
      <td>4056.989990</td>
      <td>4082.370117</td>
      <td>4032.330078</td>
      <td>4069.310059</td>
      <td>4069.310059</td>
      <td>762980000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>4186.189941</td>
      <td>4192.189941</td>
      <td>3989.709961</td>
      <td>4131.149902</td>
      <td>4131.149902</td>
      <td>1510070000</td>
      <td>0.015197</td>
      <td>0.015082</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>4020.000000</td>
      <td>4073.250000</td>
      <td>3898.229980</td>
      <td>3901.689941</td>
      <td>3901.689941</td>
      <td>1511840000</td>
      <td>-0.055544</td>
      <td>-0.057146</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>3854.350098</td>
      <td>3924.209961</td>
      <td>3734.870117</td>
      <td>3877.540039</td>
      <td>3877.540039</td>
      <td>1735670000</td>
      <td>-0.006190</td>
      <td>-0.006209</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>3834.439941</td>
      <td>3868.760010</td>
      <td>3715.620117</td>
      <td>3727.129883</td>
      <td>3727.129883</td>
      <td>1598320000</td>
      <td>-0.038790</td>
      <td>-0.039562</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>3711.090088</td>
      <td>3882.669922</td>
      <td>3711.090088</td>
      <td>3882.620117</td>
      <td>3882.620117</td>
      <td>1634930000</td>
      <td>0.041718</td>
      <td>0.040872</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>4002.229980</td>
      <td>4072.360107</td>
      <td>3958.830078</td>
      <td>4049.669922</td>
      <td>4049.669922</td>
      <td>1691710000</td>
      <td>0.043025</td>
      <td>0.042125</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>4031.379883</td>
      <td>4066.659912</td>
      <td>3904.820068</td>
      <td>3921.189941</td>
      <td>3921.189941</td>
      <td>1694460000</td>
      <td>-0.031726</td>
      <td>-0.032240</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>3950.949951</td>
      <td>3950.979980</td>
      <td>3834.530029</td>
      <td>3850.020020</td>
      <td>3850.020020</td>
      <td>1525900000</td>
      <td>-0.018150</td>
      <td>-0.018317</td>
    </tr>
    <tr>
      <th>2000-01-13</th>
      <td>3915.139893</td>
      <td>3957.469971</td>
      <td>3858.219971</td>
      <td>3957.209961</td>
      <td>3957.209961</td>
      <td>1476970000</td>
      <td>0.027841</td>
      <td>0.027461</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = nasdaq.iloc[6:, -2:]
```

### 2. 로그 수익률 데이터 가공


```python
window_length = 5
encoding_dim = 2
epochs = 100
test_samples = 200

import pandas as pd

nasdaq['pct_change'] = nasdaq['Adj Close'].pct_change()
nasdaq['log_ret'] = np.log(nasdaq['Adj Close']) - np.log(nasdaq['Adj Close'].shift(1))
```


```python
plt.figure(figsize = (10,10))
plt.suptitle('NASDAQ Adjusted Close and Percent Change')

# First subplot for Adjusted Close
plt.subplot(2, 1, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(nasdaq['Adj Close'], label='Adjusted Close')
plt.title('Adjusted Close')  # Title for the first subplot
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()  

# Second subplot for Percent Change
plt.subplot(2, 1, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(nasdaq['pct_change'], label='Percent Change', color='orange')
plt.title('Percent Change')  # Title for the second subplot
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()  


plt.show()
```


    
![png](output_11_0.png)
    


- (종가 - 전일 종가)로 로그 수익률 계산
- 로그 함수 성질 이용


```python
def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 20))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
        
        
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
```


```python
scaler = MinMaxScaler()
x_train = np.array([scaler.fit_transform(nasdaq['log_ret'].values[i-window_length:i].reshape(-1, 1)) for i in range(window_length+1,len(nasdaq['log_ret']))])
x_train = x_train[:-test_samples]
x_test = x_train[-test_samples:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print("테스트 데이터 비중: {}%".format((test_samples/len(x_train))*100))
```

    테스트 데이터 비중: 3.3709758975223325%



```python
x_train.shape
```




    (5933, 5, 1)




```python
x_test.shape
```




    (200, 5, 1)



![image-3.png](attachment:image-3.png)

### 첫 번째 모델 : 단순 MLP 오토인코더
![image.png](attachment:image.png)

- 입력 받는 10개 데이터에 10개 출력이 3개 압축층 통과하고 출력층에 10개 출력을 보냄
- 각 모델 손실값 변화에 대해, 테스트 데이터로 오토인코더 예측값 출력하여 비교


```python
x_train_sample = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_sample = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_window = Input(shape=(window_length,))
# "encoded"는 input데이터를 encode로 압축한다.
encoded = Dense(encoding_dim, activation='relu')(input_window)
# "decoded"는 압축된 데이터로 input 데이터를 최대한 표현한다.
decoded = Dense(window_length, activation='sigmoid')(encoded)

# 2개의 모델이 있다.
# 하나는 인풋데이터를 인코더 디코더를 통해 인풋값을 예측하는 모델.
autoencoder = Model(input_window, decoded)
# 다른 하나는 인풋데이터를 인코더 디코더로 압축하는 모델.
encoder = Model(input_window, encoded)

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_sample, x_train_sample,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test_sample, x_test_sample))

decoded_stocks = autoencoder.predict(x_test_sample)
plot_history(history)
plot_examples(x_test_sample, decoded_stocks)
```

    Model: "model_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_6 (InputLayer)         [(None, 5)]               0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 2)                 12        
    _________________________________________________________________
    dense_10 (Dense)             (None, 5)                 15        
    =================================================================
    Total params: 27
    Trainable params: 27
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    6/6 [==============================] - 0s 21ms/step - loss: 0.7233 - val_loss: 0.7199
    Epoch 2/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7209 - val_loss: 0.7176
    Epoch 3/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7186 - val_loss: 0.7153
    Epoch 4/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7164 - val_loss: 0.7133
    Epoch 5/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7144 - val_loss: 0.7113
    Epoch 6/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7125 - val_loss: 0.7095
    Epoch 7/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7107 - val_loss: 0.7078
    Epoch 8/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7089 - val_loss: 0.7062
    Epoch 9/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7074 - val_loss: 0.7046
    Epoch 10/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7058 - val_loss: 0.7032
    Epoch 11/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7044 - val_loss: 0.7018
    Epoch 12/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7030 - val_loss: 0.7005
    Epoch 13/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7017 - val_loss: 0.6993
    Epoch 14/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.7004 - val_loss: 0.6981
    Epoch 15/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6992 - val_loss: 0.6969
    Epoch 16/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6980 - val_loss: 0.6958
    Epoch 17/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6968 - val_loss: 0.6947
    Epoch 18/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6957 - val_loss: 0.6937
    Epoch 19/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6946 - val_loss: 0.6926
    Epoch 20/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6935 - val_loss: 0.6916
    Epoch 21/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6924 - val_loss: 0.6906
    Epoch 22/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6914 - val_loss: 0.6896
    Epoch 23/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6903 - val_loss: 0.6887
    Epoch 24/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6893 - val_loss: 0.6877
    Epoch 25/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.6883 - val_loss: 0.6867
    Epoch 26/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6872 - val_loss: 0.6858
    Epoch 27/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6862 - val_loss: 0.6848
    Epoch 28/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6853 - val_loss: 0.6838
    Epoch 29/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6843 - val_loss: 0.6829
    Epoch 30/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6833 - val_loss: 0.6819
    Epoch 31/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6823 - val_loss: 0.6810
    Epoch 32/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6813 - val_loss: 0.6801
    Epoch 33/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6803 - val_loss: 0.6791
    Epoch 34/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6794 - val_loss: 0.6782
    Epoch 35/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6784 - val_loss: 0.6772
    Epoch 36/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6774 - val_loss: 0.6762
    Epoch 37/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6764 - val_loss: 0.6752
    Epoch 38/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6754 - val_loss: 0.6743
    Epoch 39/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6745 - val_loss: 0.6733
    Epoch 40/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6735 - val_loss: 0.6722
    Epoch 41/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6725 - val_loss: 0.6712
    Epoch 42/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6715 - val_loss: 0.6701
    Epoch 43/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.6705 - val_loss: 0.6691
    Epoch 44/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6695 - val_loss: 0.6681
    Epoch 45/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6685 - val_loss: 0.6670
    Epoch 46/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6675 - val_loss: 0.6660
    Epoch 47/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6665 - val_loss: 0.6649
    Epoch 48/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.6655 - val_loss: 0.6639
    Epoch 49/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6644 - val_loss: 0.6628
    Epoch 50/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6634 - val_loss: 0.6618
    Epoch 51/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6624 - val_loss: 0.6607
    Epoch 52/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6614 - val_loss: 0.6597
    Epoch 53/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6603 - val_loss: 0.6587
    Epoch 54/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6593 - val_loss: 0.6576
    Epoch 55/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6583 - val_loss: 0.6566
    Epoch 56/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6573 - val_loss: 0.6556
    Epoch 57/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6563 - val_loss: 0.6545
    Epoch 58/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6552 - val_loss: 0.6535
    Epoch 59/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6542 - val_loss: 0.6525
    Epoch 60/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6532 - val_loss: 0.6515
    Epoch 61/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6522 - val_loss: 0.6505
    Epoch 62/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6512 - val_loss: 0.6495
    Epoch 63/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6502 - val_loss: 0.6485
    Epoch 64/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6492 - val_loss: 0.6475
    Epoch 65/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6482 - val_loss: 0.6465
    Epoch 66/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6472 - val_loss: 0.6455
    Epoch 67/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6462 - val_loss: 0.6446
    Epoch 68/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6452 - val_loss: 0.6436
    Epoch 69/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6442 - val_loss: 0.6426
    Epoch 70/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6432 - val_loss: 0.6416
    Epoch 71/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6423 - val_loss: 0.6406
    Epoch 72/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6413 - val_loss: 0.6396
    Epoch 73/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6403 - val_loss: 0.6387
    Epoch 74/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6393 - val_loss: 0.6377
    Epoch 75/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6383 - val_loss: 0.6367
    Epoch 76/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6374 - val_loss: 0.6358
    Epoch 77/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6364 - val_loss: 0.6348
    Epoch 78/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6354 - val_loss: 0.6339
    Epoch 79/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6345 - val_loss: 0.6330
    Epoch 80/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6335 - val_loss: 0.6320
    Epoch 81/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6326 - val_loss: 0.6311
    Epoch 82/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6316 - val_loss: 0.6302
    Epoch 83/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6307 - val_loss: 0.6293
    Epoch 84/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6297 - val_loss: 0.6283
    Epoch 85/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6288 - val_loss: 0.6274
    Epoch 86/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6279 - val_loss: 0.6265
    Epoch 87/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6269 - val_loss: 0.6256
    Epoch 88/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6260 - val_loss: 0.6247
    Epoch 89/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6251 - val_loss: 0.6238
    Epoch 90/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6242 - val_loss: 0.6229
    Epoch 91/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6233 - val_loss: 0.6221
    Epoch 92/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6223 - val_loss: 0.6212
    Epoch 93/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6214 - val_loss: 0.6203
    Epoch 94/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6205 - val_loss: 0.6195
    Epoch 95/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6196 - val_loss: 0.6186
    Epoch 96/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6187 - val_loss: 0.6178
    Epoch 97/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6179 - val_loss: 0.6169
    Epoch 98/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6170 - val_loss: 0.6161
    Epoch 99/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6161 - val_loss: 0.6153
    Epoch 100/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6152 - val_loss: 0.6145



    
![png](output_20_1.png)
    



    
![png](output_20_2.png)
    


### 두 번째 : Deep 오토인코더

- 다층 퍼셉트론 모델에 배치층 추가한 구조
- 인코더 + 차원 축소층, 디코더 + 차원 확장층 
- 공통적으로 배치정규화층 추가 구성


```python
x_train_sample = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_sample = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


input_window = Input(shape=(window_length,))

x = Dense(6, activation='relu')(input_window)
x = BatchNormalization()(x) # 입력값 정규화
encoded = Dense(encoding_dim, activation='relu')(x) 
# 완전 연결층 정의하면서 encoding 차원으로 축소 전달
# "decoded" is the lossy reconstruction of the input

x = Dense(6, activation='relu')(encoded) # 3차원 -> 6차원
x = BatchNormalization()(x) # 입력값 정규화
decoded = Dense(window_length, activation='sigmoid')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_window, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_window, encoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_sample, x_train_sample,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test_sample, x_test_sample))

decoded_stocks = autoencoder.predict(x_test_sample)
plot_history(history)
plot_examples(x_test_sample, decoded_stocks)
```

    Model: "model_12"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_7 (InputLayer)         [(None, 5)]               0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 6)                 36        
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 6)                 24        
    _________________________________________________________________
    dense_12 (Dense)             (None, 2)                 14        
    _________________________________________________________________
    dense_13 (Dense)             (None, 6)                 18        
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 6)                 24        
    _________________________________________________________________
    dense_14 (Dense)             (None, 5)                 35        
    =================================================================
    Total params: 151
    Trainable params: 127
    Non-trainable params: 24
    _________________________________________________________________
    Epoch 1/100
    6/6 [==============================] - 1s 29ms/step - loss: 0.7622 - val_loss: 0.6936
    Epoch 2/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.7497 - val_loss: 0.6932
    Epoch 3/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7382 - val_loss: 0.6927
    Epoch 4/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7289 - val_loss: 0.6922
    Epoch 5/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7212 - val_loss: 0.6918
    Epoch 6/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7149 - val_loss: 0.6913
    Epoch 7/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7090 - val_loss: 0.6908
    Epoch 8/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.7029 - val_loss: 0.6903
    Epoch 9/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6966 - val_loss: 0.6898
    Epoch 10/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6907 - val_loss: 0.6891
    Epoch 11/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6843 - val_loss: 0.6884
    Epoch 12/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6780 - val_loss: 0.6874
    Epoch 13/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6713 - val_loss: 0.6861
    Epoch 14/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6649 - val_loss: 0.6848
    Epoch 15/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6582 - val_loss: 0.6833
    Epoch 16/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6521 - val_loss: 0.6817
    Epoch 17/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6461 - val_loss: 0.6801
    Epoch 18/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6414 - val_loss: 0.6784
    Epoch 19/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.6378 - val_loss: 0.6767
    Epoch 20/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6349 - val_loss: 0.6750
    Epoch 21/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6323 - val_loss: 0.6734
    Epoch 22/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6300 - val_loss: 0.6720
    Epoch 23/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6278 - val_loss: 0.6706
    Epoch 24/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6258 - val_loss: 0.6693
    Epoch 25/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6240 - val_loss: 0.6680
    Epoch 26/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6221 - val_loss: 0.6666
    Epoch 27/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6204 - val_loss: 0.6652
    Epoch 28/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6189 - val_loss: 0.6639
    Epoch 29/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6176 - val_loss: 0.6625
    Epoch 30/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6163 - val_loss: 0.6612
    Epoch 31/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6149 - val_loss: 0.6598
    Epoch 32/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6138 - val_loss: 0.6585
    Epoch 33/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6126 - val_loss: 0.6570
    Epoch 34/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6115 - val_loss: 0.6557
    Epoch 35/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.6104 - val_loss: 0.6543
    Epoch 36/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6093 - val_loss: 0.6528
    Epoch 37/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6084 - val_loss: 0.6513
    Epoch 38/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6074 - val_loss: 0.6499
    Epoch 39/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6063 - val_loss: 0.6486
    Epoch 40/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6054 - val_loss: 0.6473
    Epoch 41/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6044 - val_loss: 0.6462
    Epoch 42/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6034 - val_loss: 0.6450
    Epoch 43/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6025 - val_loss: 0.6439
    Epoch 44/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6015 - val_loss: 0.6428
    Epoch 45/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6005 - val_loss: 0.6418
    Epoch 46/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5994 - val_loss: 0.6408
    Epoch 47/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5988 - val_loss: 0.6397
    Epoch 48/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5975 - val_loss: 0.6390
    Epoch 49/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5963 - val_loss: 0.6379
    Epoch 50/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5952 - val_loss: 0.6367
    Epoch 51/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5941 - val_loss: 0.6353
    Epoch 52/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5931 - val_loss: 0.6335
    Epoch 53/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5921 - val_loss: 0.6324
    Epoch 54/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5910 - val_loss: 0.6312
    Epoch 55/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5900 - val_loss: 0.6295
    Epoch 56/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5889 - val_loss: 0.6275
    Epoch 57/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5880 - val_loss: 0.6263
    Epoch 58/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5868 - val_loss: 0.6247
    Epoch 59/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.5858 - val_loss: 0.6233
    Epoch 60/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5850 - val_loss: 0.6217
    Epoch 61/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5839 - val_loss: 0.6196
    Epoch 62/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5829 - val_loss: 0.6181
    Epoch 63/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.5822 - val_loss: 0.6157
    Epoch 64/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5808 - val_loss: 0.6147
    Epoch 65/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5800 - val_loss: 0.6135
    Epoch 66/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.5789 - val_loss: 0.6110
    Epoch 67/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5778 - val_loss: 0.6090
    Epoch 68/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5767 - val_loss: 0.6066
    Epoch 69/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5761 - val_loss: 0.6038
    Epoch 70/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5747 - val_loss: 0.6018
    Epoch 71/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5735 - val_loss: 0.5988
    Epoch 72/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5725 - val_loss: 0.5961
    Epoch 73/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5712 - val_loss: 0.5937
    Epoch 74/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5702 - val_loss: 0.5904
    Epoch 75/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5688 - val_loss: 0.5875
    Epoch 76/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5675 - val_loss: 0.5848
    Epoch 77/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5659 - val_loss: 0.5821
    Epoch 78/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5647 - val_loss: 0.5799
    Epoch 79/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5632 - val_loss: 0.5781
    Epoch 80/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5615 - val_loss: 0.5763
    Epoch 81/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5601 - val_loss: 0.5751
    Epoch 82/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5583 - val_loss: 0.5739
    Epoch 83/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5566 - val_loss: 0.5735
    Epoch 84/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5549 - val_loss: 0.5732
    Epoch 85/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5533 - val_loss: 0.5730
    Epoch 86/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5519 - val_loss: 0.5730
    Epoch 87/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5502 - val_loss: 0.5732
    Epoch 88/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5486 - val_loss: 0.5741
    Epoch 89/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.5471 - val_loss: 0.5752
    Epoch 90/100
    6/6 [==============================] - 0s 7ms/step - loss: 0.5456 - val_loss: 0.5766
    Epoch 91/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5439 - val_loss: 0.5776
    Epoch 92/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5427 - val_loss: 0.5781
    Epoch 93/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5413 - val_loss: 0.5782
    Epoch 94/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5402 - val_loss: 0.5784
    Epoch 95/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5391 - val_loss: 0.5785
    Epoch 96/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5377 - val_loss: 0.5783
    Epoch 97/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5366 - val_loss: 0.5781
    Epoch 98/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5355 - val_loss: 0.5777
    Epoch 99/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5340 - val_loss: 0.5773
    Epoch 100/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.5331 - val_loss: 0.5769



    
![png](output_23_1.png)
    



    
![png](output_23_2.png)
    


- 100번 에폭 동안 손실값 꾸준히 감소
- 입력값 예측값 : 상승과 하락 재현한 부분 몇 군데 있

## 1D Convolutional autoencoder



```python
input_window = Input(shape=(window_length,1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 2, activation='relu')(x) # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_stocks = autoencoder.predict(x_test)
plot_history(history)
plot_examples(x_test, decoded_stocks)
```

    Model: "model_15"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_8 (InputLayer)         [(None, 5, 1)]            0         
    _________________________________________________________________
    conv1d_5 (Conv1D)            (None, 5, 16)             64        
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 3, 16)             0         
    _________________________________________________________________
    conv1d_6 (Conv1D)            (None, 3, 1)              49        
    _________________________________________________________________
    max_pooling1d_3 (MaxPooling1 (None, 2, 1)              0         
    _________________________________________________________________
    conv1d_7 (Conv1D)            (None, 2, 1)              4         
    _________________________________________________________________
    up_sampling1d_2 (UpSampling1 (None, 4, 1)              0         
    _________________________________________________________________
    conv1d_8 (Conv1D)            (None, 3, 16)             48        
    _________________________________________________________________
    up_sampling1d_3 (UpSampling1 (None, 6, 16)             0         
    _________________________________________________________________
    conv1d_9 (Conv1D)            (None, 6, 1)              49        
    =================================================================
    Total params: 214
    Trainable params: 214
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /tmp/ipykernel_1050/1298710627.py in <module>
         22 
         23 autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    ---> 24 history = autoencoder.fit(x_train, x_train,
         25                 epochs=epochs,
         26                 batch_size=1024,


    /opt/conda/lib/python3.9/site-packages/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1182                 _r=1):
       1183               callbacks.on_train_batch_begin(step)
    -> 1184               tmp_logs = self.train_function(iterator)
       1185               if data_handler.should_sync:
       1186                 context.async_wait()


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        883 
        884       with OptionalXlaContext(self._jit_compile):
    --> 885         result = self._call(*args, **kwds)
        886 
        887       new_tracing_count = self.experimental_get_tracing_count()


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        931       # This is the first call of __call__, so we have to initialize.
        932       initializers = []
    --> 933       self._initialize(args, kwds, add_initializers_to=initializers)
        934     finally:
        935       # At this point we know that the initialization is complete (or less


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py in _initialize(self, args, kwds, add_initializers_to)
        757     self._graph_deleter = FunctionDeleter(self._lifted_initializer_graph)
        758     self._concrete_stateful_fn = (
    --> 759         self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
        760             *args, **kwds))
        761 


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/function.py in _get_concrete_function_internal_garbage_collected(self, *args, **kwargs)
       3064       args, kwargs = None, None
       3065     with self._lock:
    -> 3066       graph_function, _ = self._maybe_define_function(args, kwargs)
       3067     return graph_function
       3068 


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/function.py in _maybe_define_function(self, args, kwargs)
       3461 
       3462           self._function_cache.missed.add(call_context_key)
    -> 3463           graph_function = self._create_graph_function(args, kwargs)
       3464           self._function_cache.primary[cache_key] = graph_function
       3465 


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/function.py in _create_graph_function(self, args, kwargs, override_flat_arg_shapes)
       3296     arg_names = base_arg_names + missing_arg_names
       3297     graph_function = ConcreteFunction(
    -> 3298         func_graph_module.func_graph_from_py_func(
       3299             self._name,
       3300             self._python_function,


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes, acd_record_initial_resource_uses)
       1005         _, original_func = tf_decorator.unwrap(python_func)
       1006 
    -> 1007       func_outputs = python_func(*func_args, **func_kwargs)
       1008 
       1009       # invariant: `func_outputs` contains only Tensors, CompositeTensors,


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py in wrapped_fn(*args, **kwds)
        666         # the function a weak reference to itself to avoid a reference cycle.
        667         with OptionalXlaContext(compile_with_xla):
    --> 668           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
        669         return out
        670 


    /opt/conda/lib/python3.9/site-packages/tensorflow/python/framework/func_graph.py in wrapper(*args, **kwargs)
        992           except Exception as e:  # pylint:disable=broad-except
        993             if hasattr(e, "ag_error_metadata"):
    --> 994               raise e.ag_error_metadata.to_exception(e)
        995             else:
        996               raise


    ValueError: in user code:
    
        /opt/conda/lib/python3.9/site-packages/keras/engine/training.py:853 train_function  *
            return step_function(self, iterator)
        /opt/conda/lib/python3.9/site-packages/keras/engine/training.py:842 step_function  **
            outputs = model.distribute_strategy.run(run_step, args=(data,))
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/distribute/distribute_lib.py:1286 run
            return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/distribute/distribute_lib.py:2849 call_for_each_replica
            return self._call_for_each_replica(fn, args, kwargs)
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/distribute/distribute_lib.py:3632 _call_for_each_replica
            return fn(*args, **kwargs)
        /opt/conda/lib/python3.9/site-packages/keras/engine/training.py:835 run_step  **
            outputs = model.train_step(data)
        /opt/conda/lib/python3.9/site-packages/keras/engine/training.py:788 train_step
            loss = self.compiled_loss(
        /opt/conda/lib/python3.9/site-packages/keras/engine/compile_utils.py:201 __call__
            loss_value = loss_obj(y_t, y_p, sample_weight=sw)
        /opt/conda/lib/python3.9/site-packages/keras/losses.py:141 __call__
            losses = call_fn(y_true, y_pred)
        /opt/conda/lib/python3.9/site-packages/keras/losses.py:245 call  **
            return ag_fn(y_true, y_pred, **self._fn_kwargs)
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:206 wrapper
            return target(*args, **kwargs)
        /opt/conda/lib/python3.9/site-packages/keras/losses.py:1809 binary_crossentropy
            backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:206 wrapper
            return target(*args, **kwargs)
        /opt/conda/lib/python3.9/site-packages/keras/backend.py:5000 binary_crossentropy
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:206 wrapper
            return target(*args, **kwargs)
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/ops/nn_impl.py:245 sigmoid_cross_entropy_with_logits_v2
            return sigmoid_cross_entropy_with_logits(
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:206 wrapper
            return target(*args, **kwargs)
        /opt/conda/lib/python3.9/site-packages/tensorflow/python/ops/nn_impl.py:132 sigmoid_cross_entropy_with_logits
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
    
        ValueError: logits and labels must have the same shape ((None, 6, 1) vs (None, 5, 1))



## LSTM(recurrent neural networks) autoencoder


```python
inputs = Input(shape=(window_length, 1))
encoded = LSTM(encoding_dim)(inputs)

decoded = RepeatVector(window_length)(encoded)
decoded = LSTM(1, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
sequence_autoencoder.summary()

sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = sequence_autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_stocks = sequence_autoencoder.predict(x_test)
plot_history(history)
plot_examples(x_test, decoded_stocks)
```

    Model: "model_16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_9 (InputLayer)         [(None, 5, 1)]            0         
    _________________________________________________________________
    lstm_4 (LSTM)                (None, 2)                 32        
    _________________________________________________________________
    repeat_vector_2 (RepeatVecto (None, 5, 2)              0         
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 5, 1)              16        
    =================================================================
    Total params: 48
    Trainable params: 48
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    6/6 [==============================] - 3s 129ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 2/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 3/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 4/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 5/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 6/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 7/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 8/100
    6/6 [==============================] - 0s 10ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 9/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 10/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 11/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 12/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 13/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 14/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 15/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 16/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 17/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 18/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 19/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 20/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 21/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 22/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 23/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 24/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 25/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 26/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 27/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 28/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 29/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 30/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 31/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 32/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 33/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 34/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 35/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 36/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 37/100
    6/6 [==============================] - 0s 9ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 38/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 39/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 40/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 41/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 42/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 43/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 44/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 45/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 46/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 47/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 48/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 49/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 50/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 51/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 52/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 53/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 54/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 55/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 56/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 57/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 58/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 59/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 60/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 61/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 62/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 63/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 64/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 65/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 66/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 67/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 68/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 69/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 70/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 71/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 72/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 73/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 74/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 75/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 76/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 77/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 78/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 79/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 80/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 81/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 82/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 83/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 84/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 85/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 86/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 87/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 88/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 89/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 90/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 91/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 92/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 93/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 94/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 95/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 96/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 97/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 98/100
    6/6 [==============================] - 0s 8ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 99/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594
    Epoch 100/100
    6/6 [==============================] - 0s 7ms/step - loss: 7.6844 - val_loss: 7.4594



    
![png](output_28_1.png)
    



    
![png](output_28_2.png)
    



```python
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Dropout
# from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dropout, RepeatVector, TimeDistributed


window_length = 5
encoding_dim = 2
epochs = 100
test_samples = 200


model = Sequential([
    LSTM(128, input_shape=(window_length, 1)),
    Dropout(0.3),
    RepeatVector(window_length),
    LSTM(128, return_sequences=True), 
    Dropout(0.3),
    TimeDistributed(Dense(1))
])


model.compile(optimizer='adam', loss='mae')
history = model.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_stocks = model.predict(x_test)
plot_history(history)
plot_examples(x_test, decoded_stocks)
```

    Epoch 1/100
    6/6 [==============================] - 3s 119ms/step - loss: 0.4514 - val_loss: 0.3610
    Epoch 2/100
    6/6 [==============================] - 0s 11ms/step - loss: 0.3587 - val_loss: 0.3493
    Epoch 3/100
    6/6 [==============================] - 0s 11ms/step - loss: 0.3440 - val_loss: 0.3323
    Epoch 4/100
    6/6 [==============================] - 0s 11ms/step - loss: 0.3399 - val_loss: 0.3258
    Epoch 5/100
    6/6 [==============================] - 0s 11ms/step - loss: 0.3299 - val_loss: 0.3225
    Epoch 6/100
    6/6 [==============================] - 0s 11ms/step - loss: 0.3248 - val_loss: 0.3139
    Epoch 7/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.3191 - val_loss: 0.3089
    Epoch 8/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.3132 - val_loss: 0.3043
    Epoch 9/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.3092 - val_loss: 0.3003
    Epoch 10/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.3035 - val_loss: 0.2951
    Epoch 11/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2976 - val_loss: 0.2876
    Epoch 12/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2916 - val_loss: 0.2812
    Epoch 13/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2861 - val_loss: 0.2761
    Epoch 14/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2826 - val_loss: 0.2722
    Epoch 15/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2787 - val_loss: 0.2683
    Epoch 16/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2758 - val_loss: 0.2665
    Epoch 17/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2745 - val_loss: 0.2649
    Epoch 18/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2736 - val_loss: 0.2645
    Epoch 19/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2719 - val_loss: 0.2635
    Epoch 20/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2712 - val_loss: 0.2623
    Epoch 21/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2703 - val_loss: 0.2614
    Epoch 22/100
    6/6 [==============================] - 0s 13ms/step - loss: 0.2690 - val_loss: 0.2606
    Epoch 23/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2683 - val_loss: 0.2599
    Epoch 24/100
    6/6 [==============================] - 0s 11ms/step - loss: 0.2679 - val_loss: 0.2594
    Epoch 25/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2669 - val_loss: 0.2590
    Epoch 26/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2660 - val_loss: 0.2581
    Epoch 27/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2651 - val_loss: 0.2572
    Epoch 28/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2640 - val_loss: 0.2559
    Epoch 29/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2629 - val_loss: 0.2546
    Epoch 30/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2616 - val_loss: 0.2531
    Epoch 31/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2604 - val_loss: 0.2520
    Epoch 32/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2591 - val_loss: 0.2503
    Epoch 33/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2572 - val_loss: 0.2473
    Epoch 34/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2555 - val_loss: 0.2451
    Epoch 35/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2533 - val_loss: 0.2435
    Epoch 36/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2522 - val_loss: 0.2400
    Epoch 37/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.2492 - val_loss: 0.2366
    Epoch 38/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2463 - val_loss: 0.2344
    Epoch 39/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2422 - val_loss: 0.2269
    Epoch 40/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2389 - val_loss: 0.2229
    Epoch 41/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2349 - val_loss: 0.2210
    Epoch 42/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2320 - val_loss: 0.2146
    Epoch 43/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2287 - val_loss: 0.2130
    Epoch 44/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2263 - val_loss: 0.2107
    Epoch 45/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2235 - val_loss: 0.2053
    Epoch 46/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2205 - val_loss: 0.2032
    Epoch 47/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2177 - val_loss: 0.2027
    Epoch 48/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2166 - val_loss: 0.1976
    Epoch 49/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2136 - val_loss: 0.1960
    Epoch 50/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2124 - val_loss: 0.1965
    Epoch 51/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2120 - val_loss: 0.1969
    Epoch 52/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2108 - val_loss: 0.1915
    Epoch 53/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2102 - val_loss: 0.1940
    Epoch 54/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2097 - val_loss: 0.1911
    Epoch 55/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2087 - val_loss: 0.1924
    Epoch 56/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2082 - val_loss: 0.1888
    Epoch 57/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2070 - val_loss: 0.1902
    Epoch 58/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2076 - val_loss: 0.1915
    Epoch 59/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2062 - val_loss: 0.1865
    Epoch 60/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.2053 - val_loss: 0.1861
    Epoch 61/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2048 - val_loss: 0.1898
    Epoch 62/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2045 - val_loss: 0.1876
    Epoch 63/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2047 - val_loss: 0.1876
    Epoch 64/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2051 - val_loss: 0.1857
    Epoch 65/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2036 - val_loss: 0.1873
    Epoch 66/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2029 - val_loss: 0.1869
    Epoch 67/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2027 - val_loss: 0.1849
    Epoch 68/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2028 - val_loss: 0.1840
    Epoch 69/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2013 - val_loss: 0.1858
    Epoch 70/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2019 - val_loss: 0.1882
    Epoch 71/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2011 - val_loss: 0.1830
    Epoch 72/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.2006 - val_loss: 0.1831
    Epoch 73/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.2012 - val_loss: 0.1814
    Epoch 74/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1995 - val_loss: 0.1834
    Epoch 75/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1999 - val_loss: 0.1816
    Epoch 76/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1990 - val_loss: 0.1820
    Epoch 77/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1985 - val_loss: 0.1807
    Epoch 78/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1979 - val_loss: 0.1795
    Epoch 79/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1980 - val_loss: 0.1788
    Epoch 80/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1975 - val_loss: 0.1795
    Epoch 81/100
    6/6 [==============================] - 0s 10ms/step - loss: 0.1970 - val_loss: 0.1798
    Epoch 82/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1961 - val_loss: 0.1775
    Epoch 83/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1956 - val_loss: 0.1774
    Epoch 84/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1954 - val_loss: 0.1776
    Epoch 85/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1949 - val_loss: 0.1783
    Epoch 86/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1956 - val_loss: 0.1803
    Epoch 87/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1956 - val_loss: 0.1803
    Epoch 88/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1937 - val_loss: 0.1805
    Epoch 89/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1934 - val_loss: 0.1771
    Epoch 90/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1918 - val_loss: 0.1726
    Epoch 91/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1901 - val_loss: 0.1738
    Epoch 92/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1888 - val_loss: 0.1711
    Epoch 93/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1881 - val_loss: 0.1676
    Epoch 94/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1861 - val_loss: 0.1667
    Epoch 95/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1841 - val_loss: 0.1652
    Epoch 96/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1830 - val_loss: 0.1655
    Epoch 97/100
    6/6 [==============================] - 0s 8ms/step - loss: 0.1808 - val_loss: 0.1666
    Epoch 98/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1788 - val_loss: 0.1606
    Epoch 99/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1747 - val_loss: 0.1581
    Epoch 100/100
    6/6 [==============================] - 0s 9ms/step - loss: 0.1715 - val_loss: 0.1467



    
![png](output_29_1.png)
    



    
![png](output_29_2.png)
    


## Encoder를 Feature로 활용하기

- 인코더 모델에서 predict() 함수 호출하여 출력된 값을 새로운 특성 데이터로 사용
- 입력 데이터층은 오토인코더와 동일하지만 출력층이 encoded 층임


```python
x_train_simple = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_simple = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_window = Input(shape=(window_length,))
encoded = Dense(encoding_dim, activation='relu')(input_window)
decoded = Dense(window_length, activation='sigmoid')(encoded)

autoencoder = Model(input_window, decoded)
encoder = Model(input_window, encoded)


autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_simple, x_train_simple,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test_simple, x_test_simple))

decoded_stocks = autoencoder.predict(x_test_simple) 
# 출력된 값 새로운 특성 데이터로 사용

#############
compress_x_train = encoder.predict(x_train_simple)
# 학습 데이터를 인코더 모델에 전달해 압축된 출력값 생성
compress_x_test = encoder.predict(x_test_simple)
# 테스트 데이터를 인코더 모델에 전달해 압축된 출력값 생성
```

    Model: "model_18"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_10 (InputLayer)        [(None, 5)]               0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 2)                 12        
    _________________________________________________________________
    dense_18 (Dense)             (None, 5)                 15        
    =================================================================
    Total params: 27
    Trainable params: 27
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    6/6 [==============================] - 0s 21ms/step - loss: 0.7146 - val_loss: 0.7022
    Epoch 2/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7108 - val_loss: 0.6988
    Epoch 3/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7073 - val_loss: 0.6957
    Epoch 4/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7039 - val_loss: 0.6926
    Epoch 5/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.7006 - val_loss: 0.6898
    Epoch 6/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6975 - val_loss: 0.6871
    Epoch 7/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6946 - val_loss: 0.6845
    Epoch 8/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6919 - val_loss: 0.6821
    Epoch 9/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6893 - val_loss: 0.6798
    Epoch 10/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6869 - val_loss: 0.6777
    Epoch 11/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6846 - val_loss: 0.6757
    Epoch 12/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6824 - val_loss: 0.6737
    Epoch 13/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6803 - val_loss: 0.6719
    Epoch 14/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6784 - val_loss: 0.6702
    Epoch 15/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6765 - val_loss: 0.6685
    Epoch 16/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6748 - val_loss: 0.6669
    Epoch 17/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6731 - val_loss: 0.6653
    Epoch 18/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6715 - val_loss: 0.6639
    Epoch 19/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6699 - val_loss: 0.6624
    Epoch 20/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6684 - val_loss: 0.6610
    Epoch 21/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6670 - val_loss: 0.6597
    Epoch 22/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6655 - val_loss: 0.6584
    Epoch 23/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6642 - val_loss: 0.6571
    Epoch 24/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6629 - val_loss: 0.6558
    Epoch 25/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6616 - val_loss: 0.6546
    Epoch 26/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6603 - val_loss: 0.6534
    Epoch 27/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6590 - val_loss: 0.6522
    Epoch 28/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6578 - val_loss: 0.6510
    Epoch 29/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6566 - val_loss: 0.6499
    Epoch 30/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6554 - val_loss: 0.6487
    Epoch 31/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6543 - val_loss: 0.6476
    Epoch 32/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6531 - val_loss: 0.6465
    Epoch 33/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6520 - val_loss: 0.6454
    Epoch 34/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6509 - val_loss: 0.6443
    Epoch 35/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6497 - val_loss: 0.6432
    Epoch 36/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6487 - val_loss: 0.6422
    Epoch 37/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6476 - val_loss: 0.6411
    Epoch 38/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6465 - val_loss: 0.6401
    Epoch 39/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6454 - val_loss: 0.6390
    Epoch 40/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6444 - val_loss: 0.6380
    Epoch 41/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6434 - val_loss: 0.6370
    Epoch 42/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6423 - val_loss: 0.6360
    Epoch 43/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6413 - val_loss: 0.6350
    Epoch 44/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6403 - val_loss: 0.6341
    Epoch 45/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6393 - val_loss: 0.6331
    Epoch 46/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6384 - val_loss: 0.6321
    Epoch 47/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6374 - val_loss: 0.6312
    Epoch 48/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6364 - val_loss: 0.6302
    Epoch 49/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6355 - val_loss: 0.6293
    Epoch 50/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6345 - val_loss: 0.6284
    Epoch 51/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6336 - val_loss: 0.6275
    Epoch 52/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6327 - val_loss: 0.6266
    Epoch 53/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6318 - val_loss: 0.6257
    Epoch 54/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6309 - val_loss: 0.6248
    Epoch 55/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6300 - val_loss: 0.6239
    Epoch 56/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6291 - val_loss: 0.6231
    Epoch 57/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6282 - val_loss: 0.6222
    Epoch 58/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6274 - val_loss: 0.6213
    Epoch 59/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6265 - val_loss: 0.6205
    Epoch 60/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6257 - val_loss: 0.6197
    Epoch 61/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6248 - val_loss: 0.6188
    Epoch 62/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6240 - val_loss: 0.6180
    Epoch 63/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6232 - val_loss: 0.6172
    Epoch 64/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6224 - val_loss: 0.6164
    Epoch 65/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6215 - val_loss: 0.6156
    Epoch 66/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6207 - val_loss: 0.6148
    Epoch 67/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6200 - val_loss: 0.6141
    Epoch 68/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6192 - val_loss: 0.6133
    Epoch 69/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6184 - val_loss: 0.6125
    Epoch 70/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6176 - val_loss: 0.6118
    Epoch 71/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6169 - val_loss: 0.6110
    Epoch 72/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6161 - val_loss: 0.6103
    Epoch 73/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6153 - val_loss: 0.6096
    Epoch 74/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6146 - val_loss: 0.6088
    Epoch 75/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6139 - val_loss: 0.6081
    Epoch 76/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6131 - val_loss: 0.6074
    Epoch 77/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6124 - val_loss: 0.6067
    Epoch 78/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6117 - val_loss: 0.6060
    Epoch 79/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6110 - val_loss: 0.6053
    Epoch 80/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6102 - val_loss: 0.6046
    Epoch 81/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6095 - val_loss: 0.6039
    Epoch 82/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6088 - val_loss: 0.6032
    Epoch 83/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6081 - val_loss: 0.6025
    Epoch 84/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6075 - val_loss: 0.6019
    Epoch 85/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6068 - val_loss: 0.6012
    Epoch 86/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6061 - val_loss: 0.6006
    Epoch 87/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6054 - val_loss: 0.5999
    Epoch 88/100
    6/6 [==============================] - 0s 6ms/step - loss: 0.6048 - val_loss: 0.5993
    Epoch 89/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6041 - val_loss: 0.5986
    Epoch 90/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6035 - val_loss: 0.5980
    Epoch 91/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6028 - val_loss: 0.5973
    Epoch 92/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6022 - val_loss: 0.5967
    Epoch 93/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6015 - val_loss: 0.5961
    Epoch 94/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6009 - val_loss: 0.5955
    Epoch 95/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.6003 - val_loss: 0.5949
    Epoch 96/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.5996 - val_loss: 0.5943
    Epoch 97/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.5990 - val_loss: 0.5937
    Epoch 98/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.5984 - val_loss: 0.5931
    Epoch 99/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.5978 - val_loss: 0.5925
    Epoch 100/100
    6/6 [==============================] - 0s 5ms/step - loss: 0.5972 - val_loss: 0.5919



```python
x_train_simple.shape
# 압축되기 전 10차원
```




    (5933, 5)




```python
compress_x_train.shape
# 압축된 새로운 특성 데이터 3차원
```




    (5933, 2)




```python
compress_x_test.shape
```




    (200, 2)




```python
# 압축된 Feature 데이터를 입력값으로 전달

new_feature = np.concatenate([compress_x_train,compress_x_test])
tmp_df = pd.DataFrame(new_feature,columns=['comp_fe1','comp_fe2'])
```


```python
tmp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp_fe1</th>
      <th>comp_fe2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.618603</td>
      <td>0.422928</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.811931</td>
      <td>0.525239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.161636</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.996170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.110866</td>
      <td>2.010788</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6128</th>
      <td>0.000000</td>
      <td>1.135982</td>
    </tr>
    <tr>
      <th>6129</th>
      <td>0.178245</td>
      <td>1.292992</td>
    </tr>
    <tr>
      <th>6130</th>
      <td>1.782091</td>
      <td>1.368582</td>
    </tr>
    <tr>
      <th>6131</th>
      <td>0.792279</td>
      <td>0.099597</td>
    </tr>
    <tr>
      <th>6132</th>
      <td>0.833195</td>
      <td>0.506897</td>
    </tr>
  </tbody>
</table>
<p>6133 rows × 2 columns</p>
</div>




```python
len(a)
```




    6139




```python
a.reset_index(inplace=True)
```


```python
a = a.join(tmp_df,how='left').set_index('Date')
```


```python
a.to_csv('/aiffel/aiffel/0000/MLStock/Data/nasdaq.csv')
```


```python
sp = pdr.get_data_yahoo('^GSPC', '1999-12-31')
snp500_df = pd.DataFrame({'Adj Close':sp['Adj Close']})
snp500_df['pct_change'] = snp500_df['Adj Close'].pct_change()
snp500_df['log_ret'] = np.log(snp500_df['Adj Close']) - np.log(snp500_df['Adj Close'].shift(1))
```
