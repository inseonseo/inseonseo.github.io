---
title: Data Cleaning _ Used Car Price
date: 2024-06-23
categories: [Data Preprocessing, Data Cleaning]
tags: [개발자 글쓰기, 모글모글, 아이펠, data preprocessing]     # TAG names should always be lowercase
---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

brand = pd.read_csv('~/data/brand.csv')
cars = pd.read_csv('~/data/cars.csv')
```


```python
brand.head()
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
      <th>title</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>skoda</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vauxhall</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hyundai</td>
      <td>South Korea</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mini</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ford</td>
      <td>United States</td>
    </tr>
  </tbody>
</table>
</div>




```python
cars.head()
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
      <th>title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Previous Owners</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Gearbox</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SKODA FABIA</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VAUXHALL CORSA</td>
      <td>1495</td>
      <td>88585</td>
      <td>2008</td>
      <td>4.0</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.2L</td>
      <td>Manual</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>Full</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HYUNDAI I30</td>
      <td>949</td>
      <td>137000</td>
      <td>2011</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MINI HATCH</td>
      <td>2395</td>
      <td>96731</td>
      <td>2010</td>
      <td>5.0</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>Euro 4</td>
      <td>Full</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VAUXHALL CORSA</td>
      <td>1000</td>
      <td>85000</td>
      <td>2013</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.3L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# Step1. 데이터 전처리
## 1.1 두 데이터 파일 합치기


```python
brand['title'].unique()
```




    array(['skoda', 'vauxhall', 'hyundai', 'mini', 'ford', 'volvo', 'peugeot',
           'bmw', 'citroen', 'mercedes-benz', 'mazda', 'saab', 'volkswagen',
           'honda', 'mg', 'toyota', 'seat', 'nissan', 'alfa', 'renault',
           'kia', 'proton', 'fiat', 'audi', 'mitsubishi', 'lexus', 'land',
           'chevrolet', 'suzuki', 'dacia', 'daihatsu', 'jeep', 'jaguar',
           'chrysler', 'rover', 'ds', 'daewoo', 'dodge', 'porsche', 'subaru',
           'infiniti', 'abarth', 'smart', 'marcos', 'maserati', 'ssangyong',
           'lagonda', 'isuzu'], dtype=object)




```python
type(cars['title'].value_counts()) #pandas.core.series.Series
```




    pandas.core.series.Series




```python
cars['first_words'] = cars['title'].apply(lambda x: x.split()[0].lower())
cars['model'] = cars['title'].apply(lambda x: ' '.join(x.split()[1:]))
cars['first_words'].unique()
```




    array(['skoda', 'vauxhall', 'hyundai', 'mini', 'ford', 'volvo', 'peugeot',
           'bmw', 'citroen', 'mercedes-benz', 'mazda', 'saab', 'volkswagen',
           'honda', 'mg', 'toyota', 'seat', 'nissan', 'alfa', 'renault',
           'kia', 'proton', 'fiat', 'audi', 'mitsubishi', 'lexus', 'land',
           'chevrolet', 'suzuki', 'dacia', 'daihatsu', 'jeep', 'jaguar',
           'chrysler', 'rover', 'ds', 'daewoo', 'dodge', 'porsche', 'subaru',
           'infiniti', 'abarth', 'smart', 'marcos', 'maserati', 'ssangyong',
           'lagonda', 'isuzu'], dtype=object)



- cars 'title' 각 앞단어가 brand 'title'과 일치   

    → first_words 기준으로 Merge


```python
merged = pd.merge(cars, brand, left_on='first_words', right_on='title', how='inner', suffixes=('_cars', '_brand'))
merged
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
      <th>title_cars</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Previous Owners</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Gearbox</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>first_words</th>
      <th>model</th>
      <th>title_brand</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SKODA FABIA</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
      <td>skoda</td>
      <td>FABIA</td>
      <td>skoda</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SKODA FABIA</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
      <td>skoda</td>
      <td>FABIA</td>
      <td>skoda</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SKODA FABIA</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.9L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>Full</td>
      <td>skoda</td>
      <td>FABIA</td>
      <td>skoda</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SKODA FABIA</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>NaN</td>
      <td>skoda</td>
      <td>FABIA</td>
      <td>skoda</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SKODA FABIA</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.2L</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 5</td>
      <td>NaN</td>
      <td>skoda</td>
      <td>FABIA</td>
      <td>skoda</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>MASERATI QUATTROPORTE</td>
      <td>10000</td>
      <td>72000</td>
      <td>2009</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Saloon</td>
      <td>4.2L</td>
      <td>Automatic</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>Full</td>
      <td>maserati</td>
      <td>QUATTROPORTE</td>
      <td>maserati</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>SSANGYONG KORANDO</td>
      <td>12995</td>
      <td>42771</td>
      <td>2018</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>2.2L</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
      <td>ssangyong</td>
      <td>KORANDO</td>
      <td>ssangyong</td>
      <td>South Korea</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>SSANGYONG KORANDO</td>
      <td>4277</td>
      <td>82400</td>
      <td>2013</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>2.0L</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 5</td>
      <td>Full</td>
      <td>ssangyong</td>
      <td>KORANDO</td>
      <td>ssangyong</td>
      <td>South Korea</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>LAGONDA LG6 ROADSTER</td>
      <td>14995</td>
      <td>84000</td>
      <td>1953</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Saloon</td>
      <td>2.6L</td>
      <td>Manual</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lagonda</td>
      <td>LG6 ROADSTER</td>
      <td>lagonda</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>ISUZU TROOPER</td>
      <td>2250</td>
      <td>147700</td>
      <td>2001</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>isuzu</td>
      <td>TROOPER</td>
      <td>isuzu</td>
      <td>Japan</td>
    </tr>
  </tbody>
</table>
<p>3687 rows × 17 columns</p>
</div>




```python
merged.drop(columns=['title_cars', 'title_brand'], inplace=True)
```


```python
merged.rename(columns={'first_words': 'Title'}, inplace=True)
columns = ['Title'] + [col for col in merged.columns if col.lower() != 'title']
merged = merged[columns]
merged

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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Previous Owners</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Gearbox</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>skoda</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skoda</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>skoda</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1.9L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>Full</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>skoda</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.4L</td>
      <td>Manual</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>skoda</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>1.2L</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 5</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>maserati</td>
      <td>10000</td>
      <td>72000</td>
      <td>2009</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Saloon</td>
      <td>4.2L</td>
      <td>Automatic</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>Euro 4</td>
      <td>Full</td>
      <td>QUATTROPORTE</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>ssangyong</td>
      <td>12995</td>
      <td>42771</td>
      <td>2018</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>2.2L</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 6</td>
      <td>NaN</td>
      <td>KORANDO</td>
      <td>South Korea</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>ssangyong</td>
      <td>4277</td>
      <td>82400</td>
      <td>2013</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>2.0L</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Euro 5</td>
      <td>Full</td>
      <td>KORANDO</td>
      <td>South Korea</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>lagonda</td>
      <td>14995</td>
      <td>84000</td>
      <td>1953</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Saloon</td>
      <td>2.6L</td>
      <td>Manual</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LG6 ROADSTER</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>isuzu</td>
      <td>2250</td>
      <td>147700</td>
      <td>2001</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>Automatic</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TROOPER</td>
      <td>Japan</td>
    </tr>
  </tbody>
</table>
<p>3687 rows × 15 columns</p>
</div>



## 1.2 카테고리형 변수를 숫자 형태로 변환하기



```python
merged['Engine'].value_counts()
```




    1.6L    736
    2.0L    553
    1.2L    521
    1.4L    421
    1.0L    326
    1.5L    294
    1.3L    170
    1.8L    158
    3.0L     79
    2.2L     75
    2.1L     63
    1.1L     39
    1.7L     35
    2.5L     33
    1.9L     31
    2.4L     28
    0.9L     15
    2.7L     11
    3.5L     10
    3.2L      7
    4.4L      5
    3.7L      5
    4.2L      4
    2.3L      4
    2.6L      4
    2.8L      3
    5.0L      3
    4.3L      2
    0.8L      2
    5.5L      1
    6.3L      1
    3.1L      1
    4.8L      1
    3.3L      1
    Name: Engine, dtype: int64




```python
merged['Engine'].dtype
```




    dtype('O')




```python
merged['Engine'] = (merged['Engine'].str.replace('L', '').astype(float))
merged['Engine']
```




    0       1.4
    1       1.4
    2       1.9
    3       1.4
    4       1.2
           ... 
    3682    4.2
    3683    2.2
    3684    2.0
    3685    2.6
    3686    NaN
    Name: Engine, Length: 3687, dtype: float64




```python
merged['Engine'].describe()
```




    count    3642.000000
    mean        1.606260
    std         0.486584
    min         0.800000
    25%         1.300000
    50%         1.600000
    75%         1.900000
    max         6.300000
    Name: Engine, dtype: float64




```python
# merged['Engine'].sort_values(ascending=False).head(30)
# 3.5 초과를 마지막 범주로 설정 -> 데이터 갯수가 30개 미만이라 그 이상 구간은 의미 없어보임
```

### 주요 범주형 카테고리 분류


```python
merged.nunique()
```




    Title                  48
    Price                 866
    Mileage(miles)       1570
    Registration_Year      40
    Previous Owners         9
    Fuel type               6
    Body type              10
    Engine                 34
    Gearbox                 2
    Doors                   4
    Seats                   6
    Emission Class          6
    Service history         1
    model                 453
    country                12
    dtype: int64




```python
nunique = merged.nunique()
category_col = nunique[nunique<=6]
category_col
```




    Fuel type          6
    Gearbox            2
    Doors              4
    Seats              6
    Emission Class     6
    Service history    1
    dtype: int64




```python
merged['Previous Owners'].unique()
```




    array([ 3.,  2., nan,  4.,  8.,  1.,  5.,  6.,  9.,  7.])




```python
merged['Engine'].max()
```




    6.3




```python
bins = [0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 6.3]
labels = [0, 1, 2, 3, 4, 5]

merged['Engine'] = pd.cut(merged['Engine'], bins=bins, labels=labels)
merged['Engine']
```




    0         0
    1         0
    2         1
    3         0
    4         0
           ... 
    3682      5
    3683      2
    3684      1
    3685      3
    3686    NaN
    Name: Engine, Length: 3687, dtype: category
    Categories (6, int64): [0 < 1 < 2 < 3 < 4 < 5]



### Emission Class


```python
merged['Emission Class'].value_counts()
```




    Euro 5    1257
    Euro 6    1109
    Euro 4    1068
    Euro 3     137
    Euro 2      25
    Euro 1       4
    Name: Emission Class, dtype: int64




```python
merged['Emission Class'] = merged['Emission Class'].str.replace('Euro ', '').astype(float)

# 'Euro' 문자열을 제외한 부분을 추출하여 숫자로 변환
# merged['Emission Class'] = merged['Emission Class'].str.extract('(\d+)').astype(float)

```

Fuel type          6
Gearbox            2
Body type 10
country 12


```python
merged.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3687 entries, 0 to 3686
    Data columns (total 15 columns):
     #   Column             Non-Null Count  Dtype   
    ---  ------             --------------  -----   
     0   Title              3687 non-null   object  
     1   Price              3687 non-null   int64   
     2   Mileage(miles)     3687 non-null   int64   
     3   Registration_Year  3687 non-null   int64   
     4   Previous Owners    2276 non-null   float64 
     5   Fuel type          3687 non-null   object  
     6   Body type          3687 non-null   object  
     7   Engine             3640 non-null   category
     8   Gearbox            3687 non-null   object  
     9   Doors              3662 non-null   float64 
     10  Seats              3652 non-null   float64 
     11  Emission Class     3600 non-null   float64 
     12  Service history    540 non-null    object  
     13  model              3687 non-null   object  
     14  country            3687 non-null   object  
    dtypes: category(1), float64(4), int64(3), object(7)
    memory usage: 435.9+ KB



```python
from tabulate import tabulate

cols = ['Fuel type', 'Gearbox', 'Body type', 'country']

for col in cols:
    data = merged[col].value_counts().reset_index()
    data.columns = [col, 'Count']
    print(f"\n{col} Counts:\n")
    print(tabulate(data, headers=data.columns, tablefmt='pretty', showindex=False)) 
```

    
    Fuel type Counts:
    
    +-----------------------+-------+
    |       Fuel type       | Count |
    +-----------------------+-------+
    |        Petrol         | 2362  |
    |        Diesel         | 1219  |
    |     Petrol Hybrid     |  47   |
    |       Electric        |  31   |
    | Petrol Plug-in Hybrid |  27   |
    |     Diesel Hybrid     |   1   |
    +-----------------------+-------+
    
    Gearbox Counts:
    
    +-----------+-------+
    |  Gearbox  | Count |
    +-----------+-------+
    |  Manual   | 2870  |
    | Automatic |  817  |
    +-----------+-------+
    
    Body type Counts:
    
    +-------------+-------+
    |  Body type  | Count |
    +-------------+-------+
    |  Hatchback  | 2280  |
    |     SUV     |  461  |
    |   Saloon    |  368  |
    |   Estate    |  171  |
    |     MPV     |  153  |
    |    Coupe    |  139  |
    | Convertible |  109  |
    |   Pickup    |   3   |
    |  Combi Van  |   2   |
    |   Minibus   |   1   |
    +-------------+-------+
    
    country Counts:
    
    +----------------+-------+
    |    country     | Count |
    +----------------+-------+
    |    Germany     |  863  |
    | United Kingdom |  729  |
    |     Japan      |  641  |
    |     France     |  522  |
    | United States  |  439  |
    |  South Korea   |  178  |
    |     Italy      |  128  |
    | Czech Republic |  63   |
    |     Spain      |  60   |
    |     Sweden     |  47   |
    |    Romania     |  14   |
    |    Malaysia    |   3   |
    +----------------+-------+


- object 형 데이터들 점검
- 원핫 인코딩으로 전부 변환 시 추후 모델 학습에 오차가 생길 수 있을 것 같다
- 정수형 인코딩 고려


```python
merged= pd.get_dummies(merged, columns=['Gearbox'], drop_first=True) # Manual, Automatic
merged
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Previous Owners</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>skoda</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skoda</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>skoda</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Full</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>skoda</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>skoda</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>maserati</td>
      <td>10000</td>
      <td>72000</td>
      <td>2009</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Saloon</td>
      <td>5</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Full</td>
      <td>QUATTROPORTE</td>
      <td>Italy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>ssangyong</td>
      <td>12995</td>
      <td>42771</td>
      <td>2018</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>KORANDO</td>
      <td>South Korea</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>ssangyong</td>
      <td>4277</td>
      <td>82400</td>
      <td>2013</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Full</td>
      <td>KORANDO</td>
      <td>South Korea</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>lagonda</td>
      <td>14995</td>
      <td>84000</td>
      <td>1953</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Saloon</td>
      <td>3</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LG6 ROADSTER</td>
      <td>United Kingdom</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>isuzu</td>
      <td>2250</td>
      <td>147700</td>
      <td>2001</td>
      <td>NaN</td>
      <td>Diesel</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TROOPER</td>
      <td>Japan</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3687 rows × 15 columns</p>
</div>




```python
merged.head()
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Previous Owners</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>skoda</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skoda</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>skoda</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>2.0</td>
      <td>Diesel</td>
      <td>Hatchback</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Full</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>skoda</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>NaN</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>skoda</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3.0</td>
      <td>Petrol</td>
      <td>Hatchback</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>Czech Republic</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import LabelEncoder

# Label Encoding을 수행할 열들 정의
columns_to_encode = ['Fuel type', 'Title', 'Body type', 'country']

# Label Encoder 객체를 각 열에 대해 생성하고 변환
label_encoders = {}
for column in columns_to_encode:
    label_encoder = LabelEncoder()
    merged[column] = label_encoder.fit_transform(merged[column])
    label_encoders[column] = label_encoder

# 변환된 데이터 확인
encoded_data = merged[columns_to_encode].head()

# 변환 전/후 매핑 출력
for column in columns_to_encode:
    mapping = dict(zip(label_encoders[column].classes_, label_encoders[column].transform(label_encoders[column].classes_)))
    print(f"\nMapping for {column}:\n{mapping}")
    print(f"\nOriginal values for {column}:\n{merged.head()[column]}")
    print(f"\nEncoded values for {column}:\n{encoded_data[column]}")
```

    
    Mapping for Fuel type:
    {'Diesel': 0, 'Diesel Hybrid': 1, 'Electric': 2, 'Petrol': 3, 'Petrol Hybrid': 4, 'Petrol Plug-in Hybrid': 5}
    
    Original values for Fuel type:
    0    0
    1    0
    2    0
    3    3
    4    3
    Name: Fuel type, dtype: int64
    
    Encoded values for Fuel type:
    0    0
    1    0
    2    0
    3    3
    4    3
    Name: Fuel type, dtype: int64
    
    Mapping for Title:
    {'abarth': 0, 'alfa': 1, 'audi': 2, 'bmw': 3, 'chevrolet': 4, 'chrysler': 5, 'citroen': 6, 'dacia': 7, 'daewoo': 8, 'daihatsu': 9, 'dodge': 10, 'ds': 11, 'fiat': 12, 'ford': 13, 'honda': 14, 'hyundai': 15, 'infiniti': 16, 'isuzu': 17, 'jaguar': 18, 'jeep': 19, 'kia': 20, 'lagonda': 21, 'land': 22, 'lexus': 23, 'marcos': 24, 'maserati': 25, 'mazda': 26, 'mercedes-benz': 27, 'mg': 28, 'mini': 29, 'mitsubishi': 30, 'nissan': 31, 'peugeot': 32, 'porsche': 33, 'proton': 34, 'renault': 35, 'rover': 36, 'saab': 37, 'seat': 38, 'skoda': 39, 'smart': 40, 'ssangyong': 41, 'subaru': 42, 'suzuki': 43, 'toyota': 44, 'vauxhall': 45, 'volkswagen': 46, 'volvo': 47}
    
    Original values for Title:
    0    39
    1    39
    2    39
    3    39
    4    39
    Name: Title, dtype: int64
    
    Encoded values for Title:
    0    39
    1    39
    2    39
    3    39
    4    39
    Name: Title, dtype: int64
    
    Mapping for Body type:
    {'Combi Van': 0, 'Convertible': 1, 'Coupe': 2, 'Estate': 3, 'Hatchback': 4, 'MPV': 5, 'Minibus': 6, 'Pickup': 7, 'SUV': 8, 'Saloon': 9}
    
    Original values for Body type:
    0    4
    1    4
    2    4
    3    4
    4    4
    Name: Body type, dtype: int64
    
    Encoded values for Body type:
    0    4
    1    4
    2    4
    3    4
    4    4
    Name: Body type, dtype: int64
    
    Mapping for country:
    {'Czech Republic': 0, 'France': 1, 'Germany': 2, 'Italy': 3, 'Japan': 4, 'Malaysia': 5, 'Romania': 6, 'South Korea': 7, 'Spain': 8, 'Sweden': 9, 'United Kingdom': 10, 'United States': 11}
    
    Original values for country:
    0    0
    1    0
    2    0
    3    0
    4    0
    Name: country, dtype: int64
    
    Encoded values for country:
    0    0
    1    0
    2    0
    3    0
    4    0
    Name: country, dtype: int64



```python
merged
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Previous Owners</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Full</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>NaN</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3.0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>FABIA</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>25</td>
      <td>10000</td>
      <td>72000</td>
      <td>2009</td>
      <td>3.0</td>
      <td>3</td>
      <td>9</td>
      <td>5</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Full</td>
      <td>QUATTROPORTE</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>41</td>
      <td>12995</td>
      <td>42771</td>
      <td>2018</td>
      <td>2.0</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>KORANDO</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>41</td>
      <td>4277</td>
      <td>82400</td>
      <td>2013</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Full</td>
      <td>KORANDO</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>21</td>
      <td>14995</td>
      <td>84000</td>
      <td>1953</td>
      <td>NaN</td>
      <td>3</td>
      <td>9</td>
      <td>3</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LG6 ROADSTER</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>17</td>
      <td>2250</td>
      <td>147700</td>
      <td>2001</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TROOPER</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3687 rows × 15 columns</p>
</div>



## 1.3 결측치 처리하기
-  Service history 변수의 결측치는 'Unknown'으로 채우고, 결측치 변수가 일정 개수 이상 포함된 데이터는 제거합니다.



```python
merged.isnull().sum()
```




    Title                   0
    Price                   0
    Mileage(miles)          0
    Registration_Year       0
    Previous Owners      1411
    Fuel type               0
    Body type               0
    Engine                 47
    Doors                  25
    Seats                  35
    Emission Class         87
    Service history      3147
    model                   0
    country                 0
    Gearbox_Manual          0
    dtype: int64




```python
merged['Service history'].fillna('Unknown', inplace=True)
```


```python
merged.isnull().sum()
```




    Title                   0
    Price                   0
    Mileage(miles)          0
    Registration_Year       0
    Previous Owners      1411
    Fuel type               0
    Body type               0
    Engine                 47
    Doors                  25
    Seats                  35
    Emission Class         87
    Service history         0
    model                   0
    country                 0
    Gearbox_Manual          0
    dtype: int64




```python
merged['Previous Owners'].value_counts()
```




    2.0    594
    1.0    523
    3.0    475
    4.0    360
    5.0    208
    6.0     60
    7.0     39
    8.0     12
    9.0      5
    Name: Previous Owners, dtype: int64




```python
merged.shape
```




    (3687, 15)




```python
merged = merged.dropna(thresh=merged.shape[1]-1) # null 2개 이상인 거 지우기 # 1388
merged.shape
```




    (3606, 15)




```python
missing_percentage = (merged.isnull().sum() / len(merged)) * 100
missing_percentage
```




    Title                 0.000000
    Price                 0.000000
    Mileage(miles)        0.000000
    Registration_Year     0.000000
    Previous Owners      37.520799
    Fuel type             0.000000
    Body type             0.000000
    Engine                0.055463
    Doors                 0.000000
    Seats                 0.110926
    Emission Class        0.249584
    Service history       0.000000
    model                 0.000000
    country               0.000000
    Gearbox_Manual        0.000000
    dtype: float64




```python
merged = pd.get_dummies(merged, columns=['Service history'])

#'full'을 남기고 다른 열 삭제
merged.drop(['Service history_Full'], axis=1, inplace=True)
```


```python
threshold = 0.7 * len(merged)  # 전체 데이터 개수의 50%
merged = merged.dropna(axis=1, thresh=threshold)
merged.head()
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>Unknown</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>Unknown</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Full</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>FABIA</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Unknown</td>
      <td>FABIA</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Emission 중앙값으로 대체


```python
null_rows = merged.isnull().any(axis=1)
merged[null_rows]
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>446</th>
      <td>45</td>
      <td>4250</td>
      <td>35616</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>COMBO</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>45</td>
      <td>9499</td>
      <td>65767</td>
      <td>2014</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>Unknown</td>
      <td>AMPERA</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>3</td>
      <td>6700</td>
      <td>105000</td>
      <td>1999</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>3 SERIES</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1773</th>
      <td>6</td>
      <td>3250</td>
      <td>126000</td>
      <td>1995</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>XM</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>27</td>
      <td>8855</td>
      <td>74280</td>
      <td>2012</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>E CLASS DIESEL ESTATE</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2091</th>
      <td>37</td>
      <td>5995</td>
      <td>157000</td>
      <td>1992</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Unknown</td>
      <td>900</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2174</th>
      <td>46</td>
      <td>6495</td>
      <td>66000</td>
      <td>2011</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>POLO</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2189</th>
      <td>46</td>
      <td>2500</td>
      <td>124000</td>
      <td>1988</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>GOLF</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2401</th>
      <td>14</td>
      <td>6700</td>
      <td>80000</td>
      <td>2012</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>Full</td>
      <td>FREED</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2512</th>
      <td>44</td>
      <td>5480</td>
      <td>52624</td>
      <td>2013</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>YARIS</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2520</th>
      <td>44</td>
      <td>5949</td>
      <td>145000</td>
      <td>2016</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>PRIUS</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2587</th>
      <td>44</td>
      <td>9495</td>
      <td>55000</td>
      <td>2012</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Full</td>
      <td>PRIUS+</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2935</th>
      <td>35</td>
      <td>2490</td>
      <td>67104</td>
      <td>2010</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>FLUENCE</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3525</th>
      <td>4</td>
      <td>2490</td>
      <td>44584</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>MATIZ</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3651</th>
      <td>33</td>
      <td>6750</td>
      <td>121000</td>
      <td>1987</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>944</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged['Emission Class'].fillna(merged['Emission Class'].median(), inplace=True)
null_rows = merged.isnull().any(axis=1)
merged[null_rows]
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>446</th>
      <td>45</td>
      <td>4250</td>
      <td>35616</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>COMBO</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>45</td>
      <td>9499</td>
      <td>65767</td>
      <td>2014</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>Unknown</td>
      <td>AMPERA</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>3</td>
      <td>6700</td>
      <td>105000</td>
      <td>1999</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>3 SERIES</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1773</th>
      <td>6</td>
      <td>3250</td>
      <td>126000</td>
      <td>1995</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>XM</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2091</th>
      <td>37</td>
      <td>5995</td>
      <td>157000</td>
      <td>1992</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Unknown</td>
      <td>900</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3525</th>
      <td>4</td>
      <td>2490</td>
      <td>44584</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>MATIZ</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
similar_rows = merged_1_c[
    (merged_1_c['title_cars'].str.contains('VAUXHALL')) &
    (merged_1_c['Registration_Year']==2014) &
    (merged_1_c['Seats']==5) &
    (merged_1_c['Doors']>=4) &
    (merged_1_c['Emission Class']==5)
]
```


```python
similar_rows['Engine'].value_counts() #Engine=1
```


```python
similar_rows = merged_1_c[
    (merged_1_c['title_cars'].str.contains('CHEVROLET')) &
    (merged_1_c['Registration_Year']==2008) &
    (merged_1_c['Seats']==5) &
    (merged_1_c['Doors']==5) &
    (merged_1_c['Emission Class']==4)
]
```


```python
similar_rows #Engine =2
```


```python
similar_rows = merged_1_c[
    (merged_1_c['title_cars'].str.contains('SMART')) 
  #  (merged_1_c['Registration_Year']==2013) 
  #  (merged_1_c['Seats']<=2) &
#     (merged_1_c['Doors']<=2) &
   # (merged_1_c['Emission Class']==5)
]
similar_rows # engine=0
```


```python
merged['Emission Class'].fillna(merged['Emission Class'].median(), inplace=True)
null_rows = merged.isnull().any(axis=1)
merged[null_rows]
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>446</th>
      <td>45</td>
      <td>4250</td>
      <td>35616</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>COMBO</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>45</td>
      <td>9499</td>
      <td>65767</td>
      <td>2014</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>Unknown</td>
      <td>AMPERA</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>3</td>
      <td>6700</td>
      <td>105000</td>
      <td>1999</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>3 SERIES</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1773</th>
      <td>6</td>
      <td>3250</td>
      <td>126000</td>
      <td>1995</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>XM</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2091</th>
      <td>37</td>
      <td>5995</td>
      <td>157000</td>
      <td>1992</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Unknown</td>
      <td>900</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3525</th>
      <td>4</td>
      <td>2490</td>
      <td>44584</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>MATIZ</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replace missing 'Engine' values for specific rows
merged.loc[(merged['Title'] == 45) & (merged['Registration_Year'] == 2014), 'Engine'] = 1
merged.loc[(merged['Title'] == 4) & (merged['Registration_Year'] == 2008), 'Engine'] = 2
```


```python
merged[null_rows]
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>Service history</th>
      <th>model</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>446</th>
      <td>45</td>
      <td>4250</td>
      <td>35616</td>
      <td>2010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>COMBO</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>45</td>
      <td>9499</td>
      <td>65767</td>
      <td>2014</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>Unknown</td>
      <td>AMPERA</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>3</td>
      <td>6700</td>
      <td>105000</td>
      <td>1999</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>3 SERIES</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1773</th>
      <td>6</td>
      <td>3250</td>
      <td>126000</td>
      <td>1995</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Unknown</td>
      <td>XM</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2091</th>
      <td>37</td>
      <td>5995</td>
      <td>157000</td>
      <td>1992</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Unknown</td>
      <td>900</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3525</th>
      <td>4</td>
      <td>2490</td>
      <td>44584</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>Unknown</td>
      <td>MATIZ</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged = merged.dropna()
merged.shape
```




    (3602, 14)



## 1.4 Scaler


```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
ss = StandardScaler()
rs = RobustScaler()
mm = MinMaxScaler()
```


```python
merged_ = merged.drop(columns=['model'])
```


```python
merged_
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
      <th>Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3680</th>
      <td>25</td>
      <td>16000</td>
      <td>66000</td>
      <td>2008</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3681</th>
      <td>25</td>
      <td>13900</td>
      <td>63000</td>
      <td>2014</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>25</td>
      <td>10000</td>
      <td>72000</td>
      <td>2009</td>
      <td>3</td>
      <td>9</td>
      <td>5</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>41</td>
      <td>12995</td>
      <td>42771</td>
      <td>2018</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>41</td>
      <td>4277</td>
      <td>82400</td>
      <td>2013</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3602 rows × 13 columns</p>
</div>




```python
ss.fit(merged_)
```




    StandardScaler()




```python
ss_df = pd.DataFrame(ss.transform(merged_), columns = merged_.columns)
```


```python
ss_df.head()
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
      <th>Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.771539</td>
      <td>0.245264</td>
      <td>-0.287084</td>
      <td>0.850853</td>
      <td>-1.394408</td>
      <td>-0.40464</td>
      <td>-0.811776</td>
      <td>0.690258</td>
      <td>0.174028</td>
      <td>1.204461</td>
      <td>-1.376648</td>
      <td>0.515399</td>
      <td>0.418116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.771539</td>
      <td>0.245264</td>
      <td>-0.287084</td>
      <td>0.850853</td>
      <td>-1.394408</td>
      <td>-0.40464</td>
      <td>-0.811776</td>
      <td>0.690258</td>
      <td>0.174028</td>
      <td>1.204461</td>
      <td>-1.376648</td>
      <td>0.515399</td>
      <td>0.418116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.771539</td>
      <td>-0.843268</td>
      <td>1.732726</td>
      <td>-1.035806</td>
      <td>-1.394408</td>
      <td>-0.40464</td>
      <td>0.409108</td>
      <td>0.690258</td>
      <td>0.174028</td>
      <td>-1.016409</td>
      <td>-1.376648</td>
      <td>0.515399</td>
      <td>-2.391683</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.771539</td>
      <td>-0.842159</td>
      <td>0.062436</td>
      <td>-0.826177</td>
      <td>0.675084</td>
      <td>-0.40464</td>
      <td>-0.811776</td>
      <td>0.690258</td>
      <td>0.174028</td>
      <td>-1.016409</td>
      <td>-1.376648</td>
      <td>0.515399</td>
      <td>0.418116</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.771539</td>
      <td>-0.731311</td>
      <td>1.078226</td>
      <td>-0.197291</td>
      <td>0.675084</td>
      <td>-0.40464</td>
      <td>-0.811776</td>
      <td>0.690258</td>
      <td>0.174028</td>
      <td>0.094026</td>
      <td>-1.376648</td>
      <td>-1.940245</td>
      <td>0.418116</td>
    </tr>
  </tbody>
</table>
</div>




```python
rs.fit(merged_)
```




    RobustScaler()




```python
rs_df = pd.DataFrame(rs.transform(merged_), columns = merged_.columns)
```


```python
rs_df.head()
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
      <th>Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.290323</td>
      <td>0.526340</td>
      <td>-0.212402</td>
      <td>0.500</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>-0.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.290323</td>
      <td>0.526340</td>
      <td>-0.212402</td>
      <td>0.500</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>-0.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.290323</td>
      <td>-0.364808</td>
      <td>1.515455</td>
      <td>-0.625</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.5</td>
      <td>-0.5</td>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.290323</td>
      <td>-0.363900</td>
      <td>0.086597</td>
      <td>-0.500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.5</td>
      <td>-0.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.290323</td>
      <td>-0.273152</td>
      <td>0.955559</td>
      <td>-0.125</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.5</td>
      <td>-1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
mm.fit(merged_)
```




    MinMaxScaler()




```python
mm_df = pd.DataFrame(mm.transform(merged_), columns = merged_.columns)
```


```python
mm_df.head()
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
      <th>Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.829787</td>
      <td>0.194030</td>
      <td>0.063227</td>
      <td>0.805556</td>
      <td>0.0</td>
      <td>0.444444</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.829787</td>
      <td>0.194030</td>
      <td>0.063227</td>
      <td>0.805556</td>
      <td>0.0</td>
      <td>0.444444</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.829787</td>
      <td>0.047463</td>
      <td>0.135122</td>
      <td>0.555556</td>
      <td>0.0</td>
      <td>0.444444</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.829787</td>
      <td>0.047612</td>
      <td>0.075668</td>
      <td>0.583333</td>
      <td>0.6</td>
      <td>0.444444</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.829787</td>
      <td>0.062537</td>
      <td>0.111825</td>
      <td>0.666667</td>
      <td>0.6</td>
      <td>0.444444</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## 1.5 PCA
- 기존 데이터의 정보 70% 이상을 가지는 수준에서 최소한의 주성분 추출 (PCA)


```python
from sklearn.decomposition import PCA
```


```python
pca = PCA()
pca.fit(merged_)
merged_
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
      <th>Title</th>
      <th>Price</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Engine</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
      <th>Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>6900</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>1990</td>
      <td>150000</td>
      <td>2007</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
      <td>1995</td>
      <td>84000</td>
      <td>2008</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39</td>
      <td>2495</td>
      <td>124138</td>
      <td>2011</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3680</th>
      <td>25</td>
      <td>16000</td>
      <td>66000</td>
      <td>2008</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3681</th>
      <td>25</td>
      <td>13900</td>
      <td>63000</td>
      <td>2014</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>25</td>
      <td>10000</td>
      <td>72000</td>
      <td>2009</td>
      <td>3</td>
      <td>9</td>
      <td>5</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>41</td>
      <td>12995</td>
      <td>42771</td>
      <td>2018</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>41</td>
      <td>4277</td>
      <td>82400</td>
      <td>2013</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3602 rows × 13 columns</p>
</div>




```python
pd.DataFrame(pca.transform(merged_))
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-11388.917960</td>
      <td>447.354897</td>
      <td>11.786525</td>
      <td>4.694754</td>
      <td>-4.404105</td>
      <td>-0.032541</td>
      <td>-1.716591</td>
      <td>-0.345115</td>
      <td>-0.442940</td>
      <td>0.116962</td>
      <td>0.319416</td>
      <td>-0.013027</td>
      <td>0.042798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-11388.917960</td>
      <td>447.354897</td>
      <td>11.786525</td>
      <td>4.694754</td>
      <td>-4.404105</td>
      <td>-0.032541</td>
      <td>-1.716591</td>
      <td>-0.345115</td>
      <td>-0.442940</td>
      <td>0.116962</td>
      <td>0.319416</td>
      <td>-0.013027</td>
      <td>0.042798</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68572.486272</td>
      <td>168.585517</td>
      <td>13.981151</td>
      <td>5.649706</td>
      <td>0.172572</td>
      <td>0.633572</td>
      <td>-1.589264</td>
      <td>-0.811324</td>
      <td>-0.255536</td>
      <td>0.229805</td>
      <td>-0.574628</td>
      <td>0.749854</td>
      <td>-0.216177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2683.011935</td>
      <td>-3649.421966</td>
      <td>9.513520</td>
      <td>6.130241</td>
      <td>-0.348004</td>
      <td>0.602755</td>
      <td>0.819177</td>
      <td>-1.096391</td>
      <td>-0.281244</td>
      <td>0.109670</td>
      <td>-0.136616</td>
      <td>-0.193920</td>
      <td>-0.071541</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42724.657109</td>
      <td>-825.296184</td>
      <td>12.661260</td>
      <td>4.920061</td>
      <td>-3.048222</td>
      <td>0.760027</td>
      <td>1.439986</td>
      <td>-0.856841</td>
      <td>0.035617</td>
      <td>0.003973</td>
      <td>0.247510</td>
      <td>-0.099069</td>
      <td>0.927448</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3597</th>
      <td>-16097.994296</td>
      <td>9289.427682</td>
      <td>2.845829</td>
      <td>3.841243</td>
      <td>11.750734</td>
      <td>5.731580</td>
      <td>-1.200581</td>
      <td>1.305947</td>
      <td>3.198580</td>
      <td>-0.276166</td>
      <td>-0.476049</td>
      <td>0.603190</td>
      <td>-0.753027</td>
    </tr>
    <tr>
      <th>3598</th>
      <td>-18971.316629</td>
      <td>7019.188019</td>
      <td>1.489752</td>
      <td>2.671886</td>
      <td>4.696125</td>
      <td>-3.023248</td>
      <td>-1.532942</td>
      <td>1.118368</td>
      <td>1.135219</td>
      <td>-0.049741</td>
      <td>-0.092160</td>
      <td>-0.339686</td>
      <td>-0.024616</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>-9760.522991</td>
      <td>3647.052242</td>
      <td>-0.422697</td>
      <td>3.516139</td>
      <td>6.798656</td>
      <td>-2.763287</td>
      <td>0.798111</td>
      <td>0.872716</td>
      <td>3.443770</td>
      <td>-0.228056</td>
      <td>-0.553636</td>
      <td>0.544139</td>
      <td>-0.417374</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>-39113.931022</td>
      <td>4943.951042</td>
      <td>15.769132</td>
      <td>-1.601283</td>
      <td>1.244852</td>
      <td>-3.250710</td>
      <td>-1.783921</td>
      <td>0.563045</td>
      <td>0.961825</td>
      <td>0.207496</td>
      <td>0.187911</td>
      <td>-0.234453</td>
      <td>0.455090</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>953.514913</td>
      <td>-1463.931022</td>
      <td>13.024181</td>
      <td>-1.339961</td>
      <td>-1.160809</td>
      <td>-4.165844</td>
      <td>-1.215973</td>
      <td>0.511554</td>
      <td>0.168770</td>
      <td>0.292104</td>
      <td>-0.516471</td>
      <td>0.607978</td>
      <td>0.979812</td>
    </tr>
  </tbody>
</table>
<p>3602 rows × 13 columns</p>
</div>




```python
pca = PCA(2)
```


```python
pd.DataFrame(pca.fit_transform(merged_), columns = ['PC1','PC2'])
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
      <th>PC1</th>
      <th>PC2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-11388.917960</td>
      <td>447.354897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-11388.917960</td>
      <td>447.354897</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68572.486272</td>
      <td>168.585517</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2683.011935</td>
      <td>-3649.421966</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42724.657109</td>
      <td>-825.296184</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3597</th>
      <td>-16097.994296</td>
      <td>9289.427682</td>
    </tr>
    <tr>
      <th>3598</th>
      <td>-18971.316629</td>
      <td>7019.188019</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>-9760.522991</td>
      <td>3647.052242</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>-39113.931022</td>
      <td>4943.951042</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>953.514913</td>
      <td>-1463.931022</td>
    </tr>
  </tbody>
</table>
<p>3602 rows × 2 columns</p>
</div>




```python
pca = PCA(2)
pd.DataFrame(pca.fit_transform(merged_), columns = ['PC1','PC2'])
print((pca.explained_variance_ratio_))
print("sum:", (pca.explained_variance_ratio_).sum())
```

    [0.99042745 0.00957238]
    sum: 0.9999998347809548



```python
pca = PCA(3)
pd.DataFrame(pca.fit_transform(merged_), columns = ['PC1','PC2', 'PC3'])
print((pca.explained_variance_ratio_))
print("sum:", (pca.explained_variance_ratio_).sum())
```

    [9.90427454e-01 9.57238125e-03 1.45248684e-07]
    sum: 0.9999999800296406


- 첫 번쨰 주성분 약 99.04%, 두 번째 주성분 0.96%, 세 번째 주성분 매우 작은 값
- 거의 분산을 설명하지 못함


```python
pca = PCA()
pd.DataFrame(pca.fit_transform(merged_))
print((pca.explained_variance_ratio_))
print("sum:", (pca.explained_variance_ratio_).sum())
```

    [9.90427454e-01 9.57238125e-03 1.45248684e-07 9.40780013e-09
     5.57552300e-09 2.66393992e-09 1.15580040e-09 5.42805898e-10
     2.67781849e-10 1.30392520e-10 8.28234672e-11 7.65008773e-11
     6.69930605e-11]
    sum: 1.0


- 주성분 개수 지정하지 않고 모든 주성분 사용
- 모든 주성분 사용하여 전체 분산 완벽히 설명하고 있다 (sum:1)

# Step2. 데이터 심화
## 2.1 국가별 총 브랜드 개수


```python
brand = pd.read_csv('~/data/brand.csv')
```


```python
pd.pivot_table(brand, columns = 'country', values = 'title', aggfunc = 'count')
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
      <th>country</th>
      <th>Czech Republic</th>
      <th>France</th>
      <th>Germany</th>
      <th>Italy</th>
      <th>Japan</th>
      <th>Malaysia</th>
      <th>Romania</th>
      <th>South Korea</th>
      <th>Spain</th>
      <th>Sweden</th>
      <th>United Kingdom</th>
      <th>United States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>title</th>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
brand.groupby('country')['title'].count().reset_index()
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
      <th>country</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Italy</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Japan</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Malaysia</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Romania</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>South Korea</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spain</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sweden</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>United Kingdom</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>United States</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2 상관관계


```python
df_corr = merged_.corr()
```


```python
df_corr.style.background_gradient()
```




<style type="text/css">
#T_5398c_row0_col0, #T_5398c_row1_col1, #T_5398c_row2_col2, #T_5398c_row3_col3, #T_5398c_row4_col4, #T_5398c_row5_col5, #T_5398c_row6_col6, #T_5398c_row7_col7, #T_5398c_row8_col8, #T_5398c_row9_col9, #T_5398c_row10_col10, #T_5398c_row11_col11 {
  background-color: #023858;
  color: #f1f1f1;
}
#T_5398c_row0_col1 {
  background-color: #c9cee4;
  color: #000000;
}
#T_5398c_row0_col2, #T_5398c_row0_col8 {
  background-color: #c6cce3;
  color: #000000;
}
#T_5398c_row0_col3, #T_5398c_row0_col10, #T_5398c_row3_col6 {
  background-color: #b8c6e0;
  color: #000000;
}
#T_5398c_row0_col4 {
  background-color: #c8cde4;
  color: #000000;
}
#T_5398c_row0_col5 {
  background-color: #f5eef6;
  color: #000000;
}
#T_5398c_row0_col6 {
  background-color: #d6d6e9;
  color: #000000;
}
#T_5398c_row0_col7, #T_5398c_row2_col7, #T_5398c_row3_col10 {
  background-color: #d1d2e6;
  color: #000000;
}
#T_5398c_row0_col9, #T_5398c_row2_col11, #T_5398c_row3_col7, #T_5398c_row4_col8, #T_5398c_row10_col1, #T_5398c_row11_col8 {
  background-color: #dcdaeb;
  color: #000000;
}
#T_5398c_row0_col11, #T_5398c_row2_col6 {
  background-color: #efe9f3;
  color: #000000;
}
#T_5398c_row1_col0 {
  background-color: #fef6fa;
  color: #000000;
}
#T_5398c_row1_col2, #T_5398c_row1_col9, #T_5398c_row1_col10, #T_5398c_row1_col11, #T_5398c_row2_col1, #T_5398c_row2_col3, #T_5398c_row2_col4, #T_5398c_row2_col8, #T_5398c_row4_col6, #T_5398c_row4_col7, #T_5398c_row5_col0, #T_5398c_row10_col5 {
  background-color: #fff7fb;
  color: #000000;
}
#T_5398c_row1_col3 {
  background-color: #045e94;
  color: #f1f1f1;
}
#T_5398c_row1_col4, #T_5398c_row7_col10 {
  background-color: #e3e0ee;
  color: #000000;
}
#T_5398c_row1_col5, #T_5398c_row7_col3 {
  background-color: #b7c5df;
  color: #000000;
}
#T_5398c_row1_col6, #T_5398c_row4_col10 {
  background-color: #d3d4e7;
  color: #000000;
}
#T_5398c_row1_col7 {
  background-color: #e6e2ef;
  color: #000000;
}
#T_5398c_row1_col8 {
  background-color: #056aa6;
  color: #f1f1f1;
}
#T_5398c_row2_col0 {
  background-color: #fbf4f9;
  color: #000000;
}
#T_5398c_row2_col5, #T_5398c_row7_col0 {
  background-color: #e0deed;
  color: #000000;
}
#T_5398c_row2_col9, #T_5398c_row8_col2, #T_5398c_row8_col9 {
  background-color: #f3edf5;
  color: #000000;
}
#T_5398c_row2_col10 {
  background-color: #f4eef6;
  color: #000000;
}
#T_5398c_row3_col0, #T_5398c_row4_col11, #T_5398c_row11_col9 {
  background-color: #ede8f3;
  color: #000000;
}
#T_5398c_row3_col1 {
  background-color: #045e93;
  color: #f1f1f1;
}
#T_5398c_row3_col2 {
  background-color: #f9f2f8;
  color: #000000;
}
#T_5398c_row3_col4, #T_5398c_row9_col11 {
  background-color: #eae6f1;
  color: #000000;
}
#T_5398c_row3_col5, #T_5398c_row11_col3 {
  background-color: #d9d8ea;
  color: #000000;
}
#T_5398c_row3_col8 {
  background-color: #034973;
  color: #f1f1f1;
}
#T_5398c_row3_col9, #T_5398c_row5_col11, #T_5398c_row8_col0, #T_5398c_row8_col4, #T_5398c_row11_col6 {
  background-color: #f0eaf4;
  color: #000000;
}
#T_5398c_row3_col11, #T_5398c_row4_col5 {
  background-color: #fef6fb;
  color: #000000;
}
#T_5398c_row4_col0, #T_5398c_row8_col7 {
  background-color: #dddbec;
  color: #000000;
}
#T_5398c_row4_col1, #T_5398c_row7_col1 {
  background-color: #b9c6e0;
  color: #000000;
}
#T_5398c_row4_col2, #T_5398c_row9_col6, #T_5398c_row11_col4, #T_5398c_row11_col10 {
  background-color: #e0dded;
  color: #000000;
}
#T_5398c_row4_col3, #T_5398c_row10_col0 {
  background-color: #cccfe5;
  color: #000000;
}
#T_5398c_row4_col9 {
  background-color: #d5d5e8;
  color: #000000;
}
#T_5398c_row5_col1 {
  background-color: #81aed2;
  color: #f1f1f1;
}
#T_5398c_row5_col2 {
  background-color: #afc1dd;
  color: #000000;
}
#T_5398c_row5_col3 {
  background-color: #adc1dd;
  color: #000000;
}
#T_5398c_row5_col4 {
  background-color: #faf2f8;
  color: #000000;
}
#T_5398c_row5_col6 {
  background-color: #abbfdc;
  color: #000000;
}
#T_5398c_row5_col7 {
  background-color: #8bb2d4;
  color: #000000;
}
#T_5398c_row5_col8 {
  background-color: #bfc9e1;
  color: #000000;
}
#T_5398c_row5_col9, #T_5398c_row8_col11 {
  background-color: #fbf3f9;
  color: #000000;
}
#T_5398c_row5_col10, #T_5398c_row7_col4 {
  background-color: #fdf5fa;
  color: #000000;
}
#T_5398c_row6_col0, #T_5398c_row8_col5, #T_5398c_row9_col0 {
  background-color: #dedcec;
  color: #000000;
}
#T_5398c_row6_col1 {
  background-color: #97b7d7;
  color: #000000;
}
#T_5398c_row6_col2 {
  background-color: #bcc7e1;
  color: #000000;
}
#T_5398c_row6_col3 {
  background-color: #86b0d3;
  color: #000000;
}
#T_5398c_row6_col4 {
  background-color: #f7f0f7;
  color: #000000;
}
#T_5398c_row6_col5 {
  background-color: #a5bddb;
  color: #000000;
}
#T_5398c_row6_col7 {
  background-color: #4a98c5;
  color: #f1f1f1;
}
#T_5398c_row6_col8 {
  background-color: #9ab8d8;
  color: #000000;
}
#T_5398c_row6_col9, #T_5398c_row11_col7 {
  background-color: #e7e3f0;
  color: #000000;
}
#T_5398c_row6_col10 {
  background-color: #d4d4e8;
  color: #000000;
}
#T_5398c_row6_col11, #T_5398c_row9_col5, #T_5398c_row11_col0 {
  background-color: #f2ecf5;
  color: #000000;
}
#T_5398c_row7_col2 {
  background-color: #9ebad9;
  color: #000000;
}
#T_5398c_row7_col5 {
  background-color: #8eb3d5;
  color: #000000;
}
#T_5398c_row7_col6 {
  background-color: #529bc7;
  color: #f1f1f1;
}
#T_5398c_row7_col8, #T_5398c_row8_col6 {
  background-color: #c2cbe2;
  color: #000000;
}
#T_5398c_row7_col9 {
  background-color: #e5e1ef;
  color: #000000;
}
#T_5398c_row7_col11 {
  background-color: #f1ebf4;
  color: #000000;
}
#T_5398c_row8_col1 {
  background-color: #0567a1;
  color: #f1f1f1;
}
#T_5398c_row8_col3 {
  background-color: #034871;
  color: #f1f1f1;
}
#T_5398c_row8_col10 {
  background-color: #d2d2e7;
  color: #000000;
}
#T_5398c_row9_col1, #T_5398c_row10_col2, #T_5398c_row10_col9 {
  background-color: #ced0e6;
  color: #000000;
}
#T_5398c_row9_col2 {
  background-color: #bbc7e0;
  color: #000000;
}
#T_5398c_row9_col3, #T_5398c_row9_col10 {
  background-color: #bdc8e1;
  color: #000000;
}
#T_5398c_row9_col4 {
  background-color: #c0c9e2;
  color: #000000;
}
#T_5398c_row9_col7 {
  background-color: #d7d6e9;
  color: #000000;
}
#T_5398c_row9_col8, #T_5398c_row10_col4 {
  background-color: #d0d1e6;
  color: #000000;
}
#T_5398c_row10_col3 {
  background-color: #a7bddb;
  color: #000000;
}
#T_5398c_row10_col6 {
  background-color: #dbdaeb;
  color: #000000;
}
#T_5398c_row10_col7 {
  background-color: #e2dfee;
  color: #000000;
}
#T_5398c_row10_col8 {
  background-color: #b1c2de;
  color: #000000;
}
#T_5398c_row10_col11, #T_5398c_row11_col5 {
  background-color: #e9e5f1;
  color: #000000;
}
#T_5398c_row11_col1 {
  background-color: #d2d3e7;
  color: #000000;
}
#T_5398c_row11_col2 {
  background-color: #9fbad9;
  color: #000000;
}
</style>
<table id="T_5398c_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Title</th>
      <th class="col_heading level0 col1" >Price</th>
      <th class="col_heading level0 col2" >Mileage(miles)</th>
      <th class="col_heading level0 col3" >Registration_Year</th>
      <th class="col_heading level0 col4" >Fuel type</th>
      <th class="col_heading level0 col5" >Body type</th>
      <th class="col_heading level0 col6" >Doors</th>
      <th class="col_heading level0 col7" >Seats</th>
      <th class="col_heading level0 col8" >Emission Class</th>
      <th class="col_heading level0 col9" >country</th>
      <th class="col_heading level0 col10" >Gearbox_Manual</th>
      <th class="col_heading level0 col11" >Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5398c_level0_row0" class="row_heading level0 row0" >Title</th>
      <td id="T_5398c_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_5398c_row0_col1" class="data row0 col1" >-0.096757</td>
      <td id="T_5398c_row0_col2" class="data row0 col2" >-0.081463</td>
      <td id="T_5398c_row0_col3" class="data row0 col3" >0.021919</td>
      <td id="T_5398c_row0_col4" class="data row0 col4" >0.107680</td>
      <td id="T_5398c_row0_col5" class="data row0 col5" >-0.108652</td>
      <td id="T_5398c_row0_col6" class="data row0 col6" >0.099627</td>
      <td id="T_5398c_row0_col7" class="data row0 col7" >0.089156</td>
      <td id="T_5398c_row0_col8" class="data row0 col8" >0.004234</td>
      <td id="T_5398c_row0_col9" class="data row0 col9" >0.101858</td>
      <td id="T_5398c_row0_col10" class="data row0 col10" >0.181561</td>
      <td id="T_5398c_row0_col11" class="data row0 col11" >-0.016480</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row1" class="row_heading level0 row1" >Price</th>
      <td id="T_5398c_row1_col0" class="data row1 col0" >-0.096757</td>
      <td id="T_5398c_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_5398c_row1_col2" class="data row1 col2" >-0.503350</td>
      <td id="T_5398c_row1_col3" class="data row1 col3" >0.780527</td>
      <td id="T_5398c_row1_col4" class="data row1 col4" >-0.026317</td>
      <td id="T_5398c_row1_col5" class="data row1 col5" >0.199939</td>
      <td id="T_5398c_row1_col6" class="data row1 col6" >0.113713</td>
      <td id="T_5398c_row1_col7" class="data row1 col7" >-0.024606</td>
      <td id="T_5398c_row1_col8" class="data row1 col8" >0.703448</td>
      <td id="T_5398c_row1_col9" class="data row1 col9" >-0.119894</td>
      <td id="T_5398c_row1_col10" class="data row1 col10" >-0.204308</td>
      <td id="T_5398c_row1_col11" class="data row1 col11" >-0.140314</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row2" class="row_heading level0 row2" >Mileage(miles)</th>
      <td id="T_5398c_row2_col0" class="data row2 col0" >-0.081463</td>
      <td id="T_5398c_row2_col1" class="data row2 col1" >-0.503350</td>
      <td id="T_5398c_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_5398c_row2_col3" class="data row2 col3" >-0.444010</td>
      <td id="T_5398c_row2_col4" class="data row2 col4" >-0.229571</td>
      <td id="T_5398c_row2_col5" class="data row2 col5" >0.023324</td>
      <td id="T_5398c_row2_col6" class="data row2 col6" >-0.034683</td>
      <td id="T_5398c_row2_col7" class="data row2 col7" >0.092801</td>
      <td id="T_5398c_row2_col8" class="data row2 col8" >-0.380299</td>
      <td id="T_5398c_row2_col9" class="data row2 col9" >-0.031590</td>
      <td id="T_5398c_row2_col10" class="data row2 col10" >-0.117852</td>
      <td id="T_5398c_row2_col11" class="data row2 col11" >0.084472</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row3" class="row_heading level0 row3" >Registration_Year</th>
      <td id="T_5398c_row3_col0" class="data row3 col0" >0.021919</td>
      <td id="T_5398c_row3_col1" class="data row3 col1" >0.780527</td>
      <td id="T_5398c_row3_col2" class="data row3 col2" >-0.444010</td>
      <td id="T_5398c_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_5398c_row3_col4" class="data row3 col4" >-0.062607</td>
      <td id="T_5398c_row3_col5" class="data row3 col5" >0.065793</td>
      <td id="T_5398c_row3_col6" class="data row3 col6" >0.214318</td>
      <td id="T_5398c_row3_col7" class="data row3 col7" >0.028659</td>
      <td id="T_5398c_row3_col8" class="data row3 col8" >0.911732</td>
      <td id="T_5398c_row3_col9" class="data row3 col9" >-0.003718</td>
      <td id="T_5398c_row3_col10" class="data row3 col10" >0.092251</td>
      <td id="T_5398c_row3_col11" class="data row3 col11" >-0.135488</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row4" class="row_heading level0 row4" >Fuel type</th>
      <td id="T_5398c_row4_col0" class="data row4 col0" >0.107680</td>
      <td id="T_5398c_row4_col1" class="data row4 col1" >-0.026317</td>
      <td id="T_5398c_row4_col2" class="data row4 col2" >-0.229571</td>
      <td id="T_5398c_row4_col3" class="data row4 col3" >-0.062607</td>
      <td id="T_5398c_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_5398c_row4_col5" class="data row4 col5" >-0.184092</td>
      <td id="T_5398c_row4_col6" class="data row4 col6" >-0.161433</td>
      <td id="T_5398c_row4_col7" class="data row4 col7" >-0.209188</td>
      <td id="T_5398c_row4_col8" class="data row4 col8" >-0.106434</td>
      <td id="T_5398c_row4_col9" class="data row4 col9" >0.138090</td>
      <td id="T_5398c_row4_col10" class="data row4 col10" >0.078614</td>
      <td id="T_5398c_row4_col11" class="data row4 col11" >-0.005135</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row5" class="row_heading level0 row5" >Body type</th>
      <td id="T_5398c_row5_col0" class="data row5 col0" >-0.108652</td>
      <td id="T_5398c_row5_col1" class="data row5 col1" >0.199939</td>
      <td id="T_5398c_row5_col2" class="data row5 col2" >0.023324</td>
      <td id="T_5398c_row5_col3" class="data row5 col3" >0.065793</td>
      <td id="T_5398c_row5_col4" class="data row5 col4" >-0.184092</td>
      <td id="T_5398c_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_5398c_row5_col6" class="data row5 col6" >0.260308</td>
      <td id="T_5398c_row5_col7" class="data row5 col7" >0.328640</td>
      <td id="T_5398c_row5_col8" class="data row5 col8" >0.035671</td>
      <td id="T_5398c_row5_col9" class="data row5 col9" >-0.086460</td>
      <td id="T_5398c_row5_col10" class="data row5 col10" >-0.189107</td>
      <td id="T_5398c_row5_col11" class="data row5 col11" >-0.026193</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row6" class="row_heading level0 row6" >Doors</th>
      <td id="T_5398c_row6_col0" class="data row6 col0" >0.099627</td>
      <td id="T_5398c_row6_col1" class="data row6 col1" >0.113713</td>
      <td id="T_5398c_row6_col2" class="data row6 col2" >-0.034683</td>
      <td id="T_5398c_row6_col3" class="data row6 col3" >0.214318</td>
      <td id="T_5398c_row6_col4" class="data row6 col4" >-0.161433</td>
      <td id="T_5398c_row6_col5" class="data row6 col5" >0.260308</td>
      <td id="T_5398c_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_5398c_row6_col7" class="data row6 col7" >0.497019</td>
      <td id="T_5398c_row6_col8" class="data row6 col8" >0.177558</td>
      <td id="T_5398c_row6_col9" class="data row6 col9" >0.047623</td>
      <td id="T_5398c_row6_col10" class="data row6 col10" >0.073879</td>
      <td id="T_5398c_row6_col11" class="data row6 col11" >-0.044786</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row7" class="row_heading level0 row7" >Seats</th>
      <td id="T_5398c_row7_col0" class="data row7 col0" >0.089156</td>
      <td id="T_5398c_row7_col1" class="data row7 col1" >-0.024606</td>
      <td id="T_5398c_row7_col2" class="data row7 col2" >0.092801</td>
      <td id="T_5398c_row7_col3" class="data row7 col3" >0.028659</td>
      <td id="T_5398c_row7_col4" class="data row7 col4" >-0.209188</td>
      <td id="T_5398c_row7_col5" class="data row7 col5" >0.328640</td>
      <td id="T_5398c_row7_col6" class="data row7 col6" >0.497019</td>
      <td id="T_5398c_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_5398c_row7_col8" class="data row7 col8" >0.022823</td>
      <td id="T_5398c_row7_col9" class="data row7 col9" >0.055857</td>
      <td id="T_5398c_row7_col10" class="data row7 col10" >-0.003037</td>
      <td id="T_5398c_row7_col11" class="data row7 col11" >-0.029299</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row8" class="row_heading level0 row8" >Emission Class</th>
      <td id="T_5398c_row8_col0" class="data row8 col0" >0.004234</td>
      <td id="T_5398c_row8_col1" class="data row8 col1" >0.703448</td>
      <td id="T_5398c_row8_col2" class="data row8 col2" >-0.380299</td>
      <td id="T_5398c_row8_col3" class="data row8 col3" >0.911732</td>
      <td id="T_5398c_row8_col4" class="data row8 col4" >-0.106434</td>
      <td id="T_5398c_row8_col5" class="data row8 col5" >0.035671</td>
      <td id="T_5398c_row8_col6" class="data row8 col6" >0.177558</td>
      <td id="T_5398c_row8_col7" class="data row8 col7" >0.022823</td>
      <td id="T_5398c_row8_col8" class="data row8 col8" >1.000000</td>
      <td id="T_5398c_row8_col9" class="data row8 col9" >-0.032114</td>
      <td id="T_5398c_row8_col10" class="data row8 col10" >0.090076</td>
      <td id="T_5398c_row8_col11" class="data row8 col11" >-0.107745</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row9" class="row_heading level0 row9" >country</th>
      <td id="T_5398c_row9_col0" class="data row9 col0" >0.101858</td>
      <td id="T_5398c_row9_col1" class="data row9 col1" >-0.119894</td>
      <td id="T_5398c_row9_col2" class="data row9 col2" >-0.031590</td>
      <td id="T_5398c_row9_col3" class="data row9 col3" >-0.003718</td>
      <td id="T_5398c_row9_col4" class="data row9 col4" >0.138090</td>
      <td id="T_5398c_row9_col5" class="data row9 col5" >-0.086460</td>
      <td id="T_5398c_row9_col6" class="data row9 col6" >0.047623</td>
      <td id="T_5398c_row9_col7" class="data row9 col7" >0.055857</td>
      <td id="T_5398c_row9_col8" class="data row9 col8" >-0.032114</td>
      <td id="T_5398c_row9_col9" class="data row9 col9" >1.000000</td>
      <td id="T_5398c_row9_col10" class="data row9 col10" >0.165250</td>
      <td id="T_5398c_row9_col11" class="data row9 col11" >0.011718</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row10" class="row_heading level0 row10" >Gearbox_Manual</th>
      <td id="T_5398c_row10_col0" class="data row10 col0" >0.181561</td>
      <td id="T_5398c_row10_col1" class="data row10 col1" >-0.204308</td>
      <td id="T_5398c_row10_col2" class="data row10 col2" >-0.117852</td>
      <td id="T_5398c_row10_col3" class="data row10 col3" >0.092251</td>
      <td id="T_5398c_row10_col4" class="data row10 col4" >0.078614</td>
      <td id="T_5398c_row10_col5" class="data row10 col5" >-0.189107</td>
      <td id="T_5398c_row10_col6" class="data row10 col6" >0.073879</td>
      <td id="T_5398c_row10_col7" class="data row10 col7" >-0.003037</td>
      <td id="T_5398c_row10_col8" class="data row10 col8" >0.090076</td>
      <td id="T_5398c_row10_col9" class="data row10 col9" >0.165250</td>
      <td id="T_5398c_row10_col10" class="data row10 col10" >1.000000</td>
      <td id="T_5398c_row10_col11" class="data row10 col11" >0.016287</td>
    </tr>
    <tr>
      <th id="T_5398c_level0_row11" class="row_heading level0 row11" >Service history_Unknown</th>
      <td id="T_5398c_row11_col0" class="data row11 col0" >-0.016480</td>
      <td id="T_5398c_row11_col1" class="data row11 col1" >-0.140314</td>
      <td id="T_5398c_row11_col2" class="data row11 col2" >0.084472</td>
      <td id="T_5398c_row11_col3" class="data row11 col3" >-0.135488</td>
      <td id="T_5398c_row11_col4" class="data row11 col4" >-0.005135</td>
      <td id="T_5398c_row11_col5" class="data row11 col5" >-0.026193</td>
      <td id="T_5398c_row11_col6" class="data row11 col6" >-0.044786</td>
      <td id="T_5398c_row11_col7" class="data row11 col7" >-0.029299</td>
      <td id="T_5398c_row11_col8" class="data row11 col8" >-0.107745</td>
      <td id="T_5398c_row11_col9" class="data row11 col9" >0.011718</td>
      <td id="T_5398c_row11_col10" class="data row11 col10" >0.016287</td>
      <td id="T_5398c_row11_col11" class="data row11 col11" >1.000000</td>
    </tr>
  </tbody>
</table>





```python
plt.figure(figsize=(15,8))
sns.heatmap(df_corr, annot=True, vmax=1, vmin=-1, cmap="coolwarm")
```




    <AxesSubplot:>




    
![output_88_1](https://github.com/inseonseo/inseonseo.github.io/assets/50574738/d6fb6a3b-2390-4cff-8b0e-39083bb8133d)    


- 원핫인코딩을 전부 정수형 인코딩으로 변환 후 다시 실행


```python
df_corr = merged_.corr()
```


```python
df_corr.style.background_gradient()
```


```python
plt.figure(figsize=(15,8))
sns.heatmap(df_corr, annot=True, vmax=1, vmin=-1, cmap="coolwarm")
```

# 가격예측


```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
```


```python
merged_model = merged.drop(columns=['Price'])

feature_names = merged_model.select_dtypes(include="number").columns
feature_names
```




    Index(['Title', 'Mileage(miles)', 'Registration_Year', 'Fuel type',
           'Body type', 'Doors', 'Seats', 'Emission Class', 'country',
           'Gearbox_Manual', 'Service history_Unknown'],
          dtype='object')




```python
label_name = "Price"
label_name
```




    'Price'




```python
X = merged_model[feature_names]
y = merged[label_name]
display(X.head(2))
display(y.head(3))
X.shape, y.shape
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
      <th>Title</th>
      <th>Mileage(miles)</th>
      <th>Registration_Year</th>
      <th>Fuel type</th>
      <th>Body type</th>
      <th>Doors</th>
      <th>Seats</th>
      <th>Emission Class</th>
      <th>country</th>
      <th>Gearbox_Manual</th>
      <th>Service history_Unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>70189</td>
      <td>2016</td>
      <td>0</td>
      <td>4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



    0    6900
    1    6900
    2    1990
    Name: Price, dtype: int64





    ((3602, 11), (3602,))




```python
split_count = int(merged_model.shape[0] * 0.8)
split_count
```




    2881




```python
X_train = X[:split_count].copy()
y_train = y[:split_count].copy()

X_test = X[split_count:].copy()
y_test = y[split_count:].copy()

print("* 데이터 나뉜 값들")

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

    * 데이터 나뉜 값들





    ((2881, 11), (721, 11), (2881,), (721,))




```python
model.fit(X_train, y_train)
```




    DecisionTreeClassifier()




```python
y_predict = model.predict(X_test)
y_predict[:5]
```




    array([3652, 2495, 8799,  945, 9149])




```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = split_count,
                                                    random_state = 42)

print("* 데이터가 잘 나뉘었는지 shape 값을 통해 확인해 보세요.")

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

    * 데이터가 잘 나뉘었는지 shape 값을 통해 확인해 보세요.





    ((2881, 11), (721, 11), (2881,), (721,))




```python
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
y_predict[:5]
```




    array([5495, 8295, 7000, 2395, 9290])




```python
model.feature_importances_
sns.barplot(x=model.feature_importances_, y=feature_names)
```




    <AxesSubplot:>




    
![output_104_1](https://github.com/inseonseo/inseonseo.github.io/assets/50574738/f5cf9cef-6d05-422c-a4f7-b646786a1efc)
    



```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict) * 100
```


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 예시: X와 y는 각각 독립변수와 종속변수에 해당하는 데이터
# X와 y 데이터는 실제 데이터에 맞게 설정해야 합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 세트를 사용하여 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

```

    Mean Squared Error: 4808952.406531508
    R-squared: 0.7329704103775868


### 회고 

- 모델링 전 데이터 전처리 
    - 결측치 처리, 이상치 제거, 피처 스케일링, 범주형 변수 처리 등이 모델 성능에 영향을 끼침
    - 상관관계 높은 피처가 모델에 동시 포함되면 다중공산성 문제 발생할 수 있고 이는 모델 안정성과 해석을 어렵게 만든다   
    
    
- 좋은 모델을 위한 데이터란? 
    - 원핫인코딩의 적절한 활용   
    : 범주형 데이터를 모델이 이해할 수 있는 형태로 변환하는 중요한 단계이다. 하지만 항상 정답은 아니다.
    - 범주 개수가 많을 때 원핫인코딩을 적용하면 feature 차원이 매우 증가하기 때문이다.   
    

- 문제해결
    - 도메인 이해 없는 방법론만 따르는 전처리는 성능 향상에 불필요하다 
    - 어떤 데이터를 찾고자, 어떤 처리를 할 지 생각해보는 게 중요하다
    - 분석하기 전 기획하고 계획하는 과정이 필요하다고 느꼈다
    - 차종, 중고차 분류 기준으로 재정리 후 성능 향상


```python

```
