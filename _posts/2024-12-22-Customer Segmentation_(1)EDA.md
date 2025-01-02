---
title: Customer Segmentation_(1)EDA
date: 2024-12-22
categories: [Data Analysis, Customer Segmentation]
tags: [customer segmentation]     
---

```python
select * 
from first-metric-410915.modulabs_project.data
limit 10
```

<img width="1437" alt="Screen%20Shot%202024-01-15%20at%2011 22 04%20AM" src="https://github.com/user-attachments/assets/46a72187-1d5b-4247-86b9-62847d0bbee2" />


```python
SELECT COUNT(*) as total_rows
FROM first-metric-410915.modulabs_project.data
```

<img width="640" alt="Screen%20Shot%202024-01-15%20at%2011 24 18%20AM" src="https://github.com/user-attachments/assets/674f5f69-b55a-4c56-a3ab-957f1b9bec9b" />



```python
# 결측치 비율 확인
SELECT COUNT(InvoiceNo) as COUNT_InvoiceNo, COUNT(StockCode) as COUNT_StockCode, COUNT(Description) as COUNT_Description, COUNT(Quantity) as COUNT_Quantity, COUNT(InvoiceDate) as COUNT_InvoiceDate, COUNT(UnitPrice) as COUNT_UnitPrice, COUNT(CustomerID) as COUNT_CustomerID, COUNT(Country) as COUNT_Country
FROM first-metric-410915.modulabs_project.data;
```

<img width="1072" alt="Screen%20Shot%202024-01-15%20at%2011 25 50%20AM" src="https://github.com/user-attachments/assets/ed218fb1-df3f-4864-91a5-c6c969412266" />

## 1-1. 데이터 전처리 : 결측치 제거


```python
#  결측치 비율 확인
SELECT
    'InvoiceNo' AS column_name,
    ROUND(SUM(CASE WHEN InvoiceNo IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'StockCode' AS column_name,
    ROUND(SUM(CASE WHEN StockCode IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'Description' AS column_name,
    ROUND(SUM(CASE WHEN Description IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'Quantity' AS column_name,
    ROUND(SUM(CASE WHEN Quantity IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'InvoiceDate' AS column_name,
    ROUND(SUM(CASE WHEN InvoiceDate IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'UnitPrice' AS column_name,
    ROUND(SUM(CASE WHEN UnitPrice IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'CustomerID' AS column_name,
    ROUND(SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
UNION ALL
SELECT
    'Country' AS column_name,
    ROUND(SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_percentage
FROM first-metric-410915.modulabs_project.data
```
<img width="391" alt="Screen%20Shot%202024-01-15%20at%2011 27 24%20AM" src="https://github.com/user-attachments/assets/6b3b865f-dc5c-411f-8f68-80e100e5cf4f" />


```python
# Description 확인
SELECT DISTINCT Description
FROM first-metric-410915.modulabs_project.data
WHERE StockCode = '85123A'
```

<img width="276" alt="Screen%20Shot%202024-01-15%20at%2011 27 52%20AM" src="https://github.com/user-attachments/assets/8bc7716c-0d13-4210-b2ce-506a8198b3b4" />


```python
# 결측치 처리
DELETE FROM first-metric-410915.modulabs_project.data
WHERE Description IS NULL 
OR CustomerID IS NULL;
```

<img width="522" alt="Screen%20Shot%202024-01-15%20at%2011 35 34%20AM" src="https://github.com/user-attachments/assets/742bacb7-01ec-440d-9d0c-ae6b77f19020" />


```python

```


```python

```

<img width="698" alt="Screen%20Shot%202024-01-15%20at%2011 36 04%20AM" src="https://github.com/user-attachments/assets/26eaafab-bc38-4085-9b2b-b6953a530d5e" />

## 1-2 데이터 전처리: 중복값 처리


```python
# 중복값 확인
WITH DUPLICATE AS (
  SELECT InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, COUNT(*) as row_count
  FROM first-metric-410915.modulabs_project.data
  GROUP BY 1,2,3,4,5,6,7,8
)
SELECT *
FROM DUPLICATE
WHERE DUPLICATE.row_count > 1
```

<img width="1569" alt="Screen%20Shot%202024-01-15%20at%2011 37 14%20AM" src="https://github.com/user-attachments/assets/bd4a0039-4c62-43ea-bb19-15edcd778d41" />


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS
SELECT DISTINCT *
FROM first-metric-410915.modulabs_project.data;
```


```python
SELECT COUNT(*) 
FROM first-metric-410915.modulabs_project.data;
```

<img width="621" alt="Screen%20Shot%202024-01-15%20at%2011 39 15%20AM" src="https://github.com/user-attachments/assets/bc2ca01f-bb5e-4601-87b9-70cef68dd65d" />


```python
SELECT COUNT(DISTINCT InvoiceNo) 
FROM first-metric-410915.modulabs_project.data;
```
<img width="618" alt="Screen%20Shot%202024-01-15%20at%2011 39 55%20AM" src="https://github.com/user-attachments/assets/59e4b15a-bd0a-429a-8bf0-e597108e7e71" />

```python
SELECT DISTINCT InvoiceNo
FROM first-metric-410915.modulabs_project.data
LIMIT 100
```

<img width="744" alt="Screen%20Shot%202024-01-15%20at%2012 02 01%20PM" src="https://github.com/user-attachments/assets/a0b1f4b0-85fe-4fe9-bd01-a0405520eec2" />

## 1-3 데이터 전처리: 오류값 처리


```python
SELECT *
FROM first-metric-410915.modulabs_project.data
WHERE InvoiceNo LIKE 'C%'
LIMIT 100
```

<img width="1435" alt="Screen%20Shot%202024-01-15%20at%2011 45 12%20AM" src="https://github.com/user-attachments/assets/956fb95e-7fa1-4d10-bcee-6c2ac35d397c" />

여기서 발견할 수 있는 특이한 경향성
100건 중 2 건 제외 전부 United Kingdom 에서 구매한 데이터 => 단순히 취소 데이터가 많은 건지 애초에 그 국가에서 많이 이용하는 건지 확인해봐야 함 <Br/>Quantity가 음수
전체 취소건은 8872건인데 이 중 취소한 물품은 3654 종류 => 여러 고객이 구매 취소하는 몇몇 물품들이 몰려 있을 것 .


```python
SELECT concat(ROUND(SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END)/ COUNT(InvoiceNo) * 100, 1), "%") as cancellations
FROM first-metric-410915.modulabs_project.data;
```

<img width="657" alt="Screen%20Shot%202024-01-15%20at%2011 57 41%20AM" src="https://github.com/user-attachments/assets/e631ca77-98dc-4eb8-aeba-4035c3fba9e4" />


```python
SELECT COUNT(DISTINCT StockCode) 
FROM first-metric-410915.modulabs_project.data
```

<img width="619" alt="Screen%20Shot%202024-01-15%20at%2011 59 19%20AM" src="https://github.com/user-attachments/assets/0d6f7f8f-2bc0-4da5-8893-b76bb03ed46e" />


```python
SELECT StockCode, COUNT(*) AS sell_cnt 
FROM first-metric-410915.modulabs_project.data
GROUP BY 1
ORDER BY sell_cnt DESC
LIMIT 10
```

<img width="759" alt="Screen%20Shot%202024-01-15%20at%2012 01 09%20PM" src="https://github.com/user-attachments/assets/807b9b44-a8c8-4bd1-a361-adbfa411b959" />

### 인사이트
먼저 파악할 수 있는 건 대부분의 StockCode들은 5-6 자리 숫자라는 점.   
그래서 StockCode 중 'POST'는 이상치 같아 보인다.   
- 그 이상의 정보는 StockCode 자체만 봤을 때는 파악하기 어렵다.   
- 다른 StockCode와는 확연히 POST가 뭔지 확인해봐야할 것 같다. 5자리 숫자 끝에 붙은 A, B는 분류 기준으로 보인다.   


```python
WITH UniqueStockCodes AS (
  SELECT DISTINCT StockCode
  FROM first-metric-410915.modulabs_project.data
)
SELECT
  LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, r'[0-9]', '')) AS number_count,
  COUNT(*) AS stock_cnt
FROM UniqueStockCodes
GROUP BY number_count
ORDER BY stock_cnt DESC;
```

<img width="525" alt="Screen%20Shot%202024-01-15%20at%2012 09 05%20PM" src="https://github.com/user-attachments/assets/c7bb25c8-9392-49cb-b53f-75aa39659769" />


```python
SELECT DISTINCT StockCode, number_count
FROM (
  SELECT StockCode,
    LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, r'[0-9]', '')) AS number_count
  FROM first-metric-410915.modulabs_project.data
) 
WHERE number_count = 0 OR number_count = 1
```

<img width="595" alt="Screen%20Shot%202024-01-15%20at%2012 14 55%20PM" src="https://github.com/user-attachments/assets/34002869-0d27-42b6-960d-f994db465a1c" />


```python
SELECT concat(ROUND(SUM(CASE WHEN StockCode IN ('POST', 'M', 'PADS', 'D', 'BANK CHARGES', 'DOT', 'CRUK', 'C2') 
                        THEN 1 ELSE 0 END)/ COUNT(StockCode) * 100, 2), "%") as abnormal_StockCode
FROM first-metric-410915.modulabs_project.data
```

<img width="644" alt="Screen%20Shot%202024-01-15%20at%2012 18 24%20PM" src="https://github.com/user-attachments/assets/df6c8281-a3cb-4fa2-8e01-f1ca83e67e09" />


```python
DELETE FROM first-metric-410915.modulabs_project.data
WHERE StockCode IN (
  SELECT DISTINCT StockCode
  FROM (
  SELECT StockCode,
    LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, r'[0-9]', '')) AS number_count
  FROM first-metric-410915.modulabs_project.data
  ) 
WHERE number_count = 0 OR number_count = 1
)
```

```python
SELECT Description, COUNT(*) AS description_cnt
FROM first-metric-410915.modulabs_project.data
GROUP BY 1
ORDER BY 2 DESC
LIMIT 30
```

<img width="1017" alt="Screen%20Shot%202024-01-15%20at%2012 31 59%20PM" src="https://github.com/user-attachments/assets/87e3703c-614f-4628-8fb9-8840d52066f9" />


```python
SELECT DISTINCT Description
FROM first-metric-410915.modulabs_project.data
WHERE REGEXP_CONTAINS(Description, r'[a-z]')
```

<img width="1000" alt="Screen%20Shot%202024-01-15%20at%2012 33 28%20PM" src="https://github.com/user-attachments/assets/106e70a7-7070-487b-a17e-efa68496e6a6" />


```python
DELETE
FROM first-metric-410915.modulabs_project.data
WHERE Description IN ('Next Day Carriage', 'High Resolution Image')
```


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS
SELECT
  * EXCEPT (Description),
  UPPER(Description) AS Description
FROM first-metric-410915.modulabs_project.data
```



```python
select * FROM first-metric-410915.modulabs_project.data_updated WHERE InvoiceNo = '560538'
```

<img width="1049" alt="Screen%20Shot%202024-01-15%20at%2012 55 03%20PM" src="https://github.com/user-attachments/assets/345e9922-312b-473b-9fe4-26f6c6eb733e" />


```python
SELECT COUNT(UnitPrice) AS cnt_quantity, MIN(Quantity) AS min_quantity, MAX(Quantity) AS max_quantity
, AVG(Quantity) AS avg_quantity
FROM first-metric-410915.modulabs_project.data
WHERE UnitPrice = 0
```

<img width="717" alt="Screen%20Shot%202024-01-15%20at%201 04 50%20PM" src="https://github.com/user-attachments/assets/a1c7d365-f244-42eb-950d-d641ab1a4da1" />


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS 
SELECT *
FROM first-metric-410915.modulabs_project.data
WHERE UnitPrice = 0
```

### 불필요한 행 제거 
서비스 관련 정보('Next Day Carriage', 'High Resolution Image')포함하는 행들 33개 제거   
399606 → 399573(=399606-33)

<img width="678" alt="Screen%20Shot%202024-01-15%20at%201 08 47%20PM" src="https://github.com/user-attachments/assets/a20959ef-2e63-4880-9e10-688b2c915dda" />