---
title: Customer Segmentation_1
date: 2024-12-22
categories: [Data Analysis, Customer Segmentation]
tags: [customer segmentation]     
---

```python
select * 
from first-metric-410915.modulabs_project.data
limit 10
```

![Screen%20Shot%202024-01-15%20at%2011.22.04%20AM.png](Screen%20Shot%202024-01-15%20at%2011.22.04%20AM.png)


```python
SELECT COUNT(*) as total_rows
FROM first-metric-410915.modulabs_project.data
```

![Screen%20Shot%202024-01-15%20at%2011.24.18%20AM.png](Screen%20Shot%202024-01-15%20at%2011.24.18%20AM.png)


```python

```


```python
# 결측치 비율 확인
SELECT COUNT(InvoiceNo) as COUNT_InvoiceNo, COUNT(StockCode) as COUNT_StockCode, COUNT(Description) as COUNT_Description, COUNT(Quantity) as COUNT_Quantity, COUNT(InvoiceDate) as COUNT_InvoiceDate, COUNT(UnitPrice) as COUNT_UnitPrice, COUNT(CustomerID) as COUNT_CustomerID, COUNT(Country) as COUNT_Country
FROM first-metric-410915.modulabs_project.data;
```

![Screen%20Shot%202024-01-15%20at%2011.25.50%20AM.png](Screen%20Shot%202024-01-15%20at%2011.25.50%20AM.png)

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

![Screen%20Shot%202024-01-15%20at%2011.27.24%20AM.png](Screen%20Shot%202024-01-15%20at%2011.27.24%20AM.png)


```python
# Description 확인
SELECT DISTINCT Description
FROM first-metric-410915.modulabs_project.data
WHERE StockCode = '85123A'
```

![Screen%20Shot%202024-01-15%20at%2011.27.52%20AM.png](Screen%20Shot%202024-01-15%20at%2011.27.52%20AM.png)


```python
# 결측치 처리
DELETE FROM first-metric-410915.modulabs_project.data
WHERE Description IS NULL 
OR CustomerID IS NULL;
```

![Screen%20Shot%202024-01-15%20at%2011.35.34%20AM.png](Screen%20Shot%202024-01-15%20at%2011.35.34%20AM.png)


```python

```


```python

```

![Screen%20Shot%202024-01-15%20at%2011.36.04%20AM.png](Screen%20Shot%202024-01-15%20at%2011.36.04%20AM.png)

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

![Screen%20Shot%202024-01-15%20at%2011.37.14%20AM.png](Screen%20Shot%202024-01-15%20at%2011.37.14%20AM.png)


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS
SELECT DISTINCT *
FROM first-metric-410915.modulabs_project.data;
```

![Screen%20Shot%202024-01-15%20at%2011.38.13%20AM.png](Screen%20Shot%202024-01-15%20at%2011.38.13%20AM.png)


```python
SELECT COUNT(*) 
FROM first-metric-410915.modulabs_project.data;
```

![Screen%20Shot%202024-01-15%20at%2011.39.15%20AM.png](Screen%20Shot%202024-01-15%20at%2011.39.15%20AM.png)


```python
SELECT COUNT(DISTINCT InvoiceNo) 
FROM first-metric-410915.modulabs_project.data;
```

![Screen%20Shot%202024-01-15%20at%2011.39.55%20AM.png](Screen%20Shot%202024-01-15%20at%2011.39.55%20AM.png)


```python
SELECT DISTINCT InvoiceNo
FROM first-metric-410915.modulabs_project.data
LIMIT 100
```

![Screen%20Shot%202024-01-15%20at%2012.02.01%20PM.png](Screen%20Shot%202024-01-15%20at%2012.02.01%20PM.png)

## 1-3 데이터 전처리: 오류값 처리


```python
SELECT *
FROM first-metric-410915.modulabs_project.data
WHERE InvoiceNo LIKE 'C%'
LIMIT 100
```

![Screen%20Shot%202024-01-15%20at%2011.45.12%20AM.png](Screen%20Shot%202024-01-15%20at%2011.45.12%20AM.png)

여기서 발견할 수 있는 특이한 경향성
100건 중 2 건 제외 전부 United Kingdom 에서 구매한 데이터 => 단순히 취소 데이터가 많은 건지 애초에 그 국가에서 많이 이용하는 건지 확인해봐야 함 <Br/>Quantity가 음수
전체 취소건은 8872건인데 이 중 취소한 물품은 3654 종류 => 여러 고객이 구매 취소하는 몇몇 물품들이 몰려 있을 것 .


```python
SELECT concat(ROUND(SUM(CASE WHEN InvoiceNo LIKE 'C%' THEN 1 ELSE 0 END)/ COUNT(InvoiceNo) * 100, 1), "%") as cancellations
FROM first-metric-410915.modulabs_project.data;
```

![Screen%20Shot%202024-01-15%20at%2011.57.41%20AM.png](Screen%20Shot%202024-01-15%20at%2011.57.41%20AM.png)


```python
SELECT COUNT(DISTINCT StockCode) 
FROM first-metric-410915.modulabs_project.data
```

![Screen%20Shot%202024-01-15%20at%2011.59.19%20AM.png](Screen%20Shot%202024-01-15%20at%2011.59.19%20AM.png)


```python
SELECT StockCode, COUNT(*) AS sell_cnt 
FROM first-metric-410915.modulabs_project.data
GROUP BY 1
ORDER BY sell_cnt DESC
LIMIT 10
```

![Screen%20Shot%202024-01-15%20at%2012.01.09%20PM.png](Screen%20Shot%202024-01-15%20at%2012.01.09%20PM.png)

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

![Screen%20Shot%202024-01-15%20at%2012.09.05%20PM.png](Screen%20Shot%202024-01-15%20at%2012.09.05%20PM.png)


```python
SELECT DISTINCT StockCode, number_count
FROM (
  SELECT StockCode,
    LENGTH(StockCode) - LENGTH(REGEXP_REPLACE(StockCode, r'[0-9]', '')) AS number_count
  FROM first-metric-410915.modulabs_project.data
) 
WHERE number_count = 0 OR number_count = 1
```

![Screen%20Shot%202024-01-15%20at%2012.14.55%20PM.png](Screen%20Shot%202024-01-15%20at%2012.14.55%20PM.png)


```python
SELECT concat(ROUND(SUM(CASE WHEN StockCode IN ('POST', 'M', 'PADS', 'D', 'BANK CHARGES', 'DOT', 'CRUK', 'C2') 
                        THEN 1 ELSE 0 END)/ COUNT(StockCode) * 100, 2), "%") as abnormal_StockCode
FROM first-metric-410915.modulabs_project.data
```

![Screen%20Shot%202024-01-15%20at%2012.18.24%20PM.png](Screen%20Shot%202024-01-15%20at%2012.18.24%20PM.png)


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

![Screen%20Shot%202024-01-15%20at%2012.28.26%20PM.png](Screen%20Shot%202024-01-15%20at%2012.28.26%20PM.png)


```python
SELECT Description, COUNT(*) AS description_cnt
FROM first-metric-410915.modulabs_project.data
GROUP BY 1
ORDER BY 2 DESC
LIMIT 30
```

![Screen%20Shot%202024-01-15%20at%2012.31.59%20PM.png](Screen%20Shot%202024-01-15%20at%2012.31.59%20PM.png)


```python
SELECT DISTINCT Description
FROM first-metric-410915.modulabs_project.data
WHERE REGEXP_CONTAINS(Description, r'[a-z]')
```

![Screen%20Shot%202024-01-15%20at%2012.33.28%20PM.png](Screen%20Shot%202024-01-15%20at%2012.33.28%20PM.png)


```python
DELETE
FROM first-metric-410915.modulabs_project.data
WHERE Description IN ('Next Day Carriage', 'High Resolution Image')
```

![Screen%20Shot%202024-01-15%20at%2012.38.06%20PM.png](Screen%20Shot%202024-01-15%20at%2012.38.06%20PM.png)


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS
SELECT
  * EXCEPT (Description),
  UPPER(Description) AS Description
FROM first-metric-410915.modulabs_project.data
```

![Screen%20Shot%202024-01-15%20at%2012.55.30%20PM.png](Screen%20Shot%202024-01-15%20at%2012.55.30%20PM.png)


```python
select * FROM first-metric-410915.modulabs_project.data_updated WHERE InvoiceNo = '560538'
```

![Screen%20Shot%202024-01-15%20at%2012.55.03%20PM.png](Screen%20Shot%202024-01-15%20at%2012.55.03%20PM.png)


```python
SELECT COUNT(UnitPrice) AS cnt_quantity, MIN(Quantity) AS min_quantity, MAX(Quantity) AS max_quantity
, AVG(Quantity) AS avg_quantity
FROM first-metric-410915.modulabs_project.data
WHERE UnitPrice = 0
```

![Screen%20Shot%202024-01-15%20at%201.04.50%20PM.png](Screen%20Shot%202024-01-15%20at%201.04.50%20PM.png)


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS 
SELECT *
FROM first-metric-410915.modulabs_project.data
WHERE UnitPrice = 0
```

### 불필요한 행 제거 
서비스 관련 정보('Next Day Carriage', 'High Resolution Image')포함하는 행들 33개 제거   
399606 → 399573(=399606-33)

![Screen%20Shot%202024-01-15%20at%201.08.47%20PM.png](Screen%20Shot%202024-01-15%20at%201.08.47%20PM.png)

## 2. RFM 스코어

### Recency


```python
SELECT *
FROM first-metric-410915.modulabs_project.data;

CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.data AS
SELECT
  * EXCEPT (InvoiceDate),
  DATE(InvoiceDate) AS InvoiceDate
FROM first-metric-410915.modulabs_project.data;

SELECT *
FROM first-metric-410915.modulabs_project.data;
```

![Screen%20Shot%202024-01-15%20at%201.45.09%20PM.png](Screen%20Shot%202024-01-15%20at%201.45.09%20PM.png)
🔼 변경 전

![Screen%20Shot%202024-01-15%20at%201.43.49%20PM.png](Screen%20Shot%202024-01-15%20at%201.43.49%20PM.png)
🔼 변경 후


```python
SELECT
    MAX(InvoiceDate) OVER () AS most_recent_date,
    InvoiceDate AS InvoiceDay,
    *
FROM first-metric-410915.modulabs_project.data;
```



![Screen%20Shot%202024-01-15%20at%202.16.59%20PM.png](Screen%20Shot%202024-01-15%20at%202.16.59%20PM.png)


```python
SELECT
    CustomerID,
    MAX(InvoiceDate) AS InvoiceDay
FROM first-metric-410915.modulabs_project.data
GROUP BY 1
ORDER BY 1;

SELECT
    DISTINCT CustomerID,
    MAX(InvoiceDate) OVER (PARTITION BY CustomerID) AS InvoiceDay
FROM first-metric-410915.modulabs_project.data
ORDER BY 1;
```

![Screen%20Shot%202024-01-15%20at%202.26.31%20PM.png](Screen%20Shot%202024-01-15%20at%202.26.31%20PM.png)


```python
SELECT
  CustomerID, 
  EXTRACT(DAY FROM MAX(InvoiceDay) OVER () - InvoiceDay) AS recency
FROM (
  SELECT 
    CustomerID,
    MAX(DATE(InvoiceDate)) AS InvoiceDay
  FROM first-metric-410915.modulabs_project.data
  GROUP BY CustomerID
)ORDER BY 1;
```

![Screen%20Shot%202024-01-15%20at%202.28.01%20PM.png](Screen%20Shot%202024-01-15%20at%202.28.01%20PM.png)


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_r AS
SELECT
  CustomerID, 
  EXTRACT(DAY FROM MAX(InvoiceDay) OVER () - InvoiceDay) AS recency
FROM (
  SELECT 
    CustomerID,
    MAX(DATE(InvoiceDate)) AS InvoiceDay
  FROM first-metric-410915.modulabs_project.data
  GROUP BY CustomerID
)
```

![Screen%20Shot%202024-01-15%20at%202.30.06%20PM.png](Screen%20Shot%202024-01-15%20at%202.30.06%20PM.png)

![Screen%20Shot%202024-01-15%20at%202.31.09%20PM.png](Screen%20Shot%202024-01-15%20at%202.31.09%20PM.png)
🔼 데이터 user_r

### Frequency


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_rf AS
-- (1) 전체 거래 건수 계산
WITH purchase_cnt AS ( 
  SELECT
    CustomerID,
    COUNT(DISTINCT InvoiceNo) AS purchase_cnt
  FROM first-metric-410915.modulabs_project.data
  GROUP BY 1
),
-- (2) 구매한 아이템 총 수량 계산
item_cnt AS (
  SELECT
    CustomerID,
    SUM(Quantity) AS item_cnt
  FROM first-metric-410915.modulabs_project.data
  GROUP BY 1
)
-- 기존의 user_r에 (1)과 (2)를 통합
SELECT
  pc.CustomerID,
  pc.purchase_cnt,
  ic.item_cnt,
  ur.recency
FROM purchase_cnt AS pc
JOIN item_cnt AS ic
  ON pc.CustomerID = ic.CustomerID
JOIN first-metric-410915.modulabs_project.user_r AS ur
  ON pc.CustomerID = ur.CustomerID;
```

![Screen%20Shot%202024-01-15%20at%202.42.30%20PM.png](Screen%20Shot%202024-01-15%20at%202.42.30%20PM.png)

![Screen%20Shot%202024-01-15%20at%202.50.41%20PM.png](Screen%20Shot%202024-01-15%20at%202.50.41%20PM.png)
🔼 데이터 user_rf

### Monetary (1) 고객별 총 지출액 계산


```python
SELECT
  CustomerID,
  ROUND(SUM(Quantity * UnitPrice)) AS user_total
FROM first-metric-410915.modulabs_project.data
GROUP BY 1;
```

![Screen%20Shot%202024-01-15%20at%203.06.00%20PM.png](Screen%20Shot%202024-01-15%20at%203.06.00%20PM.png)

### Monetary (2) 고객별 평균 거래 금액 계산


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_rfm AS   
SELECT
  rf.CustomerID AS CustomerID,
  rf.purchase_cnt,
  rf.item_cnt,
  rf.recency,
  ut.user_total,
  ROUND(ut.user_total / rf.purchase_cnt) AS user_average
FROM first-metric-410915.modulabs_project.user_rf rf
LEFT JOIN (
  -- 고객 별 총 지출액
  SELECT
    CustomerID,
    ROUND(SUM(Quantity * UnitPrice)) AS user_total
  FROM first-metric-410915.modulabs_project.data
  GROUP BY 1
) ut
ON rf.CustomerID = ut.CustomerID;

SELECT *
FROM first-metric-410915.modulabs_project.user_rfm;
```

![Screen%20Shot%202024-01-15%20at%203.11.46%20PM.png](Screen%20Shot%202024-01-15%20at%203.11.46%20PM.png)
🔼 데이터 user_rfm