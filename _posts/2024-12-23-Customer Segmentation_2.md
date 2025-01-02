---
title: Customer Segmentation_2
date: 2024-12-23
categories: [Data Analysis, Customer Segmentation]
tags: [customer segmentation]     
---

## 3. 추가 Feature 추출: 클러스터링 알고리즘

### 3-1. 구매하는 제품의 다양성


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_data AS  
WITH unique_products AS (
  SELECT
    CustomerID,
    COUNT(DISTINCT StockCode) AS unique_products
  FROM first-metric-410915.modulabs_project.data
  GROUP BY CustomerID
)
SELECT ur.*, up.* EXCEPT (CustomerID)
FROM first-metric-410915.modulabs_project.user_rfm AS ur
JOIN unique_products AS up
ON ur.CustomerID = up.CustomerID;
```

![Screen%20Shot%202024-01-15%20at%203.39.46%20PM.png](Screen%20Shot%202024-01-15%20at%203.39.46%20PM.png)
🔼 데이터 user_data

### 3-2. 평균 구매 주기


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_data AS
WITH purchase_intervals AS (
  -- (2) 고객 별 구매와 구매 사이의 평균 소요 일수
  SELECT
    CustomerID,
    CASE WHEN ROUND(AVG(interval_), 2) IS NULL THEN 0 ELSE ROUND(AVG(interval_), 2) END AS average_interval
  FROM (
    -- (1) 구매와 구매 사이에 소요된 일수
    SELECT
      CustomerID,
      DATE_DIFF(InvoiceDate, LAG(InvoiceDate) OVER (PARTITION BY CustomerID ORDER BY InvoiceDate), DAY) AS interval_
    FROM
      first-metric-410915.modulabs_project.data
    WHERE CustomerID IS NOT NULL
  )
  GROUP BY CustomerID
)

SELECT u.*, pi.* EXCEPT (CustomerID)
FROM first-metric-410915.modulabs_project.user_data AS u
LEFT JOIN purchase_intervals AS pi
ON u.CustomerID = pi.CustomerID;


select * from first-metric-410915.modulabs_project.user_data
```

![Screen%20Shot%202024-01-15%20at%203.51.10%20PM.png](Screen%20Shot%202024-01-15%20at%203.51.10%20PM.png)
🔼 업데이트된 데이터 user_data

### 3-3. 구매 취소 경향성


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_data AS

WITH TransactionInfo AS (
  SELECT
    CustomerID,
    COUNT(InvoiceNo) AS total_transactions,
    COUNTIF(InvoiceNo LIKE 'C%') AS cancel_frequency
  FROM first-metric-410915.modulabs_project.data
  GROUP BY CustomerID
)
SELECT u.*, t.* EXCEPT(CustomerID), ROUND(t.cancel_frequency / t.total_transactions * 100, 2) AS cancel_rate
FROM `first-metric-410915.modulabs_project.user_data` AS u
LEFT JOIN TransactionInfo AS t
ON u.CustomerID = t.CustomerID;
```

![Screen%20Shot%202024-01-15%20at%205.02.57%20PM.png](Screen%20Shot%202024-01-15%20at%205.02.57%20PM.png)
🔼 업데이트된 최종 데이터 user_data

### 결과 데이터 불러오기


```python
import pandas as pd 
user_data = pd.read_csv('aiffel/customer_segmentation/user_DAta.csv')
user_data.head()
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
      <th>CustomerID</th>
      <th>purchase_cnt</th>
      <th>item_cnt</th>
      <th>recency</th>
      <th>user_total</th>
      <th>user_average</th>
      <th>unique_products</th>
      <th>average_interval</th>
      <th>total_transactions</th>
      <th>cancel_frequency</th>
      <th>cancel_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16323</td>
      <td>1</td>
      <td>50</td>
      <td>196</td>
      <td>208.0</td>
      <td>208.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16093</td>
      <td>1</td>
      <td>20</td>
      <td>106</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16738</td>
      <td>1</td>
      <td>3</td>
      <td>297</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15524</td>
      <td>1</td>
      <td>4</td>
      <td>24</td>
      <td>440.0</td>
      <td>440.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17986</td>
      <td>1</td>
      <td>10</td>
      <td>56</td>
      <td>21.0</td>
      <td>21.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 회고 및 인사이트

1.  새롭게 배운 점
- 이전 노드에서 배운 RFM 적용 및 데이터 전처리. 생각보다 데이터 전처리 과정이 까다롭고 하나하나 생각해봐야 할 점이 많다. 단순히 Null 값과 중복값을 제외하는 정도라고 생각했던 것 같은데 각 데이터와 그 값이 어떤 의미인지 따져봐야 한다. 그 양이 방대한 경우에는 일부만 가지고 경향성을 판단할 수 없다. 
- 경향성 읽기
    - 예를 들어, United Kingdom에서 취소율이 높다라고 판단했는데(100 건중 2건 제외 모두) 알고보니 구매율 자체가 해당 국가에서 높았다.

2. 가장 시간이 오래 소요된 구간, 새로운 시도
- 임시테이블 만든 후 기존 테이블과 합치는 작업(Create, Replace)
    - WITH으로 만든 테이블에서 계산을 잘못하여 다시 시행하려고 하는데 이미 생성된 후라서, 컬럼명 중복에러가 났다.
    - 당연히 바로 덮어씌우기가 될 줄 알았는데 한 번 계산을 잘못하면 그냥 덮어쓰기로 단순히 처리가 되는 게 아니었다.

![Screen%20Shot%202024-01-15%20at%205.37.40%20PM.png](Screen%20Shot%202024-01-15%20at%205.37.40%20PM.png)
🔼 columns with duplicate Error


```python
CREATE OR REPLACE TABLE first-metric-410915.modulabs_project.user_data AS

WITH TransactionInfo AS (
  SELECT
    CustomerID,
    COUNT(DISTINCT InvoiceNo) AS total_transactions,
    COUNTIF(InvoiceNo LIKE 'C%') AS cancel_frequency
  FROM first-metric-410915.modulabs_project.data
  GROUP BY CustomerID
)
SELECT u.*, t.* EXCEPT(CustomerID), ROUND(t.cancel_frequency / t.total_transactions * 100, 2) AS cancel_rate
FROM `first-metric-410915.modulabs_project.user_data` AS u
LEFT JOIN TransactionInfo AS t
ON u.CustomerID = t.CustomerID;
```

🔼 CREATE OR REPLACE TABLE 코드 일부


```python
-- total_transactions 컬럼 삭제
ALTER TABLE first-metric-410915.modulabs_project.user_data
DROP COLUMN total_transactions;

-- total_transactions 컬럼 추가
ALTER TABLE first-metric-410915.modulabs_project.user_data
ADD COLUMN total_transactions INT;

-- total_transactions 업데이트
UPDATE first-metric-410915.modulabs_project.user_data AS u
SET 
  total_transactions = (
    SELECT COUNT(DISTINCT InvoiceNo)
    FROM first-metric-410915.modulabs_project.data AS d
    WHERE d.CustomerID = u.CustomerID
  ),
  cancel_rate = (
    CASE 
      WHEN u.total_transactions > 0 THEN 
        ROUND(COUNTIF(InvoiceNo LIKE 'C%') / u.total_transactions * 100, 2)
      ELSE
        0
    END
  )
WHERE EXISTS (
    SELECT 1
    FROM first-metric-410915.modulabs_project.data AS d
    WHERE d.CustomerID = u.CustomerID
);
```

    - 방법1. 테이블 새로 생성
      - 다 지우고 새로 만드는 방법. user_data 테이블을 생성하는 작업은 어렵지 않아 간단하다.
      - 하지만 이전 작업량이 많아 데이터가 축적된 경우에는 사용 불가.
    - 방법2. alter, update로 기존 테이블에서 값을 참조해서 합치는 작업
      - 본 프로젝트에서는 방법 1로도 가능하나 위와 같은 상황을 대비하여 방법2 적용해서 연습해보았다.

![Screen%20Shot%202024-01-15%20at%205.56.40%20PM.png](Screen%20Shot%202024-01-15%20at%205.56.40%20PM.png)
🔼 취소율 경향성 1

- 먼저 취소율에 대한 데이터를 봤을 때, 고객을 분류해야 할 필요성을 느꼈다.
- 원래 취소를 잘 하지 않는 고객이 취소한 물품에 대한 개선이 필요하다. 공통된 교집합에 있는 제품이 있다면 제품 자체에 문제가 있을 수도 있다.
    - 1% 미만 취소율을 가진 고객(214명)을 확인하고 나서 세 그룹으로 나누어봐야겠다고 생각했다.
    - 상, 중, 하 세 그룹이고 취소율이 낮은 고객은 개인적으로 궁금해서 더 세분화 해봤다.

![Screen%20Shot%202024-01-15%20at%205.57.14%20PM.png](Screen%20Shot%202024-01-15%20at%205.57.14%20PM.png)
🔼 취소율 경향성 2


```python
SELECT
  cancel_rate_group,
  COUNT(DISTINCT CustomerID) AS customer_count
FROM (
  SELECT
    CustomerID,
    cancel_rate,
    CASE
      WHEN cancel_rate > 0 AND cancel_rate <= 5 THEN 'Low-Low(0~5%)'
      WHEN cancel_rate > 5 AND cancel_rate < 10 THEN 'Low(5~10%)'
      WHEN cancel_rate >= 10 AND cancel_rate < 20 THEN 'Medium(10~20%)'
      WHEN cancel_rate >= 20 THEN 'High(20% 이상 취소)'
      ELSE '0 CANCEL'
    END AS cancel_rate_group
  FROM first-metric-410915.modulabs_project.user_data
) AS grouped_data
GROUP BY cancel_rate_group
ORDER BY customer_count DESC;
```

![image.png](image.png)
🔼 취소율 분류하기

- 먼저 Low 고객은 자연 취소율이라고 생각한다. 
- 20%이상 취소한 고객군은 관리가 필요하다.
    - 소량 취소 고객(1~3건) 
    - 다양한 상품 구매(15종류↑)했으나 많이 취소(50%↑)한 그룹 : 15 종류 이상 구매하였는데 100% 취소한 고객과 같이 10건 이상 구매하였는데 절반 이상 취소하였다면 유의고객으로 따로 관리할 필요가 있을 것 같다. 이들의 구매 동기와 취소 이유를 자세히 살펴보고, 만약 문제가 해결될 수 있다면 이들과의 소통과 서비스 개선이 필요할 것이다. 이탈 위험이 있다고 판단된다.
    - 같은 종류만 구매하는데 취소 했던 고객도 있다. 잘못 주문했거나 해당 거래 때 어떤 문제가 있었는지 살펴볼 필요가 있다. 해당 고객은 가장 큰 매출을 낸 고객이라 vip로 따로 관리가 필요할 것 같다.
- 특정 상품만 구매, 취소를 반복한 고객
    - 특정 상품(1~3건)만 구매하고 취소(50%↑)하는 고객은 해당 상품에 대한 문제점이나 고객의 실수일 가능성이 있다. 해당 고객들에게 추가 정보 제공이나 개선된 상품 관련 서비스를 제공하여 이탈을 방지할 수 있을 것이다.

5. RFMScore 인사이트


```python
-- RFM 테이블 생성
WITH RFM AS (
  SELECT customerid,
          recency, 
          user_average as frequency, 
          user_total as monetary
  FROM first-metric-410915.modulabs_project.user_data
)
, RFMScores AS (
  SELECT
    customerid,
    recency,
    frequency,
    monetary,
    NTILE(5) OVER (ORDER BY recency DESC) as RecencyScore,
    NTILE(5) OVER (ORDER BY frequency) as FrequencyScore,
    NTILE(5) OVER (ORDER BY monetary) as MonetaryScore
  FROM RFM
)
, RFMGroup AS (
  SELECT
    customerid,
    RecencyScore,
    FrequencyScore,
    MonetaryScore,
    RecencyScore + FrequencyScore + MonetaryScore as RFMScore
  FROM RFMScores
)
, GroupedRFM AS (
  SELECT
    customerid,
    RecencyScore,
    CASE
      WHEN RFMScore <= 5 THEN 'Low'
      WHEN RFMScore <= 10 THEN 'Medium'
      WHEN RFMScore <= 15 THEN 'High'
      ELSE 'Unknown'
    END AS RFMGroup
  FROM RFMGroup
)
SELECT
  RFMGroup,
  COUNT(customerid) as CustomerCount
FROM GroupedRFM
GROUP BY RFMGroup
ORDER BY 1;

```

![Screen%20Shot%202024-01-15%20at%206.54.13%20PM.png](Screen%20Shot%202024-01-15%20at%206.54.13%20PM.png)

![Screen%20Shot%202024-01-15%20at%206.55.01%20PM.png](Screen%20Shot%202024-01-15%20at%206.55.01%20PM.png)
🔼 RFMScores

1. 0~5점 : Low
- Recency, Frequency, Monetary 모두 낮은 수준의 활동을 보이는 고객
- 오랜 기간 동안 거래가 없거나 적은 빈도와 낮은 지출
- 리액션을 유도하기 위한 프로모션 및 할인 제공 등의 전략이 필요
2. 6~10점 : Medium
- 중간 정도의 활동을 보이는 고객
- 일정 기간 동안 일정한 빈도와 지출을 보임
- 추가적인 리워드 제공이나 새로운 제품 소개를 통한 적극적인 마케팅 전략이 가능
3. 11~15점 : High
- 최근에 활발한 거래를 보이며 높은 빈도와 지출을 가진 고객
- 프리미엄 서비스, 독점 혜택 등을 제공하여 충성도를 높이고 지속적인 관계 구축 필요
- 꾸준한 고객 서비스와 특별한 혜택을 제공하여 만족도 유지   

