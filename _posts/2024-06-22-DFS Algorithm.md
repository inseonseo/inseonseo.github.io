---
title: DFS Algorithm practice
date: 2024-06-22
categories: [Algorithm, DFS]
tags: [모글모글, 아이펠, 개발자 글쓰기, dfs]     # TAG names should always be lowercase
---


## DFS algorithm
### Depth-First Search

- 하나의 순환 알고리즘으로 백트래킹에 사용하는 대표적인 탐색 알고리즘
- 루트 노드(혹은 다른 임의의 노드)에서 시작하여 다음 분기로 넘어가기 전 해당 분기를 완벽하게 탐색하는 방식
- 주로 재귀함수 또는 Stack 으로 구현
    ex) 미로찾기 : 최대한 한 방향으로 갈 수 있을 때까지 가다가 더 이상 갈 수 없게 되면 다시 가장 가까운 갈림길로 돌아와서 그 갈림길부터 다시 다른 방향으로 탐색을 진행하는 것

![image](https://github.com/inseonseo/inseonseo.github.io/assets/50574738/4133548a-f46e-4e15-bb53-27ae2e6c5f0b)

### 시간 복잡도와 장단점
- 인접 행렬에서 시간 복잡도: O(V^2)
    - 행렬의 모든 칸을 살펴본다
- 인접 리스트에서 시간 복잡도: O(V+E)
    - 모든 노드를 살펴보면서 각 노드를 볼 때마다 갈 수 있는 모든 간선 살펴본다
- V: 정점(Vertex, 노드) 개수, E: 간선(Edge) 개수 \
    ⇢ 간선 개수가 매우 적고 노드 개수가 훨씬 많다면 ~ O(V) \
    ⇢ 간선 개수가 매우 많다면 ~ O(E)

- **장점**
    - 현재 경로상 노드들만 기억하면 되므로 저장공간이 비교적 적게 든다
    - 깊이 우선 탐색(DFS)이 너비 우선 탐색(BFS)보다 상대적 간단 ⇢ DFS, BFS 둘다 사용해도 되는 문제
        e.g.) 그래프 모든 정점을 방문하는 문제
    - 목표노드가 깊은 단계에 있을 경우 해를 빨리 구할 수 있음 
- **단점**
    - 해가 없는 경로에 깊이 빠질 가능성
        ⇢ 미리 지정한 임의의 깊이까지만 탐색하고 목표노드를 발견하지 못하면 다음 경로를 따라 탐색하는 방법 고려해볼 수 있음
    - 얻은 해가 최단 경로가 된다는 보장 없음
        - 목표에 이르는 경로가 다수인 문제에서 깊이우선 탐색은 해에 다다르면 탐색을 끝내버리므로, 이 때 얻어진 해는 최적이 아닐 수 있음
        - 처음 발견된 경로가 최단거리인지는 보장되지 않기 때문에 모든 경로를 탐색하며 그 중 가장 짧은 경로를 취해야 한다




### DFS 활용 적합한 문제
- 경로 특징을 저장해둬야 하는 문제(경로 상 노드를 기억)
     e.g.) 지나간 경로를 또 지나가면 안되는 문제 등 각 경로마다 특징을 저장해둬야 할 때
- 검색 대상 그래프 큰 문제
    ⇢ 목표노드가 깊은 단계에 있을 경우 해를 빨리 구할 수 있으므로

## Softeer. 문제(Lv.3)

![image](https://github.com/inseonseo/inseonseo.github.io/assets/50574738/86369d68-b7e0-4c55-8ee3-c4c233c766d6)
![image](https://github.com/inseonseo/inseonseo.github.io/assets/50574738/9fd40912-ea77-4066-9f1c-427145997fda)


```python
import sys
import itertools

input = sys.stdin.read
data = input().splitlines()
n, m = map(int, data[0].split())
arr =[]
```


```python
# 열매 수확량 데이터
for i in range(1, n+1):
    row = list(map(int, data[i].split()))
    arr.append(row)

# 친구들 초기 위치
initial_positions = []
for i in range(n+1, n+m+1):
    a, b = map(int, data[i].split())
    initial_positions.append((a-1, b-1))
    
# 상하좌우 이동 방향 벡터
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 각 친구 가능한 경로 DFS 사용하여 찾기
def dfs(path, x, y, routes):
    if len(path) == 4:
        routes.append(path[:])
        return
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in path:
            path.append((nx, ny))
            dfs(path, nx, ny, routes)
            path.pop()
            
# 각 친구 가능한 모든 경로
friends_routes = []
for start in initial_positions:
    routes = []
    dfs([start], start[0], start[1], routes)
    friends_routes.append(routes)
    
# 가능한 모든 경로 조합하여 최대 수확량 계산
max_total_yield = 0

for comb in itertools.product(*friends_routes):
    used = set()
    current_yield = 0
    valid = True
    
    for route in comb:
        for pos in route:
            if pos in used:
                valid = False
                break
            used.add(pos)
            current_yield += arr[pos[0]][pos[1]]
        if not valid:
            break
    if valid:
        max_total_yield = max(max_total_yield, current_yield)
        
print(max_total_yield)
```

### 문제풀이

- 친구들 간 순서를 고려하지 않아 특정케이스에서 오답
- 최대 수확량을 내기 위해 친구들 간 수확 순서까지 고려하여 문제 풀이
