# Decision Tree-basic

使用scikit-learn中的內建資料集鳶尾花，建構基本的決策分類樹


## Import

```python=
import math
import numpy as np
from sklearn import datasets
```

## Functions

### 計算亂度

計算當前資料中的混亂程度

```python=
def entropy(p1,n1): #算亂度
  if (p1==0 and n1==0):
    return 1
  elif (p1==0 or n1==0):
    return 0
  else:
    pp = p1/(p1+n1)
    np = n1/(p1+n1)
    return -pp*math.log2(pp)-np*math.log2(np)
```

### Information Gain

計算假如用這種方法切割的話，能夠對分類有多大的幫助

```python=
def IG(p1,n1,p2,n2):  #information gain 分類方法對降低亂度的幫助大小
  num1 = p1+n1
  num2 = p2+n2
  num = num1+num2
  return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)
```

### Train

實際開始訓練的程式碼

```python=
def ID3DTtrain(feature,target):
  node = dict()
  node['data'] = range(len(target)) #全部資料放在根節點，紀錄有多少資料落在這裡
  tree = [] #要output的樹
  tree.append(node) #把根結點塞進去
  
  t = 0 #一個一個決定結點的內容是否要分枝成兩個
  while (t<len(tree)):  #表示t還沒看到最後一個節點
    index = tree[t]['data'] #從0號根結點開始
    
    if(sum(target[index])==0):  #target裡面都是0、1、2，如果全為0就代表分到一群都是0的
      tree[t]['leaf'] = 1
      tree[t]['decision'] = 0 
    elif(sum(target[index])==len(index)):
      tree[t]['leaf'] = 1
      tree[t]['decision'] = 1
    else:
      bestIG = 0
      for i in range(feature.shape[1]):
        pool = list(set(feature[index,i]))  #index=第t個node中所有資料的編號，把第i個特徵拿出來比較
        pool.sort()
        
        for j in range(len(pool)-1):  #找兩兩的數，中間下去分切(1,3,5、切2,4)，間隔有len(pool)-1個
          thres = (pool[j]+pool[j+1])/2
          G1 = []
          G2 = []
          for k in index:
            if(feature[k][i]<thres):
              G1.append(k)
            else:
              G2.append(k)
          p1 = sum(target[G1]==1)
          p2 = sum(target[G1]==0)
          n1 = sum(target[G2]==1)
          n2 = sum(target[G2]==0)
          thisIG = IG(p1,p2,n1,n2)
          if(thisIG>bestIG):
            bestIG = thisIG
            bestG1 = G1
            bestG2 = G2
            bestthres = thres
            bestf = i
      if(bestIG>0): #還能再切分
        tree[t]['leaf'] = 0
        tree[t]['selectf'] = bestf
        tree[t]['threshold'] = bestthres
        tree[t]['child'] = [len(tree),len(tree)+1]
        node = dict()
        node['data'] = bestG1 #假如現在t=0 切完的右半邊會放在這裡(t=1)
        tree.append(node)
        node = dict()
        node['data'] = bestG2 #左半邊放這裡
        tree.append(node)
        
      else:
        tree[t]['leaf'] = 1
        if(sum(target[index]==1)>sum(target[index]==0)):  #target=1的數量和=2的數量比較
          tree[t]['decision'] = 1
        else:
          tree[t]['decision'] = 0
    t += 1
  return tree  
```

### Test

用來測試模型的表現

```python=
def ID3DTtest(tree,feature1):
  now = 0
  while(tree[now]['leaf']==0):
    bestf = tree[now]['selectf']
    thres = tree[now]['threshold']
    if(feature1[bestf]<thres):
      now = tree[now]['child'][0]
    else:
      now = tree[now]['child'][1]
  return tree[now]['decision']
```

### Main function

主程式的呼叫，一次使用100朵花來進行分類訓練，因為這是二元的決策分類樹。

```python=
data = datasets.load_iris()
feature = data['data']  # 每一朵的特色
target = data['target'] # 花的種類(有三種各50朵)

# 前100朵
T = ID3DTtrain(feature[0:100,:],target[0:100])
# 另外100朵
T_next = ID3DTtrain(feature[50:150,:],target[50:150]-1)
```

### 模擬測試 1

這邊使用train的資料對模型測試，很明顯因為模型是用train的資料建立，所以可以預見得到的準確率會是100%

```python=
miss = 0  #計算判斷錯誤的次數
for i in range(50,150):
  flag = ID3DTtest(T_next,feature[i,:])
  if(i<100 and flag==0):
    print(i,': ',flag,'correct')
  elif(i<100 and flag==1):
    print(i,': ',flag,'wrong')
    miss+=1

  if(i>=100 and flag==1):
    print(i,': ',flag,'correct')
  elif(i>=100 and flag==0):
    print(i,': ',flag,'wrong')
    miss+=1
print('Accuration',(1-(miss/100))*100,'%')
```

### 模擬測試 2

現在改用每個分類的前 30 筆建樹，後 20 筆測試，來看結果維何。

```python=
x = feature[50:80,:]
y = feature[100:130,:]
p = target[50:80]
q = target[100:130]
feature_new = np.concatenate((x,y),axis=0)  #把兩類的前30筆，合併起來
target_new = np.concatenate((p,q),axis=0)
miss = 0  #計算判斷錯誤的次數
T_type2 = ID3DTtrain(feature_new[0:60,:],target_new[0:60]-1)
for i in range(80,100):
  flag = ID3DTtest(T_type2,feature[i,:])
  if(flag==0):
    print(i,': ',flag,'correct')
  elif(flag==1):
    print(i,': ',flag,'wrong')
    miss+=1
print("-----------------")
for i in range(130,150):
  flag = ID3DTtest(T_type2,feature[i,:])
  if(flag==0):
    print(i,': ',flag,'wrong')
    miss+=1
  elif(flag==1):
    print(i,': ',flag,'correct')
print('Accuration',(1-(miss/40))*100,'%')
```

### 執行結果

可以看到準確率大約為95%，雖然是用內建好的資料庫做為訓練，不過仍然對於訓練模型很有幫助。

![](https://hackmd.io/_uploads/H10mSjzF2.png)


