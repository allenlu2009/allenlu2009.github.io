---
title: 增進工程師效率 Python DataFrame - CSV & Plot
date: 2018-12-21 23:29:08
categories:
- Language
tags: [python, pandas, DataFrame]
---

*Download the code: https://github.com/allenlu2009/colab/blob/master/dataframe_demo.ipynb*

## Python DataFrame
### Create DataFrame
* Direct input

* Use dict: Method 1: 一筆一筆加入。



```python
import pandas as pd
dict1 = {'Name': 'Allen' , 'Sex': 'male', 'Age': 33}
dict2 = {'Name': 'Alice' , 'Sex': 'female', 'Age': 22}
dict3 = {'Name': 'Bob' , 'Sex': 'male', 'Age': 11}
data = [dict1, dict2, dict3]
df = pd.DataFrame(data)
df
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
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allen</td>
      <td>male</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice</td>
      <td>female</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bob</td>
      <td>male</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



Method 2: 一次加入所有資料。


```python
name = ['Allen', 'Alice', 'Bob']
sex = ['male', 'female', 'male']
age = [33, 22, 11]
all_dict = {
    "Name": name,
    "Sex": sex,
    "Age": age
}
df = pd.DataFrame(all_dict)
df[['Name', 'Age']]
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
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allen</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bob</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Dataframe 的屬性
* ndim: 2 for 2D dataframe; **axis 0 => row; axis 1 => column**
* shape:  (row no. x column no.) (not including number index)
* dtypes: (object or int) of **each column**


```python
df.ndim
```




    2




```python
df.shape
```




    (3, 3)




```python
df.dtypes
```




    Name    object
    Sex     object
    Age      int64
    dtype: object




```python
df.columns
```




    Index(['Name', 'Sex', 'Age'], dtype='object')




```python
df.index
```




    RangeIndex(start=0, stop=3, step=1)



## Read CSV 
Donwload a test csv file from https://people.sc.fsu.edu/~jburkardt/data/csv/csv.html  Pick the biostats.csv

For 2, Before read csv, reference Medium article to import google drive

* Read csv 使用 read_csv function.  **但是要加上 skipinitialspace to strip the leading space!!**
* Two ways to read_csv: (1) load csv file directly; (2) load from url



```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
#!ls 'drive/My Drive/Colab Notebooks/'
df = pd.read_csv('drive/My Drive/Colab Notebooks/biostats.csv', skipinitialspace=True)
df
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive





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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
url = "https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv"
df = pd.read_csv(url, skipinitialspace=True)
df
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.columns); print(df.index)
```

    Index(['Name', 'Sex', 'Age', 'Height (in)', 'Weight (lbs)'], dtype='object')
    RangeIndex(start=0, stop=18, step=1)



```python
df.ndim
```




    2




```python
df.shape
```




    (18, 5)




```python
df.dtypes
```




    Name            object
    Sex             object
    Age              int64
    Height (in)      int64
    Weight (lbs)     int64
    dtype: object



## Basic Viewing Command


```python
df.head(3)
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(3)
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (18, 5)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18 entries, 0 to 17
    Data columns (total 5 columns):
    Name            18 non-null object
    Sex             18 non-null object
    Age             18 non-null int64
    Height (in)     18 non-null int64
    Weight (lbs)    18 non-null int64
    dtypes: int64(3), object(2)
    memory usage: 848.0+ bytes



```python
df[7:10]
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Name'][7:10]
```




    7    Hank
    8    Ivan
    9    Jake
    Name: Name, dtype: object




```python
df[['Name', 'Age', 'Sex']][7:10]
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
      <th>Name</th>
      <th>Age</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>30</td>
      <td>M</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>53</td>
      <td>M</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>32</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[7:10, ['Name', 'Age', 'Sex']] # compare with loc call
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
      <th>Name</th>
      <th>Age</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>30</td>
      <td>M</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>53</td>
      <td>M</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>32</td>
      <td>M</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>47</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.count()
```




    Name            18
    Sex             18
    Age             18
    Height (in)     18
    Weight (lbs)    18
    dtype: int64



## Basic Index Operation
Index (索引) is a very useful key for DataFrame.  The default index is the row number starting from 0 to N-1, where N is the number of data.
除了用 row number 做為 index, 一般也會使用 unique feature 例如 name, id, or phone number 做為 index.  

### 把 column 變成 index

* Method 1: 直接在 read_csv 指定 index_col.  可以看到 index number 消失，而被 Name column 取代。



```python
df = pd.read_csv('drive/My Drive/Colab Notebooks/biostats.csv', skipinitialspace=True, index_col='Name')
df
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>



* df.index shows the element in index column



```python
df.index
```




    Index(['Alex', 'Bert', 'Carl', 'Dave', 'Elly', 'Fran', 'Gwen', 'Hank', 'Ivan',
           'Jake', 'Kate', 'Luke', 'Myra', 'Neil', 'Omar', 'Page', 'Quin', 'Ruth'],
          dtype='object', name='Name')



* 使用 reset_index 又會回到 index number.


```python
df.reset_index()
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>



再看一次 df 並沒有改變。很多 DataFrame 的 function 都是保留原始的 df, create a new object, 也就是 inplace = False.   如果要取代原來的 df, **必須 inplace = True**!


```python
df
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.reset_index(inplace=True)
df
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>



如果再 reset_index(）一次，會是什麼結果？此處用 default inplace=False.
多了一個 index column


```python
df.reset_index()
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
      <th>index</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>



* Method 2: 使用 set_index()


```python
df.set_index('Name', inplace=True)
df
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>



### loc[]
使用 loc[] 配合 index label 取出資料非常方便。\
如果是 number index, 可以用 df[0], df[3], etc.\
但如果是其他 column index, e.g. **Name, df[2] 或是 df["Hank"] are wrong!, 必須用 df.loc['Hank']**\
或是 df.loc[ ['Hank', 'Ruth', 'Page'] ]


```python
df.loc['Hank']
```




    Sex               M
    Age              30
    Height (in)      71
    Weight (lbs)    158
    Name: Hank, dtype: object




```python
df.loc[:, ['Sex', 'Age']]
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
      <th>Sex</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[ ['Hank', 'Ruth', 'Page'] ]
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
  </tbody>
</table>
</div>



loc[] 可以用 row, column 得到對應的 element, 似乎是奇怪的用法


```python
df.loc['Hank', 'Age']
```




    30



### iloc[]
使用 column index 仍然可以用 iloc[] 配合 index number 取出資料。



```python
df.iloc[0]
```




    Sex               M
    Age              41
    Height (in)      74
    Weight (lbs)    170
    Name: Alex, dtype: object




```python
df.iloc[1:10]
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[[1, 4, 6]]
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>



## 排序
包含兩種排序
* sort_index()
* sort_value()


```python
df.sort_index()
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by = 'Age')
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
  </tbody>
</table>
</div>



## Rename and Drop Column(s) and Index(s)



```python
df.rename(columns={"Height (in)": "Height", "Weight (lbs)": "Weight"}, inplace=True)
df
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rename(index={"Alex": "Allen", "Bert": "Bob"}, inplace=True)
df
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Allen</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bob</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(labels=['Sex', 'Weight'], axis="columns") # axis=1 eq axis="columns"
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
      <th>Age</th>
      <th>Height</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Allen</th>
      <td>41</td>
      <td>74</td>
    </tr>
    <tr>
      <th>Bob</th>
      <td>42</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>32</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>39</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>30</td>
      <td>66</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>33</td>
      <td>66</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>26</td>
      <td>64</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>30</td>
      <td>71</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>53</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>32</td>
      <td>69</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>47</td>
      <td>69</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>34</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>23</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>36</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>38</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>31</td>
      <td>67</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>29</td>
      <td>71</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>28</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(labels=['Allen', 'Ruth'], axis="index") # axis=0 eq axis="index"
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>



## 進階技巧
### Multiple Index (多重索引)
**這是非常有用的技巧，使用 set_index with keys**


```python
df = pd.read_csv('drive/My Drive/Colab Notebooks/biostats.csv', skipinitialspace=True)
df  # show the original dataframe
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
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bert</td>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Carl</td>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dave</td>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elly</td>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fran</td>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gwen</td>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hank</td>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ivan</td>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jake</td>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kate</td>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Luke</td>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Myra</td>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Neil</td>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Omar</td>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Page</td>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Quin</td>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ruth</td>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index(keys = ['Name', 'Sex'])  
# Notice "Name" "Sex" columns header is lower than the rest
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
      <th></th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <th>M</th>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <th>M</th>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <th>M</th>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <th>M</th>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <th>F</th>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <th>F</th>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <th>F</th>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <th>M</th>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <th>M</th>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <th>M</th>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <th>F</th>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <th>M</th>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <th>F</th>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <th>M</th>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <th>M</th>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <th>F</th>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <th>M</th>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <th>F</th>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index(keys = ['Sex', 'Name'], inplace=True)
df
# Note that key sequence matters; and same index values group
# Note that inplace=True replaces the original df 
# This is useful to display sorted group
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
      <th></th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">M</th>
      <th>Alex</th>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">F</th>
      <th>Elly</th>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">M</th>
      <th>Hank</th>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>F</th>
      <th>Kate</th>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>M</th>
      <th>Luke</th>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>F</th>
      <th>Myra</th>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">M</th>
      <th>Neil</th>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>F</th>
      <th>Page</th>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>M</th>
      <th>Quin</th>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>F</th>
      <th>Ruth</th>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    MultiIndex([('M', 'Alex'),
                ('M', 'Bert'),
                ('M', 'Carl'),
                ('M', 'Dave'),
                ('F', 'Elly'),
                ('F', 'Fran'),
                ('F', 'Gwen'),
                ('M', 'Hank'),
                ('M', 'Ivan'),
                ('M', 'Jake'),
                ('F', 'Kate'),
                ('M', 'Luke'),
                ('F', 'Myra'),
                ('M', 'Neil'),
                ('M', 'Omar'),
                ('F', 'Page'),
                ('M', 'Quin'),
                ('F', 'Ruth')],
               names=['Sex', 'Name'])




```python
df.index.names
```




    FrozenList(['Sex', 'Name'])




```python
type(df.index)  # MultiIndex
```




    pandas.core.indexes.multi.MultiIndex




```python
df.sort_index(inplace=True)
df
# sorting is based on "Sex", and then "Name"
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
      <th></th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">F</th>
      <th>Elly</th>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
    <tr>
      <th rowspan="11" valign="top">M</th>
      <th>Alex</th>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>



### Groupby Command
Groupby 是 SQL 的語法。根據某一項資料做分組方便查找。\
The SQL GROUP BY Statement
The GROUP BY statement is often used with aggregate functions (COUNT, MAX, MIN, SUM, AVG) to group the result-set by one or more columns.



```python
df = pd.read_csv('drive/My Drive/Colab Notebooks/biostats.csv', index_col="Name", skipinitialspace=True)
df  # show the dataframe with "Name" index column
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Elly</th>
      <td>F</td>
      <td>30</td>
      <td>66</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Fran</th>
      <td>F</td>
      <td>33</td>
      <td>66</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Gwen</th>
      <td>F</td>
      <td>26</td>
      <td>64</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Kate</th>
      <td>F</td>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Myra</th>
      <td>F</td>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Page</th>
      <td>F</td>
      <td>31</td>
      <td>67</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Ruth</th>
      <td>F</td>
      <td>28</td>
      <td>65</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
</div>




```python
grpBySex = df.groupby('Sex')  # output is a DataFrameGroupBy object
type(grpBySex)
```




    pandas.core.groupby.generic.DataFrameGroupBy




```python
grpBySex.groups  
# output is a dict, use get_group() obtains each sub-group
```




    {'F': Index(['Elly', 'Fran', 'Gwen', 'Kate', 'Myra', 'Page', 'Ruth'], dtype='object', name='Name'),
     'M': Index(['Alex', 'Bert', 'Carl', 'Dave', 'Hank', 'Ivan', 'Jake', 'Luke', 'Neil',
            'Omar', 'Quin'],
           dtype='object', name='Name')}




```python
grpBySex.size()  # size() shows the counts of each group
```




    Sex
    F     7
    M    11
    dtype: int64




```python
grpBySex.get_group('M')  # get_group() output a DataFrame object
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
      <th>Sex</th>
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alex</th>
      <td>M</td>
      <td>41</td>
      <td>74</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Bert</th>
      <td>M</td>
      <td>42</td>
      <td>68</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Carl</th>
      <td>M</td>
      <td>32</td>
      <td>70</td>
      <td>155</td>
    </tr>
    <tr>
      <th>Dave</th>
      <td>M</td>
      <td>39</td>
      <td>72</td>
      <td>167</td>
    </tr>
    <tr>
      <th>Hank</th>
      <td>M</td>
      <td>30</td>
      <td>71</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Ivan</th>
      <td>M</td>
      <td>53</td>
      <td>72</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>M</td>
      <td>32</td>
      <td>69</td>
      <td>143</td>
    </tr>
    <tr>
      <th>Luke</th>
      <td>M</td>
      <td>34</td>
      <td>72</td>
      <td>163</td>
    </tr>
    <tr>
      <th>Neil</th>
      <td>M</td>
      <td>36</td>
      <td>75</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Omar</th>
      <td>M</td>
      <td>38</td>
      <td>70</td>
      <td>145</td>
    </tr>
    <tr>
      <th>Quin</th>
      <td>M</td>
      <td>29</td>
      <td>71</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>



### Groupby Operation
分組後可以進行各類運算：sum(), mean(), max(), min()


```python
grpBySex.sum()
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
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>218</td>
      <td>459</td>
      <td>863</td>
    </tr>
    <tr>
      <th>M</th>
      <td>406</td>
      <td>784</td>
      <td>1778</td>
    </tr>
  </tbody>
</table>
</div>




```python
grpBySex.mean()
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
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>31.142857</td>
      <td>65.571429</td>
      <td>123.285714</td>
    </tr>
    <tr>
      <th>M</th>
      <td>36.909091</td>
      <td>71.272727</td>
      <td>161.636364</td>
    </tr>
  </tbody>
</table>
</div>




```python
grpBySex.max()
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
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>47</td>
      <td>69</td>
      <td>139</td>
    </tr>
    <tr>
      <th>M</th>
      <td>53</td>
      <td>75</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>




```python
grpBySex.min()
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
      <th>Age</th>
      <th>Height (in)</th>
      <th>Weight (lbs)</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>23</td>
      <td>62</td>
      <td>98</td>
    </tr>
    <tr>
      <th>M</th>
      <td>29</td>
      <td>68</td>
      <td>143</td>
    </tr>
  </tbody>
</table>
</div>



## Wash Data with NAN
### 判斷 NAN
* isnull() 
* notnull() 

### 處理 NAN
* dropna()
* fillna()



```python
import numpy as np
import pandas as pd

groups = ["Modern Web", "DevOps", np.nan, "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, np.nan]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df.loc[:, "groups"].isnull()) # 判斷哪些組的組名是遺失值
print("---") # 分隔線
print(ironmen_df.loc[:, "ironmen"].notnull()) # 判斷哪些組的鐵人數不是遺失值

ironmen_df_na_dropped = ironmen_df.dropna() # 有遺失值的觀測值都刪除
print(ironmen_df_na_dropped)
print("---") # 分隔線
ironmen_df_na_filled = ironmen_df.fillna(0) # 有遺失值的觀測值填補 0
print(ironmen_df_na_filled)
print("---") # 分隔線
ironmen_df_na_filled = ironmen_df.fillna({"groups": "Cloud", "ironmen": 71}) # 依欄位填補遺失值
print(ironmen_df_na_filled)
```

    0    False
    1    False
    2     True
    3    False
    4    False
    5    False
    Name: groups, dtype: bool
    ---
    0     True
    1     True
    2     True
    3     True
    4     True
    5    False
    Name: ironmen, dtype: bool
           groups  ironmen
    0  Modern Web     59.0
    1      DevOps      9.0
    3    Big Data     14.0
    4    Security      6.0
    ---
           groups  ironmen
    0  Modern Web     59.0
    1      DevOps      9.0
    2           0     19.0
    3    Big Data     14.0
    4    Security      6.0
    5       自我挑戰組      0.0
    ---
           groups  ironmen
    0  Modern Web     59.0
    1      DevOps      9.0
    2       Cloud     19.0
    3    Big Data     14.0
    4    Security      6.0
    5       自我挑戰組     71.0


## Plot
**DataFrame 一個很重要的特性是利用 matplotlib.pyplot 繪圖功能 visuallize data!**\
有兩種方式：(1) 直接用 df.plot; (2) 用 pyplot 的 plot.\
(1) 是一個 quick way to plot \
(2) 可以調用 pyplot 所有的功能



```python
import matplotlib.pyplot as plt
df = pd.read_csv('drive/My Drive/Colab Notebooks/biostats.csv', index_col="Name", skipinitialspace=True)
df.plot(title="Generated Plot", grid=True, figsize=(8,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f952bc52240>




![png](/media/output_81_1.png)



```python
df.columns
plt.plot(df[['Age', 'Height (in)']])
plt.xlabel('Name')
plt.ylabel('Number')
plt.title('Generated Plot')
plt.grid()
```


![png](/media/output_82_0.png)


