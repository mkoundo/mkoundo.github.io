---
layout: posts
title: "Pandas for Data Science"
date:   2020-08-02 15:00:00 +0100
tags: [python, jupyter, pandas, numpy]
mathjax: "true"
---

## Introduction to Pandas

The following pandas for data science example is from Giles' [tutorial](https://youtu.be/MYU9W34dZh0)


```python
import numpy as np
import pandas as pd
import os
import urllib
%matplotlib inline
```


```python
url = 'https://covid19.who.int/WHO-COVID-19-global-data.csv'
file_path = os.path.join('data','covid')
```


```python
os.makedirs(file_path, exist_ok=True)
csv_path = os.path.join(file_path, 'WHO-COVID-19-global-data.csv')
urllib.request.urlretrieve(url,csv_path)
```




    ('data\\covid\\WHO-COVID-19-global-data.csv',
     <http.client.HTTPMessage at 0x2537f988c40>)




```python
df = pd.read_csv(csv_path)
```


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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-02-24</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-25</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-02-26</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-02-27</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-02-28</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
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
    </tr>
    <tr>
      <th>31874</th>
      <td>2020-07-27</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>78</td>
      <td>2512</td>
      <td>0</td>
      <td>34</td>
    </tr>
    <tr>
      <th>31875</th>
      <td>2020-07-28</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>192</td>
      <td>2704</td>
      <td>2</td>
      <td>36</td>
    </tr>
    <tr>
      <th>31876</th>
      <td>2020-07-29</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>113</td>
      <td>2817</td>
      <td>4</td>
      <td>40</td>
    </tr>
    <tr>
      <th>31877</th>
      <td>2020-07-30</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>62</td>
      <td>2879</td>
      <td>1</td>
      <td>41</td>
    </tr>
    <tr>
      <th>31878</th>
      <td>2020-07-31</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>213</td>
      <td>3092</td>
      <td>12</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
<p>31879 rows × 8 columns</p>
</div>




```python
df_index = df.index
df_index
```




    RangeIndex(start=0, stop=31879, step=1)




```python
df_cols = df.columns
df_cols
```




    Index(['Date_reported', ' Country_code', ' Country', ' WHO_region',
           ' New_cases', ' Cumulative_cases', ' New_deaths', ' Cumulative_deaths'],
          dtype='object')




```python
df_index.values
```




    array([    0,     1,     2, ..., 31876, 31877, 31878], dtype=int64)




```python
df.values
```




    array([['2020-02-24', 'AF', 'Afghanistan', ..., 5, 0, 0],
           ['2020-02-25', 'AF', 'Afghanistan', ..., 5, 0, 0],
           ['2020-02-26', 'AF', 'Afghanistan', ..., 5, 0, 0],
           ...,
           ['2020-07-29', 'ZW', 'Zimbabwe', ..., 2817, 4, 40],
           ['2020-07-30', 'ZW', 'Zimbabwe', ..., 2879, 1, 41],
           ['2020-07-31', 'ZW', 'Zimbabwe', ..., 3092, 12, 53]], dtype=object)




```python
df.dtypes
```




    Date_reported         object
     Country_code         object
     Country              object
     WHO_region           object
     New_cases             int64
     Cumulative_cases      int64
     New_deaths            int64
     Cumulative_deaths     int64
    dtype: object




```python
df.shape
```




    (31879, 8)




```python
df.head(10)
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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-02-24</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-25</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-02-26</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-02-27</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-02-28</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-02-29</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-03-01</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-03-02</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-03-03</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-03-04</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(8)
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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31871</th>
      <td>2020-07-24</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>90</td>
      <td>2124</td>
      <td>2</td>
      <td>28</td>
    </tr>
    <tr>
      <th>31872</th>
      <td>2020-07-25</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>172</td>
      <td>2296</td>
      <td>4</td>
      <td>32</td>
    </tr>
    <tr>
      <th>31873</th>
      <td>2020-07-26</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>138</td>
      <td>2434</td>
      <td>2</td>
      <td>34</td>
    </tr>
    <tr>
      <th>31874</th>
      <td>2020-07-27</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>78</td>
      <td>2512</td>
      <td>0</td>
      <td>34</td>
    </tr>
    <tr>
      <th>31875</th>
      <td>2020-07-28</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>192</td>
      <td>2704</td>
      <td>2</td>
      <td>36</td>
    </tr>
    <tr>
      <th>31876</th>
      <td>2020-07-29</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>113</td>
      <td>2817</td>
      <td>4</td>
      <td>40</td>
    </tr>
    <tr>
      <th>31877</th>
      <td>2020-07-30</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>62</td>
      <td>2879</td>
      <td>1</td>
      <td>41</td>
    </tr>
    <tr>
      <th>31878</th>
      <td>2020-07-31</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>213</td>
      <td>3092</td>
      <td>12</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 31879 entries, 0 to 31878
    Data columns (total 8 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   Date_reported       31879 non-null  object
     1    Country_code       31739 non-null  object
     2    Country            31879 non-null  object
     3    WHO_region         31879 non-null  object
     4    New_cases          31879 non-null  int64 
     5    Cumulative_cases   31879 non-null  int64 
     6    New_deaths         31879 non-null  int64 
     7    Cumulative_deaths  31879 non-null  int64 
    dtypes: int64(4), object(4)
    memory usage: 1.9+ MB
    


```python
df.describe()
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
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>31879.000000</td>
      <td>3.187900e+04</td>
      <td>31879.000000</td>
      <td>31879.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>536.591706</td>
      <td>2.717609e+04</td>
      <td>20.982779</td>
      <td>1413.004894</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3257.687811</td>
      <td>1.625662e+05</td>
      <td>127.100072</td>
      <td>7929.931079</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3525.000000</td>
      <td>1.000000e+00</td>
      <td>-514.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>4.700000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>6.260000e+02</td>
      <td>0.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112.000000</td>
      <td>5.879000e+03</td>
      <td>2.000000</td>
      <td>116.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>74354.000000</td>
      <td>4.388566e+06</td>
      <td>6409.000000</td>
      <td>150054.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[' Country']
```




    0        Afghanistan
    1        Afghanistan
    2        Afghanistan
    3        Afghanistan
    4        Afghanistan
                ...     
    31874       Zimbabwe
    31875       Zimbabwe
    31876       Zimbabwe
    31877       Zimbabwe
    31878       Zimbabwe
    Name:  Country, Length: 31879, dtype: object




```python
df[' Country'].unique()
```




    array(['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola',
           'Anguilla', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba',
           'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
           'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
           'Bermuda', 'Bhutan', 'Bolivia (Plurinational State of)',
           'Bonaire, Sint Eustatius and Saba', 'Bosnia and Herzegovina',
           'Botswana', 'Brazil', 'British Virgin Islands',
           'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi',
           'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands',
           'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
           'Comoros', 'Congo', 'Costa Rica', 'Côte d’Ivoire', 'Croatia',
           'Cuba', 'Curaçao', 'Cyprus', 'Czechia',
           'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
           'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
           'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
           'Eswatini', 'Ethiopia', 'Falkland Islands (Malvinas)',
           'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana',
           'French Polynesia', 'Gabon', 'Gambia', 'Georgia', 'Germany',
           'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada',
           'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea',
           'Guinea-Bissau', 'Guyana', 'Haiti', 'Holy See', 'Honduras',
           'Hungary', 'Iceland', 'India', 'Indonesia',
           'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Isle of Man',
           'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan',
           'Kazakhstan', 'Kenya', 'Kosovo[1]', 'Kuwait', 'Kyrgyzstan',
           "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho',
           'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
           'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
           'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico',
           'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco',
           'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands',
           'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
           'North Macedonia',
           'Northern Mariana Islands (Commonwealth of the)', 'Norway',
           'occupied Palestinian territory, including east Jerusalem', 'Oman',
           'Other', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay',
           'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico',
           'Qatar', 'Republic of Korea', 'Republic of Moldova', 'Réunion',
           'Romania', 'Russian Federation', 'Rwanda', 'Saint Barthélemy',
           'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin',
           'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines',
           'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal',
           'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
           'Sint Maarten', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa',
           'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
           'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
           'The United Kingdom', 'Timor-Leste', 'Togo', 'Trinidad and Tobago',
           'Tunisia', 'Turkey', 'Turks and Caicos Islands', 'Uganda',
           'Ukraine', 'United Arab Emirates', 'United Republic of Tanzania',
           'United States of America', 'United States Virgin Islands',
           'Uruguay', 'Uzbekistan', 'Venezuela (Bolivarian Republic of)',
           'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe'], dtype=object)




```python
df.columns = [col.strip() for col in df.columns]
df.columns
```




    Index(['Date_reported', 'Country_code', 'Country', 'WHO_region', 'New_cases',
           'Cumulative_cases', 'New_deaths', 'Cumulative_deaths'],
          dtype='object')




```python
df.Country
```




    0        Afghanistan
    1        Afghanistan
    2        Afghanistan
    3        Afghanistan
    4        Afghanistan
                ...     
    31874       Zimbabwe
    31875       Zimbabwe
    31876       Zimbabwe
    31877       Zimbabwe
    31878       Zimbabwe
    Name: Country, Length: 31879, dtype: object




```python
df.loc[1:4, 'Country']
```




    1    Afghanistan
    2    Afghanistan
    3    Afghanistan
    4    Afghanistan
    Name: Country, dtype: object




```python
df.loc[1:8, ['Country', 'New_cases']]
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
      <th>Country</th>
      <th>New_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Country == 'United States of America'
```




    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    31874    False
    31875    False
    31876    False
    31877    False
    31878    False
    Name: Country, Length: 31879, dtype: bool




```python
df[df.Country == 'United States of America']
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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30554</th>
      <td>2020-01-20</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30555</th>
      <td>2020-01-21</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30556</th>
      <td>2020-01-22</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30557</th>
      <td>2020-01-23</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30558</th>
      <td>2020-01-24</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>1</td>
      <td>6</td>
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
    </tr>
    <tr>
      <th>30743</th>
      <td>2020-07-27</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>63968</td>
      <td>4148011</td>
      <td>929</td>
      <td>145727</td>
    </tr>
    <tr>
      <th>30744</th>
      <td>2020-07-28</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>61498</td>
      <td>4209509</td>
      <td>604</td>
      <td>146331</td>
    </tr>
    <tr>
      <th>30745</th>
      <td>2020-07-29</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>54022</td>
      <td>4263531</td>
      <td>1118</td>
      <td>147449</td>
    </tr>
    <tr>
      <th>30746</th>
      <td>2020-07-30</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>59629</td>
      <td>4323160</td>
      <td>1191</td>
      <td>148640</td>
    </tr>
    <tr>
      <th>30747</th>
      <td>2020-07-31</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>65406</td>
      <td>4388566</td>
      <td>1414</td>
      <td>150054</td>
    </tr>
  </tbody>
</table>
<p>194 rows × 8 columns</p>
</div>




```python
df[df.New_deaths > 1000]
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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4038</th>
      <td>2020-05-21</td>
      <td>BR</td>
      <td>Brazil</td>
      <td>AMRO</td>
      <td>17408</td>
      <td>271628</td>
      <td>1179</td>
      <td>17971</td>
    </tr>
    <tr>
      <th>4040</th>
      <td>2020-05-23</td>
      <td>BR</td>
      <td>Brazil</td>
      <td>AMRO</td>
      <td>18508</td>
      <td>310087</td>
      <td>1188</td>
      <td>20047</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>2020-05-24</td>
      <td>BR</td>
      <td>Brazil</td>
      <td>AMRO</td>
      <td>20803</td>
      <td>330890</td>
      <td>1001</td>
      <td>21048</td>
    </tr>
    <tr>
      <th>4045</th>
      <td>2020-05-28</td>
      <td>BR</td>
      <td>Brazil</td>
      <td>AMRO</td>
      <td>16324</td>
      <td>391222</td>
      <td>1039</td>
      <td>24512</td>
    </tr>
    <tr>
      <th>4046</th>
      <td>2020-05-29</td>
      <td>BR</td>
      <td>Brazil</td>
      <td>AMRO</td>
      <td>20599</td>
      <td>411821</td>
      <td>1086</td>
      <td>25598</td>
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
    </tr>
    <tr>
      <th>30741</th>
      <td>2020-07-25</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>71714</td>
      <td>4009808</td>
      <td>1110</td>
      <td>143663</td>
    </tr>
    <tr>
      <th>30742</th>
      <td>2020-07-26</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>74235</td>
      <td>4084043</td>
      <td>1135</td>
      <td>144798</td>
    </tr>
    <tr>
      <th>30745</th>
      <td>2020-07-29</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>54022</td>
      <td>4263531</td>
      <td>1118</td>
      <td>147449</td>
    </tr>
    <tr>
      <th>30746</th>
      <td>2020-07-30</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>59629</td>
      <td>4323160</td>
      <td>1191</td>
      <td>148640</td>
    </tr>
    <tr>
      <th>30747</th>
      <td>2020-07-31</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>65406</td>
      <td>4388566</td>
      <td>1414</td>
      <td>150054</td>
    </tr>
  </tbody>
</table>
<p>125 rows × 8 columns</p>
</div>




```python
df.loc[df.New_deaths > 1000, ['New_deaths', 'Country']]
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
      <th>New_deaths</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4038</th>
      <td>1179</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>4040</th>
      <td>1188</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>1001</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>4045</th>
      <td>1039</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>4046</th>
      <td>1086</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30741</th>
      <td>1110</td>
      <td>United States of America</td>
    </tr>
    <tr>
      <th>30742</th>
      <td>1135</td>
      <td>United States of America</td>
    </tr>
    <tr>
      <th>30745</th>
      <td>1118</td>
      <td>United States of America</td>
    </tr>
    <tr>
      <th>30746</th>
      <td>1191</td>
      <td>United States of America</td>
    </tr>
    <tr>
      <th>30747</th>
      <td>1414</td>
      <td>United States of America</td>
    </tr>
  </tbody>
</table>
<p>125 rows × 2 columns</p>
</div>




```python
df.loc[(df.New_deaths > 1000) & (df.Country_code == 'US'), ['Date_reported', 'Country', 'New_cases', 'New_deaths']]
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
      <th>Date_reported</th>
      <th>Country</th>
      <th>New_cases</th>
      <th>New_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30629</th>
      <td>2020-04-04</td>
      <td>United States of America</td>
      <td>28103</td>
      <td>1061</td>
    </tr>
    <tr>
      <th>30630</th>
      <td>2020-04-05</td>
      <td>United States of America</td>
      <td>32105</td>
      <td>1166</td>
    </tr>
    <tr>
      <th>30631</th>
      <td>2020-04-06</td>
      <td>United States of America</td>
      <td>33510</td>
      <td>1338</td>
    </tr>
    <tr>
      <th>30632</th>
      <td>2020-04-07</td>
      <td>United States of America</td>
      <td>26493</td>
      <td>1201</td>
    </tr>
    <tr>
      <th>30633</th>
      <td>2020-04-08</td>
      <td>United States of America</td>
      <td>29510</td>
      <td>1286</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30741</th>
      <td>2020-07-25</td>
      <td>United States of America</td>
      <td>71714</td>
      <td>1110</td>
    </tr>
    <tr>
      <th>30742</th>
      <td>2020-07-26</td>
      <td>United States of America</td>
      <td>74235</td>
      <td>1135</td>
    </tr>
    <tr>
      <th>30745</th>
      <td>2020-07-29</td>
      <td>United States of America</td>
      <td>54022</td>
      <td>1118</td>
    </tr>
    <tr>
      <th>30746</th>
      <td>2020-07-30</td>
      <td>United States of America</td>
      <td>59629</td>
      <td>1191</td>
    </tr>
    <tr>
      <th>30747</th>
      <td>2020-07-31</td>
      <td>United States of America</td>
      <td>65406</td>
      <td>1414</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 4 columns</p>
</div>




```python
df.loc[df.Country_code == 'US', ['New_cases']].max()
```




    New_cases    74354
    dtype: int64




```python
df.loc[df.Country_code == 'US', ['New_cases']].min()
```




    New_cases    0
    dtype: int64




```python
df.loc[df.Country_code == 'US', ['New_cases']].sum()
```




    New_cases    4388566
    dtype: int64




```python
df.loc[df.Country_code == 'US', ['Cumulative_cases']].max()
```




    Cumulative_cases    4388566
    dtype: int64




```python
df.New_deaths.idxmax()
```




    30642




```python
df.loc[df.New_deaths.idxmax(),['Date_reported', 'Country', 'New_cases', 'New_deaths', 'Cumulative_deaths']]
```




    Date_reported                      2020-04-17
    Country              United States of America
    New_cases                               28711
    New_deaths                               6409
    Cumulative_deaths                       32280
    Name: 30642, dtype: object




```python
df[df.New_deaths < 0]
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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1588</th>
      <td>2020-06-04</td>
      <td>AU</td>
      <td>Australia</td>
      <td>WPRO</td>
      <td>8</td>
      <td>7229</td>
      <td>-1</td>
      <td>102</td>
    </tr>
    <tr>
      <th>4652</th>
      <td>2020-07-13</td>
      <td>BF</td>
      <td>Burkina Faso</td>
      <td>AFRO</td>
      <td>13</td>
      <td>1033</td>
      <td>-1</td>
      <td>53</td>
    </tr>
    <tr>
      <th>6551</th>
      <td>2020-06-09</td>
      <td>CG</td>
      <td>Congo</td>
      <td>AFRO</td>
      <td>0</td>
      <td>683</td>
      <td>-2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7398</th>
      <td>2020-05-14</td>
      <td>CY</td>
      <td>Cyprus</td>
      <td>EURO</td>
      <td>2</td>
      <td>905</td>
      <td>-6</td>
      <td>17</td>
    </tr>
    <tr>
      <th>7556</th>
      <td>2020-05-19</td>
      <td>CZ</td>
      <td>Czechia</td>
      <td>EURO</td>
      <td>111</td>
      <td>8586</td>
      <td>-1</td>
      <td>297</td>
    </tr>
    <tr>
      <th>7603</th>
      <td>2020-07-05</td>
      <td>CZ</td>
      <td>Czechia</td>
      <td>EURO</td>
      <td>121</td>
      <td>12440</td>
      <td>-1</td>
      <td>351</td>
    </tr>
    <tr>
      <th>7604</th>
      <td>2020-07-06</td>
      <td>CZ</td>
      <td>Czechia</td>
      <td>EURO</td>
      <td>75</td>
      <td>12515</td>
      <td>-3</td>
      <td>348</td>
    </tr>
    <tr>
      <th>7849</th>
      <td>2020-05-13</td>
      <td>DK</td>
      <td>Denmark</td>
      <td>EURO</td>
      <td>78</td>
      <td>10591</td>
      <td>-6</td>
      <td>527</td>
    </tr>
    <tr>
      <th>10217</th>
      <td>2020-05-20</td>
      <td>FR</td>
      <td>France</td>
      <td>EURO</td>
      <td>462</td>
      <td>140959</td>
      <td>-218</td>
      <td>27972</td>
    </tr>
    <tr>
      <th>12027</th>
      <td>2020-07-06</td>
      <td>GP</td>
      <td>Guadeloupe</td>
      <td>AMRO</td>
      <td>0</td>
      <td>184</td>
      <td>-2</td>
      <td>14</td>
    </tr>
    <tr>
      <th>12241</th>
      <td>2020-05-05</td>
      <td>GT</td>
      <td>Guatemala</td>
      <td>AMRO</td>
      <td>15</td>
      <td>703</td>
      <td>-2</td>
      <td>17</td>
    </tr>
    <tr>
      <th>12586</th>
      <td>2020-07-04</td>
      <td>GN</td>
      <td>Guinea</td>
      <td>AFRO</td>
      <td>71</td>
      <td>5521</td>
      <td>-13</td>
      <td>33</td>
    </tr>
    <tr>
      <th>14866</th>
      <td>2020-06-25</td>
      <td>IT</td>
      <td>Italy</td>
      <td>EURO</td>
      <td>577</td>
      <td>239410</td>
      <td>-31</td>
      <td>34644</td>
    </tr>
    <tr>
      <th>18884</th>
      <td>2020-06-16</td>
      <td>YT</td>
      <td>Mayotte</td>
      <td>AFRO</td>
      <td>28</td>
      <td>2310</td>
      <td>-1</td>
      <td>28</td>
    </tr>
    <tr>
      <th>23461</th>
      <td>2020-05-05</td>
      <td>PR</td>
      <td>Puerto Rico</td>
      <td>AMRO</td>
      <td>35</td>
      <td>1843</td>
      <td>-43</td>
      <td>54</td>
    </tr>
    <tr>
      <th>25681</th>
      <td>2020-05-10</td>
      <td>ST</td>
      <td>Sao Tome and Principe</td>
      <td>AFRO</td>
      <td>-43</td>
      <td>165</td>
      <td>-1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27436</th>
      <td>2020-05-23</td>
      <td>SS</td>
      <td>South Sudan</td>
      <td>AFRO</td>
      <td>0</td>
      <td>473</td>
      <td>-1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>30660</th>
      <td>2020-05-05</td>
      <td>US</td>
      <td>United States of America</td>
      <td>AMRO</td>
      <td>29266</td>
      <td>1154985</td>
      <td>-514</td>
      <td>67279</td>
    </tr>
    <tr>
      <th>31846</th>
      <td>2020-06-29</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>0</td>
      <td>567</td>
      <td>-1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['pc_cases'] = (df['New_cases'] / df['Cumulative_cases']) * 100
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
      <th>Date_reported</th>
      <th>Country_code</th>
      <th>Country</th>
      <th>WHO_region</th>
      <th>New_cases</th>
      <th>Cumulative_cases</th>
      <th>New_deaths</th>
      <th>Cumulative_deaths</th>
      <th>pc_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-02-24</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-25</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-02-26</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-02-27</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-02-28</td>
      <td>AF</td>
      <td>Afghanistan</td>
      <td>EMRO</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>31874</th>
      <td>2020-07-27</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>78</td>
      <td>2512</td>
      <td>0</td>
      <td>34</td>
      <td>3.105096</td>
    </tr>
    <tr>
      <th>31875</th>
      <td>2020-07-28</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>192</td>
      <td>2704</td>
      <td>2</td>
      <td>36</td>
      <td>7.100592</td>
    </tr>
    <tr>
      <th>31876</th>
      <td>2020-07-29</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>113</td>
      <td>2817</td>
      <td>4</td>
      <td>40</td>
      <td>4.011360</td>
    </tr>
    <tr>
      <th>31877</th>
      <td>2020-07-30</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>62</td>
      <td>2879</td>
      <td>1</td>
      <td>41</td>
      <td>2.153526</td>
    </tr>
    <tr>
      <th>31878</th>
      <td>2020-07-31</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
      <td>AFRO</td>
      <td>213</td>
      <td>3092</td>
      <td>12</td>
      <td>53</td>
      <td>6.888745</td>
    </tr>
  </tbody>
</table>
<p>31879 rows × 9 columns</p>
</div>




```python

```
