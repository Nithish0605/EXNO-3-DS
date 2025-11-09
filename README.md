## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
# Developed by : NITHISH S

# Reg No : 212224240105
```
import pandas as pd
df = pd.read_csv('/content/Encoding Data.csv')
df
```
<img width="864" height="473" alt="image" src="https://github.com/user-attachments/assets/46b2a901-279e-4377-b5f9-fd86dcef7f78" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="411" height="248" alt="image" src="https://github.com/user-attachments/assets/d0c6f970-c297-4061-8eb9-f48ea873a736" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="704" height="463" alt="image" src="https://github.com/user-attachments/assets/edc36e24-8624-4ae8-913c-b4682becd2aa" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="712" height="467" alt="image" src="https://github.com/user-attachments/assets/99663072-4925-48a9-b2d7-88a5db246c11" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="1216" height="450" alt="image" src="https://github.com/user-attachments/assets/f5c6fb9a-569d-4b99-9c46-e9254e00483f" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="1347" height="476" alt="image" src="https://github.com/user-attachments/assets/05e06f73-0e4a-4987-95d1-31f008ad82aa" />

```
!pip install category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="911" height="512" alt="image" src="https://github.com/user-attachments/assets/f8dad858-4c76-442a-8be8-30ebbb331691" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="751" height="452" alt="image" src="https://github.com/user-attachments/assets/8cf25665-5f2f-4ccf-adfd-17b53c9ab896" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="1024" height="531" alt="image" src="https://github.com/user-attachments/assets/87723026-c4b6-4db9-957d-5cbf319ca109" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="859" height="537" alt="image" src="https://github.com/user-attachments/assets/4f1e1b9a-21b5-4387-bf35-d90090175ffb" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="1171" height="551" alt="image" src="https://github.com/user-attachments/assets/a5dc8aee-2aec-45b9-bbb8-c3fc8bffff30" />

```
df.skew()
```
<img width="558" height="263" alt="image" src="https://github.com/user-attachments/assets/55f592c5-f090-404f-999d-57c09b951ad1" />

```
np.log(df["Highly Positive Skew"])
```
<img width="672" height="568" alt="image" src="https://github.com/user-attachments/assets/09c8aa23-9d01-4674-ae57-eb75d126db61" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="584" height="563" alt="image" src="https://github.com/user-attachments/assets/32b0f032-d9ea-4a7f-83b1-c24b9de9329f" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="539" height="576" alt="image" src="https://github.com/user-attachments/assets/91de9c5c-5aad-43a9-891d-82a6f30674db" />

```
np.square(df["Highly Positive Skew"])
```
<img width="412" height="560" alt="image" src="https://github.com/user-attachments/assets/c412ffca-eb3d-436b-94bd-46b85e817c07" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1411" height="542" alt="image" src="https://github.com/user-attachments/assets/e0b332f4-d0ae-4839-b15e-6ac180b2ad8d" />

```
df.skew()
```
<img width="559" height="319" alt="image" src="https://github.com/user-attachments/assets/5fda4ca6-877d-4f93-8dd1-9e58da1e6fa8" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="562" height="385" alt="image" src="https://github.com/user-attachments/assets/dc80c6cc-c4ef-4849-b961-7032309cadb7" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1429" height="555" alt="image" src="https://github.com/user-attachments/assets/99780113-4578-4859-b953-48904a057b7b" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="919" height="578" alt="image" src="https://github.com/user-attachments/assets/e4dd2b2a-8b55-4393-b1dc-5b746179b0d4" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="792" height="562" alt="image" src="https://github.com/user-attachments/assets/8bd696ce-057d-479a-9a01-39291248cf8b" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="832" height="562" alt="image" src="https://github.com/user-attachments/assets/ec132002-9995-420f-82cd-e89bfc9577a4" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="864" height="559" alt="image" src="https://github.com/user-attachments/assets/a5914016-c238-4154-aa08-d41c43db737d" />

```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
<img width="1417" height="632" alt="image" src="https://github.com/user-attachments/assets/a9595a19-8d18-40a0-b639-d20bf25d0a32" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
<img width="802" height="572" alt="image" src="https://github.com/user-attachments/assets/7c77d7a4-fc05-439c-81dd-6384f5fd84e0" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="809" height="590" alt="image" src="https://github.com/user-attachments/assets/6257aa22-6c7c-4b5a-aebd-811ab5edb3bb" />


       
# RESULT:
   Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully
       
