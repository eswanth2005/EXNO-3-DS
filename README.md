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

```
# NAME: ESWANTH KUMAR K
# REG NO: 212223040046

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2025-04-22 104753](https://github.com/user-attachments/assets/cc1374ab-17d0-46e5-aa08-24f79ba273f7)
```
#ordinal encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
e1 = OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/c68b963c-9a06-4d97-ae6b-4842b7715597)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/304c2f99-0b88-4393-979a-400f1c4c97f5)
```
# LabelEncoder (orders in alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a54949b6-d9b8-40b6-9f3c-718f5b0083c2)

```
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/418d8474-ec0c-4e71-a758-ff0cd1b4b353)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/f5a8e082-8ba1-42ba-8a6a-a30e4269ad2d)
```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/e9817fd7-97e6-4881-a9f0-65d3f159e023)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)

CC
```
![image](https://github.com/user-attachments/assets/c93732f5-fb5d-4dd7-9b30-108403095143)

```
#Data_to_transform
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/53a6c268-52be-4a33-aa7b-cb3789e1c810)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/a0c8c2ac-4c18-4581-9dae-31ea94fec5ed)
```
import pandas as pd
from scipy import stats
import numpy as np
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2aa83465-f5f5-430d-a1da-2eab2447c4f1)
```
# Reciprocol transformation
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/533895b7-919a-40e4-b8b7-f93f93ea5267)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a92b4a00-d56d-437c-9dae-f7149d9d6889)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0afc47db-8eb0-420d-8951-4a4cbcc92a98)

```
# power transformation
# box cox
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/966ac1bb-c272-4a68-b1f8-64093aeb2716)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/6d6dcd44-2797-471f-b45b-eb07a055fbcb)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/2c67310e-be7a-4bd8-a9ce-45931f80fbc8)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/5172e29c-9db4-4e86-8a7d-a225333bad01)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a0194bae-eee2-46c2-8dd1-db952f42aa49)

```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5e96226d-747b-4f61-8cda-ea8c547a246e)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/07378e20-1008-44e4-a4bd-e47bd8421dcb)


# RESULT:
    The program has been executed successful

       
