import numpy as np # type: ignore
import pandas as pd
import matplotlib.pyplot as mat
data=pd.read_csv('Dyslexia_detection/Data/Dyslexic/111JA2.csv')
out=data.head()
pic=data['RY'].plot(kind='bar',figsize=(20,30))
print(out)
print(pic)              