import pandas as pd
import numpy as np

def auto_encoder_filter(df, col, others):
    
    new_df = df[df[col] == 1]
    for lab in others:
        new_df = new_df[new_df[lab]!=1]
        new_df = new_df[new_df[lab]!=-1]
    return new_df

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("valid.csv")

train_size = train_df.count()[0]
print(train_df.head())
print("column names:", train_df.columns)
print("Total train records: ",train_df.count()[0])


print("this data is not super accurate as it contains null values and some zeros")
print(train_df.groupby(['Sex']).count().T/train_size)

label_columns = list(train_df.columns[5:])
info_columns = list(train_df.columns[0:4])
print("label columns: ", label_columns)
print("info columns: ", info_columns)
label_df = train_df.drop(info_columns,axis = 1)

buns = label_columns[:]
for label in buns:
    label_copy = label_columns[:]
    label_copy.remove(label)
    #print(label, "::", label_copy)
    new_df = auto_encoder_filter(train_df, label, label_copy)
    print(label, " size: ", new_df.count()[0], "percentage: ", "{:.2%}".format(new_df.count()[0]/train_size))
    
    new_df.to_csv("autoencoders/"+label.replace(" ", "_")+".csv")






