'''
Data loader for Chronic Diseases Prevalence Dataset
'''

# Necessary packages
import numpy as np
import pandas as pd


def data_loader(miss_rate, yy):
  # diease_list = ['Diabetes Mellitus (Diabetes) Prevalence','Hypertension Prevalence',]
  diease_list = ['Hypertension Prevalence','Diabetes Mellitus (Diabetes) Prevalence',]
  year = yy #target year
  diease_select_list = 0# target disease
  diease_select_list2 = 1

  target_year = 2016
  #year = 2017
  N1 = 483
  N3 = target_year - year + 1
  N33 = target_year - year + 1
  data_x = np.ones((N1, N3), dtype='float64') #numpy.ones(shape, dtype=None, order='C')，
  data_m = np.ones((N1, N3), dtype='float64')

  data_tensor = np.ones((483, target_year - year + 1, 2), dtype='float64') #（483，N3，2）的tensor

  for y in range(year,target_year+1):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    ward_code_list=list(df['Ward Code'])
    # print("list(df):",list(df))
    df1 = df[diease_list[diease_select_list]+"_"+str(y)]
    data_x[:,y - year] = df1.values

  for y in range(year, target_year + 1):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    df1 = df[diease_list[diease_select_list] + "_" + str(y)]
    data_tensor[:,y - year,0] = df1.values

    df2 = df[diease_list[diease_select_list2]+"_"+str(y)]
    data_tensor[:,y - year,1] = df2.values


  miss_data_x = data_x.copy()

  ward_number = int(N1 * (100 - miss_rate*100) / 100)  # select wards number

  for y in range(N33 - 1, N33):
    data_year = data_x[:,y]
    ward_list = []  # training set
    ward_nor_list = []  # testing set
    num = 0
    df_ward = pd.read_csv("./data/Variance_2008_2017_"+diease_list[diease_select_list]+"_NORMALIZE.csv")
    df_diease = pd.read_csv("./data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])
    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)]) #Variance_2008_2017_xxx文档中的内容
    iii = 0
    while num < ward_number:
      id = ward_var[iii]
      iii += 1
      ward_code = ward_code_old[id]
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)
        diease_rate = data_year[index1]
        # print("diease_rate1:",diease_rate)
        if diease_rate!=0:
          num += 1
          ward_list.append(index1)

    for i in range(N1):
      if i in ward_list:
        continue
      ward_nor_list.append(i)
      data_m[i,N33-1] = 0
      data_tensor[i,-1,0] = 0

  for y in range(N33 - 1, N33):
    data_year = data_tensor[:, y, 1]

    ward_list2 = []
    ward_nor_list2 = []
    num = 0
    df_ward = pd.read_csv("./data/Variance_2008_2017_"+diease_list[diease_select_list2]+"_NORMALIZE.csv")
    # print('df_ward:',df_ward)

    df_diease = pd.read_csv("./data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])
    # print('ward_code_old:',ward_code_old)

    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
    # print("ward_var:",ward_var)
    iii = 0
    while num < ward_number:
      id = ward_var[iii]
      # print("id",id)
      iii += 1
      ward_code = ward_code_old[id]
      # print('ward_code:',ward_code)
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)
        # print('index1:',index1)
        diease_rate = data_year[index1]
        # print("diease_rate2:", diease_rate)
        if diease_rate!=0:
          num += 1
          ward_list2.append(index1)
    # print("ward_list2:", len(ward_list2))

    for i in range(N1):
      if i in ward_list2:
        continue
      ward_nor_list2.append(i)
      data_tensor[i,-1,1] = 0

  miss_data_x[data_m == 0] = np.nan
  data_image = data_tensor.reshape(1, 483, target_year - year + 1, 2)

  return data_x, miss_data_x, data_m, ward_nor_list, data_image

