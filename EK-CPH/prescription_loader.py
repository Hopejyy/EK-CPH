import numpy as np
import pandas as pd

def prescription_loader(yy):
    # prescription_list = [ 'Diabetes','Hypertension',]
    prescription_list = ['Hypertension','Diabetes',]
    prescription_select_list1 = 0 # target prescription
    prescription_select_list2 = 1

    target_year = 2016
    N1 = 483
    N2 = 184
    month = 113
    year = yy
    N3 = target_year - year + 1
    N33 = target_year - year + 1
    prescription_z = np.ones((N1,N3),dtype='float64')
    prescription_tensor = np.ones((483,N3,2),dtype='float64')

    for y in range(year,target_year+1):
        df = pd.read_csv('./data/prescription.csv')
        ward_code_list = list(df['Ward Code'])
        df1 = df[prescription_list[prescription_select_list1] + "_" + str(y)]
        # print(df1)
        prescription_z[:, y - year] = df1.values

    for y in range(year,target_year+1):
        df = pd.read_csv('./data/prescription.csv')
        df1 = df[prescription_list[prescription_select_list1] + '_' + str(y)]
        prescription_tensor[:, y - year, 0] = df1.values

        df2 = df[prescription_list[prescription_select_list2] + '_' + str(y)]
        prescription_tensor[:, y - year, 1] = df2.values

    # ward_number = 184
    ward_number = int( N1 * (100 - 0.75 * 100) / 100)  # select wards number

    for y in range(N33-1,N33):
        prescription_year = prescription_z[:, y]
        ward_list = []
        ward_nor_list = []
        num = 0
        df_ward = pd.read_csv("./data/Prescription_" + prescription_list[prescription_select_list1] + "_Variance.csv")
        df_prescription = pd.read_csv('./data/Ward_code_list.csv')
        ward_code_old = list(df_prescription['Ward Code'])

        ward_var = list(df_ward["Ward_id_" + str(year)])
        # print(ward_var)
        iii = 0
        while num < ward_number:
            id = ward_var[iii]
            iii += 1
            ward_code = ward_code_old[id]
            if ward_code in ward_code_list:
                index1 = ward_code_list.index(ward_code)
                prescription_rate = prescription_year[index1]
                # print(prescription_rate)
                if prescription_rate != 0:
                    num += 1
                    ward_list.append(index1)
        # print(ward_list)

        for i in range(N1):
            if i in ward_list:
                continue
            ward_nor_list.append(i)
            prescription_tensor[i,-1,0] = 0


    for y in range(N33 - 1, N33):
        prescription_year = prescription_z[:, y]
        ward_list = []
        ward_nor_list = []
        num = 0
        df_ward = pd.read_csv("./data/Prescription_" + prescription_list[prescription_select_list2] + "_Variance.csv")
        df_prescription = pd.read_csv('./data/Ward_code_list.csv')
        ward_code_old = list(df_prescription['Ward Code'])
        ward_var = list(df_ward["Ward_id_" + str(year)])
        # print(ward_var)
        iii = 0
        while num < ward_number:
            id = ward_var[iii]
            iii += 1
            ward_code = ward_code_old[id]
            if ward_code in ward_code_list:
                index1 = ward_code_list.index(ward_code)
                prescription_rate = prescription_year[index1]
                if prescription_rate != 0:
                    num += 1
                    ward_list.append(index1)
        # print(len(ward_list))

        for i in range(N1):
            if i in ward_list:
                continue
            ward_nor_list.append(i)
            prescription_tensor[i, -1, 1] = 0
    # print("prescription_tensor.shape", prescription_tensor.shape)
    # print("ward_nor_list:", len(ward_nor_list))
    # where_are_nan = np.isnan(prescription_z)
    # prescription_z[where_are_nan] = 0
    # print(prescription_z, prescription_z.shape)

    prescription_image = prescription_tensor.reshape(1, 483, target_year-year+1, 2)
    # print(prescription_tensor)
    # print("prescription_image.shape:",prescription_image.shape)
    # print(prescription_z.shape)
    return prescription_z,ward_nor_list,prescription_image

