'''
Main function for CPH.
'''
# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
from data_loader import data_loader
from prescription_loader import prescription_loader
from EK_CPH import ek_cph
from utils import rmse_loss
from math import *
import random

def xtest_loss(ori_data_x,imputed_data_x,ward_nor_list):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    ward_code_list=list(df['Ward Code'])
    n=0
    y=0
    y_mae=0
    no, dim = ori_data_x.shape
    dim2 = int(dim)
    print("dim1,dim2:",dim,dim2)
    R_original=ori_data_x[:,dim2-1]
    R_result = imputed_data_x[:,dim2-1]
    yy_mae = []
    for id in ward_nor_list:
        result=R_result[id]
        origial=R_original[id]
        # print("id:",id,"origial:",origial,"result:",result)
        if str(origial)!="nan" and origial!=0:
            y=y+pow((origial-result),2)
            n+=1
            y_mae = y_mae + abs(origial - result)
            yy_mae.append(abs(origial - result))
    RMSE=sqrt(y/n)
    MAE=y_mae/n
    # print(y,y_mae)
    # print("RMSE:",RMSE)
    # print("MAE:",MAE)
    #print()
    return RMSE, MAE

def main (args,yy):
  '''
  Args:
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse, mae
  '''

  miss_rate = args.miss_rate
  
  cph_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, ward_nor_list_x, data_image = data_loader(miss_rate, yy)
  prescription_z,ward_nor_list_z,prescription_image = prescription_loader(yy)
  # Impute missing data
  imputed_data_x = ek_cph(miss_data_x, cph_parameters,data_image,prescription_z,prescription_image)
  RMSE, MAE = xtest_loss(ori_data_x,imputed_data_x,ward_nor_list_z)
  return RMSE, MAE

if __name__ == '__main__':
    for i in range(20,100,10):
        missing_rate = i
        rmse = []
        mae = []
        for yy in range(2010,2011):
            mmin = 100
            print("Diabetes, year:" + str(yy) + "-2017")
            for i in range(10):

                # Inputs for the main function
                parser = argparse.ArgumentParser()
                parser.add_argument(
                  '--miss_rate',
                  help='missing data probability',
                  default=missing_rate/100,
                  type=float)
                parser.add_argument(
                  '--batch_size',
                  help='the number of samples in mini-batch',
                  default=483,
                  type=int)
                parser.add_argument(
                  '--hint_rate',
                  help='hint probability',
                  default=0.9,
                  type=float)
                parser.add_argument(
                  '--alpha',
                  help='hyperparameter',
                  default=100,
                  type=float)
                parser.add_argument(
                  '--iterations',
                  help='number of training interations',
                  default=10000,
                  type=int)

                args = parser.parse_args()
                print(args)
                #sys.exit(1)
                # Calls main function
                RMSE, MAE = main(args,yy)
                rmse.append(RMSE)
                mae.append(MAE)
            average_rmse = np.mean(rmse)
            average_mae = np.mean(mae)
            print("average_rmse:", average_rmse)
            print("average_mae:", average_mae)


