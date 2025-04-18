
#==============================================================#Funktion zum Generieren variierter Datensätze -  DATENSÄTZE  RE UND IMAG als Funktion der Frequenz======================

from tkinter import filedialog
import pandas as pd
import numpy as np
import os


number_data = 10
variation_range_real = 0.0005 # Streuung in X-Richtung! (Absolutwerte)
variation_range_imag = 0.0005 # Streuung in X-Richtung! (Absolutwerte)


# Read in Nyquist_Data_extrapolation (Data base)  

def read_data_from_excel():
    # Load Excel file
    file_path = './HMS_Data base/01_Nyquist_Data_extrapolation_new.xlsx'
    nyquist_data = pd.ExcelFile(file_path)

    # Load specific Excel-sheet 'Real' and 'Imag' in Excel file 
    df_real = nyquist_data.parse('Real', delimiter='\t', decimal=",")
    df_imag = nyquist_data.parse('Imag', delimiter='\t', decimal=",")

    # Real part and imaginary part as Pandas Dataframe
    data_real=pd.DataFrame(df_real)
    data_imag=pd.DataFrame(df_imag)
    print("data_real:\n", data_real)
    print("data_imag:\n", data_imag)

    return(data_real,data_imag)

#[data_real, data_imag]=read_data_from_excel()



def read_data_from_csv():
    # Load csv files (time efficient)
    # Real part and imaginary part as Pandas Dataframe
    data_real = pd.DataFrame(pd.read_csv('./HMS_Classification/exported_failure_data_low_real.csv', delimiter = ';', header = 0, decimal = '.'))# read as pandas DataFrame 
    data_imag = pd.DataFrame(pd.read_csv('./HMS_Data base/Nyquist_Data_adapted_Imag_part.csv', delimiter = ';', header = 0, decimal = '.'))# read as pandas DataFrame 

    return(data_real,data_imag)

[data_real, data_imag]=read_data_from_csv()
print("data_real:\n", data_real)


# Funktion zur Erzeugung von Datenvariationen

# Generiere Datensätze für Realpart
def augment_data_real(data_real_op_point, number_data, variation_range_real,):
    augmented_data_real = pd.DataFrame(data_real_op_point)
    #print("augmented data_real:\n", augmented_data_real)
    
    for _ in range(number_data):
        
        # Erstellen einer Kopie der originalen Daten
        temp_df_real = data_real_op_point.iloc[0,:].copy()
        #print("temp_df_real:\n", temp_df_real)
        
        
        # Hinzufügen zufälliger Variationen
        temp_df_real.iloc[3:] += np.random.uniform(-variation_range_real, variation_range_real, size=data_real_op_point.shape[0])      
        #temp_df_real=data_real_op_point.iloc[0,:3].append(temp_df_real, ignore_index=True)
        #temp_df_real=temp_df_real.transpose()
        #print("temp_df_real_variation:\n", temp_df_real)
        
        # Hinzufügen der variierten Daten zum erweiterten Datensatz
        augmented_data_real = augmented_data_real.append(temp_df_real, ignore_index=True)
        
    #print("augmented data_real:\n", augmented_data_real)
    return augmented_data_real
    

 
# Generiere Datensätze für Imaginärpart          
def augment_data_imag(data_imag_op_point, number_data, variation_range_imag):
    augmented_data_imag = pd.DataFrame()
    print("augemented data_imag:\n", augmented_data_imag)
    
    for _ in range(number_data):
        
        # Erstellen einer Kopie der originalen Daten
        temp_df_imag = data_imag_op_point.iloc[0,3:].copy()
        
        # Hinzufügen zufälliger Variationen
        temp_df_imag += np.random.uniform(-variation_range_imag, variation_range_imag, size=data_imag_op_point.shape[0])
        #temp_df_real=temp_df['real'].transpose()
        
        #print("tem_df_real:\n", temp_df_real)

    
        # Hinzufügen der variierten Daten zum erweiterten Datensatz
        augmented_data_imag = augmented_data_imag.append(temp_df_imag, ignore_index=True)
        
    #print("augmented data:\n", augmented_data)
    return  augmented_data_imag  



def generate_variated_datasets():
    
    data_base=pd.DataFrame()

    number_of_rows=len(data_real)
    i=0
    print("number of rows: ", number_of_rows)
    #exit()
    while i<=(number_of_rows-1):

        # Realpart
        copy_real = pd.DataFrame(data_real.iloc[i,:]).transpose()
        data_real_op_point = copy_real.dropna()
        print("\ndata_real_op_point:\n", data_real_op_point)
        
        # Imagpart 
        #copy_imag = data_imag.where((data_real.RH_Air == rel_air_humi) & (data_real.Lambda == Lambda) & (data_real.Temperature == Temp))
        #data_imag_op_point = copy_imag.dropna()      
        #print("\ndata_imag_op_point:\n", data_imag_op_point)       

        
        # Generieren von augmentierten Daten
        augmented_data_real = augment_data_real(data_real_op_point, number_data, variation_range_real)   # Anzahl Datensätze, Variation Range (siehe Codeanfang)
        #augmented_data_imag = augment_data_imag(data_imag_op_point, number_data, variation_range_imag)   # Anzahl Datensätze, Variation Range (siehe Codeanfang)

        # Hinzufügen der generierten/augmentierten Daten zur Datenbasis
        data_base = data_base.append(augmented_data_real, ignore_index=True)

        print("\naugmented_data_real: ",augmented_data_real)
        print("\ni: ",i)
                
        i+=1            

 
    print("\nData Base:\n", data_base)

    #exit()

    # Speichern der Datenbasis
    data_base.to_csv(f'./HMS_Classification/Nyquist_Data_classification_extended_Real_part_TBD.csv', index=False)
    


    #print("\nData Base:\n", data_base)
    #print("\naugmented data imag:\n", augmented_data_imag)
     
    print("\nDatensätze wurden erfolgreich generiert.\n")

   
generate_variated_datasets()


#==============================================================# Ende: Funktion zum Generieren variierter Datensätze -  DATENSÄTZE  RE UND IMAG als Funktion der Frequenz================================