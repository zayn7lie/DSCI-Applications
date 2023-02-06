import numpy as np
import xlrd
import ResNet

def rData(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [ [int(table.row_values(i,0,1)[0])] + table.row_values(i,-8) for i in range(1,table.nrows)]
    return np.array(data)
    #data: num, bool: NDGCAHMO

if __name__ == "__main__":
    print("finished")
