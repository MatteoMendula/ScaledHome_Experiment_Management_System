import simulations
import datasets
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import settings

if __name__ == '__main__':
    data_file_uri = 'data/mi_meteo_2001.csv'
    simulations.simulateRealDataTemperatures(3, data_file_uri)
    # simulations.performRandomActionsFixedDuration(3600*2, 60)