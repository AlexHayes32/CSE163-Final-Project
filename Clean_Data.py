'''
CSE 163 Final Project
Alex Hayes
Sedona Munguia
Loads in datasets, alters them for analysis in Main.Py
'''


import pandas as pd

avocado = pd.read_csv('Dataavocado.csv')
cities = pd.read_csv('uscities.csv')
    

def main():
    '''
    Runs the program
    '''
    load_in_data()



if __name__ == '__main__':
    main()