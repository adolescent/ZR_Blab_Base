'''

This stuct will pack single cell info of 

This class is usually generated from response matrix, 
you need to provide :
    - FOB 
    - raw response matrix
These response should be unceilied and un averaged, raw response must be provided.

Return will be a class object, using method can get cell's ceiling index, preference, etc...
NOTE : Response here will not indicate any stim-related info, so for specific stim set, you need to find it by yourself.
    
Input:

'''

class Cell_Info(object):

    name='Single cell property and response'

    def __init__(self):
        pass





#%% Testrun parts

if __name__ == '__main__':

    wp=r'E:\#Preprocessed_Data\GoodUnits'
    