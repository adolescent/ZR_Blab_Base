'''
A small test for monkey logic bhv2 file reading

'''

#%%

import os
import struct


with open(r'D:\#Data\Loc_Example\Example_Data\241026_MaoDan_YJ_WordLOC.bhv2', 'rb') as file:
    data_bytes = file.read()


c_var_name_len = struct.unpack('Q',data_bytes[:8])[0]
c_var_name = struct.unpack(f'{c_var_name_len}s',data_bytes[8:8+c_var_name_len])
cloc = 8+c_var_name_len

c_var_type_len = struct.unpack('Q',data_bytes[cloc:cloc+8])[0]
cloc+=8
c_var_type = struct.unpack(f'{c_var_type_len}s',data_bytes[cloc:cloc+c_var_type_len])
cloc+=c_var_type_len
c_var_len = struct.unpack('Q',data_bytes[cloc:cloc+8])[0]
cloc+=8
c_var = struct.unpack(f'{c_var_len}Q',data_bytes[cloc:cloc+c_var_len*8])
cloc+=c_var_len*8

def Read_Var(bytes):
    # first 8 bit, Length of var name
    c_var_name_len = struct.unpack('Q',bytes[:8])[0]
    var_name = struct.unpack(f'{c_var_name_len}s',data_bytes[8:8+c_var_name_len])
    cloc = 8+c_var_name_len
    # then 8 bit of var type
    c_var_type_len = struct.unpack('Q',data_bytes[cloc:cloc+8])[0]
    cloc+=8
    c_var_type = struct.unpack(f'{c_var_type_len}s',data_bytes[cloc:cloc+c_var_type_len])
    cloc+=c_var_type_len
    # then vars saved in uint 64.
    c_var_len = struct.unpack('Q',data_bytes[cloc:cloc+8])[0]
    cloc+=8
    c_var = struct.unpack(f'{c_var_len}Q',data_bytes[cloc:cloc+c_var_len*8])
    cloc+=c_var_len*8
    rest_bytes = bytes[cloc:]
    # then read the real data here.
    if c_var_type == b'double':
        pass
    elif c_var_type == b'struct':
        pass 
    elif c_var_type == b'cell':
        pass

    return var_name,c_var_type,c_var,rest_bytes

