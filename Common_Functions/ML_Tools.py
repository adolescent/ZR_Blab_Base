'''
These functions will read monkeylogic generated graph matrix,return it into a python-readable dict.


'''
#%%

from OS_Tools import *
import numpy as np
import copy 


def ML_Meta(data_bytes,decoder = 'utf-8'):
    # unpack var name
    # print('ping')
    # print(f'Current Length:{len(data_bytes)}')
    rest_bytes,name_len = Bin_Unpack(data_bytes,8,1,'Q')
    name_len=name_len[0]
    if name_len != 0:
        rest_bytes,var_name = Bin_Unpack(rest_bytes,name_len,name_len,'s')
        var_name = var_name[0].decode(decoder)
    else:
        var_name = ''

    # unpack var type,b'dobule'/b'struct'/b'cell'
    rest_bytes,var_type_len = Bin_Unpack(rest_bytes,8,1,'Q')
    var_type_len = var_type_len[0]
    rest_bytes,var_type = Bin_Unpack(rest_bytes,var_type_len,var_type_len,'s')
    var_type = var_type[0].decode(decoder)

    # unpack var dim, will return a turple.
    rest_bytes,var_dim_len = Bin_Unpack(rest_bytes,8,1,'Q')
    var_dim_len = var_dim_len[0]
    rest_bytes,var_dim = Bin_Unpack(rest_bytes,8*var_dim_len,var_dim_len,'Q')

    # print(var_type)
    return rest_bytes,var_name,var_type,var_dim


def ML_Double_Unpack(data_bytes,var_dim):
    # total num of vars
    product = lambda t: eval('*'.join(map(str, t)))
    var_len = product(var_dim)
    # unpack target length of double.
    rest_bytes,double_data = Bin_Unpack(data_bytes,8*var_len,var_len,'d')
    double_var = np.array(double_data)
    # then reshape double data into column major order.
    double_var = double_var.reshape(var_dim,order='F') # F as column major order.
    return rest_bytes,double_var

def Char_Bit_Checker(data, char_length):
    # 读取固定字节长度（最大可能长度 = 字符数 * 4）
    max_byte_length = char_length * 4
    unpacked = struct.unpack(f"{max_byte_length}s", data[:max_byte_length])[0]
    
    # 检测有效汉字字节序列 (UTF-8模式)
    valid_bytes = bytearray()
    char_count = 0
    i = 0
    
    while i < len(unpacked) and char_count < char_length:
        byte = unpacked[i]
        
        # ASCII字符 (0-127)
        if byte <= 0x7F:
            valid_bytes.append(byte)
            i += 1
            char_count += 1
        
        # 2字节字符 (部分汉字)
        elif 0xC0 <= byte <= 0xDF:
            if i + 1 < len(unpacked):
                valid_bytes.extend(unpacked[i:i+2])
                i += 2
                char_count += 1
        
        # 3字节字符 (大多数汉字)
        elif 0xE0 <= byte <= 0xEF:
            if i + 2 < len(unpacked):
                valid_bytes.extend(unpacked[i:i+3])
                i += 3
                char_count += 1
        
        # 4字节字符 (扩展汉字)
        elif 0xF0 <= byte <= 0xF7:
            if i + 3 < len(unpacked):
                valid_bytes.extend(unpacked[i:i+4])
                i += 4
                char_count += 1
        
        # 无效字节或结束
        else:
            break
    
    # 转换为字符串并返回
    # return bytes(valid_bytes).decode('utf-8', errors='ignore'), char_count
    return len(valid_bytes)


def ML_Char_Unpack(data_bytes,var_dim,decoder = 'utf-8'):

    product = lambda t: eval('*'.join(map(str, t)))
    length = product(var_dim) # warning, length of chars, not length of bytes, some utf char will cause trouble!
    real_bytes = Char_Bit_Checker(data_bytes,length)

    rest_bytes,char_string = Bin_Unpack(data_bytes,real_bytes,real_bytes,'s')
    char_string = char_string[0].decode(decoder)
    return rest_bytes,char_string

def ML_Logical_Unpack(data_bytes,var_dim):
    product = lambda t: eval('*'.join(map(str, t)))
    length = product(var_dim)
    rest_bytes,c_bool = Bin_Unpack(data_bytes,length,length,'?')
    return rest_bytes,c_bool

def ML_Cell_Unpack(data_bytes,var_dim):
    product = lambda t: eval('*'.join(map(str, t)))
    cell_dict = {}
    cell_dict['Original_Shape'] = var_dim
    # cycle each element of cell.
    length = product(var_dim)
    rest_bytes = data_bytes
    
    for i in range(length):
        # cell has no name, so skip it's name uint64 char.
        rest_bytes,var_name,field_type,field_dim = ML_Meta(rest_bytes)
        if field_type == 'char':
            rest_bytes,cell_var = ML_Char_Unpack(rest_bytes,field_dim)
        elif field_type == 'double':
            rest_bytes,cell_var = ML_Double_Unpack(rest_bytes,field_dim)
        elif field_type == 'function_handle': # skip it. keep its location
            rest_bytes,cell_var = ML_Char_Unpack(rest_bytes,field_dim)
        elif field_type == 'struct':
            rest_bytes,cell_var = ML_Struck_Unpack(rest_bytes,field_dim)
        else:
            print('Cell_Unpack_Error')
            print(field_type)
            raise ValueError('Invalid Data Type !')
        
        cell_dict[i]= cell_var

    return rest_bytes,cell_dict

def ML_Struck_Unpack(data_bytes,var_dim):
    '''
    Unpack struct, save each field into a dict. Struct is a little different, so make sure this is a real struct.
    '''
    rest_bytes,N_field = Bin_Unpack(data_bytes,8,1,'Q')
    # print(N_field)
    N_field = N_field[0]
    # cycle each struck var, and for each var, cycle its field.
    struct_dict = {}
    struct_dict['Orignial_Shape'] = var_dim
    product = lambda t: eval('*'.join(map(str, t)))
    struct_len = product(var_dim)
    for i in range(struct_len):
        struct_dict[i]={} # generate sub-field data type for reading.
        for j in range(N_field):
            # print(len(rest_bytes))
            rest_bytes,field_name,field_type,field_dim = ML_Meta(rest_bytes)
            # print(field_name)
            # print(field_dim)
            # print(field_type)
            if field_type == 'char':
                rest_bytes,field_var = ML_Char_Unpack(rest_bytes,field_dim)
            elif field_type == 'double':
                rest_bytes,field_var = ML_Double_Unpack(rest_bytes,field_dim)
            elif field_type == 'cell':
                rest_bytes,field_var = ML_Cell_Unpack(rest_bytes,field_dim)
            elif field_type == 'logical':
                rest_bytes,field_var = ML_Logical_Unpack(rest_bytes,field_dim)
            elif field_type == 'struct':
                rest_bytes,field_var = ML_Struck_Unpack(rest_bytes,field_dim)
            elif field_type == 'function_handle': # skip it. keep its location
                print('Aha!')
                rest_bytes,field_var = ML_Char_Unpack(rest_bytes,field_dim)
            else:
                print('Struck_Unpack_Error')
                raise ValueError('Invalid Data Type !')
            struct_dict[i][field_name] = field_var
    # print('End of struct.')
    return rest_bytes,struct_dict

#%% Testrun part
if __name__ == '__main__':

    with open(r'D:\#Data\Loc_Example\Example_Data\241026_MaoDan_YJ_WordLOC.bhv2', 'rb') as file:
        data_bytes = file.read()

    rest_bytes = data_bytes
    #%% test third var
    rest_bytes,var_name,var_type,var_dim = ML_Meta(rest_bytes) 
    print(f'Current var type:{var_type}')
    if var_type == 'double':
        rest_bytes,var = ML_Double_Unpack(rest_bytes,var_dim)
    elif var_type == 'struct':
        rest_bytes,var = ML_Struck_Unpack(rest_bytes,var_dim)

#%% bug testing
    test_bytes = data_bytes[-11627284:]
    decoder = 'utf-8'
    rest_bytes,var_name,var_type,var_dim = ML_Meta(test_bytes)

    # rest_bytes,var_name,var_type,var_dim = ML_Meta(test_bytes) 
    buggy_start = copy.deepcopy(rest_bytes)
    test_len = 16
    # rest_bytes,var = ML_Char_Unpack(rest_bytes,var_dim)
    rest_bytes,char_string = Bin_Unpack(buggy_start,test_len,test_len,'s')

    print(char_string[0].decode('utf-8'))
