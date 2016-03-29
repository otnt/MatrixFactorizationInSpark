import sys
import numpy as np
import random
import time

fileName = 'ratings_1M.csv' #sys.argv[1]
f = sc.textFile('hdfs:///input/%s' % fileName)

max_block_id = (f
# same key to send all lines to same place
.map(lambda line:(0, (int(line.split(',')[0]), int(line.split(',')[1]))))
# get maximum id
.reduceByKey(lambda a,b: (max(a[0],b[0]), max(a[1],b[1])))
)

# e.g. (6040, 3952)
max_block_id = max_block_id.collect()[0][1]

# e.g. 16, 100
K = 30 #int(sys.argv[2])
N = 1 #int(sys.argv[3])
eta = 0.001
eta_decay = 0.99

# x_block_dim, y_block_dim, K, N, eta
global constants_bc 
constants_bc = sc.broadcast([(max_block_id[0]+N)/N, (max_block_id[1]+N)/N, K, N, eta, eta_decay]) 

# initialize factorized matrix, x/y-N-inblockindex
def initialize_matrix():
    random.seed(1)
    x_block_dim = constants_bc.value[0]
    y_block_dim = constants_bc.value[1]
    K = constants_bc.value[2]
    N = constants_bc.value[3]
    total_list = []
    for n in range(0, N):
        for x in range(0, x_block_dim):
            new_list = []
            for k in range(0, K):
                new_list.append(random.random())
            total_list.append(('x-%d-%d' % (n, x), np.asarray(new_list, dtype=float)))
        for y in range(0, y_block_dim):
            new_list = []
            for k in range(0, K):
                new_list.append(random.random())
            total_list.append(('y-%d-%d' % (n, y), np.asarray(new_list, dtype=float)))
    return total_list

# input line from original file, output (block_id, [(x_id, y_id, rating_value)])
def map_to_target_value(line):
    x_block_dim = constants_bc.value[0]
    y_block_dim = constants_bc.value[1]
    N = constants_bc.value[3]
    inputs = line.split(',')
    x_id = int(inputs[0])
    y_id = int(inputs[1])
    value = (x_id % x_block_dim, y_id % y_block_dim, float(inputs[2]))
    x_block_id = (x_id/x_block_dim)
    y_block_id = (y_id/y_block_dim)
    block_id = x_block_id * N + y_block_id
    return (block_id, value)

# each row in matrix would be mapped to multiple block in target matrex
# input is ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
# output is (100, ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])))
def map_to_block_id_row_data_list(row):
    dimension = row[0].split('-')[0]
    index = int(row[0].split('-')[1])
    N = constants_bc.value[3]
    block_id_row_data_list = []
    for n in range(0, N):
        block_id = None
        if dimension is 'x':
            block_id = index * N + n
        else:
            block_id = n * N + index
        block_id_row_data_list.append((block_id, row))
    return block_id_row_data_list

# input is (0, (<pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00610>, <pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00b10>))
# each target_info is (x_block_index, y_block_index, rating_value)
# each matrix_info is ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
# output is ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
def sdg_go(block_info):
    x_block_dim = constants_bc.value[0]
    eta = constants_bc.value[4]
    target_info = list(block_info[1][0])
    matrix_info = list(block_info[1][1]) #matrix we're going to calculated
    matrix_info.sort(key=lambda x:(x[0].split('-')[0], int(x[0].split('-')[2])))
    for target in target_info:
        x_id = target[0]
        y_id = target[1]
        rating = target[2]
        x_row = matrix_info[x_id][1]
        y_row = matrix_info[x_block_dim + y_id][1]
        diff = rating - np.dot(x_row, y_row)
        W_gradient = -2.0 * diff * y_row
        H_gradient = -2.0 * diff * x_row
        x_row -= eta * W_gradient
        y_row -= eta * H_gradient
        matrix_info[x_id] = (matrix_info[x_id][0], x_row)
        matrix_info[x_block_dim + y_id] = (matrix_info[x_block_dim + y_id][0], y_row)
    return matrix_info

# input is (0, (<pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00610>, <pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00b10>))
def calculate_error(block_info):
    x_block_dim = constants_bc.value[0]
    target_info = list(block_info[1][0])
    matrix_info = list(block_info[1][1]) #matrix we're going to calculated
    matrix_info.sort(key=lambda x:(x[0].split('-')[0], int(x[0].split('-')[2])))
    error = 0.0
    for target in target_info:
        x_id = target[0]
        y_id = target[1]
        rating = target[2]
        x_row = matrix_info[x_id][1]
        y_row = matrix_info[x_block_dim + y_id][1]
        diff = rating - np.dot(x_row, y_row)
        error += diff ** 2
    return (0, (error, len(target_info)))

# input is ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
# output is ('x-73-2', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])) 
def sdg_merge(line1, line2):
    if not line1:
        return line2
    result1 = line1
    result2 = line2
    return result1+result2

#(block_id, [(x_block_index, y_block_index rating_value)])
target_value = (f
.map(map_to_target_value)
.cache()
)

# (block_id(global), ('dim-block_id(within each dimension)-block_index', array))
# (0, ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])))
total_list = (sc
.parallelize(initialize_matrix())
.flatMap(map_to_block_id_row_data_list)
)

def sdg_compute(target_value, total_list):
    # e.g. (0, (<pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00610>, <pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00b10>))
    block_id_total = target_value.cogroup(total_list).cache() 
    # e.g. (0, (856589.5871305099, 1000209))
    error = (
    block_id_total
    .map(calculate_error)
    .aggregateByKey(
        (0.0, 0),
        lambda a,b: (a[0]+b[0], a[1]+b[1]),
        lambda a,b: (a[0]+b[0], a[1]+b[1]),
        )
    )
    # ('73', ('x-73-2', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])))
    total_list = (
    block_id_total
    .flatMap(sdg_go) #('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
    .reduceByKey(sdg_merge) #('x-73-2', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0., 0.]))
    .flatMap(map_to_block_id_row_data_list)
    )
    return error, total_list

t1 = time.clock()
for i in range(0, 10):
    e, total_list = sdg_compute(target_value, total_list)
    eta = constants_bc.value[4]
    eta_decay = constants_bc.value[5]
    eta *= eta_decay
    constants_bc = sc.broadcast([(max_block_id[0]+N)/N, (max_block_id[1]+N)/N, K, N, eta, eta_decay]) 
    e = e.first()
    t2 = time.clock()
    print '*' * 100
    print (i, int(t2 - t1), e[1][0], np.sqrt(e[1][0]/ e[1][1]))
    print '*' * 100

