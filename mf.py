import sys
import numpy as np
import random
import time
from pyspark import SparkContext, SparkConf

def initialize_matrix():
    '''
    Initialize matrix to be factorized, a list of tupe.

    The tuple consists of two parts, an identifier, and numpy array
    Identifier has three components: dimension(x or y), block_id, and in_block_index. 
    They are concatenated using hyphon. The numpy array is of size K, type float.
    '''

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

def map_to_target_value(line):
    '''
    Generate wanted user-movie-rating value from input file.

    Input: 
    line: line from original file

    Output:
    a tuple (block_id, [(x_in_block_index, y_in_block_index, rating_value)])
    '''

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

def map_to_block_id_row_data_list(row):
    '''
    Each row in matrix to be factorized would be mapped to multiple block in target matrix.
    In this way, we could parallelize calculation of difference in multiple parts.

    Input:
    row: a tuple of a line in matrix to be factorized
    e.g. ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))

    Output:
    a tuple of line that could be matched with a block in target matrix
    e.g. (100, ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])))
    '''

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

def sdg_once(block_info):
    '''
    Use SDG algorithm to calculate and update matrix.

    Input:
    block_info: a tuple, key is block_id, value is a tuple pair, whose first element is a
    block of target matrix, and second element is a block of matrix to be factorized. Notice
    both blocks are represented using iterator.
    e.g. (0, (<pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00610>, <pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00b10>))
    e.g. target block: list of (x_block_index, y_block_index, rating_value)
    e.g. matrix block: list of ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))

    Output: the block of matrix to be factorized
    e.g. ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
    '''

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
        #matrix_info[x_id] = (matrix_info[x_id][0], x_row)
        #matrix_info[x_block_dim + y_id] = (matrix_info[x_block_dim + y_id][0], y_row)
    return matrix_info

def calculate_error(block_info):
    '''
    Calculate square error.

    Input:
    block_info: a tuple, key is block_id, value is a tuple pair, whose first element is a
    block of target matrix, and second element is a block of matrix to be factorized. Notice
    both blocks are represented using iterator.
    e.g. (0, (<pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00610>, <pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00b10>))
    e.g. target block: list of (x_block_index, y_block_index, rating_value)
    e.g. matrix block: list of ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))

    Output: a tuple, key is 0 (set to constant so that we could aggregate all these error later),
    value is tuple pair, first element is square error, second element is element number
    '''

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

def sdg_merge(line1, line2):
    '''
    Add two SDG result numpy array.

    Input:
    line1: ('x-0-0', array([ 0.,  0.,  0.2,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
    line2: ('x-0-0', array([ 0.,  0.1,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))

    Output: 
    ('x-0-0', array([ 0.,  0.1,  0.2,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])) 
    '''
    if not line1:
        return line2
    result1 = line1
    result2 = line2
    return result1+result2

def sdg_compute(target_value, total_list):
    '''
    Use SDG algorithm to compute and update matrix to be factorized.
    It pairs target matrix and matrix to be factorized, to matching them using block_id.
    Then, within each block, it update the matrix to be factorized using SDG algorithm.
    Finally, it merged all matrix together, to gather the difference and error.

    Input:
    target_value: list of tuple (block_id, [(x_in_block_index, y_in_block_index rating_value)])
    total_list: list of tuple (block_id, ('dim-block_id(within each dimension)-block_index', array))
    e.g. (0, ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])))

    Output: (error, total_list)
    error: (0, (square error, total element number)), e.g. (0, (856589.5871305099, 1000209))
    total_list: same as input
    '''

    # e.g. (0, (<pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00610>, <pyspark.resultiterable.ResultIterable object at 0x7fb0d6d00b10>))
    block_id_total = target_value.cogroup(total_list).persist() 
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
    .flatMap(sdg_once) #('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
    .reduceByKey(sdg_merge) #('x-73-2', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0., 0.]))
    .flatMap(map_to_block_id_row_data_list)
    )
    return error, total_list

def main():
    '''
    Launch guide:

    On a machine/cluster with Spark installed.
    Use following command to start the program.
    ./bin/pyspark --master MASTER --py-files mf.py
    where MASTER set to 'local[*]' for local debug, set to 'yarn' for client or cluster mode
    for more info on this setting, see: http://spark.apache.org/docs/latest/submitting-applications.html

    Optimization guide:

    One important parameter for parallel collections is the number of partitions to cut the dataset into.
    Spark will run one task for each partition of the cluster. Typically you want 2-4 partitions for each
    CPU in your cluster. Normally, Spark tries to set the number of partitions automatically based on your
    cluster. However, you can also set it manually by passing it as a second parameter to parallelize 
    (e.g. sc.parallelize(data, 10)).
    '''
    conf = SparkConf().setAppName('matrix_factorization')
    sc = SparkContext(conf=conf)

    # parameters
    K = 30 #int(sys.argv[2])
    N = 1 #int(sys.argv[3])
    eta = 0.001
    eta_decay = 0.99
    PARTITION_NUM = 2 * 4 * 2

    fileName = 'ratings_1M.csv' #sys.argv[1]
    f = sc.textFile('hdfs:///input/%s' % fileName, PARTITION_NUM)
    
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
    
    #(block_id, [(x_block_index, y_block_index rating_value)])
    target_value = (f
    .map(map_to_target_value)
    .persist()
    )
    
    # (block_id(global), ('dim-block_id(within each dimension)-block_index', array))
    # (0, ('x-0-0', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.])))
    total_list = (sc
    .parallelize(initialize_matrix())
    .flatMap(map_to_block_id_row_data_list)
    )
    
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

