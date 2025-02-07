import numpy as np

def noisy_max(counter, sigma, num_classes):
    noisy_count_list = [0]*num_classes
    for i in range(num_classes) :
        if i in counter.keys() :
            noisy_count_list[i] = counter[i] + np.random.normal(0, sigma)
        else : 
            noisy_count_list[i] = np.random.normal(0, sigma)
    return np.argmax(noisy_count_list)

def per_query_epsilon(num_examples, test_size, epsilon) :
    return epsilon/(num_examples*test_size)