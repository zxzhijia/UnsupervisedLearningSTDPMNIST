'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
#import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
from struct import unpack
from brian2 import *
from brian2tools import *
# specify the location of the MNIST data
MNIST_data_path = '/home/xu/Downloads/BMEClassProjectMaterials/stdp-mnist-master_brian2/'

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input                
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
#         connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName][:]
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in xrange(n_e):#
                connection[:,j] *= colFactors[j]
            
def get_2d_input_weights(weightMatrix):
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = connections[name][:]
    weight_matrix = weightMatrix#np.copy(connMatrix)
        
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights(weightMatrix):
    name = 'XeAe'
    weights = get_2d_input_weights(weightMatrix)
    fig = figure(fig_num, figsize = (18, 18))
    im2 = imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    colorbar(im2)
    title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig
    
def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    ylim(ymax = 100)
    title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance
    
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
    
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start


#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

#b.set_global_preferences( 
#                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
#                        useweave = True, # Defines whether or not functions should use inlined compiled C code where defined.
#                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
#                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on 
#                        usecodegen = True,  # Whether or not to use experimental code generation support.
#                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
#                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
#                        usecodegenthreshold = False,  # Whether or not to use experimental code generation support on thresholds.
#                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
#                        usecstdp = True,  # Whether or not to use experimental new C STDP.
#                       ) 


np.random.seed(0)
data_path = './'
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'  
    num_examples = 60000 * 3
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:    
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True


ending = ''
n_input = 784
n_e = 400
n_i = n_e 
single_example_time =   0.35 * second #
resting_time = 0.15 * second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:    
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * mV 
v_rest_i = -60. * mV 
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. *mV
v_thresh_i = -40. * mV
refrac_e = 5 * ms
refrac_i = 2 * ms

conn_structure = 'dense'
weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*ms,10*ms)
delay['ei_input'] = (0*ms,5*ms)
input_intensity = 0.
start_input_intensity = input_intensity

tc_pre_ee = 20*ms
tc_post_1_ee = 20*ms
tc_post_2_ee = 40*ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

if test_mode:
    scr_e = '''v = v_reset_e
    timer = 0*ms'''    #old version scr_e = 'v = v_reset_e; timer = 0*ms' 
else:
    tc_theta = 1e7 * ms
    theta_plus_e = 0.05 * mV
    scr_e = '''v = v_reset_e
               theta += theta_plus_e
               timer = 0*ms'''
offset = 20.0*mV
#v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer/ms>refrac_e/ms)'
#v_thresh_e = 'v>(theta - offset + ' + str(v_thresh_e) + ')'
#v_thresh_e = '(timer/ms>refrac_e/ms)'
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 100.0  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1
            '''
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1;w=clip(w-nu_ee_pre,0,wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.;w=clip(w+nu_ee_post,0,wmax_ee)'
    
ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold='v>theta-offset+v_thresh_e', refractory= refrac_e, reset= scr_e,method='euler')
neuron_groups['i'] = NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= 'v>v_thresh_i', refractory= refrac_i, reset= 'v=v_reset_i',method='euler')


#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------
groupindex=0 
for name in population_names:
    print 'create neuron group', name
    groupeindx=groupindex*n_e
    groupeindxend=groupindex*n_e+n_e
     
    groupiindx=groupindex*n_i
    groupiindxend=groupindex*n_i+n_i     

    neuron_groups[name+'e'] = neuron_groups['e'][groupeindx:groupeindxend]
    neuron_groups[name+'i'] = neuron_groups['i'][groupiindx:groupiindxend]

    groupindex+=1
    
    neuron_groups[name+'e'].v = v_rest_e - 40. * mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * mV
    if test_mode or weight_path[-8:] == 'weights/':
        tempload=np.load(weight_path + 'theta_' + name + ending + '.npy')
        neuron_groups['e'].theta = tempload*mV
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*mV
    
    print 'create recurrent connections'
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy')
        state='g'+conn_type[0] 
        connections[connName] = Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], model='w:1',on_pre=state+'+=w')
        #connections[connName].connect(i=[x for x in range(len(weightMatrix)) for _ in range(len(weightMatrix[0]))], j=[x for _ in range(len(weightMatrix)) for x in range(len(weightMatrix[0]))])
        connections[connName].connect()         
        weightmatrixtemp= weightMatrix
        connections[connName].w= weightmatrixtemp.reshape(len(weightMatrix)*len(weightMatrix[0]))
    
    if ee_STDP_on:
        if 'ee' in recurrent_conn_names:
            stdp_methods[name+'e'+name+'e'] = Synapses(connections[name+'e'+name+'e'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                           post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

    print 'create monitors for', name
    rate_monitors[name+'e'] = PopulationRateMonitor(neuron_groups[name+'e'])
    rate_monitors[name+'i'] = PopulationRateMonitor(neuron_groups[name+'i'])
    spike_counters[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])
    
    if record_spikes:
        spike_monitors[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = SpikeMonitor(neuron_groups[name+'i'])

if record_spikes:
    figure(fig_num)
    fig_num += 1
    ion()
    subplot(211)
    plot(spike_monitors['Ae'].t/ms, spike_monitors['Ae'].i)
    subplot(212)
    plot(spike_monitors['Ai'].t/ms, spike_monitors['Ai'].i)


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 
pop_values = [0,0,0]
print('create input population')
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = PoissonGroup(n_input, rates=0*Hz)
    rate_monitors[name+'e'] = PopulationRateMonitor(input_groups[name+'e'])

for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        state='g'+conn_type[0]        
        connections[connName] = Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],  model='w:1',on_pre=state+'+=w')
        #connections[connName].connect(i=[x for x in range(len(weightMatrix)) for _ in range(len(weightMatrix[0]))], j=[x for _ in range(len(weightMatrix)) for x in range(len(weightMatrix[0]))])
        connections[connName].connect()  
        min_delay, max_delay = delay[connType]
        connections[connName].delay = 'min_delay + (max_delay - min_delay)*rand()'    
        weightmatrixtemp= weightMatrix      
        connections[connName].w= weightmatrixtemp.reshape(len(weightMatrix)*len(weightMatrix[0])) 
        
    if ee_STDP_on:
        print 'create STDP for connection', name[0]+'e'+name[1]+'e'      
        connName2 = name[0] + 'e' + name[1] + 'e'
        stdp_methods[name[0]+'e'+name[1]+'e'] = Synapses(input_groups[connName2[0:2]], neuron_groups[connName2[2:4]], eqs_stdp_ee, on_pre = '''pre = 1.
                                                                                                                        w -= nu_ee_pre * post1
                                                                                                                        w=clip(w-nu_ee_pre,0.,wmax_ee)''', 
                                                       on_post = '''post2before = post2
                                                                  w += nu_ee_post * pre * post2before
                                                                  post1 = 1.
                                                                  post2 = 1.
                                                                  w=clip(w+nu_ee_post,0.,wmax_ee)''')

net = Network(collect())
net.add(connections)
net.add(spike_monitors)
net.add(rate_monitors)
net.add(input_groups)
net.add(neuron_groups)
net.add(spike_counters)
net.add(stdp_methods)
#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights(weightMatrix)
    fig_num += 1
if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0
net.run(0*ms)
j = 0
while j < (int(num_examples)/1000):
    print('start running')
    if test_mode:
        if use_testing_set:
            rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
        else:
            rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = rate*Hz
    print 'run number:', j+1, 'of', int(num_examples)
    net.run(single_example_time, report='text')
            
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
    
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    print(current_spike_count)
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0
        net.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 100 == 0 and j > 0:
            print 'runs done:', j, 'of', int(num_examples)
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                print 'Classification performance', performance[:(j/float(update_interval))+1]
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1


#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        subplot(len(rate_monitors), 1, i+1)
        plot(rate_monitors[name].t/second, rate_monitors[name].smooth_rate(window='flat', width=single_example_time+resting_time)/Hz, '.')
        title('Rates of population ' + name)
    
if spike_monitors:
    figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        subplot(len(spike_monitors), 1, i+1)
        plot(spike_monitors[name].t/ms,spike_monitors[name].i,'.')
        title('Spikes of population ' + name)
        
if spike_counters:
    figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        subplot(len(spike_counters), 1, i+1)
        plot(spike_counters['Ae'].count[:])
        title('Spike count of population ' + name)


brian_plot(connections['XeAe'])


plot_2d_input_weights(weightMatrix)
ioff()
show()



