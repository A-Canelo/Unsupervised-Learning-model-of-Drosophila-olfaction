# This code tries to reproduce the results from the paper:
# P. Arena, L. Patané and P. S. Termini, "An insect brain inspired neural model for object representation
# and expectation," The 2011 International Joint Conference on Neural Networks

# Angel Canelo 2021.03.06
###### import ######################
from brian2 import *
####################################
prefs.codegen.target = 'numpy'  # this makes the simulation faster
######## Auxiliar functions ########
# Visualize synapse connections
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

# Visualize the neurons in MB layer
def visualise_layer(G, S, rows, cols, grid_dist):
    figure()
    for color in ['y']:
        plot(G.x_grid[S.j[:, :]] / umeter, G.y_grid[S.j[:, :]] / umeter, color + '.')     # plotting every dot of the grid
    for color in ['b']: #for color in ['g', 'b', 'm']:
        #neuron_idx = np.random.randint(0, rows*cols)
        neuron_idx = 40 # showing the neuron in the center of the lattice
        plot(G.x_grid[neuron_idx] / umeter, G.y_grid[neuron_idx] / umeter, 'o', mec=color, mfc='none')
        plot(G.x_grid[S.j[neuron_idx, :]] / umeter, G.y_grid[S.j[neuron_idx, :]] / umeter, color + '.')   # plotting neighbours
    xlim((-rows / 2.0 * grid_dist) / umeter, (rows / 2.0 * grid_dist) / umeter)
    ylim((-cols / 2.0 * grid_dist) / umeter, (cols / 2.0 * grid_dist) / umeter)
    title('MB layer')
    xlabel('x')
    ylabel('y', rotation='horizontal')
    axis('equal')
    tight_layout()
####################################
### Izhikevich model parameters ####
a = 0.02
b = -0.1
c = -55
d = 6
tau = 10*ms

values = zeros((1000,16))
values[:,0] = 70
values[:,5] = 70
values[:,8] = 70
values[:,15] = 70
I_ext = TimedArray(values, dt=1*ms)     # External input current
# AL neurons
eqs_AL = '''
dv/dt = (0.04*v**2+5*v+140-u+ I_ext(t,i) + IAL_syn_exc + IAL_syn_inh + IMB_AL)/tau : 1
du/dt = (a*(b*v-u))/tau: 1
IAL_syn_exc : 1
IAL_syn_inh : 1
IMB_AL : 1
'''
# MB neurons
eqs_MB = '''
dv/dt = (0.04*v**2+5*v+140-u+ IAL + IMB_syn_exc + IMB_syn_inh + IMB_del)/tau : 1
du/dt = (a*(b*v-u))/tau: 1
IMB_syn_exc : 1
IMB_syn_inh : 1
IAL : 1
IMB_del : 1
x_grid : meter
y_grid : meter
'''
#####################################
####### synapse model parameters ####
delta_w_AL = 2
delta_w_MBAL = 2 # original value 1
Apre = 5         # original value 0.05
Apost = -5       # original value -0.05
taupre = 20*ms   # original value 20*ms
taupost = 20*ms  # original value 20*ms
wmax = 100
wmax_inh = -30
tau_I = 20*ms       # original value 5*ms
tau_near = 10*ms    # original value 1*ms
tau_far = 20*ms     # original value 1*ms
tau_ALMB = 20*ms    # original value 1*ms
tau_fast = 20*ms    # original value 1*ms
tau_del = 400*ms
#####################################
start_scope()
######### Layers of neurons #########
groups, g_neurons = 4, 4
AL = NeuronGroup(groups*g_neurons, eqs_AL, threshold='v>=30', reset='v = c; u = u + d', method='euler')  # Antennal Lobe layer 4x4
rows, cols = 9, 9
MB = NeuronGroup(rows*cols, eqs_MB, threshold='v>=30', reset='v = c; u = u + d', method='euler')  # Mushroom Body layer 9x9
grid_dist = 0.5*umeter   #distance between neurons
MB.x_grid = '(i // rows) * grid_dist - rows/2.0 * grid_dist'
MB.y_grid = '(i % rows) * grid_dist - cols/2.0 * grid_dist'
#####################################
######### Synapses ##################
# AL synapses (synapse model is alpha synapse)
SAL_exc = Synapses(AL, AL, ''' w_exc : 1
depsi/dt = (x-epsi)/tau_I : 1 (clock-driven)
dx/dt = -x/tau_I : 1 (clock-driven)
''',
on_pre='''
IAL_syn_exc = epsi
w_exc += delta_w_AL
w_exc = clip(w_exc, 0, wmax)
x = w_exc
''', method='euler')
SAL_exc.connect(condition='(i<=3 and j>3) or ((i>3 and i<=7) and (j>7 or j<4)) or ((i>7 and i<=11) and (j>11 or j<8)) or ((i>11 and i<=15) and j<12)')
#visualise_connectivity(SAL_exc)

SAL_inh = Synapses(AL, AL, '''w_inh : 1
depsi/dt = (x-epsi)/tau_I : 1 (clock-driven)
dx/dt = -x/tau_I : 1 (clock-driven)
''', on_pre='''
IAL_syn_inh = epsi
x = w_inh
''', method='euler')
SAL_inh.connect(condition='i!=j and ((i<=3 and j<=3) or ((i>3 and i<=7) and (j>3 and j<=7)) or ((i>7 and i<=11) and (j>7 and j<=11)) or ((i>11 and i<=15) and (j>11 and j<=15)))')
#visualise_connectivity(SAL_inh)

# MB synapses
SMB_exc = Synapses(MB, MB, '''
w_exc : 1
depsi/dt = (x-epsi)/tau_near : 1 (clock-driven)
dx/dt = -x/tau_near : 1 (clock-driven)
''', on_pre='''
IMB_syn_exc = epsi
x = w_exc
''', method='euler')
distance = 1*umeter
SMB_exc.connect(condition='sqrt((x_grid_pre - x_grid_post)**2 + (y_grid_pre - y_grid_post)**2) < distance')
visualise_layer(MB, SMB_exc, rows, cols, grid_dist)     # visualize the MB layer topology

SMB_inh = Synapses(MB, MB, '''
w_inh : 1
depsi/dt = (x-epsi)/tau_far : 1 (clock-driven)
dx/dt = -x/tau_far : 1 (clock-driven)
''', on_pre='''
IMB_syn_inh = epsi
x = w_inh
''', method='euler')
SMB_inh.connect(condition='sqrt((x_grid_pre - x_grid_post)**2 + (y_grid_pre - y_grid_post)**2) > distance')

# AL with MB synapses
S_AL_MB = Synapses(AL, MB, '''w_exc : 1
depsi/dt = (x-epsi)/tau_fast : 1 (clock-driven)
dx/dt = -x/tau_fast : 1 (clock-driven)
''', on_pre='''
IAL = epsi
x = w_exc
''', method='euler')
S_AL_MB.connect(p=0.25)     # connecting layer 1 and 2 pairs with prob 0.25
#visualise_connectivity(S_AL_MB)

# MB delayed synapses
SMB_del = Synapses(MB, MB, '''
w_exc : 1
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
depsi/dt = (x-epsi)/tau_del : 1 (clock-driven)
dx/dt = -x/tau_del : 1 (clock-driven)
''', on_pre='''
IMB_del = epsi
apre += Apre
w_exc = clip(w_exc*0.999+apost, 0, wmax)
x = w_exc
''',
on_post='''
apost += Apost
w_exc = clip(w_exc*0.999+apre, 0, wmax)
x = w_exc
''', method='euler')
SMB_del.connect(p = 1)    # connect neuron i with all neurons j
#visualise_connectivity(SMB_del)
# Feedback connections from MB to AL
S_MB_AL = Synapses(MB, AL,'''
w_exc : 1
depsi/dt = (x-epsi)/tau_fast : 1 (clock-driven)
dx/dt = -x/tau_fast : 1 (clock-driven)
''', on_pre='''
IMB_AL = epsi
w_exc += delta_w_MBAL
w_exc = clip(w_exc, 0, wmax)
x = w_exc
''', method='euler')
S_MB_AL.connect(p=1)    # connect neuron i with all neurons j
#visualise_connectivity(S_MB_AL)
#####################################
############ Monitoring #############
potential_AL = StateMonitor(AL, 'v', record=True)
potential_MB = StateMonitor(MB, 'v', record=True)
weights = StateMonitor(SAL_exc, ['w_exc', 'IAL_syn_exc'], record=True)
weights_inh = StateMonitor(SAL_inh, ['w_inh', 'IAL_syn_inh'], record=True)
w_MB = StateMonitor(SMB_del, ['w_exc', 'IMB_del'], record=True)
spikemon = SpikeMonitor(AL)
#####################################
########### Initial values ##########
T = 1200*ms     # simulation time
AL.v = -70
MB.v = -70
SAL_exc.w_exc = 0
SAL_inh.w_inh = -3
SMB_exc.w_exc = 50   # original value 5
SMB_inh.w_inh = -30  # original value -3
S_AL_MB.w_exc = 100   # original value 10
SMB_del.w_exc = 0
S_MB_AL.w_exc = 0
#####################################
run(T)    # Simulation starts

########### Plotting ################
# plotting AL activity
kk = 0
fig, axs = plt.subplots(groups, g_neurons)
for i in range(groups):
    for j in range(g_neurons):
        axs[j,i].plot(potential_AL.t/ms, potential_AL.v[kk])
        axs[j,i].set_title('neuron ind: %i' %(i+1) + ',%i' %(j+1))
        axs[j,i].get_xaxis().set_visible(False)
        kk += 1

# figure(); clf(); subplot(411)
# plot(potential_AL.t/ms, potential_AL.v[0])
# xlabel('Time (ms)')
# ylabel('v')
# subplot(412)
# plot(weights.t/ms, reshape(weights[SAL_exc[0,5]].IAL_syn_exc,-1))
# subplot(413)
# plot(weights.t/ms, reshape(weights[SAL_exc[0,5]].w_exc,-1))
# subplot(414)
# plot(weights_inh.t/ms, reshape(weights_inh[SAL_inh[0,2]].w_inh,-1))

figure(); clf()
plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time (ms)'); xlim((0, 1000))
ylabel('Neuron index'); ylim((0, 15))

# plotting heat map of AL
h_map_pre = zeros(16)
for i in range(16):
    h_map_pre[i] = mean(potential_AL.v[i])
h_map = reshape(h_map_pre, (4,4))
h_map_org = array([h_map_pre[[0,1,4,5]],
                   h_map_pre[[2,3,6,7]],
                   h_map_pre[[8,9,12,13]],
                   h_map_pre[[10,11,14,15]]])
figure(); clf()
pos = imshow(h_map_org, cmap='jet')
cbar = colorbar(pos, label = 'AL mean potential (mV)')

# plotting MB activity
# figure(); clf(); subplot(311)
# plot(potential_AL.t/ms, potential_MB.v[25])
# xlabel('Time (ms)')
# ylabel('v')
# print(weights[SAL_exc].IAL_syn_exc.shape)
# print(weights_inh[SAL_inh].IAL_syn_inh.shape)
# print(w_MB[SMB_del].IMB_del.shape)
# subplot(312)
# plot(w_MB.t/ms, swapaxes(w_MB[SMB_del[25,1]].IMB_del, 0,1))
# subplot(313)
# plot(w_MB.t/ms, swapaxes(w_MB[SMB_del[25,1]].w_exc, 0,1))

# plotting heat map of MB
h_map_MB = zeros(rows*cols)
for i in range(rows*cols):
    h_map_MB[i] = mean(potential_MB.v[i])
h_map_MB_reshaped = reshape(h_map_MB, (rows,cols))
figure(); clf()
pos = imshow(h_map_MB_reshaped, cmap='jet')
cbar = colorbar(pos, label = 'MB mean potential (mV)')

# MB mean weight values matrix
# figure(); clf()
# w_map_MB = zeros((rows,cols))
# for i in range(rows):
#     for j in range(cols):
#         w_map_MB[i,j] = mean(w_MB[SMB_del[i,j]].w_exc)
# pos = imshow(w_map_MB); colorbar(pos, label='MB mean weight %')
# print(w_MB.w_exc.shape)

# AL learned weights matrix
figure(); clf()
wAL_learned = np.zeros((len(AL), len(AL)))
wAL_learned[SAL_exc.i[:], SAL_exc.j[:]] = SAL_exc.w_exc[:]
pos = imshow(wAL_learned); colorbar(pos, label='AL learned weights %')

# MB learned weights matrix
figure(); clf()
wMB_learned = np.zeros((len(MB), len(MB)))
wMB_learned[SMB_del.i[:], SMB_del.j[:]] = SMB_del.w_exc[:]
pos = imshow(wMB_learned); colorbar(pos, label='MB learned weights %')
show()
#####################################