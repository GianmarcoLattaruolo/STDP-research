
def base_simulation(
        pars,
        spk_input, # input spike train, a numpy vector of shape (time_steps, N_pre)
        neuron_type, # class for the neuron type
        weight_rule = None, # class for the weights update
        N_pre = 100, # number of pre-synaptic neurons
        W_init = None, # initial weights
        neuron_params = {}, # parameters for the neuron model
        weight_update_params = {}, # parameters for the weight update
):
    N_post = 1 # this is the base simulation, just one output neuron
    num_steps = np.shape(spk_input)[0]

    # Initialize the post-syn neuron
    my_neuron = neuron_type(pars, **neuron_params)
    

    # Check if the given inital weight are valid
    if W_init is None:
        W_init = np.random.rand(N_post, N_pre)
    else:
        assert np.shape(W_init)[1] == N_pre, 'W must have the same number of rows as the number of pre-synaptic neurons'

    # Initialize the synapses with the given update rule
    if weight_rule is not None:
        my_synapses = weight_rule(pars, N_pre, N_post, W_init = W_init,**weight_update_params)
    else:
        W = W_init 

    # start the simulation
    for t in range(num_steps):
        
        # get the spikes from the pre-synaptic neurons at time t
        pre_syn_spikes = spk_input[t,:] 
        # get the weights from the synapse
        if weight_rule is not None:
            W = my_synapses.W
        
        # compute the input current for the postsynaptic neuron
        #I = np.dot(pre_syn_spikes, W[:,0])
        I = (W @ pre_syn_spikes)[0]

        # run the neuron model
        _ , spk = my_neuron.forward(I)

        # update the weights
        if weight_rule is not None:
            spikes = [pre_syn_spikes, spk]
            my_synapses.update_weights(spikes)

    if weight_rule is not None:
        return my_neuron, my_synapses
    else:
        return my_neuron
    







    
def LIF(
    pars,                        # parameters dictionary
    I_inj,                       # input current [pA]
    mem = None,                  # membrane potential [mV]
    refractory_time = False,     
    dynamic_threshold = False,   
    hard_reset = True,           
    noisy_input = False,
    **kwargs,
):
    """
    INPUTS:
    - pars: parameter dictionary
    - I_inj: input current [pA]
    - mem: membrane potential [mV]
    - refractory_time: boolean, if True the neuron has a refractory period
    - dynamic_threshold: boolean, if True the threshold is dynamic
    - hard_reset: boolean, if True the reset is hard
    - noisy_input: boolean, if True the input is noisy
    
    REUTRNS:
    - membrane potential at the next time step
    - spikes produced by the neuron (0 or 1)
    """
    
    #retrive parameters
    tau_m = pars.get('tau_m', 10)
    R = pars.get('R', 0.1)
    U_resting = pars.get('U_resting', -75)
    threshold = pars.get('threshold', -55)
    dt = pars.get('dt', 0.1)
    if mem is None:
        mem = pars.get('U_init', -75)
    if noisy_input:
        sigma = pars.get('sigma', 1)
        I_inj += np.random.normal(0, sigma, 1)

    #mem = mem + (dt/tau_m)*(U_resting-mem + I_inj*R)
    spk = mem > threshold
    if spk:
        if hard_reset:
            mem = pars['U_reset']
        else:
            mem = mem - (threshold-U_resting)

    mem = (1-dt/tau_m)*mem + dt/tau_m * U_resting + dt/tau_m  * R * I_inj

    return mem, int(spk)



def Poisson_neuron(pars, # parameters dictionary
        I_inj, # input current [pA] 
        mem = None, # membrane potential [mV]
        refractory_time = False,
        dynamic_threshold = False,
):
    """ Simulate a Poisson neuron

    INPUT:
    - pars                : parameter dictionary
    - I_inj               : input current [pA]
    - mem                 : membrane potential [mV]
    - refractory_time     : boolean, if True the neuron has a refractory period 
    - dynamic_threshold   : boolean, if True the neuron has a dynamic threshold

    RETURNS:
    - mem                 : membrane potential for the next time step
    - spk                 : output spike  (0 if no spike, 1 if spike)

    """

    # retrive parameters
    tau_m = pars.get('tau_m', 10)
    R = pars.get('R', 10)   
    U_resting = pars.get('U_resting', -75)
    threshold = pars.get('threshold', -55)
    dt = pars.get('dt', 0.1)
    alpha = pars.get('alpha', 0.1)
    if mem is None:
        mem = pars.get('U_init', -75)
        
    mem = mem + (dt/tau_m)*(U_resting-mem + I_inj*R)     
    rate = alpha * (mem - threshold)
    spk = 1.*  np.random.random() < rate 
    if spk:
        mem = pars['U_resting']

    return mem, spk





def STDP_rule(
        pars, # parameters dictionary
        W, # current weights
        traces, # STDP traces list [pre_synaptic, post_synaptic]  
        spikes, # Spikes [pre_synaptic, post_synaptic]
        **kwargs,
):
    """
    INPUTS:
    - pars    : parameters dictionary
    - W       : current weights of shape: (N_post, N_pre)
    - traces  : STDP traces list [pre_synaptic, post_synaptic] of shape (N_pre,), (N_pre,) respectively
    - spikes  : Spikes [pre_synaptic, post_synaptic] of shape (N_pre,), (N_post,) respectively

    RETURNS:
    - W       : updated weights    
    - traces  : updated STDP traces list [pre_synaptic, post_synaptic]

    """

    # retrive parameters
    dt = pars.get('dt', 0.1)
    A_plus = pars.get('A_plus', 0.01)
    A_minus = pars.get('A_minus', 0.01)
    tau_plus = pars.get('tau_plus', 20)
    tau_minus = pars.get('tau_minus', 20)
    Wmax = pars.get('Wmax', 1)
    Wmin = pars.get('Wmin', 0)

    if traces is None:
        traces = [np.zeros(np.shape(W)[1]), np.zeros(np.shape(W)[0])]
    # unpack the variables
    pre_syn_trace, post_syn_trace = traces
    pre_syn_spk, post_syn_spk = spikes

    # what the correct order of weight and traces update?

    # update the traces
    pre_syn_trace = pre_syn_trace*(tau_plus-1)/tau_plus + pre_syn_spk
    post_syn_trace = post_syn_trace*(tau_minus-1)/tau_minus + post_syn_spk

    # update the weights
    #print('depression',post_syn_trace)
    W =  W + A_plus*np.outer(post_syn_spk,pre_syn_trace) - A_minus*np.outer(post_syn_trace,pre_syn_spk)

    return W, [pre_syn_trace, post_syn_trace]




    def simple_pars(**kwargs):
    '''
    Define the default parameters
    values come from COURSE 2 
    '''

    pars = {}

    # typical neuron parameters
    pars['threshold'] = 1.    # spike threshold [mV]
    pars['tau_m'] = 10.         # membrane time constant [ms]
    pars['R'] = 1            # leak resistance [Ohm] with this resistance input current must be of the order of 100 mV
    pars['U_init'] = 0.       # initial potential [mV]
    pars['U_reset'] = 0.      # reset potential [mV]
    pars['U_resting'] = 0.    # leak reversal potential [mV]
    pars['t_ref'] = 2.          # refractory time (ms)

    # in the case of dynamic threshold
    pars['tau_thr'] = 20         # threshold time constant [ms]
    pars['ratio_thr'] = 1.1     # relative increment in the threshold due to a spike

    # in the case of soft reset
    ## ?? some way to lower the membrane potential not to a constant value

    # random seed
    pars['my_seed'] = 42

    # time steps
    pars['dt'] = 1             # simulation time step [ms]

    # for Poisson models
    pars['alpha'] = 0.1          # scaling factor for the membrane to the rate

    # STDP parameters
    pars['A_plus'] = 0.008                   # magnitude of LTP
    pars['A_minus'] = pars['A_plus'] * 1.10  # magnitude of LTD 
    pars['tau_plus'] = 20                    # LTP time constant [ms]
    pars['tau_minus'] = pars['tau_plus']     # LTD time constant [ms]

    # weight parameters
    pars['w_max'] = 1.            # maximum weight
    pars['w_min'] = 0.            # minimum weight


    # external parameters if any #
    for k in kwargs:
        pars[k] = kwargs[k]

    return pars

s_pars = simple_pars()



    w_max_widget = widgets.FloatSlider(
         value=5,
         min=0,
         max=60,
         step=1,
         description='w_max',
         layout=widgets.Layout(width='400px'),
         tooltip = 'Maximum weight value  for STDP',
         continuous_update=False
    )
    w_min_widget = widgets.FloatSlider(
         value=0,
         min=-40,
         max=20,
         step=1,
         description='w_min',
         layout=widgets.Layout(width='400px'),
         tooltip = 'Minimum weight value  for STDP',
         continuous_update=False
    )


    
    R_widget = widgets.FloatLogSlider(
        value = 1,
        base = 2,
        min = -10,
        max = 6,
        step = 0.5,
        description = 'R',
        layout = widgets.Layout(width='400px'),
        tooltip = 'Resistence of the membrane',
        continuous_update=False
    )


constrain = widgets.ToggleButtons(
            options = ['None','Hard', 'Dynamic'],
            value = 'None',
            description = 'Constrain on the weights',
            disabled = False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips = ['No constrain', 'Hard constrain', 'Dynamic constrain'],
        )

        neuron_plot = widgets.ToggleButtons(
            options = ['None', 'Spikes', 'Mem & Spk'],
            value = 'None',
            description = 'Selected Neuron plots',
            disabled = False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips = ['No neuron plot', 'Spikes train in output plot', 'Membrane and spikes plots'],
        ),

        short_memory_trace = widgets.Checkbox(
            value=False,
            description='Short memory trace for STDP',
            disabled=False,
            indent=False
        ),


    dynamic_weight_exponent_widget = widgets.FloatLogSlider(
         value=0.01,
         base=2,
         min=-10, # max exponent of base
         max=2, # min exponent of base
         step=0.5, # exponent step
         description='Exponent',
         layout=widgets.Layout(width='400px'),
         tooltip = 'Exponent for the dynamic weight constrain',
         continuous_update=False
    )


N_pre = 100
num_steps = 1000
N_post = 100

I = np.concatenate([np.round(np.random.random((num_steps//3,N_pre))), np.round(np.random.random((num_steps//3,N_pre))-0.30),np.round(np.random.random((num_steps//3,N_pre)))], axis=0) 
pars = default_pars(R = 1, tau_thr = 40, ratio_thr = 1.5,
                    refractory_time = True, 
                    dynamic_threshold = True, 
                    hard_reset = False)

# first simulation
neurons = simulation(pars , N_post = N_post, spk_input= I , neuron_type=LIFNeuron)

post_spk = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = I, post_syn_spk = post_spk, title = 'Raster plot of the input and output spikes')

# second simulation
neurons = simulation(pars , N_post = N_post, spk_input= post_spk, neuron_type=LIFNeuron)

post_spk_2 = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = post_spk, post_syn_spk = post_spk_2, title = 'Raster plot of second output spikes', pre_syn_plot=False)

# third simulation
neurons = simulation(pars , N_post = N_post, spk_input= post_spk_2, neuron_type=LIFNeuron)

post_spk_3 = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = post_spk_2, post_syn_spk = post_spk_3, title = 'Raster plot of third output spikes', pre_syn_plot=False)

# fourth simulation
neurons = simulation(pars , N_post = N_post, spk_input= post_spk_3, neuron_type=LIFNeuron)

post_spk_4 = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = post_spk_3, post_syn_spk = post_spk_4, title = 'Raster plot of fourth output spikes', pre_syn_plot=False)



def syn_plot(pars, syn, 
             manual_update = True, 
             time_step = 1, 
             subsampling = False,   
             time_in_ms = False):
    """
    Plot the weights changes during the simulation as graph and as distribution at a given time step

    INPUT:
    - pars: parameter of the simulation
    - syn: synapse object containing the weights and the traces records
    - manual_update: if True the plot is updated only when the button is pressed
    - time_step: time step to plot the distribution
    - subsampling: subsampling of the weights plot in time
    - post_index: index of the post synaptic neuron
    - time_in_ms: if True the x axis is in ms, otherwise in time steps

    RETURN:
    Interactive demo, Visualization of synaptic weights as graph and distribution at a given time step
    """

    # useful values
    weights_history = syn.get_records()['W']
    N_post = weights_history.shape[1]
    num_steps = weights_history.shape[0]
    

    # check if we want the time in ms
    if time_in_ms:
        dt=pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    # time steps for the x axis
    time_steps = np.arange(0, num_steps, 1)*dt

    # set the default time step
    if time_step is None:
        time_step = num_steps-10
    elif time_step > num_steps:
        print(f'Time step must be less than {num_steps}')
        return
    
    def main_plot(post_index, time_step = time_step, subsampling = False):
        # check i f subsampling is less than the number of time steps
        if subsampling:
            s = num_steps//10
        else:
            s = 1

        fig,ax = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [1.5, 1]})#, sharex=True)

        # plot the weights
        x = time_steps[::s]
        y = weights_history[ ::s,post_index,:]
        ax[0].plot(x, y, lw=1., alpha=0.7)
        ax[0].axvline(time_step, 0., 1., color='red', ls='--')
        ax[0].set_xlabel(label_x)
        ax[0].set_ylabel('Weight')


        # plot the weights distribution
        w_min = np.min(weights_history[time_step,:])-0.1
        w_max = np.max(weights_history[time_step,:])+0.1
        width = (w_max - w_min)/51
        bins = np.arange(w_min, w_max, width)
        #g_dis, _ = np.histogram(weights_history[time_step,:], bins)
        #ax[1].bar(bins[1:], g_dis, color='b', alpha=0.5, width=width)
        ax[1].hist(weights_history[time_step,post_index,:], bins, color='b', alpha=0.5, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
        ax[1].set_xlabel('weights ditribution')
        ax[1].set_ylabel('Number')
        #ax[1].set_title(f'Time step: {time_step}')
        plt.tight_layout()
        plt.show()
        return

    my_layout.width = '620px'


    interactive_plot = widgets.interactive(main_plot, 
                                            {'manual': manual_update, 'manual_name': 'Update plots'},
                                            post_index=widgets.IntSlider(
                                                min=0,
                                                max=N_post-1,
                                                step=1,
                                                layout=my_layout),
                                            time_step=widgets.IntSlider(
                                                value=time_step, 
                                                min=0, 
                                                max=num_steps, 
                                                step=num_steps//100,
                                                description='Time step',
                                                layout=my_layout),
                                            subsampling=subsampling)
    #output = interactive_plot.children[-1]
    #output.layout.height = '400px'    
    return interactive_plot
 

