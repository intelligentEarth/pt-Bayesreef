 
  
#Main Contributers:   Rohitash Chandra, Ratneel Deo and Jodie Pall  Email: c.rohitash@gmail.com 

#rohitash-chandra.github.io

#  : Parallel tempering for multi-core systems - PT-BayesReef

#related: https://github.com/pyReef-model/pt-BayesReef


from __future__ import print_function, division
import multiprocessing
import gc

import os
import math
import time
import random
import csv
import numpy as np
from numpy import inf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from matplotlib import ticker
from matplotlib.cm import terrain, plasma, Set2
from pylab import rcParams
from pyReefCore.model import Model
from pyReefCore import plotResults
from cycler import cycler
from scipy import stats 

import sys

from sys import getsizeof


config = 2 # for parameter limits config

cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c
 


class ptReplica(multiprocessing.Process):


    def __init__(self, samples,filename,xmlinput,vis,num_communities, vec_parameters, realvalues, maxlimits_vec,minlimits_vec,stepratio_vec,
        swap_interval,simtime,    core_depths, core_data, tempr, parameter_queue,event , main_proc, burn_in,  pt_stage):

    #self.chains.append(ptReplica(self.NumSamples,self.folder,self.xmlinput, self.vis, self.communities, vec_parameters, self.realvalues, self.maxlimits_vec, self.minlimits_vec, self.stepratio_vec, self.swap_interval, self.simtime,  self.core_depths, self.core_data,
                #self.temperature[i], self.chain_parameters[i], self.event[i], self.wait_chain[i], self.burn_in))
    

        #--------------------------------------------------------
        multiprocessing.Process.__init__(self)

        self.samples = samples
        self.filename = filename
        self.input = xmlinput  
        self.vis = vis
        self.communities = num_communities
        self.vec_parameters =  vec_parameters
   
        self.swap_interval = swap_interval
        self.simtime = simtime
        self.realvalues_vec = realvalues # true values of free parameters for comparision. Note this will not be avialable in real world application
        self.num_param =  vec_parameters.size
      
        self.temperature = tempr
        self.adapttemp  = tempr
        self.processID = tempr      
        self.parameter_queue = parameter_queue
        self.event = event
        self.signal_main = main_proc

        # self.run_nb = run_nb 
        self.burn_in = burn_in

        self.sedsim = True
        self.flowsim = True  

        self.d_sedprop = float(np.count_nonzero(core_data[:,self.communities]))/core_data.shape[0]
        self.initial_sed = []
        self.initial_flow = []

        self.font = 10
        self.width = 1 
        
        self.core_depths = core_depths 
        self.core_data =  core_data 

        self.runninghisto = False  # if you want to have histograms of the chains during runtime in pos_variables folder NB: this has issues in Artimis
 
  
        self.maxlimits_vec = maxlimits_vec
        self.minlimits_vec = minlimits_vec 
        self.stepratio_vec = stepratio_vec 

        self.sedlimits = [0, self.maxlimits_vec[2] ]
        self.flowlimits = [0, self.maxlimits_vec[3] ]

        self.pt_stage =  pt_stage

 
 


    def run_Model(self,  input_vector):


        reef = Model()

        reef.convert_vector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.initial_sed, self.initial_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)


        #print(self.initial_sed, self.initial_flow , '   * initial sed and initial flow')
        if self.vis[0] == True:
            reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        if self.vis[1] == True:
            from matplotlib.cm import terrain, plasma
            nbcolors = len(reef.core.coralH)+10
            colors = terrain(np.linspace(0, 1.8, nbcolors))
            nbcolors = len(reef.core.layTime)+3
            colors2 = plasma(np.linspace(0, 1, nbcolors))
            #reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=300)
        output_core = reef.plot.core_timetodepth(self.communities, self.core_depths) #modelPlot.py
        #predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
        #return predicted_core 
        return output_core


    def convert_core_format(self, core, communities):
        vec = np.zeros(core.shape[0])
        for n in range(len(vec)):
            idx = np.argmax(core[n,:])# get index,
            vec[n] = idx+1 # +1 so that zero is preserved as 'none'
        return vec

 

    def give_weight(self, arr):   
        index_array = np.zeros(arr.shape[0]) 
        for i in range(0, arr.shape[0]): 
            if (arr[i] == 0): 
                index_array[i] = 1
            else:  
                index_array[i] = 0 

        return index_array


    def convertmat_assemindex(self, arr):   
        index_array = np.zeros(arr.shape[0]) 
        for i in range(0, arr.shape[0]): 
            for j in range(0, arr.shape[1]):  
                if (arr[i][j] == 1): 
                    index_array[i] = j 
        return index_array

    def score_updated(self, predictions, targets):
        # where there is 1 in the sed column, count

        predictions = np.where(predictions > 0.5, 1, 0) 

        p = self.convertmat_assemindex(predictions) #predictions.dot(1 << np.arange(predictions.shape[-1])) 

        a =  self.convertmat_assemindex(self.core_data)  
 
  

        diff = np.absolute( p-a) 

        weight_array = self.give_weight(diff)

        score = np.sum(weight_array)/weight_array.shape[0]
  

        
        return (1- score) * 100  #+ sedprop 


    def likelihood_func(self,  core_data, input_v):

         
 

        sed1=[0.0009, 0.0015, 0.0023]  # true values for synthetic 3 asssemblege problem (flow and sed)
        sed2=[0.0015, 0.0017, 0.0024]
        sed3=[0.0016, 0.0028, 0.0027]
        sed4=[0.0017, 0.0031, 0.0043]
        flow1=[0.055, 0.008 ,0.]
        flow2=[0.082, 0.051, 0.]
        flow3=[0.259, 0.172, 0.058] 
        flow4=[0.288, 0.185, 0.066]  


        v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
        #input_v = np.append(v_proposal,(ax,ay,mal))

        input_v[0: 24] = v_proposal

        print(input_v, ' ** ')


        pred_core = self.run_Model( input_v)
        pred_core = pred_core.T
        intervals = pred_core.shape[0]
        z = np.zeros((intervals,self.communities+1))   

        #print(z, intervals, ' is z int') 
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(pred_core[n,:])
            if ((pred_core[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1 

        #diff = self.diffScore(sim_prop_d,gt_prop_d, intervals)
        diff_ = self.score_updated(pred_core, core_data)
 
        
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        loss = np.log(z)
        sum_loss = np.sum(loss)
        print ('sum of loss:', sum_loss , diff_)        
        return [sum_loss *(1.0/self.adapttemp), pred_core, diff_]
 

     

    def save_core(self,reef,naccept):
        path = '%s/%s' % (self.filename, naccept)
        if not os.path.exists(path):
            os.makedirs(path)
        
        #     Initial settings     #
        reef.core.initialSetting(size=(8,2.5), size2=(8,4.5), dpi=300, fname='%s/a_thres_%s_' % (path, naccept))        
        #      Community population evolution    #
        reef.plot.speciesDepth(colors=self.colors, size=(8,4), font=8, dpi=300, fname =('%s/b_popd_%s.png' % (path,naccept)))
        reef.plot.speciesTime(colors=self.colors, size=(8,4), font=8, dpi=300,fname=('%s/c_popt_%s.png' % (path,naccept)))
        reef.plot.accomodationTime(size=(8,4), font=8, dpi=300, fname =('%s/d_acct_%s.pdf' % (path,naccept)))
        
        #      Draw core      #
        reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300, 
                           figname=('%s/e_core_%s' % (path, naccept)), filename=('%s/core_%s.csv' % (path, naccept)), sep='\t')
        return
        
    
   



    def proposal_vec(self, v_current):


        size_sed = 4 * self.communities
        size_flow = 4 * self.communities

        max_a = self.maxlimits_vec[1]
        max_m = self.maxlimits_vec[0]
        step_sed = self.stepratio_vec[2] 
        step_flow= self.stepratio_vec[3] 

 

        #if self.sedsim == True:
        tmat = v_current[0:size_sed]#np.concatenate((sed1,sed2,sed3,sed4)).reshape(4,self.communities)
        tmat = tmat.reshape(4,self.communities)

         
        tmatrix = tmat.T
 
        
        t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
 
        for x in range(self.communities):#-3):
            for s in range(tmatrix.shape[1]):
                t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,step_sed)
           
                if t2matrix[x,s] > self.sedlimits[1]:
                    t2matrix[x,s] = tmatrix[x,s]

                elif t2matrix[x,s] < self.sedlimits[0]:
                    t2matrix[x,s] = tmatrix[x,s]



                '''if t2matrix[x,s] >= self.sedlimits[x,1]:
                    t2matrix[x,s] = tmatrix[x,s]
                elif t2matrix[x,s] <= self.sedlimits[x,0]:
                    t2matrix[x,s] = tmatrix[x,s]'''
            # reorder each row , then transpose back as sed1, etc.
        tmp = np.zeros((self.communities,4))
        for x in range(t2matrix.shape[0]):
            a = np.sort(t2matrix[x,:])
            tmp[x,:] = a
        tmat = tmp.T
        p_sed1 = tmat[0,:]
        p_sed2 = tmat[1,:]
        p_sed3 = tmat[2,:]
        p_sed4 = tmat[3,:]
            
        #if self.flowsim == True:
        tmat = v_current[size_sed:size_sed+size_flow] #np.concatenate((flow1,flow2,flow3,flow4)).reshape(4,self.communities)
        tmat = tmat.reshape(4,self.communities)

        tmatrix = tmat.T
        t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
        for x in range(self.communities):#-3):
            for s in range(tmatrix.shape[1]):
                t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,step_flow)
                if t2matrix[x,s] > self.flowlimits[1]:
                    t2matrix[x,s] = tmatrix[x,s]
                elif t2matrix[x,s] < self.flowlimits[0]:
                    t2matrix[x,s] = tmatrix[x,s]
            # reorder each row , then transpose back as flow1, etc.
        tmp = np.zeros((self.communities,4))
        for x in range(t2matrix.shape[0]):
            a = np.sort(t2matrix[x,:])
            tmp[x,:] = a
        tmat = tmp.T
        p_flow1 = tmat[0,:]
        p_flow2 = tmat[1,:]
        p_flow3 = tmat[2,:]
        p_flow4 = tmat[3,:]

  

        cm_ax = v_current[size_sed+size_flow] 
        cm_ay = v_current[size_sed+size_flow+1] 
        m = v_current[size_sed+size_flow+2] 


        #stepsize_ratio = [step_m, step_a, step_sed, step_flow]
        #max_limits = [max_m, max_a, sedlim[1], flowlim[1]]
        #min_limits = [0, 0, sedlim[0], flowlim[0] ]

        step_a = self.stepratio_vec[1] 
        step_m = self.stepratio_vec[0] 


        p_ax = cm_ax + np.random.normal(0,step_a,1)
        if p_ax > 0:
            p_ax = cm_ax
        elif p_ax < max_a:
            p_ax = cm_ax
        p_ay = cm_ay + np.random.normal(0,step_a,1)
        if p_ay > 0:
            p_ay = cm_ay
        elif p_ay < max_a:
            p_ay = cm_ay   
        p_m = m + np.random.normal(0,step_m,1)
        if p_m < 0:
            p_m = m
        elif p_m > max_m:
            p_m = m  

        glv_pro = np.array([p_ax,p_ay,p_m])
 

        v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4,p_flow1,p_flow2,p_flow3,p_flow4))
        

        for a in glv_pro:
            v_proposal = np.append(v_proposal, a)
 
        return v_proposal  #np.ravel(v_proposal)


    def run(self):
        # Note this is a chain that is distributed to many cores. The chain is also known as Replica in Parallel Tempering



        data_size = self.core_data.shape[0] 
        x_data = self.core_depths
        y_data = self.core_data 

        data_vec = self.convert_core_format(self.core_data, self.communities)


        samples = self.samples
 

        burnin = int(self.burn_in * samples)
        #pt_stage = int(0.99 * samples) # paralel tempering is used only for exploration, it does not form the posterior, later mcmc in parallel is used with swaps 
        pt_samples = int(self.pt_stage * samples) # paralel tempering is used only for exploration, it does not form the posterior, later mcmc in parallel is used with swaps
        

        #pt_samples = (self.samples * 0.9)

 
        communities = self.communities
        num_param = self.num_param
        burnsamples = int(samples*self.burn_in) 

        count_list = [] 
        # initial values of the parameters to be passed to Blackbox model 




        v_proposal = self.vec_parameters
        v_current = v_proposal # to give initial value of the chain 


        # Create memory to save all the accepted proposals of parameters, model predictions and likelihood

        pos_param = np.empty((samples,v_current.size))   
        pos_param[0,:] = v_proposal # assign first proposal 

        

        #----------------------------------------------------------------------------

 




        for i in range(3):

            likelihood, rep_predcore_, rep_diffscore = self.likelihood_func(  self.core_data, v_proposal) 

             

            predcore  = self.convert_core_format(rep_predcore_, self.communities)
     

            pos_samples_d = np.empty((samples, self.core_depths.size)) # list of all accepted (plus repeats) of pred cores 
     
            pos_samples_d[0,:] = predcore # assign the first core pred
            
            pos_likl = np.empty((samples, 2)) # one for posterior of likelihood and the other for all proposed likelihood
            pos_likl[0,:] = [-10000, -10000] # to avoid prob in calc of 5th and 95th percentile later

            list_diffscore =  np.empty(samples) 


            print (i, '\tInitial likelihood:', likelihood, 'and  diff:', rep_diffscore)
        #---------------------------------------

        print (  '\t done: ----------------------------------+++++++++++++++++++++++++++--')









        count_list.append(0) # To count number of accepted for each chain (replica)
        accept_list = np.empty(samples)
        start = time.time() 
        num_accepted = 0
        
        with file(('%s/description.txt' % (self.filename)),'a') as outfile:
            outfile.write('\nChain Temp: {0}'.format(self.temperature))
            outfile.write('\n\tSamples: {0}'.format(self.samples))  
            outfile.write('\n\tInitial proposed vector\n\t{0}'.format(v_proposal))   

        #---------------------------------------
        
        print('Begin sampling using MCMC random walk')
        #b = 0

        init_count = 0
        
        for i in range(samples-1):


            if i < pt_samples:
                self.adapttemp =  1 #self.temperature #* ratio  #

            if i == pt_samples and init_count ==0: # move to MCMC canonical
                self.adapttemp = 1 
                likelihood_proposal, rep_predcore_, rep_diffscore = self.likelihood_func(  self.core_data, v_proposal)  

                init_count = 1

            #print(v_current, ' v_current')

 
 
            v_proposal = self.proposal_vec(v_current)  
 
            likelihood_proposal, rep_predcore_, rep_diffscore = self.likelihood_func(  self.core_data, v_proposal)  
            predcore  = self.convert_core_format(rep_predcore_, self.communities)
 
            diff_likelihood = likelihood_proposal - likelihood 

            print(likelihood_proposal, diff_likelihood, '  + --------------------  + ')
 

            try:
                mh_prob = min(1, math.exp(diff_likelihood))
            except OverflowError as e:
                mh_prob = 1

            u = random.uniform(0,1)

            #print('u:', u, 'MH probability:', mh_prob)
            #print((i % self.swap_interval), i,  self.swap_interval, 'mod swap')

            pos_likl[i+1,0] = likelihood_proposal 

            list_diffscore[i +1] = rep_diffscore
            accept_list[i+1] = num_accepted


            if u < mh_prob: # Accept sample
                #b = b+1
                print ('Accepted Sample',i, ' \n\tLikelihood ', likelihood_proposal,'\n\tTemperature:', self.temperature,'\n\t  accepted, sample, diff ----------------->:', num_accepted,  rep_diffscore)
                count_list.append(i)            # Append sample number to accepted list
                num_accepted = num_accepted + 1  

                v_current = v_proposal 

                #print(v_proposal)
                likelihood = likelihood_proposal 
                pos_likl[i + 1,1]=likelihood  # contains  all proposal liklihood (accepted and rejected ones) 
                pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector)  
                pos_samples_d[i+1,:] =  predcore
            else: # Reject sample
                #b = i
                pos_likl[i + 1, 1] = pos_likl[i,1]  
                pos_param[i+1,:] = pos_param[i,:]   
                pos_samples_d[i+1,:] = pos_samples_d[i,:] 
 

             
            
            

            #----------------------------------------------------------------------------------------
            if ( i % self.swap_interval == 0 ):  

 
                others = np.asarray([likelihood])
                param = np.concatenate([v_current,others])     


                # paramater placed in queue for swapping between chains
                self.parameter_queue.put(param)
 
    

                #signal main process to start and start waiting for signal for main
                self.signal_main.set()              
                self.event.wait()
                

                # retrieve parametsrs fom ques if it has been swapped
                if not self.parameter_queue.empty() : 
                    try:
                        result =  self.parameter_queue.get()
 
                        
                        v_current= result[0:v_current.size]     
                        likelihood = result[v_current.size]

                        del result

                    except:
                        print ('error')  

                    


        accepted_count =  len(count_list) 
        accept_ratio = accepted_count / (samples * 1.0) * 100


        #--------------------------------------------------------------- 

        others = np.asarray([ likelihood])
        param = np.concatenate([v_current,others])   

        self.parameter_queue.put(param)

        file_name = self.filename+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,pos_likl, fmt='%1.2f')
 
        file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
        np.savetxt(file_name, [accept_ratio], fmt='%1.2f')

        file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, accept_list, fmt='%1.2f')
 
        file_name = self.filename+'/posterior/pos_parameters/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,pos_param ) 

        file_name = self.filename+'/posterior/accept_list/diffscorechain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, list_diffscore)   

        file_name = self.filename+'/posterior/predicted_core/pos_samples_d/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, pos_samples_d, fmt='%1.2f')
  
        self.signal_main.set()

        return


class ParallelTempering:

    def __init__(self, problem, num_chains,communities, NumSample,fname,xmlinput,num_param,maxtemp,swap_interval,simtime,true_vec_parameters,   core_depths, core_data, vis,    maxlimits_vec, minlimits_vec , stepratio_vec,  burn_in,  pt_stage):

        self.num_chains = num_chains
        self.communities = communities
        self.NumSamples = int(NumSample/self.num_chains)
        self.folder = fname
        self.xmlinput = xmlinput
        self.num_param = num_param
        self.maxtemp = maxtemp
        self.swap_interval = swap_interval
        self.simtime = simtime
        self.realvalues  =  true_vec_parameters
        self.core_depths = core_depths
        self.core_data = core_data 

        self.chains = []
        self.temperature = []
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile. Takes more space 

        # Create queues for transfer of parameters between process chain
        self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]

        self.geometric =  True

        # Two ways events are used to synchronise chains
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        self.vis = vis
        self.communities = communities
        self.maxlimits_vec = maxlimits_vec
        self.minlimits_vec = minlimits_vec 
        self.stepratio_vec = stepratio_vec
        self.burn_in = burn_in
        self.pt_stage =  pt_stage
        self.problem = problem

        self.initial_sed = []
        self.initial_flow = []


 

 
    # Assign temperature dynamically 


    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                        1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                        1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                        1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                        1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                        1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                        1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                        1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                        1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                        1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                        1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                        1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                        1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                        1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                        1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                        1.26579, 1.26424, 1.26271, 1.26121,
                        1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas
        
        
    def assign_temperatures(self):
        # #Linear Spacing
        # temp = 2
        # for i in range(0,self.num_chains):
        #   self.temperatures.append(temp)
        #   temp += 2.5 #(self.maxtemp/self.num_chains)
        #   print (self.temperatures[i])
        #Geometric Spacing

        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)      
            for i in range(0, self.num_chains):         
                self.temperature.append(np.inf if betas[i] is 0 else 1.0/betas[i])
                print (self.temperature[i])
        else:

            tmpr_rate = (self.maxtemp /self.num_chains)
            temp = 1
            print("Temperatures...")
            for i in xrange(0, self.num_chains):            
                self.temperatures.append(temp)
                temp += tmpr_rate
                print(self.temperature[i])

    '''def assign_temptarures(self):
        tmpr_rate = (self.maxtemp /self.num_chains)
        temp = 1
        for i in xrange(0, self.num_chains):            
            self.temperature.append(temp)
            temp += tmpr_rate
            print('self.temperature[%s]' % i,self.temperature[i])'''
             
    def initialise_chains (self):

        self.assign_temperatures()
        for i in xrange(0, self.num_chains): 
            vec_parameters = self.initial_replicaproposal()
            print(vec_parameters, ' vec init ', i)
            self.chains.append(ptReplica(self.NumSamples,self.folder,self.xmlinput, self.vis, self.communities, vec_parameters, self.realvalues, self.maxlimits_vec, self.minlimits_vec, self.stepratio_vec, self.swap_interval, self.simtime,  self.core_depths, self.core_data,
                self.temperature[i], self.chain_parameters[i], self.event[i], self.wait_chain[i], self.burn_in, self.pt_stage))
    
    def run_chains (self):
        
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1) 
        
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))  
        lhood = np.zeros(self.num_chains)

        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1

        number_exchange = np.zeros(self.num_chains)

        # filen = open(self.folder + '/num_exchange.txt', 'a')
        
        #-------------------------------------------------------------------------------------
        # run the MCMC chains
        #-------------------------------------------------------------------------------------
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        
        #-------------------------------------------------------------------------------------
        # run the MCMC chains
        #-------------------------------------------------------------------------------------
        for j in range(0,self.num_chains):        
            self.chains[j].start()

        flag_running = True 

        
        while flag_running:          

            #-------------------------------------------------------------------------------------
            # wait for chains to complete one pass through the samples
            #-------------------------------------------------------------------------------------

            for j in range(0,self.num_chains): 
                #print (j, ' - waiting')
                self.wait_chain[j].wait()
            

            
            #-------------------------------------------------------------------------------------
            #get info from chains
            #-------------------------------------------------------------------------------------
            
            for j in range(0,self.num_chains): 
                if self.chain_parameters[j].empty() is False :
                    result =  self.chain_parameters[j].get()
                    replica_param[j,:] = result[0:self.num_param]   
                    lhood[j] = result[self.num_param]

                    del result
 
 

            # create swapping proposals between adjacent chains
            for k in range(0, self.num_chains-1): 
                swap_proposal[k]=  (lhood[k]/[1 if lhood[k+1] == 0 else lhood[k+1]])*(1/self.temperature[k] * 1/self.temperature[k+1])

            #print(' before  swap_proposal  --------------------------------------+++++++++++++++++++++++=-')

            for l in range( self.num_chains-1, 0, -1):
                #u = 1
                u = random.uniform(0, 1)
                swap_prob = swap_proposal[l-1]



                if u < swap_prob : 

                    number_exchange[l] = number_exchange[l] +1  

                    others = np.asarray([  lhood[l-1] ]  ) 
                    para = np.concatenate([replica_param[l-1,:],others])   
 
                   
                    self.chain_parameters[l].put(para) 

                    others = np.asarray([ lhood[l] ] )
                    param = np.concatenate([replica_param[l,:],others])
 
                    self.chain_parameters[l-1].put(param)

                    del para
                    del others
                    del param
                    
                else:


                    others = np.asarray([  lhood[l-1] ])
                    para = np.concatenate([replica_param[l-1,:],others]) 
 
                    self.chain_parameters[l-1].put(para) 

                    others = np.asarray([  lhood[l]  ])
                    param = np.concatenate([replica_param[l,:],others])
 
                    self.chain_parameters[l].put(param)

                    del para
                    del others
                    del param

                del u
                del swap_prob



            #-------------------------------------------------------------------------------------
            # resume suspended process
            #-------------------------------------------------------------------------------------
            for k in range (self.num_chains):
                    self.event[k].set()

                                

            #-------------------------------------------------------------------------------------
            #check if all chains have completed runing
            #-------------------------------------------------------------------------------------
            count = 0
            for i in range(self.num_chains):
                if self.chains[i].is_alive() is False:
                    count+=1
                    while self.chain_parameters[i].empty() is False:
                        dummy = self.chain_parameters[i].get()
                        del dummy

            if count == self.num_chains :
                flag_running = False
            
            del count

            
            gc.collect() # fLet the main threag constantly be removing files from memory


        #-------------------------------------------------------------------------------------
        #wait for all processes to jin the main process
        #-------------------------------------------------------------------------------------     
        for j in range(0,self.num_chains): 
            self.chains[j].join()

        print('Process ended.\n\tNo. exchange:', number_exchange)

        burnin, pos_param, likelihood_rep, accept_list,  accept,  list_predcore_d, diffscore = self.show_results('chain_')  


        for s in range( 0, self.num_param):   
            
            self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s),  self.realvalues[s] ,self.num_chains ) 

        self.pos_sedflow(pos_param) 
   
        
        #self.summ_stats(self.folder, pos_param)

        optimal_likl, optimal_para, para_5pcent, para_95pcent = self.get_optimal(likelihood_rep, pos_param)
        print('optimal_likl', optimal_likl)

        outfile = open(self.folder+'/optimal_percentile_para.txt', 'a+')
        hdr = np.array(['optimal_para', 'para_5pcent', 'para_95pcent'])
        np.savetxt(outfile, hdr, fmt="%s", delimiter=' ')
        np.savetxt(outfile, [optimal_para,para_5pcent,para_95pcent],fmt='%1.2ff',delimiter=' ')
        np.savetxt(outfile,['optimal likelihood'], fmt='%s')
        np.savetxt(outfile,[optimal_likl], fmt='%1.2f')
        # np.savetxt(outfile, [np.array(['Optimal Parameters']),optimal_para], fmt='%1.2f',delimiter=' ')
        # np.savetxt(outfile,[np.array(['para_5pcent']), para_5pcent], fmt='%1.2f',delimiter=' ')
        # np.savetxt(outfile,[np.array(['para_95pcent']), para_95pcent], fmt='%1.2f',delimiter=' ')
        x_tick_labels = ['Shallow', 'Mod-deep', 'Deep', 'Sediment','No growth']
        x_tick_values = [1,2,3,4,5]
        #plotResults.plotPosCore(self.folder,list_predcore_d.T,list_predcore_t.T, self.gt_vec_d, self.gt_vec_t, self.gt_depths, self.gt_timelay, x_tick_labels,x_tick_values, 9)
        #plotResults.boxPlots(self.communities, pos_param, True, True, 9, 1, self.folder)

        sample_range = np.arange(burnin+1,self.NumSamples+1, 1)
        '''for s in range(self.num_param):  
            self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s), self.realvalues[s], sample_range) 
        '''
        return (pos_param,likelihood_rep, accept_list,  list_predcore_d, diffscore)

    # Merge different MCMC chains y stacking them on top of each other
    def show_results(self, filename):

        burnin = int(self.NumSamples * self.burn_in)
        print('Burnin:',burnin)

        
        #file_name = self.folder + '/posterior/pos_parameters/'+filename + str(self.temperature[0]) + '.txt'
        #dat_dummy = np.loadtxt(file_name) 

        #print(dat_dummy.shape)

        pos_param = np.zeros((self.num_chains, self.NumSamples -burnin, self.num_param)) 
        pred_d = np.zeros((self.num_chains, self.NumSamples-burnin, self.core_depths.shape[0]))
 

        file_name = self.folder + '/posterior/pos_likelihood/'+filename + str(self.temperature[0]) + '.txt'
        dat = np.loadtxt(file_name) 
        likelihood_rep = np.zeros((self.num_chains, dat.shape[0]-burnin, 2 )) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        
        '''
        pos_param = np.zeros((self.num_chains, self.NumSamples - burnin , self.num_param))
        print('Pos_param.shape:', pos_param.shape)
        pred_t = np.zeros((self.num_chains, self.NumSamples - burnin, self.gt_vec_t.shape[0]))
        pred_d = np.zeros((self.num_chains, self.NumSamples - burnin, self.gt_vec_d.shape[0]))
        print('pred_t:',pred_t,'pred_t.shape:', pred_t.shape)
        print('gt_prop_t.shape:',self.gt_prop_t.shape)
        
 
        likelihood_rep = np.zeros((self.num_chains, self.NumSamples - burnin, 2 )) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        '''

        accept_percent = np.zeros((self.num_chains, 1))

        accept_list = np.zeros((self.num_chains, self.NumSamples )) 

        diffscore_list = np.zeros((self.num_chains, self.NumSamples )) 
 
 
        for i in range(self.num_chains):
            file_name = self.folder + '/posterior/pos_parameters/'+filename + str(self.temperature[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            print('dat.shape:',dat.shape) 
            pos_param[i, :, :] = dat[burnin:,:]
             

            file_name = self.folder + '/posterior/predicted_core/pos_samples_d/chain_'+  str(self.temperature[i]) + '.txt'
            dat = np.loadtxt(file_name)
            pred_d[i, :, :] = dat[burnin:,:] 

            file_name = self.folder + '/posterior/pos_likelihood/'+filename + str(self.temperature[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            likelihood_rep[i, :] = dat[burnin:]

 

            file_name = self.folder + '/posterior/accept_list/' + 'diffscore'+ filename + str(self.temperature[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            diffscore_list[i, :] = dat 

            file_name = self.folder + '/posterior/accept_list/' + filename + str(self.temperature[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            accept_list[i, :] = dat 

            file_name = self.folder + '/posterior/accept_list/' + filename + str(self.temperature[i]) + '_accept.txt'
            dat = np.loadtxt(file_name) 
            accept_percent[i, :] = dat

        likelihood_vec = likelihood_rep.transpose(2,0,1).reshape(2,-1) 
        posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1) 
        list_predcore_d = pred_d.transpose(2,0,1).reshape(self.core_depths.shape[0],-1)
 
        accept = np.sum(accept_percent)/self.num_chains

        np.savetxt(self.folder + '/pos_param.txt', posterior.T) 
        
        np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

        np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')

        np.savetxt(self.folder + '/diffscore_list.txt', diffscore_list, fmt='%1.2f')
  
        np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')
 

        return burnin, posterior, likelihood_vec.T, accept_list, accept,   list_predcore_d, diffscore_list

    def find_nearest(self, array,value): 
        # Find nearest index for a particular value
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    def get_optimal(self, likelihood_rep, pos_param): 

        likelihood_pos = likelihood_rep[:,1]
        
        # Find 5th and 95th percentile of a posterior
        a = np.percentile(likelihood_pos, 5)   
        # Find nearest value of 5th/95th percentiles in posterior likelihood 
        lhood_5thpercentile, index_5th = self.find_nearest(likelihood_pos,a)  
        b = np.percentile(likelihood_pos, 95) 
        lhood_95thpercentile, index_95th = self.find_nearest(likelihood_pos,b)  

        # Find max of pos liklihood to get the max or optimal posterior value  
        max_index = np.argmax(likelihood_pos) 
        optimal_likelihood = likelihood_pos[max_index]  
        optimal_para = pos_param[:, max_index] 
        
        para_5thperc = pos_param[:, index_5th]
        para_95thperc = pos_param[:, index_95th] 

        return optimal_likelihood, optimal_para, para_5thperc, para_95thperc
 


    def plot_figure(self, list, title, real_value, nreplicas  ): 

        list_points =  list
        fname = self.folder
         


        size = 14
 


        fig, ax = plt.subplots()
   


        ax.hist(list_points,    bins=20, rwidth=0.9,   color='#607c8e')
 
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        #ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}')) 


        fmtr = ticker.StrMethodFormatter(('{x:,g}'))
        ax.yaxis.set_major_formatter(fmtr)
 
        ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.3f}'))

        ax.set_xlabel("Parameter", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15) 

        if self.problem == 1:
            plt.axvline(x=real_value, linewidth=2, color='b')
 
        
        ax.grid(linestyle='-', linewidth='0.2', color='grey')
        plt.tight_layout()  
        plt.savefig(fname   +'/posterior/'+ title  + '_posterior.pdf')
        plt.clf()

 


        fig, ax = plt.subplots()

        listx = np.asarray(np.split(list_points,  nreplicas ))
        ax.plot(listx.T)    

        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}')) 
 
        ax.yaxis.set_minor_formatter( ticker.StrMethodFormatter(('{x:,g}')) )
  
        ax.set_xlabel("Iterations", fontsize=15)
        ax.set_ylabel("Parameter", fontsize=15) 
        
        ax.grid(linestyle='-', linewidth='0.2', color='grey')
        plt.tight_layout()  

        plt.savefig(fname  +'/posterior/'+ title  + '_trace.pdf')
        plt.clf()

 
    def pos_sedflow(self, pos):


        pos_sed1 = pos[0:3,:]   
        pos_sed2 = pos[3:6,:]
        pos_sed3 = pos[6:9,:]
        pos_sed4 = pos[9:12,:]
        pos_flow1 = pos[12:15,:]
        pos_flow2 = pos[15:19,:]
        pos_flow3 = pos[19:22,:] 
        pos_flow4 = pos[22:25,:] 


        nb_bins=30
        slen = np.arange(0,pos_sed1.shape[0],1)
 
 


        # PLOT SEDIMENT AND FLOW RESPONSE THRESHOLDS #

        if self.communities == 3:
            a_labels = ['Shallow windward', 'Moderate-deep windward', 'Deep windward']#, 'Shallow leeward', 'Moderate-deep leeward', 'Deep leeward']
        else:
            a_labels = ['Windward Shallow', 'Windward Mod-deep', 'Windward Deep', 'Sediment','Leeward Shallow', 'Leeward Mod-deep', 'Leeward Deep']
         
        
        sed1_mu, sed1_ub, sed1_lb, sed2_mu, sed2_ub, sed2_lb, sed3_mu, sed3_ub, sed3_lb, sed4_mu, sed4_ub, sed4_lb = (np.zeros(self.communities) for i in range(12))
        if (True):
            for a in range(self.communities):
                sed1_mu[a] = np.mean(pos_sed1[:,a])
                sed1_ub[a] = np.percentile(pos_sed1[:,a], 95, axis=0)
                sed1_lb[a] = np.percentile(pos_sed1[:,a], 5, axis=0)
                
                sed2_mu[a] = np.mean(pos_sed2[:,a])
                sed2_ub[a] = np.percentile(pos_sed2[:,a], 95, axis=0)
                sed2_lb[a] = np.percentile(pos_sed2[:,a], 5, axis=0)
                
                sed3_mu[a] = np.mean(pos_sed3[:,a])
                sed3_ub[a] = np.percentile(pos_sed3[:,a], 95, axis=0)
                sed3_lb[a] = np.percentile(pos_sed3[:,a], 5, axis=0)
                
                sed4_mu[a] = np.mean(pos_sed4[:,a])
                sed4_ub[a] = np.percentile(pos_sed4[:,a], 95, axis=0)
                sed4_lb[a] = np.percentile(pos_sed4[:,a], 5, axis=0)

                sed1_mu_=sed1_mu[a]
                sed2_mu_=sed2_mu[a]
                sed3_mu_=sed3_mu[a]
                sed4_mu_=sed4_mu[a]
                sed1_min=sed1_lb[a]
                sed2_min=sed2_lb[a]
                sed3_min=sed3_lb[a]
                sed4_min=sed4_lb[a]
                sed1_max=sed1_ub[a]
                sed2_max=sed2_ub[a]
                sed3_max=sed3_ub[a]
                sed4_max=sed4_ub[a]
                sed1_med=np.median(pos_sed1[:,a])
                sed2_med=np.median(pos_sed2[:,a])
                sed3_med=np.median(pos_sed3[:,a])
                sed4_med=np.median(pos_sed4[:,a])
                sed1_mode, count=stats.mode(pos_sed1[:,a])
                sed2_mode, count=stats.mode(pos_sed2[:,a])
                sed3_mode, count=stats.mode(pos_sed3[:,a])
                sed4_mode, count=stats.mode(pos_sed4[:,a])


                with file(('%s/summ_stats.txt' % (self.folder)),'a') as outfile:
                    #outfile.write('\n# Sediment threshold: {0}\n'.format(a_labels[a]))
                    outfile.write('5TH %ILE, 95TH %ILE, MEAN, MEDIAN\n')
                    outfile.write('Sed1\n{0}, {1}, {2}, {3}\n'.format(sed1_min,sed1_max,sed1_mu_,sed1_med))
                    outfile.write('Sed2\n{0}, {1}, {2}, {3}\n'.format(sed2_min,sed2_max,sed2_mu_,sed2_med))
                    outfile.write('Sed3\n{0}, {1}, {2}, {3}\n'.format(sed3_min,sed3_max,sed3_mu_,sed3_med))
                    outfile.write('Sed4\n{0}, {1}, {2}, {3}\n'.format(sed4_min,sed4_max,sed4_mu_,sed4_med))
                    outfile.write('Modes\n\tSed1:\t{0}\n\tSed2:\t{1}\n\tSed3:\t{2}\n\tSed4:\t{3}'.format(sed1_mode,sed2_mode,sed3_mode,sed4_mode))

                cy = [0,100,100,0]
                cmu = [sed1_mu[a], sed2_mu[a], sed3_mu[a], sed4_mu[a]]
                c_lb = [sed1_mu[a]-sed1_lb[a], sed2_mu[a]-sed2_lb[a], sed3_mu[a]-sed3_lb[a], sed4_mu[a]-sed4_lb[a]]
                c_ub = [sed1_ub[a]-sed1_mu[a], sed2_ub[a]-sed2_mu[a], sed3_ub[a]-sed3_mu[a], sed4_ub[a]-sed4_mu[a]]
                
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                ax.set_facecolor('#f2f2f3')
                #if self.problem ==1:
                    #ax.plot(self.initial_sed[a,:], cy, linestyle='--', linewidth=1, marker='.',color='k', label='True')
                ax.plot(cmu, cy, linestyle='-', linewidth=1,marker='.', color='sandybrown', label='Estimate')
                ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1, color='darksalmon',mfc='darksalmon',fmt='.',label=None)
                ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1, color='sienna',mfc='sienna',fmt='.',label=None)

                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='minor', labelsize=10)

                ax.set_xlabel("Sediment input (m/year)", fontsize=11)
                ax.set_ylabel("Max. growth rate", fontsize=11) 
                


                plt.title('Sediment exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=11, y=1.06)
                #plt.ylabel('Proportion of maximum growth rate [%]',size=self.font+1)
                #plt.xlabel('Sediment input [m/year]',size=self.font+1)
                plt.ylim(-2.,110)
                lgd = plt.legend(frameon=False, prop={'size':10}, bbox_to_anchor = (1.,0.2))
                plt.savefig('%s/sediment_response_%s.pdf' % (self.folder, a+1), bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=300,transparent=False)
                plt.clf()

        flow1_mu, flow1_ub,flow1_lb, flow2_mu, flow2_ub,flow2_lb, flow3_mu, flow3_ub,flow3_lb, flow4_mu, flow4_ub,flow4_lb = (np.zeros(self.communities) for i in range(12))
        if (True):
            for a in range(self.communities):
                flow1_mu[a] = np.mean(pos_flow1[:,a])
                flow1_ub[a] = np.percentile(pos_flow1[:,a], 95, axis=0)
                flow1_lb[a] = np.percentile(pos_flow1[:,a], 5, axis=0)
                
                flow2_mu[a] = np.mean(pos_flow2[:,a])
                flow2_ub[a] = np.percentile(pos_flow2[:,a], 95, axis=0)
                flow2_lb[a] = np.percentile(pos_flow2[:,a], 5, axis=0)
                
                flow3_mu[a] = np.mean(pos_flow3[:,a])
                flow3_ub[a] = np.percentile(pos_flow3[:,a], 95, axis=0)
                flow3_lb[a] = np.percentile(pos_flow3[:,a], 5, axis=0)
                
                flow4_mu[a] = np.mean(pos_flow4[:,a])
                flow4_ub[a] = np.percentile(pos_flow4[:,a], 95, axis=0)
                flow4_lb[a] = np.percentile(pos_flow4[:,a], 5, axis=0)

                flow1_mu_ = flow1_mu[a]
                flow2_mu_ = flow2_mu[a]
                flow3_mu_ = flow3_mu[a]
                flow4_mu_ = flow4_mu[a]
                flow1_min= flow1_lb[a]
                flow1_max=flow1_ub[a]
                flow1_med=np.median(pos_flow1[:,a])
                flow2_min=flow2_lb[a]
                flow2_max=flow2_ub[a]
                flow2_med=np.median(pos_flow2[:,a])
                flow3_min=flow3_lb[a]
                flow3_max=flow3_ub[a]
                flow3_med=np.median(pos_flow3[:,a])
                flow4_min=flow4_lb[a]
                flow4_max=flow4_ub[a]
                flow4_med=np.median(pos_flow4[:,a])
                flow1_mode, count= stats.mode(pos_flow1[:,a])
                flow2_mode, count= stats.mode(pos_flow2[:,a])
                flow3_mode, count= stats.mode(pos_flow3[:,a])
                flow4_mode, count= stats.mode(pos_flow4[:,a])

                with file(('%s/summ_stats.txt' % ( self.folder)),'a') as outfile:
                    #outfile.write('\n# Water flow threshold: {0}\n'.format(a_labels[a]))
                    outfile.write('#5TH %ILE, 95TH %ILE, MEAN, MEDIAN\n')
                    outfile.write('# flow1\n{0}, {1}, {2}, {3}\n'.format(flow1_min,flow1_max,flow1_mu_,flow1_med))
                    outfile.write('# flow2\n{0}, {1}, {2}, {3}\n'.format(flow2_min,flow2_max,flow2_mu_,flow2_med))
                    outfile.write('# flow3\n{0}, {1}, {2}, {3}\n'.format(flow3_min,flow3_max,flow3_mu_,flow3_med))
                    outfile.write('# flow4\n{0}, {1}, {2}, {3}\n'.format(flow4_min,flow4_max,flow4_mu_,flow4_med))
                    outfile.write('Modes\n\tFlow1:\t{0}\n\tFlow2:\t{1}\n\tFlow3:\t{2}\n\tFlow4:\t{3}'.format(flow1_mode,flow2_mode,flow3_mode,flow4_mode))

                cy = [0,100,100,0]
                cmu = [flow1_mu[a], flow2_mu[a], flow3_mu[a], flow4_mu[a]]
                c_lb = [flow1_mu[a]-flow1_lb[a], flow2_mu[a]-flow2_lb[a], flow3_mu[a]-flow3_lb[a], flow4_mu[a]-flow4_lb[a]]
                c_ub = [flow1_ub[a]-flow1_mu[a], flow2_ub[a]-flow2_mu[a], flow3_ub[a]-flow3_mu[a], flow4_ub[a]-flow4_mu[a]]

                
                fig = plt.figure(figsize=(6,4))

                params = {'legend.fontsize': 15, 'legend.handlelength': 2}
                plt.rcParams.update(params)
                ax = fig.add_subplot(111)
                ax.set_facecolor('#f2f2f3')
                #if self.problem ==1:
                    #ax.plot(self.initial_flow[a,:], cy, linestyle='--', linewidth=1, marker='.', color='k',label='True')
                ax.plot(cmu, cy, linestyle='-', linewidth=1, marker='.', color='steelblue', label='Estimate')
                ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1,color='lightsteelblue',mfc='lightsteelblue',fmt='.',label=None)
                ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1,color='lightslategrey',mfc='lightslategrey',fmt='.',label=None)

                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='minor', labelsize=10)

                ax.set_xlabel("Fluid flow (m/sec)", fontsize=11)
                ax.set_ylabel("Max. growth rate", fontsize=11) 

                plt.title('Hydrodynamic energy exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=11, y=1.06)
                #plt.ylabel('Proportion of maximum growth rate [%]', size=self.font+1)
                #plt.xlabel('Fluid flow [m/sec]', size=self.font+1)
                plt.ylim(-2.,110.)
                lgd = plt.legend(frameon=False, prop={'size':10}, bbox_to_anchor = (1.,0.2))
                plt.savefig('%s/flow_response_%s.pdf' % (self.folder, a+1),  bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300,transparent=False)
                plt.clf()

 


    def initial_replicaproposal(self): 
     

        sed1 = np.zeros(self.communities)
        sed2 = np.zeros(self.communities)
        sed3 = np.zeros(self.communities)
        sed4 = np.zeros(self.communities)

        flow1 = np.zeros(self.communities)
        flow2 = np.zeros(self.communities)
        flow3 = np.zeros(self.communities)
        flow4 = np.zeros(self.communities)


        for s in range(self.communities): 

            sed1[s] = np.random.uniform(0,self.maxlimits_vec[2])
            sed2[s] = np.random.uniform(sed1[s],self.maxlimits_vec[2])
            sed3[s] = np.random.uniform(sed2[s],self.maxlimits_vec[2])
            sed4[s] = np.random.uniform(sed3[s],self.maxlimits_vec[2])


        for x in range(self.communities): 

            flow1[x] = np.random.uniform(0, self.maxlimits_vec[3])
            flow2[x] = np.random.uniform(flow1[x], self.maxlimits_vec[3])
            flow3[x] = np.random.uniform(flow2[x], self.maxlimits_vec[3])
            flow4[x] = np.random.uniform(flow3[x], self.maxlimits_vec[3])


       
        cm_ax   = np.random.uniform(self.maxlimits_vec[1], 0.)
        cm_ay  = np.random.uniform(self.maxlimits_vec[1], 0.)
        m   = np.random.uniform(0., self.maxlimits_vec[0])
     

        glv_pro = np.array([cm_ax,cm_ay,m]) 
        v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4)) 

        return np.hstack((v_proposal,glv_pro)) #np.ravel(v_proposal) #, m, cm_ax, cm_ay

     
 

def mean_sqerror(  pred_erodep, pred_elev,  real_elev,  real_erodep_pts):
         
        elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
        sed =  np.sqrt(  np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 

        return elev + sed, sed

def find_limits(communities, num_sed, num_flow, sedlim, flowlim,  min_a, max_a, min_m, max_m):

    sedmax_vec =  np.repeat(sedlim[1], communities * num_sed )  # vec size =12
    sedmin_vec =  np.repeat(sedlim[0], communities * num_sed ) 

    flowmax_vec =  np.repeat(flowlim[1], communities * num_flow )  
    flowmin_vec =  np.repeat(flowlim[0], communities * num_flow) 
 
    glv_max = np.array([  max_a, max_a, max_m]) 
    glv_min = np.array([  min_a, min_a, min_m])

    maxlimits_vec = np.concatenate((sedmax_vec , flowmax_vec, glv_max))
    minlimits_vec = np.concatenate((sedmin_vec , flowmin_vec, glv_min))


    return   maxlimits_vec, minlimits_vec

'''def initial_vec(communities, num_sed, num_flow, sedlim, flowlim, min_a, max_a, min_m, max_m):
    print('communities',communities)
    print('num_sed',num_sed)
    print('num_flow',num_flow)
    print('sedlim',sedlim,'flowlim',flowlim,'min_a',min_a, 'max a',max_a,'min m',min_m,'max m', max_m)

    sed1 = np.empty(communities)
    sed2 = np.empty(communities)
    sed3 = np.empty(communities)
    sed4 = np.empty(communities)
    pr_flow2 = np.empty(communities)
    pr_flow3 = np.empty(communities)
    pr_flow4 = np.empty(communities)
    pr_sed2 = np.empty(communities)
    pr_sed3 = np.empty(communities)
    pr_sed4 = np.empty(communities)
 
    for s in range(communities):

        sed1[s] = np.random.uniform(sedlim[0],sedlim[1])
        sed2[s] = np.random.uniform(sed1[s],sedlim[1])
        sed3[s] = np.random.uniform(sed2[s],sedlim[1])
        sed4[s] = np.random.uniform(sed3[s],sedlim[1])

    flow1 = np.empty(communities)
    flow2 = np.empty(communities)
    flow3 = np.empty(communities)
    flow4 = np.empty(communities) 
            
    for x in range(communities):
        flow1[x] = np.random.uniform(flowlim[0], flowlim[1])
        flow2[x] = np.random.uniform(flow1[x], flowlim[1])
        flow3[x] = np.random.uniform(flow2[x], flowlim[1])
        flow4[x] = np.random.uniform(flow3[x], flowlim[1])
        
    cm_ax = np.random.uniform(min_a,max_a)
    cm_ay = np.random.uniform(min_a,max_a)
    m = np.random.uniform(min_m,max_m)
    # # If fixing parameters
    # maxlimits_vec[24] = true_vec_parameters[24]
    # maxlimits_vec[25] = true_vec_parameters[25]
    # vec_parameters = true_vec_parameters

    for c in range(communities):
        pr_flow2[c] = flowlim[1] - flow1[c]
        pr_flow3[c] = flowlim[1] - flow2[c]
        pr_flow4[c] = flowlim[1] - flow3[c]
        pr_sed2[c] = sedlim[1] - sed1[c]
        pr_sed3[c] = sedlim[1] - sed2[c]
        pr_sed4[c] = sedlim[1] - sed3[c]
    prs_flow = np.array([pr_flow2,pr_flow3,pr_flow4])
    c_pr_flow = np.prod(prs_flow)
    prs_sed = np.array([pr_sed2,pr_sed3,pr_sed4])
    c_pr_sed = np.prod(prs_sed)

    init_pro = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
    init_pro = np.append(init_pro,(cm_ax,cm_ay,m)) 

    print('Initial parameters:', init_pro) 

    return init_pro, c_pr_flow, c_pr_sed'''



def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_core_format(core, communities):
    vec = np.zeros(core.shape[0])
    for n in range(len(vec)):
        idx = np.argmax(core[n,:])# get index,
        vec[n] = idx+1 # +1 so that zero is preserved as 'none'
    return vec

 

def core_convertbinary(core_data): 

    core_binary = np.zeros((core_data.shape[0], 7))


    for i in range(core_data.shape[0]):  
        assem_num = int(round(core_data[i] * 7) -1)
        core_binary[i,assem_num] = 1
        #print(assem_num, ' assem_num')

    #print(core_binary)

    return core_binary
 

def main():

    random.seed(time.time()) 


    if(len(sys.argv)!=5):
        sys.exit('not right input format.  ')



    problem = int(sys.argv[1])  # get input

    num_chains = int(sys.argv[2])

    swap_interval = int(sys.argv[3])

    samples = int(sys.argv[4])

    print (problem, num_chains,   swap_interval)

    #-------------------------------------------------------------------------------------
    # Number of chains of MCMC required to be run
    # PT is a multicore implementation must num_chains >= 2
    # Choose a value less than the numbe of core available (avoid context swtiching)
    #-------------------------------------------------------------------------------------
    #samples = 5000    # total number of samples by all the chains (replicas) in parallel tempering
    #num_chains = 10 # number of Replica's that will run on separate cores. Note that cores will be shared automatically - if enough cores not available
      
    burn_in = 0.2 


    #parameters for Parallel Tempering
    maxtemp = 5 

    pt_stage = 0.90
    
     
 

    config = 0

    if config ==0: 
        step_m = 0.02 
        step_a = 0.02  
        max_a = -0.15
        max_m = 0.15 
        sedlim = [0., 0.005]
        flowlim = [0.,0.3]

        step_sed = 0.001 
        step_flow = 0.05

    elif config ==1:
        step_m = 0.1 
        step_a = 0.02   
        step_sed = 0.005 
        step_flow = 0.1 
        
        max_a = -0.15 
        max_m = 0.25 
        sedlim = [0., 0.01]
        flowlim = [0.,0.5]

    elif config == 2: 
        step_m = 0.2
        step_a = 0.2   
        step_sed = 0.01 
        step_flow = 0.2 

        max_a = -0.5 
        max_m = 0.5 
        sedlim = [0., 0.05]
        flowlim = [0.,1] 
    else: 
        print(' input pl ')
        



    stepsize_ratio = [step_m, step_a, step_sed, step_flow]
    max_limits = [max_m, max_a, sedlim[1], flowlim[1]]
    min_limits = [0, 0, sedlim[0], flowlim[0] ]

    print(stepsize_ratio, ' stepsize_ratio')

    print(max_limits, '  max_limits')

    print(min_limits, ' min_limits')


    
    if problem == 0:
        '''num_communities = 3 # can be 6 for real probs
        num_flow = 4
        num_sed = 4
        simtime = 8500 
        sedlim = [0., 0.005]
        flowlim = [0.,0.3]
        min_a = -0.15 # Community interaction matrix diagonal and sub-diagnoal limits
        max_a = 0.
        min_m = 0.
        max_m = 0.15 # Malthusian parameter limit

        maxlimits_vec, minlimits_vec = find_limits(num_communities, num_sed, num_flow, sedlim, flowlim, min_a, max_a, min_m, max_m)

        stepsize_ratio  = 0.05 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
        stepratio_vec =  np.repeat(stepsize_ratio, maxlimits_vec.size) 
        num_param = maxlimits_vec.size 

        problemfolder = 'SyntheticProblem/'  # change for other reef-core (This is synthetic core) 

        true_vec_parameters = np.loadtxt(problemfolder +'data_new/core_3asemb/true_values.txt') '''

    '''elif problem ==1:


        num_communities = 3 # can be 6 for real probs
        num_param = 27

        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_synth_.xml'
        datafile = 'data/synth_core.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data = np.loadtxt('data/synth_core_bi.txt')

        true_vec_parameters = np.loadtxt('data/true_values.txt')  
        problemfolder = 'Syntheticreef_results/'  # change for other reef-core (This is synthetic core) 


    elif problem ==2:

        num_communities = 3 # can be 6 for real probs
        num_param = 27
        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_hi3_threeasembleges.xml'
        datafile = 'data/hi3.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data = np.loadtxt('data/hi3_binary.txt')


        true_vec_parameters = [] 

        problemfolder = 'Henonreef_results/'  # change for other reef-core (This is synthetic core) 


    elif problem ==3:
        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        #xmlinput = 'input_synth.xml'
        #datafile = 'data/synth_core.txt'
        #core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        #core_data = np.loadtxt('data/synth_core_bi.txt')'''

    if problem ==1:
        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_synth_.xml'
        datafile = 'data/synth_core.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data = np.loadtxt('data/synth_core_bi.txt')

        true_vec_parameters = np.loadtxt('data/true_values.txt')
        problemfolder = 'Syntheticreef_results/' 


        nCommunities = 3 

    elif problem ==2:

        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_hi3.xml'
        datafile = 'data/hi3.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt') 
        problemfolder = 'Henonreef_results/'
 
        nCommunities = 6

    elif problem ==3:
        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_oti5.xml'
        datafile = 'data/oti5.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt') 
        problemfolder = 'Onetreereef_results/'

        nCommunities = 6
    elif problem ==4:
        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_oti2.xml'
        datafile = 'data/oti2.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt') 

        nCommunities = 6
        problemfolder = 'Onetreereef_results/'

 

    vis = [False, False] # first for initialisation, second for cores
    sedsim, flowsim = True, True  # can pass this to pt class later

    fname = ""
    run_nb = 0
    while os.path.exists(problemfolder +'results_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists(problemfolder +'results_%s' % (run_nb)):
        os.makedirs(problemfolder +'results_%s' % (run_nb))
        fname = (problemfolder +'results_%s' % (run_nb))
 
    make_directory((fname + '/posterior/pos_parameters'))  
    make_directory((fname + '/posterior/predicted_core/pos_samples_d'))
    make_directory((fname + '/posterior/pos_likelihood'))
    make_directory((fname + '/posterior/accept_list')) 

    make_directory((fname + '/plot_pos'))

    run_nb_str = 'results_' + str(run_nb)

    #-------------------------------------------------------------------------------------
    #Create A a Patratellel Tempering object instance 
    #-------------------------------------------------------------------------------------
    timer_start = time.time()


    num_param = 3 + (nCommunities * 8 )  # 3  for the mal, cim_ax, cim_ay 


    #def __init__(self,num_chains,communities, NumSample,fname,xmlinput,num_param,maxtemp,swap_interval,simtime,true_vec_parameters,   core_depths, core_data, vis, num_communities,   maxlimits_vec, minlimits_vec , stepratio_vec,     burn_in):

    

    pt = ParallelTempering(problem, num_chains,nCommunities, samples,fname,xmlinput,num_param,maxtemp,swap_interval,simtime, true_vec_parameters, core_depths, core_data, vis, max_limits, min_limits, stepsize_ratio, burn_in,  pt_stage)
     
    pt.initialise_chains()
    #-------------------------------------------------------------------------------------
    #run the chains in a sequence in ascending order
    #-------------------------------------------------------------------------------------
    pos_param,likelihood_rep,  rep_acceptlist,   predcore_list, rep_diffscore = pt.run_chains()


    np.savetxt(fname+'/rep_diffscore.txt', rep_diffscore, fmt='%1.2f')  
    np.savetxt(fname+'/predcore_list.txt',  predcore_list, fmt='%1.2f')  
    np.savetxt(fname+'/posterior.txt', pos_param, fmt='%1.4e')


    sed_pos = pos_param[0:12,:] 
    flow_pos = pos_param[12:24,:]
    glv_pos =   pos_param[24:,]
 

    

    print('Successfully sampled') 


    plt.plot( likelihood_rep.flatten()) 
    plt.xlabel('Samples')
    plt.ylabel('likelihood') 
    plt.savefig( fname+'/rep_likelihoodlist.png')





    plt.plot( rep_diffscore.flatten())
    plt.title('Difference Score Evolution')
    plt.xlabel('Samples')
    plt.ylabel('Score') 
    plt.savefig( fname+'/rep_diffscore.png')
    plt.clf()


    timer_end = time.time() 
    likelihood = likelihood_rep[:,0] # just plot proposed likelihood  
    likelihood = np.asarray(np.split(likelihood,  num_chains ))
    
    s_range = np.arange(int((burn_in * samples)/num_chains),(samples/num_chains)+1, 1)
    sample_range = np.zeros((len(s_range), num_chains))
    # sample_range = np.zeros((num_chains,len(s_range)))
    #for i in range(num_chains):
        #sample_range[:,i] = s_range
    # sample_range = np.arange(int((burn_in * samples)/num_chains)+1,samples+1, 1)


    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111) 
    size = 12 
    ax.tick_params(labelsize=size) 
    plt.legend(loc='upper right')  
    ax.boxplot(sed_pos.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Sediment Posterior', fontsize=size) 
    plt.title("Boxplot of Sediment Posterior", fontsize=size) 
    plt.savefig(fname+'/sed_pos.pdf')
    plt.clf()


    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111) 
    size = 12
    ax.tick_params(labelsize=size)
    ax.boxplot(flow_pos.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Flow Posterior', fontsize=size) 
    plt.title("Boxplot of Flow Posterior", fontsize=size) 
    plt.savefig(fname+'/flow_pos.pdf')
    plt.clf()


    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111) 
    size = 12 
    ax.tick_params(labelsize=size)
    ax.boxplot(glv_pos.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('GLV Posterior', fontsize=size) 
    plt.title("Boxplot of GLV Posterior", fontsize=size) 
    plt.savefig(fname+'/glv_pos.pdf')
    plt.clf()



    plt.plot( rep_acceptlist.flatten()) 
    plt.xlabel('Samples')
    plt.ylabel('accepted') 
    plt.savefig( fname+'/rep_acceptlist.png')
    plt.clf()

    
    fx_mu = predcore_list.mean(axis=1)
    fx_high = np.percentile(predcore_list, 95, axis=1)
    fx_low = np.percentile(predcore_list, 5, axis=1)

    x_data =  core_depths

    data_vec = convert_core_format(core_data, nCommunities) 
  

    font = 8

    x_labels = ['Shallow', 'Mod-deep', 'Deep', 'Sediment','No growth']
    x_values = [1,2,3,4,5]


    fig = plt.figure(figsize=(4,4))
    suptitle = fig.suptitle('')
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor('#f2f2f3')
    ax1.plot(data_vec, x_data, label='Ground truth', color='k',linewidth=0.7)
    ax1.plot(fx_mu, x_data, label='Pred. (mean)',linestyle='--', linewidth=0.7)
    ax1.plot(fx_high, x_data, label='Pred. (5th percentile)',linestyle='--',linewidth=0.7)
    ax1.plot(fx_low, x_data, label='Pred. (95th percentile)',linestyle='--',linewidth=0.7)
    ax1.fill_betweenx(x_data, fx_low, fx_high, facecolor='mediumaquamarine', alpha=0.4)
    ax1.set_ylabel('Depth (meters)')
    ax1.set_ylim([0,np.amax(core_depths)])
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_labels, rotation=70)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)
 

    lgd = fig.legend(frameon=False,bbox_to_anchor = (0.45,0.19), borderpad=2., prop={'size':font-1})
    plt.tight_layout(pad=2.5)
    fig.savefig('%s/core_prediction.pdf' % (fname), bbox_extra_artists=(lgd,suptitle), bbox_inches='tight',dpi=200,transparent=False)
    plt.close('all')


if __name__ == "__main__": main() 