##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesReef modelling software                         ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to plot histograms of parameters used in BayesReef.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

def plotResults(fname, sedsim, flowsim,communities, 
    pos_m, pos_ax, pos_ay, true_m, true_ax, true_ay,
    pos_sed1, pos_sed2, pos_sed3, pos_sed4, true_sed,
    pos_flow1, pos_flow2, pos_flow3, pos_flow4,true_flow):
    nb_bins=30
    slen = np.arange(0,pos_m.shape[0],1)
    font = 10
    width = 1

    #########################
    #   MALTHUS PARAMETER   #
    #########################
    mmin, mmax = min(pos_m), max(pos_m)
    mspace = np.linspace(mmin,mmax,len(pos_m))
    mm,ms = stats.norm.fit(pos_m)
    pdf_m = stats.norm.pdf(mspace,mm,ms)
    mmean=np.mean(pos_m)
    mmedian=np.median(pos_m)
    mmode,count=stats.mode(pos_m)

    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_title(' Malthusian Parameter', fontsize= font+2)#, y=1.02)
    
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_m, bins=25, alpha=0.5, facecolor='sandybrown', normed=True)
    ax1.axvline(true_m, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_ylabel('Frequency',size=font+1)
    ax1.set_xlabel(r'$\varepsilon$', size=font+1)
    
    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(slen,pos_m,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_title(r'Trace of $\varepsilon$',size=font+2)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel(r'$\varepsilon$', size=font+1)
    ax2.set_xlim([0,np.amax(slen)])
    
    # fig.subplots_adjust(top=0.88)
    plt.savefig('%s/malthus.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
    plt.clf()
    
    #########################
    #    COMMUNITY MATRIX   #
    #########################
    a1min, a1max = min(pos_ax), max(pos_ax)
    a1space = np.linspace(a1min,a1max,len(pos_ax))
    a1m,a1s = stats.norm.fit(pos_ax)
    pdf_a1 = stats.norm.pdf(a1space,a1m,a1s)
    a2min, a2max = min(pos_ay), max(pos_ay)
    a2space = np.linspace(a2min,a2max,len(pos_ay))
    a2m,a2s = stats.norm.fit(pos_ay)
    pdf_a2 = stats.norm.pdf(a2space,a2m,a2s)
    a1min=a1min
    a1max=a1max
    a1mean=np.mean(pos_ax)
    a1median=np.median(pos_ax)
    a1mode,count=stats.mode(pos_ax)
    a2min=a2min
    a2max=a2max
    a2mean=np.mean(pos_ay)
    a2median=np.median(pos_ay)
    a2mode,count=stats.mode(pos_ay)

    ###################
    #   MAIN DIAGONAL #
    ###################
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_title('Community Interaction Matrix Parameters', fontsize= font+2, y=1.03)
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_ax, bins=25, alpha=0.5, facecolor='mediumaquamarine', normed=True)
    ax1.axvline(true_ax, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_ylabel('Frequency',size=font+1)
    ax1.set_title(r'Main diagonal value ($\alpha_{m}$)',size=font+2)
    ax1.set_xlabel(r'$\alpha_{m}$', size=font+1)
    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(slen,pos_ax,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel(r'$\alpha_{m}$', size=font+1)
    ax2.set_title(r'Trace of $\alpha_{m}$',size=font+2)
    ax2.set_xlim([0,np.amax(slen)])
    
    # fig.subplots_adjust(top=0.88)
    plt.savefig('%s/comm_ax.png'% (fname),bbox_inches='tight', dpi=300,transparent=False)
    plt.clf()

    #############################
    #   SUPER-/SUB-DIAGONALS    #
    ############################# 
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_title('Community Interaction Matrix Parameters', fontsize= font+2, y=1.03)
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_ay, bins=25, alpha=0.5, facecolor='mediumaquamarine', normed=True)
    ax1.axvline(true_ay, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_title(r'Super- and sub-diagonal values ($\alpha_{s}$)',size=font+2)
    ax1.set_xlabel(r'$\alpha_{s}$', size=font+1)
    ax1.set_ylabel('Frequency',size=font+1)
    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(slen,pos_ay,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_title(r'Trace of $\alpha_{s}$',size=font+2)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel(r'$\alpha_{s}$', size=font+1)
    ax2.set_xlim([0,np.amax(slen)])
    
    # fig.subplots_adjust(top=0.88)
    plt.savefig('%s/comm_ay.png' % (fname), dpi=300, bbox_inches='tight',transparent=False)
    plt.clf()

    if not os.path.isfile(('%s/summ_stats.txt' % (fname))):
        with file(('%s/summ_stats.txt' % (fname)),'w') as outfile:
            outfile.write('SUMMARY STATISTICS\n')
            outfile.write('MIN, MAX, MEAN, MEDIAN, MODE\n')
            outfile.write('Malthusian parameter\n{0}, {1}, {2}, {3}, \n{4}\n'.format(mmin,mmax,mmean,mmedian,mmode))
            outfile.write('Main diagonal\n{0}, {1}, {2}, {3}, \n{4}\n'.format(a1min,a1max,a1mean,a1median,a1mode))
            outfile.write('Super-/Sub-diagonal\n{0}, {1}, {2}, {3}, \n{4}\n'.format(a2min,a2max,a2mean,a2median,a2mode))

    #############################################
    #   SEDIMENT AND FLOW RESPONSE THRESHOLDS   #
    #############################################

    a_labels = ['Shallow windward', 'Moderate-deep windward', 'Deep windward']#, 'Shallow leeward', 'Moderate-deep leeward', 'Deep leeward']
    
    sed1_mu, sed1_ub, sed1_lb, sed2_mu, sed2_ub, sed2_lb, sed3_mu, sed3_ub, sed3_lb, sed4_mu, sed4_ub, sed4_lb = (np.zeros(communities) for i in range(12))
    if ((sedsim != False)):
        for a in range(communities):
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
            sed1_mode,count=stats.mode(pos_sed1[:,a])
            sed2_mode,count=stats.mode(pos_sed2[:,a])
            sed3_mode,count=stats.mode(pos_sed3[:,a])
            sed4_mode,count=stats.mode(pos_sed4[:,a])


            with file(('%s/summ_stats.txt' % (fname)),'a') as outfile:
                outfile.write('\n# Sediment threshold: {0}\n'.format(a_labels[a]))
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
            ax.plot(true_sed[a,:], cy, linestyle='--', linewidth=width, marker='.',color='k', label='Synthetic data')
            ax.plot(cmu, cy, linestyle='-', linewidth=width,marker='.', color='sandybrown', label='Mean')
            ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1, color='darksalmon',mfc='darksalmon',fmt='.',label=None)
            ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1, color='sienna',mfc='sienna',fmt='.',label=None)
            plt.title('Sediment exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=font+2, y=1.06)
            plt.ylabel('Proportion of maximum growth rate [%]',size=font+1)
            plt.xlabel('Sediment input [m/year]',size=font+1)
            plt.ylim(-2.,110)
            lgd = plt.legend(frameon=False, prop={'size':font+1}, bbox_to_anchor = (1.,0.2))
            plt.savefig('%s/sediment_response_%s.png' % (fname, a+1), bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=300,transparent=False)
            plt.clf()

            ####################################################
            #   TRACE PLOTS OF SEDIMENT EXPOSURE THRESHOLDS    #
            ####################################################

            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.set_title('Trace plots of sediment input thresholds: %s ' % a_labels[a], fontsize= font+2, y=1.03)

            ax1 = fig.add_subplot(221)
            ax1.set_facecolor('#f2f2f3')
            ax1.plot(slen,pos_sed1[:,a], linestyle='-', color='k', linewidth=width)
            ax1.set_xlabel('Samples', size=font+1)
            ax1.set_ylabel(r'$f^1_{sed}$', size=font+1)
            ax1.set_xlim([0, np.amax(slen)])

            ax2 = fig.add_subplot(222)
            ax2.set_facecolor('#f2f2f3')
            ax2.plot(slen,pos_sed2[:,a], linestyle='-', color='k', linewidth=width)
            ax2.set_xlabel('Samples', size=font+1)
            ax2.set_ylabel(r'$f^2_{sed}$', size=font+1)
            ax2.set_xlim([0, np.amax(slen)])

            ax3 = fig.add_subplot(223)
            ax3.set_facecolor('#f2f2f3')
            ax3.plot(slen,pos_sed3[:,a], linestyle='-', color='k', linewidth=width)
            ax3.set_xlabel('Samples', size=font+1)
            ax3.set_ylabel(r'$\it{f}^3_{sed}$', size=font+1)
            ax3.set_xlim([0, np.amax(slen)])

            ax4 = fig.add_subplot(224)
            ax4.set_facecolor('#f2f2f3')
            ax4.plot(slen,pos_sed4[:,a], linestyle='-', color='k', linewidth=width)
            ax4.set_xlabel('Samples', size=font+1)
            ax4.set_ylabel(r'$\it{f}^4_{sed}$', size=font+1)
            ax4.set_xlim([0, np.amax(slen)])                
            
            fig.tight_layout()
            plt.savefig('%s/sed_threshold_trace_%s.png'% (fname, a),bbox_inches='tight', dpi=300,transparent=False)
            plt.clf()


    flow1_mu, flow1_ub,flow1_lb, flow2_mu, flow2_ub,flow2_lb, flow3_mu, flow3_ub,flow3_lb, flow4_mu, flow4_ub,flow4_lb = (np.zeros(communities) for i in range(12))
    if (flowsim != False):
        for a in range(communities):
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
            flow1_mode,count= stats.mode(pos_flow1[:,a])
            flow2_mode,count= stats.mode(pos_flow2[:,a])
            flow3_mode,count= stats.mode(pos_flow3[:,a])
            flow4_mode,count= stats.mode(pos_flow4[:,a])

            with file(('%s/summ_stats.txt' % (fname)),'a') as outfile:
                outfile.write('\n# Water flow threshold: {0}\n'.format(a_labels[a]))
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
            ax = fig.add_subplot(111)
            ax.set_facecolor('#f2f2f3')
            ax.plot(true_flow[a,:], cy, linestyle='--', linewidth=width, marker='.', color='k',label='Synthetic data')
            ax.plot(cmu, cy, linestyle='-', linewidth=width, marker='.', color='steelblue', label='Mean')
            ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1,color='lightsteelblue',mfc='lightsteelblue',fmt='.',label=None)
            ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1,color='lightslategrey',mfc='lightslategrey',fmt='.',label=None)
            plt.title('Hydrodynamic energy exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=font+2, y=1.06)
            plt.ylabel('Proportion of maximum growth rate [%]', size=font+1)
            plt.xlabel('Fluid flow [m/sec]', size=font+1)
            plt.ylim(-2.,110.)
            lgd = plt.legend(frameon=False, prop={'size':font+1}, bbox_to_anchor = (1.,0.2))
            plt.savefig('%s/flow_response_%s.png' % (fname, a+1),  bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300,transparent=False)
            plt.clf()

            #########################################################
            #   TRACE PLOTS OF FLOW VELOCITY EXPOSURE THRESHOLDS    #
            #########################################################
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.set_title('Trace plots of flow velocity Thresholds: %s ' % a_labels[a], fontsize= font+2, y=1.03)

            ax1 = fig.add_subplot(221)
            ax1.set_facecolor('#f2f2f3')
            ax1.plot(slen,pos_flow1[:,a], linestyle='-', color='k', linewidth=width)
            ax1.set_xlabel('Samples', size=font+1)
            ax1.set_ylabel(r'$f^1_{flow}$', size=font+1)
            ax1.set_xlim([0, np.amax(slen)])

            ax2 = fig.add_subplot(222)
            ax2.set_facecolor('#f2f2f3')
            ax2.plot(slen,pos_flow2[:,a], linestyle='-', color='k', linewidth=width)
            ax2.set_xlabel('Samples', size=font+1)
            ax2.set_ylabel(r'$f^2_{flow}$', size=font+1)
            ax2.set_xlim([0, np.amax(slen)])

            ax3 = fig.add_subplot(223)
            ax3.set_facecolor('#f2f2f3')
            ax3.plot(slen,pos_flow3[:,a], linestyle='-', color='k', linewidth=width)
            ax3.set_xlabel('Samples', size=font+1)
            ax3.set_ylabel(r'$\it{f}^3_{flow}$', size=font+1)
            ax3.set_xlim([0, np.amax(slen)])

            ax4 = fig.add_subplot(224)
            ax4.set_facecolor('#f2f2f3')
            ax4.plot(slen,pos_flow4[:,a], linestyle='-', color='k', linewidth=width)
            ax4.set_xlabel('Samples', size=font+1)
            ax4.set_ylabel(r'$\it{f}^4_{flow}$', size=font+1)
            ax4.set_xlim([0, np.amax(slen)])                
            
            fig.tight_layout()
            plt.savefig('%s/flow_threshold_trace_%s.png'% (fname, a),bbox_inches='tight', dpi=300,transparent=False)
            plt.clf()