##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReefCore synthetic coral reef core model app.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module builds the core records through time based on coral species evolution and
the interactions between the active forcing paramters.
"""
import os
import numpy
import pandas as pd
import skfuzzy as fuzz

import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class coreData:
    """
    This class defines the core parameters
    """

    def __init__(self, input = input):
        """
        Constructor.
        """

        self.dt = input.tCarb

        # Initial core depth
        self.topH = input.depth0

        # Production rate for each carbonate
        self.prod = input.speciesProduction
        self.names = input.speciesName

        # Core parameters size based on layer number
        self.layNb = int((input.tEnd - input.tStart)/input.laytime)+1
        self.thickness = numpy.zeros(self.layNb,dtype=float)
        self.coralH = numpy.zeros((input.speciesNb+1,self.layNb),dtype=float)

        # Diagonal part of the community matrix (coefficient ii)
        self.communityMatrix = input.communityMatrix
        self.alpha = input.communityMatrix.diagonal()
        self.layTime = numpy.arange(input.tStart, input.tEnd+input.laytime, input.laytime)
        self.sealevel = numpy.zeros(len(self.layTime),dtype=float)
        self.sedinput = numpy.zeros(len(self.layTime),dtype=float)
        self.waterflow = numpy.zeros(len(self.layTime),dtype=float)
        self.maxpop = input.maxpop

        # Shape functions
        self.seaOn = input.seaOn
        self.edepth = numpy.array([[0.,0.,1000.,1000.],]*input.speciesNb)
        if input.seaOn:
            self.edepth = input.enviDepth
        self.flowOn = input.flowOn
        self.eflow = numpy.array([[0.,0.,5000.,5000.],]*input.speciesNb)
        if input.flowOn:
            self.eflow = input.enviFlow
        self.sedOn = input.sedOn
        self.esed = numpy.array([[0.,0.,500.,500.],]*input.speciesNb)
        if input.sedOn:
            self.esed = input.enviSed

        # Environmental forces functions
        self.seatime = None
        self.sedtime = None
        self.flowtime = None
        # self.seaFunc = None
        # self.sedFunc = None
        # self.flowFunc = None
        self.seaFunc = input.sedfile
        self.sedFunc = input.sedfunc
        self.flowFunc = input.flowfunc
        self.sedfctx = None
        self.sedfcty = None
        self.flowfctx = None
        self.flowfcty = None
        #### JODIE ADDITION ####
        self.inputFile = input.inputfile

        return

    def _plot_fuzzy_curve(self, xd, xs, xf, dtrap, strap, ftrap, size,
                          dpi, font, colors, width, fname):

        matplotlib.rcParams.update({'font.size': font})

        for s in range(len(self.names)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=size, sharey=True, dpi=dpi)
            ax1.set_facecolor('#f2f2f3')
            ax2.set_facecolor('#f2f2f3')
            ax3.set_facecolor('#f2f2f3')
            fig.tight_layout()
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.plot(xd, dtrap[s], linewidth=width, label=self.names[s],c=colors[s])
            ax2.plot(xs, strap[s], linewidth=width, label=self.names[s],c=colors[s])
            ax3.plot(xf, ftrap[s], linewidth=width, label=self.names[s],c=colors[s])
            ax1.set_ylabel('Environment Factor',size=font+3)
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_xlabel('Water Depth [m]',size=font+2)
            ax3.set_xlabel('Flow Velocity [m/sec]',size=font+2)
            ax3.set_ylim(-0.1, 1.1)
            ax2.set_xlabel('Sediment Input [m/year]',size=font+2)
            ax2.set_ylim(-0.1, 1.1)
            ax3.yaxis.set_label_position("right")
            ax3.set_ylabel(self.names[s],size=font+3,fontweight='bold')
            # plt.show()
            if fname is not None:
                fig.savefig(fname+self.names[s]+'.pdf', bbox_inches='tight')
            plt.close()
        return

    def initialSetting(self, font=8, size=(8,2.5), size2=(8,3.5), width=3, dpi=80, fname=None):
        """
        Visualise the initial conditions of the model run.

        Parameters
        ----------
        variable : font
            Environmental shape figures font size

        variable : size
            Environmental shape figures size

        variable : size2
            Environmental function figures size

        variable : width
            Environmental shape figures line width

        variable : dpi
            Figure resolution

        variable : fname
            Save filename.
        """

        from matplotlib.cm import terrain
        # nbcolors = len(self.names)+3
        # JODIE EDIT: colour range from 0-1.8 from 0-1
        nbcolors = len(self.names)+3
        colors = terrain(numpy.linspace(0, 1.25, nbcolors))

        # print 'Community matrix aij representing the interactions between species:'
        # print ''
        cols = []
        ids = []
        for i in range(len(self.names)):
            cols.append('a'+str(i))
            ids.append('a'+str(i)+'j')
        df = pd.DataFrame(self.communityMatrix, index=ids)
        df.columns = cols
        # print df

        ############## <START JODIE ADDITION> ##############
        import os
        commstring = str(df)
        ############## <END JODIE ADDITION> ##############

        # print ''
        # print 'Species maximum production rates [m/y]:'
        # print ''
        index = [self.names]
        df = pd.DataFrame(self.prod,index=index)
        df.columns = ['Prod.']
        # print df

        ############# <START JODIE ADDITION> ##############
        # prodstring = str(df)
        # edepthstring = str(self.edepth)
        # eflowstring = str(self.eflow)
        # esedstring = str(self.esed)

        # if self.inputFile == 'input_synth.xml':
        #     # print "coreData.py -> input file:", self.inputFile
        #     if not os.path.exists('data/synthetic_core'):
        #         os.makedirs('data/synthetic_core')
        #     filename = ('data/synthetic_core')
        #     with file(('%s/initial_conditions.txt' % (filename)),'w') as outfile:
        #         outfile.write('Community matrix:\n')
        #         outfile.write(commstring)
        #         outfile.write('\n\nCommunity maximum production rates [m/y]:\n')
        #         outfile.write(prodstring)
        #         outfile.write('\n\nCommunity depth functions [m]:\n')
        #         outfile.write(edepthstring)
        #         outfile.write('\n\nCommunity sediment tolerance functions [m/year]:\n')
        #         outfile.write(esedstring)
        #         outfile.write('\n\nCommunity flow tolerance functions [m/sec]:\n')
        #         outfile.write(eflowstring)
        # elif self.inputFile == 'input_synth_windward_2.xml':
        #     # print "coreData.py -> input file:", self.inputFile
        #     if not os.path.exists('data/synthetic_core'):
        #         os.makedirs('data/synthetic_core')
        #     filename = ('data/synthetic_core')
        #     with file(('%s/initial_conditions_windward.txt' % (filename)),'w') as outfile:
        #         outfile.write('Community matrix:\n')
        #         outfile.write(commstring)
        #         outfile.write('\n\nCommunity maximum production rates [m/y]:\n')
        #         outfile.write(prodstring)
        #         outfile.write('\n\nCommunity depth functions [m]:\n')
        #         outfile.write(edepthstring)
        #         outfile.write('\n\nCommunity sediment tolerance functions [m/year]:\n')
        #         outfile.write(esedstring)
        #         outfile.write('\n\nCommunity flow tolerance functions [m/sec]:\n')
        #         outfile.write(eflowstring)
        # elif self.inputFile == 'input_synth_leeward.xml':
        #     # print "coreData.py -> input file:", self.inputFile
        #     if not os.path.exists('data/synthetic_core'):
        #         os.makedirs('data/synthetic_core')
        #     filename = ('data/synthetic_core')
        #     with file(('%s/initial_conditions_leeward.txt' % (filename)),'w') as outfile:
        #         outfile.write('Community matrix:\n')
        #         outfile.write(commstring)
        #         outfile.write('\n\nCommunity maximum production rates [m/y]:\n')
        #         outfile.write(prodstring)
        #         outfile.write('\n\nCommunity depth functions [m]:\n')
        #         outfile.write(edepthstring)
        #         outfile.write('\n\nCommunity sediment tolerance functions [m/year]:\n')
        #         outfile.write(esedstring)
        #         outfile.write('\n\nCommunity flow tolerance functions [m/sec]:\n')
        #         outfile.write(eflowstring)
        # else:
        #     print "coreData.py -> input file:", self.inputFile
        ############# <END JODIE ADDITION> ##############
        
        # print ''
        # print 'Environmental trapezoidal shape functions:'

        # Visualise fuzzy production curve
        xs = numpy.linspace(0, self.esed.max(), num=201, endpoint=True)
        xf = numpy.linspace(0, self.eflow.max(), num=201, endpoint=True)
        xd = numpy.linspace(0, self.edepth.max(), num=201, endpoint=True)
        dtrap = []
        strap = []
        ftrap = []
        for s in range(0,len(self.names)):
            dtrap.append(fuzz.trapmf(xd, self.edepth[s,:]))
            strap.append(fuzz.trapmf(xs, self.esed[s,:]))
            ftrap.append(fuzz.trapmf(xf, self.eflow[s,:]))

        self._plot_fuzzy_curve(xd, xs, xf, dtrap, strap, ftrap, size,
                               dpi, font, colors, width, fname)

        if self.seaFunc is None and self.sedFunc is None and self.flowFunc is None:
            if self.sedfctx is None and self.flowfctx is None:
                return

        # print ''
        # print 'Environmental functions:'

        if self.seaFunc is not None and self.sedFunc is not None and self.flowFunc is not None:
            matplotlib.rcParams.update({'font.size': font})
            fig = plt.figure(figsize=size2, dpi=dpi)
            gs = gridspec.GridSpec(1,12)
            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:8]) #, sharey=ax1)
            ax3 = fig.add_subplot(gs[8:12]) #, sharey=ax1)
            ax1.set_facecolor('#f2f2f3')
            ax2.set_facecolor('#f2f2f3')
            ax3.set_facecolor('#f2f2f3')
            # Legend, title and labels
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.locator_params(axis='x', nbins=4)
            ax2.locator_params(axis='x', nbins=5)
            ax3.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=10)
            ax1.plot(self.seaFunc(self.seatime), self.seatime, linewidth=width, c='slateblue')
            ax1.set_xlim(self.seaFunc(self.seatime).min()-0.0001, self.seaFunc(self.seatime).max()+0.0001)
            ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax2.set_xlim(self.sedFunc(self.sedtime).min(), self.sedFunc(self.sedtime).max())
            ax3.plot(self.flowFunc(self.flowtime), self.flowtime, linewidth=width, c='darkcyan')
            ax3.set_xlim(self.flowFunc(self.flowtime).min()-0.0001, self.flowFunc(self.flowtime).max()+0.0001)
            # Axis
            ax1.set_ylabel('Time [years]', size=font+2)
            # Title
            tt1 = ax1.set_title('Sea-level [m]', size=font+3)
            tt2 = ax2.set_title('Water flow [m/sec]', size=font+3)
            tt3 = ax3.set_title('Sediment input [m/year]', size=font+3)
            tt1.set_position([.5, 1.03])
            tt2.set_position([.5, 1.03])
            tt3.set_position([.5, 1.03])
            fig.tight_layout()
            # plt.show()
            plt.close()
            return

        if self.seaFunc is not None and self.sedFunc is not None:
            matplotlib.rcParams.update({'font.size': font})
            fig = plt.figure(figsize=size2, dpi=dpi)
            gs = gridspec.GridSpec(1,12)
            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:8], sharey=ax1)
            ax1.set_facecolor('#f2f2f3')
            ax2.set_facecolor('#f2f2f3')
            # Legend, title and labels
            ax1.grid()
            ax2.grid()
            ax1.locator_params(axis='x', nbins=4)
            ax2.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=10)
            ax1.plot(self.seaFunc(self.seatime), self.seatime, linewidth=width, c='slateblue')
            ax1.set_xlim(self.seaFunc(self.seatime).min()-0.0001, self.seaFunc(self.seatime).max()+0.0001)
            ax2.plot(self.sedFunc(self.sedtime), self.sedtime, linewidth=width, c='sandybrown')
            ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax2.set_xlim(self.sedFunc(self.sedtime).min(), self.sedFunc(self.sedtime).max())
            # Axis
            ax1.set_ylabel('Time [years]', size=font+2)
            # Title
            tt1 = ax1.set_title('Sea-level [m]', size=font+2)
            tt2 = ax2.set_title('Sediment input [m/year]', size=font+2)
            tt1.set_position([.5, 1.03])
            tt2.set_position([.5, 1.03])
            fig.tight_layout()
            # plt.show()
            plt.close()
            if self.flowfcty is not None:
                fig = plt.figure(figsize=size2, dpi=dpi)
                gs = gridspec.GridSpec(1,12)
                ax1 = fig.add_subplot(gs[:4])
                ax1.set_facecolor('#f2f2f3')
                # Legend, title and labels
                ax1.grid()
                ax1.locator_params(axis='x', nbins=4)
                ax1.locator_params(axis='y', nbins=10)
                ax1.plot(self.flowfctx, self.flowfcty, linewidth=width, c='darkcyan')
                ax1.set_xlim(self.flowfctx.min(), self.flowfctx.max())
                # Axis
                ax1.set_ylabel('Depth [m]', size=font+2)
                # Title
                tt1 = ax1.set_title('Water flow [m/sec]', size=font+3)
                tt1.set_position([.5, 1.03])
                # plt.show()
                plt.close()

            return

        if self.seaFunc is not None and self.flowFunc is not None:
            matplotlib.rcParams.update({'font.size': font})
            fig = plt.figure(figsize=size2, dpi=dpi)
            gs = gridspec.GridSpec(1,12)
            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:8], sharey=ax1)
            ax1.set_facecolor('#f2f2f3')
            ax2.set_facecolor('#f2f2f3')
            # Legend, title and labels
            ax1.grid()
            ax2.grid()
            ax1.locator_params(axis='x', nbins=4)
            ax2.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=10)
            ax1.plot(self.seaFunc(self.seatime), self.seatime, linewidth=width, c='slateblue')
            ax1.set_xlim(self.seaFunc(self.seatime).min()-0.0001, self.seaFunc(self.seatime).max()+0.0001)
            ax2.plot(self.flowFunc(self.sedtime), self.sedtime, linewidth=width, c='darkcyan')
            ax2.set_xlim(self.flowFunc(self.sedtime).min(), self.flowFunc(self.sedtime).max())
            # Axis
            ax1.set_ylabel('Time [years]', size=font+2)
            # Title
            tt1 = ax1.set_title('Sea-level [m]', size=font+2)
            tt2 = ax2.set_title('Water flow [m/sec]', size=font+2)
            tt1.set_position([.5, 1.03])
            tt2.set_position([.5, 1.03])
            fig.tight_layout()
            # plt.show()
            plt.close()

            if self.sedfcty is not None:
                fig = plt.figure(figsize=size2, dpi=dpi)
                gs = gridspec.GridSpec(1,12)
                ax1 = fig.add_subplot(gs[:4])
                ax1.set_facecolor('#f2f2f3')
                # Legend, title and labels
                ax1.grid()
                ax1.locator_params(axis='x', nbins=4)
                ax1.locator_params(axis='y', nbins=10)
                ax1.plot(self.sedfctx, self.sedfcty, linewidth=width, c='sandybrown')
                ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.set_xlim(self.sedfctx.min(), self.sedfctx.max())
                # Axis
                ax1.set_ylabel('Depth [m]', size=font+2)
                # Title
                tt1 = ax1.set_title('Sediment input [m/year]', size=font+2)
                tt1.set_position([.5, 1.03])
                # plt.show()
                plt.close()

            return

        else:
            matplotlib.rcParams.update({'font.size': font})
            fig = plt.figure(figsize=size2, dpi=dpi)
            gs = gridspec.GridSpec(1,12)
            ax1 = fig.add_subplot(gs[:4])
            ax1.set_facecolor('#f2f2f3')
            # Legend, title and labels
            ax1.grid()
            ax1.locator_params(axis='x', nbins=4)
            ax1.locator_params(axis='y', nbins=10)
            if self.seaFunc is not None:
                ax1.plot(self.seaFunc(self.seatime), self.seatime, linewidth=width, c='slateblue')
                ax1.set_xlim(self.seaFunc(self.seatime).min()-0.0001, self.seaFunc(self.seatime).max()+0.0001)
            else:
                ax1.plot(numpy.zeros(len(self.layTime)), self.layTime, linewidth=width, c='slateblue')
                ax1.set_xlim(-0.1, 0.1)
            # Axis
            ax1.set_ylabel('Time [years]', size=font+2)
            # Title
            tt1 = ax1.set_title('Sea-level [m]', size=font+2)
            tt1.set_position([.5, 1.03])
            # plt.show()
            plt.close()

            if self.sedfcty is not None:
                fig = plt.figure(figsize=size2, dpi=dpi)
                gs = gridspec.GridSpec(1,12)
                ax1 = fig.add_subplot(gs[:4])
                ax1.set_facecolor('#f2f2f3')
                # Legend, title and labels
                ax1.grid()
                ax1.locator_params(axis='x', nbins=4)
                ax1.locator_params(axis='y', nbins=10)
                ax1.plot(self.sedfctx, self.sedfcty, linewidth=width, c='sandybrown')
                ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.set_xlim(self.sedfctx.min(), self.sedfctx.max())
                # Axis
                ax1.set_ylabel('Depth [m]', size=font+2)
                # Title
                tt1 = ax1.set_title('Sediment input [m/year]', size=font+2)
                tt1.set_position([.5, 1.03])
                # plt.show()
                plt.close()

            if self.flowfcty is not None:
                fig = plt.figure(figsize=size2, dpi=dpi)
                gs = gridspec.GridSpec(1,12)
                ax1 = fig.add_subplot(gs[:4])
                ax1.set_facecolor('#f2f2f3')
                # Legend, title and labels
                ax1.grid()
                ax1.locator_params(axis='x', nbins=4)
                ax1.locator_params(axis='y', nbins=10)
                ax1.plot(self.flowfctx, self.flowfcty, linewidth=width, c='darkcyan')
                ax1.set_xlim(self.flowfctx.min(), self.flowfctx.max())
                # Axis
                ax1.set_ylabel('Depth [m]', size=font+2)
                # Title
                tt1 = ax1.set_title('Water flow [m/sec]', size=font+2)
                tt1.set_position([.5, 1.03])
                # plt.show()
                plt.close

            # plt.show()
            plt.close()

        return

    def coralProduction(self, layID, coral, epsilon, sedh):
        """
        This function estimates the coral growth based on newly computed population.

        Parameters
        ----------

        variable : layID
            Index of current stratigraphic layer.

        variable : coral
            Species population distribution at current time step.

        variable : epsilon
            Intrinsic rate of a population species (malthus parameter)

        variable : sedh
            Silicilastic sediment input m/d
        """

        # Compute production for the given time step [m]
        production = numpy.zeros((coral.shape))
        ids = numpy.where(epsilon>0.)[0]
        production[ids] = self.prod[ids] * coral[ids] * self.dt / self.maxpop
        maxProd = self.prod * self.dt
        tmpids = numpy.where(production>maxProd)[0]
        production[tmpids] = maxProd[tmpids]

        # Total thickness deposited
        sh = sedh * self.dt
        toth = production.sum() + sh

        # In case there is no accomodation space
        if self.topH < 0.:
            # Do nothing
            return

        # If there is some accomodation space but it is all filled by sediment
        elif self.topH > 0. and self.topH - sh < 0.:
            # Just add the sediments to the sea-level
            self.coralH[len(self.prod),layID] += self.topH
            # Update current layer thickness
            self.thickness[layID] += self.topH
            # Update current layer top elevation
            self.topH = 0.

        # If there is some accomodation space that will disappear due to a
        # combination of carbonate growth and sediment input
        elif self.topH > 0. and self.topH - toth < 0:
            maxcarbh = self.topH - sh
            frac = maxcarbh/production.sum()
            production *= frac
            toth = production.sum() + sh

            # Update current layer composition
            self.coralH[0:len(self.prod),layID] += production
            # Convert sediment input from m/d to m/a
            self.coralH[len(self.prod),layID] += sh
            # Update current layer thickness
            self.thickness[layID] += toth
            # Update current layer top elevation
            self.topH -= toth

        # Otherwise
        elif self.topH > 0.:
            # Update current layer composition
            self.coralH[0:len(self.prod),layID] += production
            # Convert sediment input from m/d to m/a
            self.coralH[len(self.prod),layID] += sh
            # Update current layer thickness
            self.thickness[layID] += toth
            # Update current layer top elevation
            self.topH -= toth

        return
