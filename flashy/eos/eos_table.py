"""This module contains the class EosTable that reads in an equation of 
state file as provided by https://sntheory.org/equationofstate
The EosTable ist callable in dens,temp,ye mode"""

import sys
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

class EosTable(h5py.File):
    """This class stores a given EoS table and contains the relevant 
    methods to EoS calls.
    
    Args:
      pathToFile (str): path to the EoS hdf5 file
      
    Attributes:
      ktoMeV (float): unit conversion
    
    Methods:
      __call__: Returns the full EoS information for a given state
    
    Note: 
      At this stage there is only a density temperature ye mode!
    """
    def __init__(self, pathToFile):
        """The constructor of the EosTable."""
        # call the constructor of the parent class
        super(EosTable, self).__init__(pathToFile,'r')
        
        # unit conversions
        self._KtoMeV = 8.617385687984878e-11
        self._MeVtoK = 1./self._KtoMeV
        self._MeVtoErg = 1.60217733e-6
        
        # load defined EoS variables and initalize interpolator
        self._loadEoS()
    
    def _loadEoS(self):
        """method that loads the data for all defined variables"""
        
        # load nodes
        self.logDensityNodes = self['/logrho'][()]
        self.logTemperatureNodes = self['/logtemp'][()]
        self.yeNodes = self['/ye'][()]
        
        
        nDensNodes = len(self.logDensityNodes)
        nTempNodes = len(self.logTemperatureNodes)
        nYeNodes = len(self.yeNodes)
        self._dataShape = (nYeNodes,nTempNodes,nDensNodes)

        
        # find and sort defined variables
        self.definedVars = [k for k in list(self.keys()) 
                            if self[k].shape==self._dataShape]
        self.definedVars.sort()
        
        # find the energy and pressure index
        self._enerIdx, self._presIdx = [idx for idx,var 
                        in enumerate(self.definedVars) if (var=='logenergy' 
                                                        or var=='logpress')]        

        # load EoS data
        varData = np.array([self['/' + var][()] for var in self.definedVars])
        # change the shape for the interpolator
        self._eosData = np.rollaxis(varData,0,varData.ndim)
        #print self._dataShape
        
        
        # load EoS energy shift
        self._energyShift = self['/energy_shift'][()][0]
        
        
        # initalize interpolator
        #TDOO: Add the kwargs as input for the interpolator
        self._tableInterpolator = RegularGridInterpolator(
                                    (self.yeNodes,
                                     self.logTemperatureNodes,
                                     self.logDensityNodes
                                    ), self._eosData)

        
    def __repr__(self):
        """print representation"""
        return "EoS Table" + \
               "\n filename = " + '"' + self.filename + '"' \
               "\n content: " + str(self.definedVars)
        
    def __call__(self,density,temperature,ye):
        """The call to the EoS
        It linearly interpolates the tabulated equation of state tabel.
        
        Args:
          density (float/numpy.array): density
          temperature (float/numpy.array): temperature
          ye (float/numpy.array): electron fraction
          
        Kwargs:
          TODO: add other modes, like density, internal energy mode...
        
        Retruns:
          content(list): names of the output data
          state(numpy.array): thermodynamic  state vector
        """
        
        ##check user input:
        try:
            densCoords = np.array(density)
            tempCoords = np.array(temperature)
            yeCoords = np.array(ye)
        except:
            raise RuntimeError("Cannot cast input to numpy arrays!")
        
        tempCoords *= self._KtoMeV
        
        densCoords = np.log10(densCoords)
        tempCoords = np.log10(tempCoords)
        

        
        # setup interpolation
        zyx = np.array([yeCoords,tempCoords,densCoords]).T
        
        result = self._tableInterpolator(zyx)
        
        # apply energy shift and unit conversion
        result[:,self._enerIdx] = 10.**result[:,self._enerIdx] - self._energyShift
#        result[:,self._enerIdx] *= self._MeVtoErg # table energy is in erg/g
        result[:,self._presIdx] = 10.**result[:,self._presIdx]
    

        
        return result
        
