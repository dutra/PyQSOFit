# PyQSOFit: A code for quasar spectrum fitting
# Auther: Hengxiao Guo AT SHAO
# Email: hengxiaoguo AT gmail DOT com
# Co-Auther Yue Shen, Shu Wang, Wenke Ren, Colin J. Burke
# Email: rwk AT mail DOT ustc DOT edu DOT cn
#
# -------------------------------------------------
# Version 1.2
# 10/26/2022 
# Key updates:
#     1) change the kmpfit to lmfit
#     2) no limit now for tie-line function
#     3) error estimation with MC or MCMC
#     4) coefficients in host decomposition are forced to be positive now
#     5) option for masking absorption pixels in emission line fitting.
#
# Version 1.2.1
# 07/12/2023
# Bug fix:
#     1) Add close fig function to avoid memory leak.
#        When savefig is set to True, the figure will be closed automatically. If one would like to show the figure
#        first, he/she can set savefig to False, call self.fig to exhibit the figure and save it manually.
#     2) Change the package sfdmap to sfdmap2.
#        Since the sfdmap is no longer maintained and has been conflicted with the latest numpy package, we change it
#        to a forked repository sfdmap2. ref: https://github.com/AmpelAstro/sfdmap2
#     3) Set the default jitter to 0.0 to avoid the unrepeatable fitting results. (temporary)
#        We do think the jitter will help to get unbiased fitting results especially when most of the initial parameters
#        are 0. However, since the jitter is not a physical parameter, it is not reasonable to set it as a default
#        global absolute value. Besides, variable results in repeated running are confusing to users. We will find out a
#        more feasible way to use the jitter in the future.
#
# Version 1.2.2
# 07/14/2023
# New features:
#     1) Add the second hdu table in qsopar.fits allow user to costume their results.
#        For now, there are only two params for user: the continuum luminosity wavelength and the Fe flux measurement
#        ranges.
#     2) Open the continuum luminosity measurement positions for user
#        Referencing the example of this code, one can add/delete/modify the wavelength positions for continuum
#        luminosity measurements in our final fits file by changing the params in qsopar.fits.
#     3) We adopt all local warning message in verbose switch.
#     4) Add a dict to better display the emission line names.
# Bug fix:
#     1) Improved the Luminosity measurement function
#     2) Fixed the error estimation function
#
# Version 1.2.3
# 07/19/2023
# Bug fix:
#     1) Add a few bug report in the to~do list
#     2) Automatically choose lstsq for PCA template
#     3) We uniformly set the default value of our code to -1.
#     4) Modify the function calculating the SN to avoid error arose by spectral discontinuous or low resolution.
#     5) Use the same initial logic for fur_result group and avoid the errors if the emission line is discard during
#     fitting procedure.
#     6）Amend the workflow of rejecting absorption line
#
# Version 1.2.4
# 07/24/2023
# Routine update:
#     1) Improve the host decomposition function using BC03 templates
#     2) Open the adjustment to the precision parameters for lmfit. By decrease the precision for continuum fitting,
#     the time consumption can be reduced and less likely tracked into local minimum.
# -------------------------------------------------
# Version 2.0
# 07/26/2023
# New Stable version for PyQSOFit supporting joint fitting of reverberation mapping spectra!
# New features:
#     1) Add new parameters to measure the host fraction at 5100A and 3000A.
#
# Version 2.1
# 07/28/2023
# New features:
#     1) Add a new method decompose the host component. In this method, we employ the prior of each PCQ template. By
#     applying a penalty, we restrict the factor of the last few component not to be too dominant. In this way, we
#     efficiently avoid the degeneration of the PCQ templates.
#     2) We rebuild the host decomposition module to make it more flexible. Now, the decomposition is performed through
#     HostDecomp.py module.
#     3) In HostDecomp.py module, we add the sigma measurements and Dn4000 estimation of the decomposed host component.
# -------------------------------------------------

import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

from sfdmap2 import sfdmap
from scipy import integrate, interpolate
from scipy.stats import median_abs_deviation
from lmfit import minimize, Parameters, report_fit

from PyAstronomy import pyasl
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM

from astropy.modeling.physical_models import BlackBody

from astropy.table import Table

from .HostDecomp import Prior_decomp
from .HostDecomp import Linear_decomp
from .HostDecomp import ppxf_kinematics

import warnings

warnings.filterwarnings("ignore")


class QSOFit():

    def __init__(self, lam, flux, err, z, ra=-999, dec=-999, plateid=None, mjd=None, fiberid=None, path=None,
                 and_mask_in=None, or_mask_in=None, wdisp=None):
        """
        Get the input data prepared for the QSO spectral fitting
        
        Parameters:
        -----------
        lam: 1-D array with Npix
             Observed wavelength in unit of Angstrom
             
        flux: 1-D array with Npix
             Observed flux density in unit of 10^{-17} erg/s/cm^2/Angstrom
        
        err: 1-D array with Npix
             1 sigma err with the same unit of flux
             
        z: float number
            redshift
        
        ra, dec: float number, optional 
            the location of the source, right ascension and declination. The default number is 0
        
        plateid, mjd, fiberid: integer number, optional
            If the source is SDSS object, they have the plate ID, MJD and Fiber ID in their file herader.
            
        path: str
            the path to the parameter file
            
        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube.

        wdisp: float or 1-D array with Npix, optional
            The instrumental dispersion of the spectra in unit of pixel. If it is a float, we will assume the
            dispersion is uniform for full spectra; If a 1-D array is given, then we use the array to calculate the
            dispersion for each pixel separately. If that value is not given, we will deem it as 69 km/s, the average
            value for SDSS spectra. This value is only useful when calculating the kinematics features of the host.

        """

        self.lam_in = np.asarray(lam, dtype=np.float64)
        self.flux_in = np.asarray(flux, dtype=np.float64)
        self.err_in = np.asarray(err, dtype=np.float64)
        self.z = z
        self.and_mask_in = and_mask_in
        self.or_mask_in = or_mask_in
        self.wdisp = wdisp
        self.ra = ra
        self.dec = dec
        self.plateid = plateid
        self.mjd = mjd
        self.fiberid = fiberid
        self.path = path
        self.install_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = path

    def Fit(self, name=None, nsmooth=1, and_mask=False, or_mask=False, reject_badpix=True, deredden=True,
            wave_range=None,
            wave_mask=None, decompose_host=True, host_prior=False, host_prior_scale=0.2, host_line_mask=True,
            decomp_na_mask=False,
            qso_type='global', npca_qso=10, host_type='BC03', npca_gal=5, Fe_uv_op=True,
            poly=False, BC=False, rej_abs_conti=False, rej_abs_line=False, initial_guess=None,
            n_pix_min_conti=100, param_file_name='qsopar.fits', MC=False, MCMC=False, save_fits_name=None,
            nburn=20, nsamp=200, nthin=10, epsilon_jitter=0., linefit=True, save_result=True, plot_fig=True,
            save_fits_path='.',
            save_fig=True, plot_corner=True, verbose=False, kwargs_plot={}, kwargs_conti_emcee={},
            kwargs_line_emcee={}):

        """
        Fit the QSO spectrum and get different decomposed components and corresponding parameters
        
        Parameter:
        ----------
        name: str, optional
            source name, Default is None. If None, it will use plateid+mjd+fiberid as the name. If there are no
            such parameters, it will be empty.
            
        nsmooth: integer number, optional
            do n-pixel smoothing to the raw input flux and err spectra. The default is set to 1 (no smooth).
            It will return the same array size. We note that smooth the raw data is not suggested, this function is in case of some fail-fitted low S/N spectra.
              
        and_mask: bool, optional
            If True, and and_mask or or_mask is not None, it will delete the masked pixels, and only return the remained pixels. Default: False
            
        or_mask: bool, optional
            If True, and and_mask or or_mask is not None, it will delete the masked pixels, and only return the remained pixels. Default: False
            
        reject_badpix: bool, optional
            reject 10 most possible outliers by the test of pointDistGESD. One important Caveat here is that this process will also delete narrow emission lines
            in some high SN ratio object e.g., [OIII]. Only use it when you are definitely clear about what you are doing. It will return the remained pixels.
        
        deredden: bool, optional
            correct the Galactic extinction only if the RA and Dec are available. It will return the corrected flux with the same array size. Default: True.
        
        wave_range: 2-element array, optional
            trim input wavelength (lam) according to the min and max range of the input 2-element array, e.g.,
            np.array([4000.,6000.]) in Rest frame range. Default: None
        
        wave_mask: 2-D array
            mask some absorption lines or sky lines in spectrum, e.g., np.array([[2200.,2300.]]), np.array([[5650.,5750.],[5850.,5900.]])
            
        decompose_host: bool, optional    
            If True, the host galaxy-QSO decomposition will be applied. If no more than 100 pixels are negative, the result will be applied. The Decomposition is
            based on the PCA method of Yip et al. 2004 (AJ, 128, 585) & (128, 2603). Now the template is only available for redshift < 1.16 in specific absolute
            magnitude bins. For galaxy, the global model has 10 PCA components and first 5 will enough to reproduce 98.37% galaxy spectra. For QSO, the global model
            has 50, and the first 20 will reproduce 96.89% QSOs. If have i-band absolute magnitude, the Luminosity-redshift binned PCA components are available.
            Then the first 10 PCA in each bin is enough to reproduce most QSO spectrum. Default: False

        host_prior: bool, optional
            This parameter is only useful when the decompose_host is True and BC03 is False. If True, the code will
            adopt the prior parameters given in the pca file to perform host decomposition. See arXiv:2406.17598 for
            more description about the functionality of this prior.

        host_prior_scale: float, optional
            If the prior decomposition is performed, the code will use this parameter to scale the prior penalty to the
            original chi2. Default: 0.2

        host_line_mask: bool, optional
            If True, the line region of galaxy will be masked when subtracted from original spectra. Default: True

        decomp_na_mask: bool, optional
            If True, the narrow line region will be masked when perform decomposition so that the model would not be
            affected by the emission lines. In cases we are using PCA templates to perform the decomposition,
            restricted by the template numbers, the model may not enough to recover all the emission lines with various
            width and strength. For purpose for only separating host continuum, we suggest to set this option as True.

        qso_type: str, optional
            The name of quasar PCA templates used in the host decomposition. This parameter can be set as 'global' or
            '{1}ZBIN{2}' where 1 is the luminosity bin from one of [A, B, C, D] and 2 is the redshift bin from one of
            [1, 2, 3, 4, 5]. Yip et al. (2004) built a series sets of quasar PCA templates based on different redshift
            and absolute i-band magnitude subsamples. Check https://doi.org/10.1086/425626 for more detail. If the
            host_prior is set to True, then only 'DZBIN1' and 'CZBIN1' is supported.

        npca_gal: int, optional
            the number of galaxy PCA components applied. It only works when decompose_host is True. The default is 5,
            which is already account for 98.37% galaxies.

        host_type: str, optional
            The name of galaxy templates used in the host decomposition. We have two tested build-in options for this
            parameter: PCA, BC03. Only PCA option is allowed if host_prior=True. For pro user who want to customize
            their own templates, please check Class host_template in HostDecomp.py.

        npca_qso: int, optional
            the number of QSO PCA components applied. It only works when decompose_host is True. The default is 20,
            No matter the global or luminosity-redshift binned PCA is used, it can reproduce > 92% QSOs. The binned PCA
            is better if have Mi information.

        BC03: bool, optional -- Unavailable
            if True, it will use Bruzual1 & Charlot 2003 host model to fit spectrum, high shift host will be low resolution R ~ 300, the rest is R ~ 2000. Default: False

        Mi: float, optional -- Unavailable
            i-band absolute magnitude. It only works when decompose_host is True. If not None, the Luminosity redshift binned PCA will be used to decompose
            the spectrum. Default: None

        Fe_uv_op: bool, optional
            if True, fit continuum with UV and optical FeII template. Default: True

        poly: bool, optional
            if True, fit continuum with the polynomial component to account for the dust reddening. Default: False
        
        BC: bool, optional
            if True, fit continuum with Balmer continua from 1000 to 3646A. Default: False
            
        rej_abs_conti: bool, optional
            if True, it will iterate the continuum fitting once, rejecting 3 sigma outlier absorption pixels in the continuum
            (< 3500A), which might fall into the broad absorption lines. Default: False
            
        rej_abs_line: bool, optional
            if True, it will iterate the emission line fitting twice, rejecting 3 sigma outlier absorption pixels
            which might fall into the broad absorption lines. Default: False
        
        n_pix_min_conti: float, optional
            minimum number of negative pixels for host continuuum fit to be rejected. Default: 100
            
        param_file_name: str, optional
            name of the qso fitting parameter FITS file. Default: 'qsopar.fits'
        
        MC: bool, optional 
            if True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array.
            if False, the code will not save the MLE minimization error produced by lmfit since it is biased and can not be trusted.
            But it can be still output by using the lmfit attribute. Default: False
            
        MCMC: bool, optional 
            if True, do Markov Chain Monte Carlo sampling of the posterior probability densities after MLE fitting to produce the error array.
            Note: An error will be thrown if both MC and MCMC are True. Default: False
            
        nburn: int, optional
            the number of burn-in samples to run MCMC chain if MCMC=True. It only works when MCMC is True. Default: 20
        
        nsamp: int, optional
            the number of trials of the MC process to produce the error array (if MC=True) or number samples to run MCMC chain (if MCMC=True). Should be larger than 20. It only works when either MC or MCMC is True. Default: 200
            
        linefit: bool, optional
            if True, the emission line will be fitted. Default: True
           
        save_result: bool, optional
            if True, all the fitting results will be saved to a fits file, Default: True
            
        plot_fig: bool, optional
            if True, the fitting results will be plotted. Default: True
                    
        save_fig: bool, optional
            if True, the figure will be saved, and the path can be set by "save_fig_path". Default: True
            
        plot_corner: bool, optinoal
            whether or not to plot the corner plot results if MCMC=True. Default: True
        
        save_fig_path: str, optional
            the output path of the figure. If None, the default "save_fig_path" is set to "path"
        
        save_fits_path: str, optional
            the output path of the result fits. If None, the default "save_fits_path" is set to "path"
        
        save_fits_name: str, optional
            the output name of the result fits. Default: "result.fits"
            
        verbose: bool, optional
            turn on (True) or off (False) debugging output. Default: False
            
        kwargs_plot: dict, optional
            extra aguments for plot_fig for plotting results. See LINK TO PLOT_FIG_DOC. Default: {}
            
        kwargs_conti_emcee: dict, optional
            extra aguments for emcee Sampler for continuum fitting. Default: {}
            
        kwargs_line_emcee: dict, optional
            extra arguments for emcee Sampler for line fitting. Default: {}
            
        Return:
        -----------
        
        
        
        Properties:
        -----------
        .wave: array
            the rest wavelength, some pixels have been removed.
            
        .flux: array
            the rest flux. Dereddened and *(1+z) flux.  
            
        .err: array
            the error.
        
        .wave_prereduced: array
            the wavelength after removing bad pixels, masking, deredden, spectral trim, and smoothing.
            
        .flux_prereduced: array
            the flux after removing bad pixels, masking, deredden, spectral trim, and smoothing.
            
        .err_prereduced: array
            the error after removing bad pixels, masking, deredden, spectral trim, and smoothing.
            
        .host: array
            the model of host galaxy from PCA method
               
        .qso: array
            the model of a quasar from PCA method.
            
        .SN_ratio_conti: float
            the mean S/N ratio of 1350, 3000 and 5100A.
            
        .conti_fit.: structure 
            all the continuum fitting results, including best-fit parameters and Chisquare, etc. For details,
            see https://lmfit.github.io/lmfit-py/fitting.html
            
        .f_conti_model: array
            the continuum model including power-law, polynomial, optical/UV FeII, Balmer continuum.
            
        .f_bc_model: array
            the Balmer continuum model.
            
        .f_fe_uv_model: array
            the UV FeII model.
            
        .f_fe_op_model: array
            the optical FeII model.
            
        .f_pl_model: array
            the power-law model.
            
        .f_poly_model: array
            the polynomial model.
            
        .PL_poly_BC: array
            The combination of Powerlaw, polynomial and Balmer continuum model.
            
        .line_flux: array
            the emission line flux after subtracting the .f_conti_model.
        
        .line_fit: structrue
            Line fitting results for last complexes (From Lya to Ha) , including best-fit parameters, errors (lmfit derived) and Chisquare, etc. For details,
            see https://lmfit.github.io/lmfit-py/fitting.html
        
        .gauss_result: array
            3*n Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.
            ADD UNITS
            
        gauss_result_all: array
            [nsamp, 3*n] Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of Gaussians for all complexes.
            ADD UNITS
            
        .conti_result: array
            continuum parameters, including widely used continuum parameters and monochromatic flux at 1350, 3000
            and 5100 Angstrom, etc. The corresponding names are listed in .conti_result_name. For all continuum fitting results,
            go to .conti_fit.params. 
            
        .conti_result_name: array
            the names for .conti_result.
            
        .fur_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered.
            The corresponding names are listed in .line_result_name.
            
        .fur_result_name: array
            the names for .fur_result.
            
        .line_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad emission line covered,
            and fitting parameters of each Gaussian component. The corresponding names are listed in .line_result_name.
            
        .line_result_name: array
            the names for .line_result.
            
        .uniq_linecomp_sort: array
            the sorted complex names.
            
        .all_comp_range: array
            the start and end wavelength for each complex. e.g., Hb is [4640.  5100.] AA.
            
        .linelist: array
            the information listed in the param_file_name (qsopar.fits).
        """

        # Parameters that are set here should generally not be changed unless you know what you are doing

        self.name = name
        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.decompose_host = decompose_host
        self.linefit = linefit
        self.host_line_mask = host_line_mask
        self.host_type = host_type
        self.qso_type = qso_type
        self.npca_gal = npca_gal
        self.npca_qso = npca_qso
        self.maxOLs = 10
        self.alpha = 0.05
        self.initial_guess = initial_guess
        self.Fe_uv_op = Fe_uv_op
        self.poly = poly
        self.BC = BC
        self.rej_abs_conti = rej_abs_conti
        self.rej_abs_line = rej_abs_line
        self.rej_abs_line_max_niter = 2
        self.n_pix_min_conti = n_pix_min_conti  # pixels
        self.MC = MC
        self.MCMC = MCMC
        self.nburn = nburn
        self.nsamp = nsamp
        self.nthin = nthin
        self.epsilon_jitter = epsilon_jitter
        self.kwargs_conti_emcee = kwargs_conti_emcee
        self.kwargs_line_emcee = kwargs_line_emcee
        self.save_fig = save_fig
        self.plot_corner = plot_corner
        self.verbose = verbose
        self.param_file_name = param_file_name

        # Initial precision parameters for lmfit
        self.xtol_conti = 1e-8
        self.ftol_conti = 1e-10
        self.xtol_line = 1e-10
        self.ftol_line = 1e-10

        # Initial parameters for prior decomposition
        self.host_prior = host_prior
        self.host_prior_scale = host_prior_scale
        self.decomp_na_mask = decomp_na_mask

        self.read_out_params(os.path.join(self.path, self.param_file_name))

        # get the source name in plate-mjd-fiber, if no then None
        if name is None:
            if np.array([self.plateid, self.mjd, self.fiberid]).any() is not None:
                self.sdss_name = str(self.plateid).zfill(4) + '-' + str(self.mjd) + '-' + str(self.fiberid).zfill(4)
            else:
                self.sdss_name = ''
        else:
            self.sdss_name = name

        if self.plateid is None:
            self.plateid = 0
        if self.mjd is None:
            self.mjd = 0
        if self.fiberid is None:
            self.fiberid = 0

        # set default path for figure and fits
        if save_fits_name == None:
            if self.sdss_name == '':
                save_fits_name = 'result'
            else:
                save_fits_name = self.sdss_name
        else:
            save_fits_name = save_fits_name

        dustmap_path = os.path.join(self.install_path, 'sfddata')

        # Clean the data

        # Remove with error equal to 0 or inifity
        ind_gooderror = np.where(
            (self.err_in > 0) & np.isfinite(self.err_in) & (self.flux_in != 0) & np.isfinite(self.flux_in), True, False)
        self.err = self.err_in[ind_gooderror]
        self.flux = self.flux_in[ind_gooderror]
        self.lam = self.lam_in[ind_gooderror]

        # Renew And/or mask index
        if (self.and_mask_in is not None) and (self.or_mask_in is not None):
            self.and_mask_in = self.and_mask_in[ind_gooderror]
            self.or_mask_in = self.or_mask_in[ind_gooderror]
        else:
            self.and_mask_in = None
            self.or_mask_in = None

        # Clean and/or mask
        if (and_mask == True) and (self.and_mask_in is not None):
            self._MaskSdssAndOr(self.lam, self.flux, self.err, and_mask, or_mask)
        # Clean bad pixel
        if reject_badpix == True:
            self._RejectBadPix(self.lam, self.flux, self.err)
        # Smooth the data
        if nsmooth is not None:
            self.flux = self.Smooth(self.flux, nsmooth)
            self.err = self.Smooth(self.err, nsmooth)
        # Set fitting wavelength range
        if wave_range is not None:
            self._WaveTrim(self.lam, self.flux, self.err, self.z)
        # Set manual wavelength mask
        if wave_mask is not None:
            self._WaveMsk(self.lam, self.flux, self.err, self.z)
        # Deredden
        if deredden == True and self.ra != -999. and self.dec != -999.:
            self._DeRedden(self.lam, self.flux, self.err, self.ra, self.dec, dustmap_path)

        self._RestFrame(self.lam, self.flux, self.err, self.z)
        self._CalculateSN(self.wave, self.flux)
        self._OrignialSpec(self.wave, self.flux, self.err)

        """
        Do host decomposition
        """
        if decompose_host == True:
            self.decompose_host_qso(self.wave, self.flux, self.err, self.install_path)
        else:
            self.decomposed = False
            self.host_result = np.array([])
            self.host_result_type = np.array([])
            self.host_result_name = np.array([])

            self.frac_host_4200 = -1.
            self.frac_host_5100 = -1.

        """
        Fit the continuum
        """
        self.fit_continuum(self.wave, self.flux, self.err, self.ra, self.dec, self.plateid, self.mjd, self.fiberid)

        """
        Fit the emission lines
        """
        if linefit == True:
            self.fit_lines(self.wave, self.line_flux, self.err, self.conti_fit)
        else:
            self.ncomp = 0
            self.line_result = np.array([])
            self.line_result_type = np.array([])
            self.line_result_name = np.array([])
            self.gauss_result = np.array([])
            self.all_comp_range = np.array([])
            self.uniq_linecomp_sort = np.array([])

        """
        Save the results
        """
        if save_result == True:
            self.save_result(self.conti_result, self.conti_result_type, self.conti_result_name, self.line_result,
                             self.line_result_type, self.line_result_name, save_fits_path, save_fits_name)

        """
        Plot the results
        """
        if plot_fig == True:
            self.plot_fig(**kwargs_plot)

        return

    def _MaskSdssAndOr(self, lam, flux, err, and_mask, or_mask):
        """
        Remove SDSS and_mask and or_mask points are not zero
        Parameter:
        ----------
        lam: wavelength
        flux: flux
        err: 1 sigma error
        and_mask: SDSS flag "and_mask", mask out all non-zero pixels
        
        Retrun:
        ---------
        return the same size array of wavelength, flux, error
        """
        if (and_mask == True) and (or_mask == True):
            ind = np.where((self.and_mask_in == 0) & (self.and_mask_in == 0), True, False)
        if (and_mask == True) and (or_mask == False):
            ind = np.where(self.and_mask_in == 0, True, False)
        if (and_mask == False) and (or_mask == True):
            ind = np.where(self.or_mask == 0, True, False)

        self.lam, self.flux, self.err = lam[ind], flux[ind], err[ind]

    def _RejectBadPix(self, lam, flux, err, maxOLs=10, alpha=0.05):
        """
        Reject outliers in spectrum such as cosmic rays.
        See https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/outlier.html
        
        Parameter:
        ----------
        lam: array, required
            wavelength
            
        flux: array, required
            flux
            
        err: array, required
            1 sigma error
            
        maxOLS: int, optional
            Maximum number of outliers to reject. Default: 10
            
        alpha: float, optional
            Significance. Default: 0.05
            
        Return:
        ---------
        
        """
        # -----remove bad pixels, but not for high SN spectrum------------
        ind_bad = pyasl.pointDistGESD(flux, maxOLs, alpha)
        wv = np.asarray([i for j, i in enumerate(lam) if j not in ind_bad[1]], dtype=np.float64)
        fx = np.asarray([i for j, i in enumerate(flux) if j not in ind_bad[1]], dtype=np.float64)
        er = np.asarray([i for j, i in enumerate(err) if j not in ind_bad[1]], dtype=np.float64)
        # TODO: Below lines are confusing and generally bad practice
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = wv, fx, er
        return self.lam, self.flux, self.err

    def _WaveTrim(self, lam, flux, err, z):
        """
        Trim spectrum with a range in the rest frame. 
        """
        # trim spectrum e.g., local fit emiision lines
        ind_trim = np.where((lam / (1 + z) > self.wave_range[0]) & (lam / (1 + z) < self.wave_range[1]), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError("No enough pixels in the input wave_range!")
        return self.lam, self.flux, self.err

    def _WaveMsk(self, lam, flux, err, z):
        """Block the bad pixels or absorption lines in spectrum."""

        for msk in range(len(self.wave_mask)):
            try:
                ind_not_mask = ~np.where(
                    (lam / (1 + z) > self.wave_mask[msk, 0]) & (lam / (1 + z) < self.wave_mask[msk, 1]),
                    True, False)
            except IndexError:
                raise RuntimeError("Wave_mask should be 2D array, e.g., np.array([[2000,3000],[3100,4000]]).")
            # TODO: test if these arrays are needed to be deleted
            del self.lam, self.flux, self.err
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err

    def _DeRedden(self, lam, flux, err, ra, dec, dustmap_path):
        """Correct the Galactic extinction"""
        m = sfdmap.SFDMap(dustmap_path)
        zero_flux = np.where(flux == 0, True, False)
        flux[zero_flux] = 1e-10
        flux_unred = pyasl.unred(lam, flux, m.ebv(ra, dec))
        err_unred = err * flux_unred / flux
        flux_unred[zero_flux] = 0
        del self.flux, self.err
        self.flux = flux_unred
        self.err = err_unred
        return self.flux

    def _RestFrame(self, lam, flux, err, z):
        """Move wavelenth and flux to rest frame"""
        self.wave = lam / (1 + z)
        self.flux = flux * (1 + z)
        self.err = err * (1 + z)
        return self.wave, self.flux, self.err

    def _OrignialSpec(self, wave, flux, err):
        """Save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _CalculateSN(self, wave, flux, alter=True):
        """
        Calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots
        This function will automatically check if the 50A vicinity of at the default three wavelength contain more than
        10 pixels. If so, this function will calculate the continuum SN ratio from available regions. If not, it may
        imply that the give spectrum are very low resolution or have frequent gaps in their wavelength coverage. We
        provide another algorithm to calculate the SNR regardless of the continuum.
        :param wave:
        :param flux:
        :return:
        """
        ind5100 = np.where((wave > 5080) & (wave < 5130), True, False)
        ind3000 = np.where((wave > 3000) & (wave < 3050), True, False)
        ind1350 = np.where((wave > 1325) & (wave < 1375), True, False)

        if np.all(np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) < 10):

            if alter is False:
                self.SN_ratio_conti = -1.
                return self.SN_ratio_conti

            # referencing: www.stecf.org/software/ASTROsoft/DER_SNR/
            input_data = np.array(flux)
            # Values that are exactly zero (padded) are skipped
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            # For spectra shorter than this, no value can be returned
            if (n > 4):
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                self.SN_ratio_conti = float(signal / noise)
            else:
                self.SN_ratio_conti = -1.

        else:
            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(),
                               flux[ind1350].mean() / flux[ind1350].std()])
            tmp_SN = tmp_SN[np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) > 10]
            if not np.all(np.isnan(tmp_SN)):
                self.SN_ratio_conti = np.nanmean(tmp_SN)
            else:
                self.SN_ratio_conti = -1.

        return self.SN_ratio_conti

    def decompose_host_qso(self, wave, flux, err, path):
        """Decompose the host galaxy from QSO"""
        # Initialize default values
        self.host = np.zeros(len(wave))
        self.decomposed = True
        self.host_result = np.array([])
        self.host_result_type = np.array([])
        self.host_result_name = np.array([])

        if self.host_prior is True:
            prior_fitter = Prior_decomp(self.wave, self.flux, self.err, self.npca_gal, self.npca_qso,
                                        path, host_type=self.host_type, qso_type=self.qso_type,
                                        na_mask=self.decomp_na_mask)
            if prior_fitter.assertion is True:
                datacube, frac_host_4200, frac_host_5100, qso_par, gal_par = prior_fitter.auto_decomp(self.host_prior_scale)
            else:
                self.decomposed = False
                return self.wave, self.flux, self.err
        else:
            linear_fitter = Linear_decomp(self.wave, self.flux, self.err, self.npca_gal, self.npca_qso, path,
                                          host_type=self.host_type, qso_type=self.qso_type,
                                          na_mask=self.decomp_na_mask)
            if linear_fitter.assertion is True:
                datacube, frac_host_4200, frac_host_5100, qso_par, gal_par = linear_fitter.auto_decomp()
            else:
                self.decomposed = False
                return self.wave, self.flux, self.err

        # for some negative host template, we do not do the decomposition # not apply anymore
        # For a few cases, the host template is too weak that the host spectra (data - qso) would be mostly negative
        # through the data itself wouldn't
        flux_level = np.median(np.abs(datacube[1, :]))
        host_spec = datacube[1, :] - datacube[4, :]
        if np.sum(np.where(datacube[3, :] < 0, True, False) | np.where(datacube[4, :] < 0, True, False)) > 0.1 * \
                datacube.shape[1] or np.median(datacube[3, :]) < 0.01 * flux_level or np.median(host_spec) < 0:
            self.decomposed = False
            if self.verbose:
                print('Got negative host galaxy / QSO flux over 10% of coverage, decomposition is not applied!')
        else:
            self.wave = datacube[0, :]

            rchi2_decomp = np.sum((datacube[1, :] - datacube[4, :] - datacube[3, :]) ** 2 / datacube[2, :] ** 2) / (
                    len(datacube[1, :]) - self.npca_qso - self.npca_gal)

            # Block OIII, Ha, NII, SII, OII, Ha, Hb, Hr, Hdelta
            if self.host_line_mask == True:
                line_mask = np.where(
                    (self.wave < 4970) & (self.wave > 4950) | (self.wave < 5020) & (self.wave > 5000) |
                    (self.wave < 6590) & (self.wave > 6540) | (self.wave < 6740) & (self.wave > 6710) |
                    (self.wave < 3737) & (self.wave > 3717) | (self.wave < 4872) & (self.wave > 4852) |
                    (self.wave < 4350) & (self.wave > 4330) | (self.wave < 4111) & (self.wave > 4091),
                    True, False)
            else:
                line_mask = np.full(len(self.wave), False)

            f = interpolate.interp1d(self.wave[~line_mask], datacube[3, :][~line_mask], bounds_error=False,
                                     fill_value=0)
            masked_host = f(self.wave)
            self.spec = datacube[1, :]  # Original spectra flux
            self.flux = datacube[1, :] - masked_host  # QSO flux without host
            self.err = datacube[2, :]
            self.host = datacube[3, :]
            self.qso = datacube[4, :]
            self.host_data = datacube[1, :] - self.qso

            if self.MC is True:
                n_iter = self.nsamp
            else:
                n_iter = 0
            fit_range = (4000, 5350)
            try:
                fit_pp, ppxf_mask, ppxf_model = ppxf_kinematics(self.wave, self.host_data, self.err, path, fit_range,
                                                                MC_iter=n_iter)
                sigma, sigma_err, v_off, v_off_err, rchi2_ppxf = fit_pp
            except:
                ppxf_model = np.zeros(len(self.wave))
                ppxf_mask = np.ones(len(self.wave), dtype=bool)
                sigma, sigma_err, v_off, v_off_err, rchi2_ppxf = -999, -999, -999, -999, -999
            self.ppxf_mask = ppxf_mask
            self.ppxf_model = ppxf_model

            # TODO: A very messy way to get the SN ratio, will integrated to the _Calculate_SN function in the future.
            input_data = np.array(self.host_data)
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            # For spectra shorter than this, no value can be returned
            if (n > 4):
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                SN_host = float(signal / noise)
            else:
                SN_host = -1.

            # measure the Dn4000 of the host galaxy
            # TODO: will be separated as a independent function in the future
            lower_idx = np.where((self.wave > 3850) & (self.wave < 3950), True, False)
            upper_idx = np.where((self.wave > 4000) & (self.wave < 4100), True, False)
            if np.sum(lower_idx) > 10 and np.sum(upper_idx) > 10:
                # Convert the flux in unit of f_lambda into f_nu
                flux_lower = np.mean(self.host[lower_idx] * self.wave[lower_idx] ** 2) # * 3.34e-19
                flux_upper = np.mean(self.host[upper_idx] * self.wave[upper_idx] ** 2) # * 3.34e-19
                Dn4000 = flux_upper / flux_lower
            else:
                Dn4000 = -1.

            self.host_result = np.array(
                [SN_host, rchi2_decomp, frac_host_4200, frac_host_5100, Dn4000, sigma, sigma_err, v_off, v_off_err,
                 rchi2_ppxf])
            self.host_result_type = np.full(len(self.host_result), 'float')
            self.host_result_name = np.array(
                ['SN_host', 'rchi2_decomp', 'frac_host_4200', 'frac_host_5100', 'Dn4000', 'sigma', 'sigma_err', 'v_off',
                 'v_off_err', 'rchi2_ppxf'])

            self.host_result = np.concatenate([self.host_result, gal_par, qso_par])
            self.host_result_type = np.concatenate(
                [self.host_result_type, np.full(len(gal_par), 'float'), np.full(len(qso_par), 'float')])
            self.host_result_name = np.concatenate([self.host_result_name,
                                                    np.array(['gal_par_' + str(i) for i in range(len(gal_par))]),
                                                    np.array(['qso_par_' + str(i) for i in range(len(qso_par))])])

        return self.wave, self.flux, self.err

    def fit_continuum(self, wave, flux, err, ra, dec, plateid, mjd, fiberid):
        """
        Fit the continuum with PL/FeII/Balmer/poly components and (optionally) estimate uncertainties.

        All component functions are called with a full parameter dict (names->values), never slices.
        """

        # ---------- static resources ----------
        self.fe_uv = np.genfromtxt(os.path.join(self.install_path, "fe_uv.txt"))
        self.fe_op = np.genfromtxt(os.path.join(self.install_path, "fe_optical.txt"))

        # Read param table & continuum windows
        contilist, window_all = read_conti_params(os.path.join(self.path, self.param_file_name))
        param_names = contilist["parname"]  # array of strings

        # ---------- build global "in-window" mask ----------
        in_any_window = np.zeros_like(wave, dtype=bool)
        for lo, hi in window_all:
            in_any_window |= (wave > lo) & (wave < hi)

        n_pix_total = int(np.count_nonzero(in_any_window))
        if n_pix_total < 10 and getattr(self, "verbose", False):
            print("Less than 10 total pixels in the continuum windows to be fit.")
        if n_pix_total == 0:
            print("No pixels in the continuum windows to be fit.")  # hard warning

        # Convenience slices of windowed arrays
        w_win = wave[in_any_window]
        f_win = flux[in_any_window]
        e_win = err[in_any_window]

        # ---------- prepare lmfit.Parameters with small jitter ----------
        rng = np.random.default_rng()
        fit_params = Parameters()
        for name, init, vmin, vmax, vary in zip(
            contilist["parname"],
            contilist["initial"],
            contilist["min"],
            contilist["max"],
            contilist["vary"],
        ):
            init_jittered = init + abs(rng.normal(0.0, getattr(self, "epsilon_jitter", 0.0)))
            fit_params.add(name, value=init_jittered, min=vmin, max=vmax, vary=bool(vary))

        if getattr(self, "verbose", False):
            print("Parameters loaded:", list(param_names))

        # ---------- helper: freeze a component quickly ----------
        def _freeze(norm_name, names):
            if norm_name in fit_params:
                fit_params[norm_name].value = 0.0
            for nm in names:
                if nm in fit_params:
                    fit_params[nm].vary = False

        # ---------- enable/disable components by coverage ----------
        # UV FeII region
        uv_mask = (w_win > 1200.0) & (w_win < 3500.0)
        uv_ok = int(np.count_nonzero(uv_mask)) > getattr(self, "n_pix_min_conti", 20)
        if not (getattr(self, "Fe_uv_op", True) and uv_ok):
            _freeze("Fe_uv_norm", ["Fe_uv_norm", "Fe_uv_FWHM", "Fe_uv_shift"])

        # Optical FeII region
        op_mask = (w_win > 3686.0) & (w_win < 7484.0)
        op_ok = int(np.count_nonzero(op_mask)) > getattr(self, "n_pix_min_conti", 20)
        if not (getattr(self, "Fe_uv_op", True) and op_ok):
            _freeze("Fe_op_norm", ["Fe_op_norm", "Fe_op_FWHM", "Fe_op_shift"])

        # Balmer continuum (blueward of 3646 Å)
        bc_mask = (w_win < 3646.0)
        bc_ok = int(np.count_nonzero(bc_mask)) > 100
        if not (getattr(self, "BC", True) and bc_ok):
            _freeze("Balmer_norm", ["Balmer_norm", "Balmer_Te", "Balmer_Tau", "Balmer_vel"])

        # Polynomial component
        if not getattr(self, "poly", True):
            for nm in ["conti_a_0", "conti_a_1", "conti_a_2"]:
                if nm in fit_params:
                    fit_params[nm].value = 0.0
                    fit_params[nm].vary = False

        # ---------- model: components accept full dict ----------
        def _conti_model(x, p):
            pd = p.valuesdict()
            y = self.PL(x, pd)
            if getattr(self, "Fe_uv_op", True):
                y += self.Fe_flux_mgii(x, pd)    # UV FeII (Mg II template region)
                y += self.Fe_flux_balmer(x, pd)  # Optical FeII template (your naming kept)
            if getattr(self, "BC", True):
                y += self.Balmer_conti(x, pd)
            if getattr(self, "poly", True):
                y += self.F_poly_conti(x, pd)
            return y

        # ---------- tiny helpers used below ----------
        def _vec_to_dict(vec, names):
            return {nm: float(v) for nm, v in zip(names, vec)}

        def _evaluate_components(x, p_dict):
            pl = self.PL(x, p_dict)
            feUV = self.Fe_flux_mgii(x, p_dict) if getattr(self, "Fe_uv_op", True) else 0.0
            feOp = self.Fe_flux_balmer(x, p_dict) if getattr(self, "Fe_uv_op", True) else 0.0
            bc = self.Balmer_conti(x, p_dict) if getattr(self, "BC", True) else 0.0
            polyc = self.F_poly_conti(x, p_dict) if getattr(self, "poly", True) else 0.0
            total = pl + feUV + feOp + bc + polyc
            return pl, feUV, feOp, bc, polyc, total

        def _interleave(vals, errs):
            # [v0,e0,v1,e1,...]
            vals = np.asarray(vals)
            errs = np.asarray(errs)
            return list(np.ravel(np.column_stack([vals, errs])))

        def _append_fields(names_base, vals, arr, arr_types, arr_names):
            arr = np.append(arr, vals)
            arr_types = np.append(arr_types, ["float"] * len(vals))
            arr_names = np.append(arr_names, names_base if isinstance(names_base[0], str) else list(names_base))
            return arr, arr_types, arr_names

        # ---------- (optional) pre-fit diagnostics ----------
        if getattr(self, "verbose", False):
            fit_params.pretty_print()
            print("Fitting continuum (initial pass)…")

        # ---------- FIRST PASS ----------
        conti_fit = minimize(
            self._residuals,
            fit_params,
            args=(w_win, f_win, e_win, _conti_model),
            calc_covar=False,
            xtol=getattr(self, "xtol_conti", 1e-8),
            ftol=getattr(self, "ftol_conti", 1e-8),
        )

        # ---------- optional BAL trough rejection & SECOND PASS ----------
        params_dict = conti_fit.params.valuesdict()
        if getattr(self, "rej_abs_conti", True):
            model_win = _conti_model(w_win, conti_fit.params)
            ind_noBAL = ~(((f_win < model_win - 3.0 * e_win) & (w_win < 3500.0)))
            # If you *really* want smoothing, uncomment the next line (can bias fits):
            # f_refit = self.Smooth(f_win[ind_noBAL], 10)
            f_refit = f_win[ind_noBAL]
            conti_fit = minimize(
                self._residuals,
                conti_fit.params,
                args=(w_win[ind_noBAL], f_refit, e_win[ind_noBAL], _conti_model),
                calc_covar=False,
                xtol=getattr(self, "xtol_conti", 1e-8),
                ftol=getattr(self, "ftol_conti", 1e-8),
            )
            params_dict = conti_fit.params.valuesdict()
        else:
            ind_noBAL = np.ones_like(w_win, dtype=bool)

        if getattr(self, "verbose", False):
            print("Fit report")
            report_fit(conti_fit.params)

        # ---------- normalize best-fit structures ----------
        par_names = list(params_dict.keys())
        params_vec = list(params_dict.values())
        pd_best = params_dict

        # ---------- compute point-estimate products ----------
        L = self._L_conti(wave, pd_best, self.L_conti_wave)
        L_int = self._L_conti(wave, pd_best, self.L_conti_wave, poly=False)

        Fe_flux_result = []
        Fe_flux_type = []
        Fe_flux_name = []
        if (self.Fe_flux_range is not None) and getattr(self, "Fe_uv_op", True):
            Fe_flux_result, Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, pd_best)

        # decompose model on full wavelength grid
        pl_m, feUV_m, feOp_m, bc_m, poly_m, total_m = _evaluate_components(wave, pd_best)
        self.f_pl_model = pl_m
        self.f_fe_mgii_model = feUV_m
        self.f_fe_balmer_model = feOp_m
        self.f_bc_model = bc_m
        self.f_poly_model = poly_m
        self.f_conti_model = total_m
        self.line_flux = flux - total_m
        self.PL_poly_BC = pl_m + poly_m + bc_m

        # ---------- uncertainty estimation (MCMC / MC) ----------
        do_uncert = (getattr(self, "MCMC", False) or getattr(self, "MC", False)) and (getattr(self, "nsamp", 0) > 0)
        if do_uncert:
            if getattr(self, "MCMC", False) and not getattr(self, "MC", False):
                conti_samples = minimize(
                    self._residuals,
                    params=conti_fit.params,
                    args=(w_win[ind_noBAL], f_win[ind_noBAL], e_win[ind_noBAL], _conti_model),
                    method="emcee",
                    nan_policy="omit",
                    burn=getattr(self, "nburn", 200),
                    steps=getattr(self, "nsamp", 1000),
                    thin=getattr(self, "nthin", 1),
                    xtol=getattr(self, "xtol_conti", 1e-8),
                    ftol=getattr(self, "ftol_conti", 1e-8),
                    **getattr(self, "kwargs_conti_emcee", {}),
                    is_weighted=True,
                )
                samples_df = conti_samples.flatchain  # pandas.DataFrame

                if getattr(self, "verbose", False):
                    acc = np.asarray(conti_samples.acceptance_fraction)
                    acc_mean = np.nanmean(acc) if acc.size else np.nan
                    acc_std = np.nanstd(acc) if acc.size else np.nan
                    print(f"acceptance fraction = {acc_mean:.3f} +/- {acc_std:.3f}")
                    print("median of posterior probability distribution")
                    print("--------------------------------------------")
                    report_fit(conti_samples.params)

                # Ensure fixed params appear in samples with fixed values
                for name in par_names:
                    if name not in samples_df.columns:
                        samples_df[name] = params_dict[name]
                samples_df = samples_df[par_names]
                samples = samples_df.to_numpy()

                if getattr(self, "plot_corner", False):
                    try:
                        import corner
                        truths = [params_dict[k] for k in samples_df.columns.values.tolist()]
                        corner.corner(
                            samples_df.values,
                            labels=samples_df.columns.values.tolist(),
                            quantiles=[0.16, 0.5, 0.84],
                            truths=truths,
                        )
                    except Exception:
                        pass

            elif (not getattr(self, "MCMC", False)) and getattr(self, "MC", False):
                nsamp = int(getattr(self, "nsamp", 100))
                samples = np.zeros((nsamp, len(par_names)))
                for k in range(nsamp):
                    flux_resampled = flux + rng.normal(0.0, 1.0, size=flux.size) * err
                    fR_win = flux_resampled[in_any_window][ind_noBAL]
                    conti_fit_k = minimize(
                        self._residuals,
                        conti_fit.params,
                        args=(w_win[ind_noBAL], fR_win, e_win[ind_noBAL], _conti_model),
                        calc_covar=False,
                        xtol=getattr(self, "xtol_conti", 1e-8),
                        ftol=getattr(self, "ftol_conti", 1e-8),
                    )
                    samples[k] = list(conti_fit_k.params.valuesdict().values())
            else:
                raise RuntimeError("MCMC and MC modes are both True")

            # parameter errors
            params_err = get_err(samples, axis=0)

            # products for samples
            samples_pd = [_vec_to_dict(vec, par_names) for vec in samples]

            Ls = np.empty((len(samples_pd), len(self.L_conti_wave)))
            Ls_int = np.empty_like(Ls)
            for k, pd in enumerate(samples_pd):
                Ls[k] = self._L_conti(wave, pd, self.L_conti_wave)
                Ls_int[k] = self._L_conti(wave, pd, self.L_conti_wave, poly=False)

            L_std = get_err(Ls)
            L_int_std = get_err(Ls_int)

            Fe_flux_std = None
            if (self.Fe_flux_range is not None) and getattr(self, "Fe_uv_op", True):
                Fe_flux_results = []
                for pd in samples_pd:
                    fr, Fe_flux_type, Fe_flux_name = self.Get_Fe_flux(self.Fe_flux_range, pd)
                    Fe_flux_results.append(fr)
                Fe_flux_results = np.asarray(Fe_flux_results)
                Fe_flux_std = get_err(Fe_flux_results)

        # ---------- pack results cleanly ----------
        # Header
        self.conti_result = np.array(
            [ra, dec, str(plateid), str(mjd), str(fiberid), self.z, self.SN_ratio_conti], dtype=object
        )
        self.conti_result_type = np.array(["float", "float", "int", "int", "int", "float", "float"], dtype=object)
        self.conti_result_name = np.array(
            ["ra", "dec", "plateid", "MJD", "fiberid", "redshift", "SN_ratio_conti"], dtype=object
        )

        if do_uncert:
            # interleave param values & errors
            par_err_names = [n + "_err" for n in par_names]
            inter_vals = _interleave(params_vec, params_err)
            inter_names = list(np.ravel(np.column_stack([par_names, par_err_names])))

            self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                inter_names, inter_vals, self.conti_result, self.conti_result_type, self.conti_result_name
            )

            # L and L_int + errors
            L_names = [f"L{int(l):d}" for l in self.L_conti_wave]
            L_inter = _interleave(L, L_std)
            L_name_inter = list(np.ravel(np.column_stack([L_names, [n + "_err" for n in L_names]])))
            self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                L_name_inter, L_inter, self.conti_result, self.conti_result_type, self.conti_result_name
            )

            L_int_names = [f"L{int(l):d}_int" for l in self.L_conti_wave]
            L_int_inter = _interleave(L_int, L_int_std)
            L_int_name_inter = list(np.ravel(np.column_stack([L_int_names, [n + "_err" for n in L_int_names]])))
            self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                L_int_name_inter, L_int_inter, self.conti_result, self.conti_result_type, self.conti_result_name
            )

            # FeII flux + errors (only if enabled & requested)
            if (self.Fe_flux_range is not None) and getattr(self, "Fe_uv_op", True) and len(Fe_flux_result):
                Fe_err_names = [n + "_err" for n in Fe_flux_name]
                Fe_inter = _interleave(Fe_flux_result, Fe_flux_std if Fe_flux_std is not None else np.zeros_like(Fe_flux_result))
                Fe_name_inter = list(np.ravel(np.column_stack([Fe_flux_name, Fe_err_names])))
                self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                    Fe_name_inter, Fe_inter, self.conti_result, self.conti_result_type, self.conti_result_name
                )

        else:
            # point-only
            self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                par_names, list(pd_best.values()), self.conti_result, self.conti_result_type, self.conti_result_name
            )

            L_names = [f"L{int(l):d}" for l in self.L_conti_wave]
            self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                L_names, L, self.conti_result, self.conti_result_type, self.conti_result_name
            )

            L_int_names = [f"L{int(l):d}_int" for l in self.L_conti_wave]
            self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                L_int_names, L_int, self.conti_result, self.conti_result_type, self.conti_result_name
            )

            if (self.Fe_flux_range is not None) and getattr(self, "Fe_uv_op", True) and len(Fe_flux_result):
                self.conti_result, self.conti_result_type, self.conti_result_name = _append_fields(
                    Fe_flux_name, Fe_flux_result, self.conti_result, self.conti_result_type, self.conti_result_name
                )

        # ---------- add host products ----------
        self.conti_result = np.append(self.conti_result, self.host_result)
        self.conti_result_type = np.append(self.conti_result_type, self.host_result_type)
        self.conti_result_name = np.append(self.conti_result_name, self.host_result_name)

        # ---------- stash fit artifacts ----------
        self.conti_fit = conti_fit
        self.conti_params = params_vec
        self.tmp_all = in_any_window  # keep prior attribute name

        return self.conti_result, self.conti_result_name


    # def _L_conti(self, wave, pp, waves=np.array([1350, 3000, 5100])):
    #     """
    #     Calculate continuum Luminoisity at given waves
    #     """
    #     waves = np.array(waves)
    #     L = np.full(len(waves), -1.0)  # to save the luminosity results
    #     valid_idx = np.where((waves < np.max(wave)) & (waves > np.min(wave)), True, False)
    #     conti_flux = self.PL(waves[valid_idx], pp) + self.F_poly_conti(waves[valid_idx], pp[12:])
    #     Llam = waves[valid_idx] * self.flux2L(conti_flux, self.z)
    #     Llam[Llam <= 0] = 1e-1  # to make the log of these invalid values to be -1.
    #     L[valid_idx] = np.log10(Llam)

    #     return L

    def _L_conti(self, wave, param_dict, waves=np.array([1350, 3000, 5100]), poly=True):
            """
            Calculate continuum Luminoisity at given waves
            """
            waves = np.array(waves)

            # Add these lines:
            minw = np.min([np.min(wave), 2300])
            maxw = np.max([np.max(wave), 2700])
            wave = np.linspace(minw, maxw, 2000)  # ensure the waves are within the range of the spectrum

            L = np.full(len(waves), -1.0)  # to save the luminosity results
            valid_idx = np.where((waves < np.max(wave)) & (waves > np.min(wave)), True, False)
            conti_flux = self.PL(waves[valid_idx], param_dict)
            if poly == True:
                conti_flux += self.F_poly_conti(waves[valid_idx], param_dict)
            Llam = waves[valid_idx] * self.flux2L(conti_flux, self.z)
            Llam[Llam <= 0] = 1e-1  # to make the log of these invalid values to be -1.
            L[valid_idx] = np.log10(Llam)

            return L

    def _residuals(self, param_dict, xval, yval, weight, _conti_model):
        """Continual residual function used in lmfit"""
        #pp = list(p.valuesdict().values())
        return (yval - _conti_model(xval, param_dict)) / weight

    def fit_lines(self, wave, line_flux, err, f):
        """Fit the emission lines with Gaussian profiles."""

        rng = np.random.default_rng()
        eps = getattr(self, "epsilon_jitter", 0.0)

        # -------------------------------
        # 1) Precompute masks & metadata
        # -------------------------------
        # Mask: remove absorption trough pixels in typical emission regions
        _rej_ranges = np.array([
            [2700., 2900.],
            [1700., 1970.],
            [1500., 1700.],
            [1290., 1450.],
            [1150., 1290.],
        ], dtype=float)

        rej_mask = np.zeros_like(wave, dtype=bool)
        for lo, hi in _rej_ranges:
            rej_mask |= (wave > lo) & (wave < hi)
        # keep pixels that are not significantly negative
        ind_neg_line = ~((rej_mask) & (line_flux < -err))

        # Read line parameter file (table with config for complexes/lines)
        linelist = read_line_params(os.path.join(self.path, self.param_file_name))
        self.linelist = linelist

        # Complexes covered by spectrum
        covered = (linelist["lambda"] > wave.min()) & (linelist["lambda"] < wave.max())
        if not np.any(covered):
            # nothing to fit
            self.ncomp = 0
            self.uniq_linecomp_sort = np.array([])
            print("No line to fit! Please set line_fit to FALSE or enlarge wave_range!")
            # Save minimal outputs
            self.comp_result = np.array([])
            self.gauss_result = np.array([])
            self.gauss_result_all = np.array([])
            self.gauss_result_name = np.array([])
            self.fur_result = np.array([])
            self.fur_result_type = np.array([])
            self.fur_result_name = np.array([])
            self.line_result = np.array([])
            self.line_result_type = np.array([])
            self.line_result_name = np.array([])
            self.line_flux = line_flux
            self.all_comp_range = np.array([])
            self.f_line_model = np.zeros_like(wave)
            return self.line_result, self.line_result_name

        uniq_linecomp, uniq_idx = np.unique(linelist["compname"][covered], return_index=True)
        # Sort complexes by their representative wavelength
        sort_keys = linelist["lambda"][covered][uniq_idx].argsort()
        uniq_linecomp_sort = uniq_linecomp[sort_keys]
        allcompcenter = np.sort(linelist["lambda"][covered][uniq_idx])
        ncomp = len(uniq_linecomp_sort)

        # -------------------------------
        # 2) Containers for results
        # -------------------------------
        line_result = []
        line_result_type = []
        line_result_name = []

        comp_result = []
        comp_result_type = []
        comp_result_name = []

        gauss_result = []
        gauss_result_all = []  # only when sampling
        gauss_result_type = []
        gauss_result_name = []

        fur_result = []
        fur_result_type = []
        fur_result_name = []

        all_comp_range = []
        self.f_line_model = np.zeros_like(wave)  # final model on data grid

        # MC/MCMC flags
        do_uncert = (getattr(self, "MCMC", False) or getattr(self, "MC", False)) and (getattr(self, "nsamp", 0) > 0)
        mc_flag = 2 if do_uncert else 1  # how many numbers per parameter when interleaving val/err upstream

        # -------------------------------
        # 3) Helpers
        # -------------------------------
        def _init_fit_params(linelist_fit, ngauss_fit):
            """Build lmfit.Parameters for all lines in a complex, with jitter."""
            fit_params = Parameters()
            ln_lambda_0s = []  # for each Gaussian component in order of creation

            # loop over lines inside the complex
            for n in range(len(linelist_fit)):
                if int(ngauss_fit[n]) <= 0:
                    continue

                line_name = linelist_fit["linename"][n]
                ln_lambda_0 = np.log(linelist_fit["lambda"][n])  # line center (ln space)
                voff = linelist_fit["voff"][n]

                # sigma limits
                sig_0  = linelist_fit["inisig"][n] + abs(rng.normal(0.0, eps))
                sig_lo = linelist_fit["minsig"][n]
                sig_hi = linelist_fit["maxsig"][n]

                # scale limits
                sca_0  = linelist_fit["inisca"][n] + abs(rng.normal(0.0, eps))
                sca_lo = linelist_fit["minsca"][n]
                sca_hi = linelist_fit["maxsca"][n]

                vary = bool(linelist_fit["vary"][n])
                dw0  = rng.normal(0.0, eps)  # small wavelength offset jitter

                for nn in range(int(ngauss_fit[n])):
                    fit_params.add(f"{line_name}_{nn+1}_scale", value=sca_0, min=sca_lo, max=sca_hi, vary=vary)
                    fit_params.add(f"{line_name}_{nn+1}_dwave", value=dw0,  min=-voff,  max=voff,  vary=vary)
                    fit_params.add(f"{line_name}_{nn+1}_sigma", value=sig_0, min=sig_lo, max=sig_hi, vary=vary)
                    ln_lambda_0s.append(ln_lambda_0)

            return fit_params, np.array(ln_lambda_0s, dtype=float)

        def _apply_ties(fit_params, linelist_fit, ngauss_fit):
            """Tie velocity offsets, widths, and flux ratios via lmfit expr, within the same complex."""
            nline_fit = len(linelist_fit)
            # Tie velocity (dwave)
            for n in range(nline_fit):
                line_name = linelist_fit["linename"][n]
                vindex = linelist_fit["vindex"][n]
                if vindex <= 0:
                    continue
                mask_idx = linelist_fit["vindex"] == vindex
                ref_name = linelist_fit["linename"][mask_idx][0]
                expr = f"{ref_name}_1_dwave"
                for nn in range(int(ngauss_fit[n])):
                    pname = f"{line_name}_{nn+1}_dwave"
                    if pname != expr:
                        fit_params[pname].expr = expr

            # Tie width (sigma)
            for n in range(nline_fit):
                line_name = linelist_fit["linename"][n]
                windex = linelist_fit["windex"][n]
                if windex <= 0:
                    continue
                mask_idx = linelist_fit["windex"] == windex
                ref_name = linelist_fit["linename"][mask_idx][0]
                expr = f"{ref_name}_1_sigma"
                for nn in range(int(ngauss_fit[n])):
                    pname = f"{line_name}_{nn+1}_sigma"
                    if pname != expr:
                        fit_params[pname].expr = expr

            # Tie flux ratios (scale)
            for n in range(nline_fit):
                line_name = linelist_fit["linename"][n]
                findex = linelist_fit["findex"][n]
                if findex <= 0:
                    continue
                mask_idx = linelist_fit["findex"] == findex
                ref_name = linelist_fit["linename"][mask_idx][0]
                expr_base = f"{ref_name}_1_scale"
                f_ref = linelist_fit["fvalue"][mask_idx][0]
                f_here = linelist_fit["fvalue"][n]
                fratio = f_here / f_ref if f_ref != 0 else 0.0
                expr = f"{fratio} * {expr_base}"
                for nn in range(int(ngauss_fit[n])):
                    pname = f"{line_name}_{nn+1}_scale"
                    if pname != expr_base:
                        fit_params[pname].expr = expr

        def _pack_gauss_names_types(linelist_fit, ngauss_fit, with_errors=False):
            """Return lists of names/types for Gaussian params."""
            names = []
            types = []
            for n in range(len(linelist_fit)):
                for nn in range(int(ngauss_fit[n])):
                    prefix = f"{linelist_fit['linename'][n]}_{nn+1}"
                    if with_errors:
                        names.append([f"{prefix}_scale", f"{prefix}_scale_err",
                                    f"{prefix}_centerwave", f"{prefix}_centerwave_err",
                                    f"{prefix}_sigma", f"{prefix}_sigma_err"])
                        types.append(['float'] * 6)
                    else:
                        names.append([f"{prefix}_scale", f"{prefix}_centerwave", f"{prefix}_sigma"])
                        types.append(['float'] * 3)
            return names, types

        # -------------------------------
        # 4) Complex-by-complex fitting
        # -------------------------------
        for ii in range(ncomp):
            compname = uniq_linecomp_sort[ii]
            compcenter = allcompcenter[ii]
            sel_comp = (linelist["compname"] == compname)
            linelist_fit = linelist[sel_comp]
            nline_fit = np.sum(sel_comp)
            ngauss_fit = np.asarray(linelist_fit["ngauss"], dtype=int)

            # Complex fitting window from table (assume same for all lines in complex)
            comp_range = [linelist_fit[0]["minwav"], linelist_fit[0]["maxwav"]]
            all_comp_range = np.concatenate([all_comp_range, comp_range])

            # pixels inside complex & not-neg mask
            ind_n = (wave > comp_range[0]) & (wave < comp_range[1]) & ind_neg_line

            # crude DoF check: require at least ~3 pixels per gaussian param (like original)
            if np.count_nonzero(ind_n) <= 3 * np.sum(ngauss_fit):
                if getattr(self, "verbose", False):
                    print("Less than 10 pixels in line fitting!")
                continue

            # Initialize parameters (+ jitter)
            fit_params, ln_lambda_0s = _init_fit_params(linelist_fit, ngauss_fit)
            _apply_ties(fit_params, linelist_fit, ngauss_fit)

            # First pass fit
            if getattr(self, "verbose", False):
                fit_params.pretty_print()
                print(fr'Fitting complex {compname}')

            ind_line_abs = np.full(len(wave), True)
            args = (np.log(wave[ind_n & ind_line_abs]),
                    line_flux[ind_n & ind_line_abs],
                    err[ind_n & ind_line_abs],
                    ln_lambda_0s)

            line_fit = minimize(self._residual_line, fit_params, args=args,
                                calc_covar=False, xtol=getattr(self, "xtol_line", 1e-8),
                                ftol=getattr(self, "ftol_line", 1e-8))

            # Optional iterative rejection of strong absorption/residual outliers
            if getattr(self, "rej_abs_line", False):
                redchi = line_fit.redchi
                for _ in range(getattr(self, "rej_abs_line_max_niter", 0)):
                    resid_full = np.zeros_like(wave)
                    resid_full[ind_n & ind_line_abs] = line_fit.residual
                    # reject |resid| > 3
                    ind_line_abs_tmp = ind_line_abs & (resid_full > -3) & (resid_full < 3)

                    # keep enough pixels: (#valid - 10) > #params
                    if np.count_nonzero(ind_n & ind_line_abs_tmp) - 10 < len(fit_params):
                        break

                    args = (np.log(wave[ind_n & ind_line_abs_tmp]),
                            line_flux[ind_n & ind_line_abs_tmp],
                            err[ind_n & ind_line_abs_tmp],
                            ln_lambda_0s)

                    line_fit_tmp = minimize(self._residual_line, fit_params, args=args,
                                            calc_covar=False, xtol=getattr(self, "xtol_line", 1e-8),
                                            ftol=getattr(self, "ftol_line", 1e-8))

                    if line_fit_tmp.redchi >= redchi:
                        break
                    redchi = line_fit_tmp.redchi
                    ind_line_abs = ind_line_abs_tmp
                    line_fit = line_fit_tmp

            params_dict = line_fit.params.valuesdict()
            par_names = list(params_dict.keys())
            params_vec = list(params_dict.values())

            if getattr(self, "verbose", False):
                print("Fit report")
                report_fit(line_fit.params)

            # -------------------------------
            # 5) Uncertainties (MC/MCMC)
            # -------------------------------
            if do_uncert:
                if getattr(self, "MCMC", False) and not getattr(self, "MC", False):
                    # MCMC sampling around best-fit
                    args = (np.log(wave[ind_n & ind_line_abs]),
                            line_flux[ind_n & ind_line_abs],
                            err[ind_n & ind_line_abs],
                            ln_lambda_0s)
                    line_samples = minimize(
                        self._residual_line, params=line_fit.params, args=args,
                        method='emcee', nan_policy='omit',
                        burn=getattr(self, "nburn", 200),
                        steps=getattr(self, "nsamp", 1000),
                        thin=getattr(self, "nthin", 1),
                        is_weighted=True,
                        xtol=getattr(self, "xtol_line", 1e-8),
                        ftol=getattr(self, "ftol_line", 1e-8),
                        **getattr(self, "kwargs_line_emcee", {})
                    )
                    df_samples = line_samples.flatchain

                    # ensure fixed parameters are present
                    for name in par_names:
                        if name not in df_samples.columns:
                            df_samples[name] = params_dict[name]
                    df_samples = df_samples[par_names]
                    samples = df_samples.to_numpy()

                    if getattr(self, "verbose", False):
                        acc = np.asarray(line_samples.acceptance_fraction)
                        print(f'acceptance fraction = {np.nanmean(acc):.3f} +/- {np.nanstd(acc):.3f}')
                        print('median of posterior probability distribution')
                        print('--------------------------------------------')
                        report_fit(line_samples.params)

                    if getattr(self, "plot_corner", False):
                        try:
                            import corner
                            truths = [params_dict[k] for k in df_samples.columns.values.tolist()]
                            corner.corner(df_samples.values, labels=df_samples.columns.values.tolist(),
                                        quantiles=[0.16, 0.5, 0.84], truths=truths)
                        except Exception:
                            pass

                elif (not getattr(self, "MCMC", False)) and getattr(self, "MC", False):
                    nsamp = int(getattr(self, "nsamp", 100))
                    samples = np.zeros((nsamp, len(par_names)))
                    for k in range(nsamp):
                        lfr = line_flux + rng.normal(0.0, 1.0, size=line_flux.size) * err
                        args = (np.log(wave[ind_n & ind_line_abs]),
                                lfr[ind_n & ind_line_abs],
                                err[ind_n & ind_line_abs],
                                ln_lambda_0s)
                        line_fit_k = minimize(self._residual_line, line_fit.params, args=args,
                                            calc_covar=False, xtol=getattr(self, "xtol_line", 1e-8),
                                            ftol=getattr(self, "ftol_line", 1e-8))
                        samples[k] = list(line_fit_k.params.valuesdict().values())
                else:
                    raise RuntimeError("MCMC and MC modes cannot both be True.")

                # parameter errors across samples
                params_err = get_err(samples)

            # -------------------------------
            # 6) Save per-complex results
            # -------------------------------
            chisqr = line_fit.chisqr
            bic    = line_fit.bic
            redchi = line_fit.redchi

            # Reshape param vector as [ngauss, 3] and convert center to absolute ln(lambda)
            ngauss_total = len(params_vec) // 3
            params_mat = np.reshape(params_vec, (ngauss_total, 3))
            params_mat[:, 1] += ln_lambda_0s  # center: dln + ln(lambda0)
            params_vec_abs = params_mat.reshape(-1)

            comp_name = compname
            line_status = int(line_fit.success)
            comp_result.append([comp_name, line_status, chisqr, bic, redchi, line_fit.nfev, line_fit.nfree])
            comp_result_type.append(['str', 'int', 'float', 'float', 'float', 'int', 'int'])
            comp_result_name.append([
                f"{ii+1}_complex_name", f"{ii+1}_line_status", f"{ii+1}_line_min_chi2",
                f"{ii+1}_line_bic", f"{ii+1}_line_red_chi2", f"{ii+1}_niter", f"{ii+1}_ndof"
            ])

            br_name = compname  # broad-component "bundle" name

            if do_uncert:
                # Reshape samples similarly and adjust centers
                samples_shaped = np.reshape(samples, (samples.shape[0], ngauss_total, 3))
                samples_shaped[:, :, 1] += ln_lambda_0s
                samples_adj = samples_shaped.reshape(samples.shape[0], -1)

                # Parameter uncertainties
                params_err = get_err(samples_adj)

                # Save interleaved (val, err) for each gaussian param
                gauss_result.append(list(chain.from_iterable(zip(params_vec_abs, params_err))))
                gauss_result_all.append(samples_adj)

                # Gaussian names/types with errors
                names_err, types_err = _pack_gauss_names_types(linelist_fit, ngauss_fit, with_errors=True)
                gauss_result_name.extend(names_err)
                gauss_result_type.extend(types_err)

                # Physical broad properties (vectorized across samples)
                fur_result_temp = np.zeros((6, samples_adj.shape[0]))
                for k, s in enumerate(samples_adj):
                    fur_result_temp[:, k] = self.line_prop(compcenter, s, 'broad')
                fur_std = get_err(fur_result_temp, axis=1)
                fur_val = self.line_prop(compcenter, params_vec_abs, 'broad')

                fur_result.append(list(chain.from_iterable(zip(fur_val, fur_std))))
                fur_result_type.append(['float'] * 12)
                fur_result_name.append([
                    f"{br_name}_whole_br_fwhm", f"{br_name}_whole_br_fwhm_err",
                    f"{br_name}_whole_br_sigma", f"{br_name}_whole_br_sigma_err",
                    f"{br_name}_whole_br_ew",    f"{br_name}_whole_br_ew_err",
                    f"{br_name}_whole_br_peak",  f"{br_name}_whole_br_peak_err",
                    f"{br_name}_whole_br_area",  f"{br_name}_whole_br_area_err",
                    f"{br_name}_whole_br_snr",   f"{br_name}_whole_br_snr_err",
                ])
            else:
                # Save plain gaussian params
                gauss_result.append(params_vec_abs)
                names_noerr, types_noerr = _pack_gauss_names_types(linelist_fit, ngauss_fit, with_errors=False)
                gauss_result_name.extend(names_noerr)
                gauss_result_type.extend(types_noerr)

                # Physical broad properties (point estimate)
                fur_result.append(self.line_prop(compcenter, params_vec_abs, 'broad'))
                fur_result_type.append(['float'] * 6)
                fur_result_name.append([
                    f"{br_name}_whole_br_fwhm", f"{br_name}_whole_br_sigma",
                    f"{br_name}_whole_br_ew",   f"{br_name}_whole_br_peak",
                    f"{br_name}_whole_br_area", f"{br_name}_whole_br_snr",
                ])

        # -------------------------------
        # 7) Flatten & finalize arrays
        # -------------------------------
        if len(comp_result) > 0:
            comp_result       = np.concatenate(comp_result)
            comp_result_type  = np.concatenate(comp_result_type)
            comp_result_name  = np.concatenate(comp_result_name)

            gauss_result      = np.concatenate(gauss_result)
            if do_uncert and len(gauss_result_all) > 0:
                gauss_result_all = np.concatenate(gauss_result_all, axis=1)
            gauss_result_type = np.concatenate(gauss_result_type)
            gauss_result_name = np.concatenate(gauss_result_name)

            fur_result        = np.concatenate(fur_result)
            fur_result_type   = np.concatenate(fur_result_type)
            fur_result_name   = np.concatenate(fur_result_name)
        else:
            comp_result = np.array([])
            comp_result_type = np.array([])
            comp_result_name = np.array([])
            gauss_result = np.array([])
            gauss_result_all = np.array([])
            gauss_result_type = np.array([])
            gauss_result_name = np.array([])
            fur_result = np.array([])
            fur_result_type = np.array([])
            fur_result_name = np.array([])

        # Merge all line results
        line_result = np.concatenate([comp_result, gauss_result, fur_result]) if comp_result.size else np.array([])
        line_result_type = np.concatenate([comp_result_type, gauss_result_type, fur_result_type]) if comp_result.size else np.array([])
        line_result_name = np.concatenate([comp_result_name, gauss_result_name, fur_result_name]) if comp_result.size else np.array([])

        # -------------------------------
        # 8) Build final line model on data grid
        # -------------------------------
        self.f_line_model = np.zeros_like(wave)
        if gauss_result.size:
            if do_uncert:
                # Interleaved (val, err) per param → take values at step 2 across every 2
                # Original code takes [::2] (values) for each (val, err) pair
                vals_only = gauss_result[::2]
                # Each Gaussian has 3 params
                for p in range(len(vals_only) // 3):
                    coeffs = vals_only[p * 3:(p + 1) * 3]
                    self.f_line_model += self.Onegauss(np.log(wave), coeffs)
            else:
                for p in range(len(gauss_result) // 3):
                    coeffs = gauss_result[p * 3:(p + 1) * 3]
                    self.f_line_model += self.Onegauss(np.log(wave), coeffs)

        # -------------------------------
        # 9) Save properties to self
        # -------------------------------
        self.comp_result       = np.array(comp_result)
        self.gauss_result      = np.array(gauss_result)
        self.gauss_result_all  = np.array(gauss_result_all)
        self.gauss_result_name = np.array(gauss_result_name)

        self.fur_result        = np.array(fur_result)
        self.fur_result_type   = np.array(fur_result_type)
        self.fur_result_name   = np.array(fur_result_name)

        self.line_result       = np.array(line_result)
        self.line_result_type  = np.array(line_result_type)
        self.line_result_name  = np.array(line_result_name)

        self.ncomp             = ncomp
        self.line_flux         = line_flux
        self.all_comp_range    = np.array(all_comp_range)
        self.uniq_linecomp_sort = uniq_linecomp_sort

        return self.line_result, self.line_result_name


    def line_prop_from_name(self, line_name, line_type='broad', sample_index=-1, ln_sigma_br=0.0017):
        """
        line_name: line name e.g., 'Ha_br'
        """

        # Get the complex center wavelength of the line_name component
        mask_name = self.linelist['linename'] == line_name

        # Check if no line exists
        if np.count_nonzero(mask_name) == 0:
            return 0, 0, 0, 0, 0, 0

        # Get each Gaussian component
        compcenter = self.linelist[mask_name]['lambda'][0]
        ngauss = int(self.linelist[mask_name]['ngauss'][0])
        pp_shaped = np.zeros((ngauss, 3))

        # Check if no component is fit
        mask_result_name = self.line_result_name == f'{line_name}_{1}_scale'
        if np.count_nonzero(mask_result_name) == 0:
            return 0, 0, 0, 0, 0, 0

        # Number of Gaussian components loop
        for n in range(ngauss):

            mask_scale = self.gauss_result_name == f'{line_name}_{n+1}_scale'
            mask_center = self.gauss_result_name == f'{line_name}_{n+1}_centerwave'
            mask_sigma = self.gauss_result_name == f'{line_name}_{n+1}_sigma'
            
            if sample_index == -1:
                # Get the Gaussian properties
                pp_shaped[n,0] = float(self.gauss_result[mask_scale][0])
                pp_shaped[n,1] = float(self.gauss_result[mask_center][0])
                pp_shaped[n,2] = float(self.gauss_result[mask_sigma][0])
            else:
                index_scale = np.nonzero(mask_scale)[0]//2
                index_center = np.nonzero(mask_center)[0]//2
                index_sigma = np.nonzero(mask_sigma)[0]//2

                if len(self.gauss_result_all) == 0:
                    return 0, 0, 0, 0, 0, 0
                
                pp_shaped[n,0] = float(self.gauss_result_all[sample_index, index_scale][0])
                pp_shaped[n,1] = float(self.gauss_result_all[sample_index, index_center][0])
                pp_shaped[n,2] = float(self.gauss_result_all[sample_index, index_sigma][0])

        # Flatten
        pp = pp_shaped.reshape(-1)

        return self.line_prop(compcenter, pp, line_type, ln_sigma_br)

    def line_prop(self, compcenter, pp, linetype='broad', ln_sigma_br=0.0017):
        """
        Calculate broad/narrow line properties from Gaussian components:
        returns (fwhm[km/s], sigma[km/s], ew[Å], peak[Å], area[flux], snr).

        Parameters
        ----------
        compcenter : float
            Theoretical vacuum wavelength of the complex (Å) used for velocity normalization.
        pp : array-like, shape (3*ngauss,)
            Flat parameter vector [scale, ln_center, ln_sigma]*N for the relevant complex.
        linetype : {'broad','narrow'}
            Which subset of components to measure based on ln_sigma threshold.
        ln_sigma_br : float
            Threshold in ln(sigma[Å]) separating broad vs narrow components.
        """
        # -- normalize inputs
        pp = np.asarray(pp, dtype=float).ravel()
        if pp.size == 0 or (pp.size % 3) != 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Select components by ln(sigma)
        ln_sig_all = pp[2::3]
        if linetype.lower() == 'broad':
            which = (ln_sig_all > ln_sigma_br) & (ln_sig_all > 0)
        elif linetype.lower() == 'narrow':
            which = (ln_sig_all <= ln_sigma_br) & (ln_sig_all > 0)
        else:
            raise RuntimeError("line type should be 'broad' or 'narrow'!")

        if not np.any(which):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Build boolean mask repeated per triplet to pick selected Gaussians
        ind_mask = np.repeat(which, 3)
        pp_br = pp[ind_mask]
        ngauss_br = pp_br.size // 3

        # Also keep the full set (for residual S/N calc)
        pp_all_shaped = pp.reshape((-1, 3))
        pp_br_shaped = pp_br.reshape((ngauss_br, 3))

        # ---- construct an ln(λ) grid covering ±3σ of selected components
        cen_ln = pp_br_shaped[:, 1]
        sig_ln = pp_br_shaped[:, 2]
        left  = float(np.min(cen_ln - 3.0 * sig_ln))
        right = float(np.max(cen_ln + 3.0 * sig_ln))
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Match your original spacing in ln(λ)
        disp = 1e-4 * np.log(10.0)
        npix = max(int((right - left) / disp), 10)
        xx = np.linspace(left, right, npix)        # ln(λ)
        lam = np.exp(xx)                            # λ in Å

        # ---- evaluate the selected broad/narrow model in ln-space
        yy_br = self._Manygauss(xx, pp_br_shaped)   # flux density on ln grid

        # ---- continuum for EW: use dict-based continuum components
        #     (avoid positional parameter slicing)
        param_dict = self.conti_fit.params.valuesdict()
        conti = self.PL(lam, param_dict)
        if getattr(self, "poly", True):
            conti = conti + self.F_poly_conti(lam, param_dict)
        if getattr(self, "BC", True):
            conti = conti + self.Balmer_conti(lam, param_dict)

        # ---- peak position (Å) and S/N from data residuals around the complex
        ypeak = float(np.nanmax(yy_br)) if yy_br.size else 0.0
        if ypeak > 0.0:
            peak_idx = int(np.nanargmax(yy_br))
            peak = float(lam[peak_idx])
        else:
            peak = 0.0

        # Residual noise from ±400 Å window around compcenter
        mask_complex = (self.wave > compcenter - 400.0) & (self.wave < compcenter + 400.0)
        if np.any(mask_complex):
            model_all = self._Manygauss(np.log(self.wave), pp_all_shaped)  # full (broad+narrow) in data space
            residual = self.line_flux - model_all
            noise = median_abs_deviation(residual[mask_complex], scale='normal')
            if not np.isfinite(noise) or noise <= 0:
                # mild fallback if MAD fails
                noise = np.nanstd(residual[mask_complex])
        else:
            noise = 0.0
        snr = float(ypeak / noise) if noise and np.isfinite(noise) and noise > 0 else 0.0

        # ---- FWHM via spline roots (in ln-space then convert to km/s)
        try:
            spline = interpolate.UnivariateSpline(xx, yy_br - 0.5 * np.nanmax(yy_br), s=0)
            roots = spline.roots()
        except Exception:
            roots = []

        c_kms = const.c.to(u.km / u.s).value
        if len(roots) >= 2:
            fwhm_left, fwhm_right = float(np.min(roots)), float(np.max(roots))
            fwhm = abs(np.exp(fwhm_right) - np.exp(fwhm_left)) / float(compcenter) * c_kms
            # ---- moments & EW on linear-λ grid
            line_flux = yy_br
            lambda0 = integrate.trapezoid(line_flux, lam)  # area (total flux)
            if lambda0 > 0:
                lambda1 = integrate.trapezoid(line_flux * lam, lam)
                lambda2 = integrate.trapezoid(line_flux * lam * lam, lam)
                # guard division for EW (avoid zero/near-zero continuum)
                mask_pos = conti > 0
                if np.any(mask_pos):
                    ew = integrate.trapezoid(np.abs(line_flux[mask_pos] / conti[mask_pos]), lam[mask_pos])
                else:
                    ew = 0.0
                sigma = np.sqrt(max(lambda2 / lambda0 - (lambda1 / lambda0) ** 2, 0.0)) / float(compcenter) * c_kms
                area = lambda0
            else:
                sigma = 0.0
                ew = 0.0
                area = 0.0
        else:
            fwhm = 0.0
            sigma = 0.0
            ew = 0.0
            area = 0.0

        return float(fwhm), float(sigma), float(ew), float(peak), float(area), float(snr)

    def _residual_line(self, params, xval, yval, weight, ln_lambda_0s):
        """
        Calculate total residual for fitting of line complexes
        """

        pp = list(params.valuesdict().values())

        # Reshape parameters array for vectorization
        ngauss = len(pp) // 3
        pp_shaped = np.reshape(pp, (ngauss, 3))
        pp_shaped[:, 1] += ln_lambda_0s  # Transform ln lambda ~ d ln lambda + ln lambda0

        resid = (yval - self._Manygauss(xval, pp_shaped)) / weight

        return resid

    def save_result(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type,
                    line_result_name, save_fits_path, save_fits_name):
        """Save all data to fits"""
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])

        t = Table(self.all_result, names=(self.all_result_name), dtype=self.all_result_type)
        t.write(os.path.join(save_fits_path, save_fits_name + '.fits'), format='fits', overwrite=True)
        return

    def set_mpl_style(fsize=15, tsize=18, tdir='in', major=5.0, minor=3.0, lwidth=1.8, lhandle=2.0):

        """Function to set MPL style"""

        plt.style.use('default')
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = fsize
        plt.rcParams['legend.fontsize'] = tsize
        plt.rcParams['xtick.direction'] = tdir
        plt.rcParams['ytick.direction'] = tdir
        plt.rcParams['xtick.major.size'] = major
        plt.rcParams['xtick.minor.size'] = minor
        plt.rcParams['ytick.major.size'] = 5.0
        plt.rcParams['ytick.minor.size'] = 3.0
        plt.rcParams['axes.linewidth'] = lwidth
        plt.rcParams['legend.handlelength'] = lhandle

        return

    def plot_fig(
        self,
        save_fig_path=".",
        broad_fwhm=1200,
        plot_line_name=True,
        plot_legend=True,
        ylims=None,
        plot_residual=True,
        show_title=True,
        plot_br_prop=False,
    ):
        """Plot continuum + (optional) line complexes, using dict-based components."""

        # ---------- helpers ----------
        def _pretty_name(plain_name):
            special = {
                "Ha": r"H\alpha", "Hb": r"H\beta", "Hg": r"H\gamma", "Hd": r"H\delta",
                "Hep": r"H\epsilon", "Lya": r"Ly\alpha",
            }
            if plain_name in special:
                s = special[plain_name]
            elif "I" in plain_name:
                i = plain_name.find("I")
                s = plain_name[:i] + r"\," + plain_name[i:]
            else:
                s = plain_name
            return rf"$\mathrm{{{s}}}$"

        def _apply_nan_mask(x, y, mask_ranges):
            """Return copies of y where masked wavelength spans are set to NaN (breaks lines)."""
            if mask_ranges is None or len(mask_ranges) == 0:
                return y
            y2 = y.copy()
            for lo, hi in np.asarray(mask_ranges):
                sel = (x >= lo) & (x <= hi)
                y2[sel] = np.nan
            return y2

        def _compute_continuum_grid(param_dict, xgrid):
            """Evaluate & cache all continuum pieces once."""
            pl   = self.PL(xgrid, param_dict)
            feuv = self.Fe_flux_mgii(xgrid, param_dict) if getattr(self, "Fe_uv_op", True) else 0.0
            feop = self.Fe_flux_balmer(xgrid, param_dict) if getattr(self, "Fe_uv_op", True) else 0.0
            bc   = self.Balmer_conti(xgrid, param_dict) if getattr(self, "BC", True) else 0.0
            poly = self.F_poly_conti(xgrid, param_dict) if getattr(self, "poly", True) else 0.0
            tot  = pl + feuv + feop + bc + poly
            return pl, poly, bc, feuv, feop, tot

        def _robust_ylim(y, fallback_low=-1.0):
            """Return (lo, hi) with padding, durable when arrays are empty."""
            if y.size == 0 or not np.isfinite(y).any():
                return fallback_low * 0.9, 1.1
            yfin = y[np.isfinite(y)]
            lo, hi = np.min(yfin), np.max(yfin)
            if hi == lo:
                hi = lo + 1.0
            pad = 0.1 * (hi - lo)
            return lo - pad, hi + pad

        # ---------- params & style ----------
        matplotlib.rc("xtick", labelsize=20)
        matplotlib.rc("ytick", labelsize=20)
        pdict = self.conti_fit.params.valuesdict()

        wmin = float(np.min(self.wave))
        wmax = float(np.max(self.wave))
        wave_eval = np.linspace(wmin - 200, wmax + 200, 5000)

        # Continuum pieces (evaluate once)
        pl_eval, poly_eval, bc_eval, fe_uv_eval, fe_op_eval, conti_eval = _compute_continuum_grid(pdict, wave_eval)

        # Residual arrays (compute once)
        # When lines are fitted, residuals are (data - continuum - lines); else (data - continuum).
        if getattr(self, "linefit", False) and (len(getattr(self, "line_result", [])) > 0):
            # These get filled below while we build line models
            f_line_model_eval = np.zeros_like(wave_eval)
            self.f_line_narrow_model = np.zeros_like(self.wave)
            self.f_line_br_model = np.zeros_like(self.wave)
        else:
            f_line_model_eval = np.zeros_like(wave_eval)

        # ---------- figure layout ----------
        has_lines = getattr(self, "linefit", False) and (len(getattr(self, "line_result", [])) > 0)
        mc_flag = 2 if ((getattr(self, "MCMC", False) or getattr(self, "MC", False)) and getattr(self, "nsamp", 0) > 0) else 1
        ncomp_fit = (len(getattr(self, "fur_result", [])) // (mc_flag * 6)) if has_lines else 0

        if has_lines:
            fig, axn = plt.subplots(nrows=2, ncols=max(ncomp_fit, 1), figsize=(15, 8), squeeze=False, sharex=False)
            # Merge top row cells into one big main axis
            gs = axn[0, 0].get_gridspec()
            for axi in axn[0, :]:
                axi.remove()
            ax = fig.add_subplot(gs[0, :])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

        # ---------- build line models if present ----------
        lines_total_eval = np.zeros_like(wave_eval)
        if has_lines:
            line_order = {"r": 3, "g": 7}  # zorder: narrow (g) above broad (r)
            n_gauss = len(self.gauss_result) // (mc_flag * 3)
            for p in range(n_gauss):
                # params for this gaussian component (amp, mu, sigma log-space)
                gp = self.gauss_result[p * 3 * mc_flag : (p + 1) * 3 * mc_flag : mc_flag]

                is_narrow = self.CalFWHM(self.gauss_result[(2 + p * 3) * mc_flag]) < broad_fwhm
                color = "g" if is_narrow else "r"

                # Evaluate
                line_eval = self.Onegauss(np.log(wave_eval), gp)
                lines_total_eval += line_eval
                ax.plot(wave_eval, conti_eval + line_eval, color=color, zorder=5)

                # Accumulate full-resolution line models on data grid
                this_line_on_data = self.Onegauss(np.log(self.wave), gp)
                if is_narrow:
                    self.f_line_narrow_model += this_line_on_data
                else:
                    self.f_line_br_model += this_line_on_data

            # store full line model on data grid
            self.f_line_model = self.f_line_narrow_model + self.f_line_br_model

            # main axis: total lines + continuum
            ax.plot(wave_eval, conti_eval + lines_total_eval, "b", label="line", zorder=6)

            # per-complex mini-panels
            for c in range(ncomp_fit):
                axc = axn[1][c]
                axc.plot(wave_eval, lines_total_eval, color="b", zorder=10)
                f_line_br_interp = interpolate.interp1d(self.wave, self.f_line_br_model, bounds_error=False, fill_value=0)
                f_line_br_eval = f_line_br_interp(wave_eval)
                axc.plot(wave_eval, f_line_br_eval, color="r", zorder=6)
                f_line_narrow_interp = interpolate.interp1d(self.wave, self.f_line_narrow_model, bounds_error=False, fill_value=0)
                f_line_narrow_eval = f_line_narrow_interp(wave_eval)
                axc.plot(wave_eval, f_line_narrow_eval, color="g", zorder=8)
                lo_c, hi_c = self.all_comp_range[2 * c : 2 * c + 2]
                axc.set_xlim(lo_c, hi_c)

                # robust ylim from residuals in this complex
                mask_c = (self.wave > lo_c) & (self.wave < hi_c)
                resid_c = (self.line_flux - self.f_line_model)[mask_c] if np.any(mask_c) else np.array([])
                lo_y, hi_y = _robust_ylim(self.f_line_model[mask_c]) if ylims is None else (ylims[0], ylims[1])
                axc.set_ylim(lo_y, hi_y)

                # ticks & labels
                axc.set_xticks([lo_c, np.round((lo_c + hi_c) / 2, -1), hi_c])
                axc.text(0.02, 0.90, _pretty_name(self.uniq_linecomp_sort[c]), fontsize=20, transform=axc.transAxes)
                axc.text(0.02, 0.825, r"$\chi^2_\nu=$" + str(np.round(float(self.comp_result[c * 7 + 4]), 2)),
                        fontsize=12, transform=axc.transAxes)

                if plot_br_prop:
                    fwhm = self.fur_result[self.fur_result_name == f"{self.uniq_linecomp_sort[c]}_whole_br_fwhm"][0]
                    area = self.fur_result[self.fur_result_name == f"{self.uniq_linecomp_sort[c]}_whole_br_area"][0]
                    snr  = self.fur_result[self.fur_result_name == f"{self.uniq_linecomp_sort[c]}_whole_br_snr"][0]
                    if mc_flag == 2:
                        fwhm_err = self.fur_result[self.fur_result_name == f"{self.uniq_linecomp_sort[c]}_whole_br_fwhm_err"][0]
                        area_err = self.fur_result[self.fur_result_name == f"{self.uniq_linecomp_sort[c]}_whole_br_area_err"][0]
                        axc.text(0.02, 0.75,
                                fr"$L_{{\rm br}}=10^{{{{{np.round(np.log10(self.flux2L(area)), 2)}}}\pm{{{np.round(0.434 * self.flux2L(area_err) / self.flux2L(area), 2)}}}}}$"
                                + r"$\ \rm{erg}\ \rm{s}^{-1}$", fontsize=12, transform=axc.transAxes)
                        axc.text(0.02, 0.675,
                                fr"${{\rm FWHM}}_{{\rm br}}={{{int(np.round(fwhm, 0))}}}\pm{{{int(np.round(fwhm_err, 0))}}}$"
                                + r"$\ \rm{km}\ \rm{s}^{-1}$", fontsize=12, transform=axc.transAxes)
                    else:
                        axc.text(0.02, 0.75,
                                fr"$L_{{\rm br}}=10^{{{np.round(np.log10(self.flux2L(area)), 1)}}}$"
                                + r"$\ \rm{erg}\ \rm{s}^{-1}$", fontsize=12, transform=axc.transAxes)
                        axc.text(0.02, 0.675,
                                fr"${{\rm FWHM}}_{{\rm br}}={{{int(np.round(fwhm, 0))}}}$"
                                + r"$\ \rm{km}\ \rm{s}^{-1}$", fontsize=12, transform=axc.transAxes)

                    axc.text(0.02, 0.60, fr"$S/N_{{\rm br}}={np.round(snr, 1)}$", fontsize=12, transform=axc.transAxes)

                # add masked spans + residuals if requested (NaN masking keeps lines broken)
                line_flux_nan = _apply_nan_mask(self.wave, self.line_flux, self.wave_mask)
                axc.plot(self.wave, line_flux_nan, "k", label="data", lw=1, zorder=2)
                if plot_residual:
                    axc.axhline(-5, color="k", zorder=0, lw=0.5)
                    resid_nan = _apply_nan_mask(self.wave, self.line_flux - self.f_line_model - 5, self.wave_mask)
                    axc.plot(self.wave, resid_nan, "gray", linestyle="dotted", lw=1, zorder=3)

        # ---------- main panel plotting ----------
        # masked versions of observed spectra (break lines across masked spans)
        flux_main = _apply_nan_mask(self.wave_prereduced, self.flux_prereduced, self.wave_mask)
        ax.plot(self.wave_prereduced, flux_main, "k", label="data", lw=1, zorder=2)

        # residuals on main panel
        if plot_residual:
            if has_lines:
                resid_main = _apply_nan_mask(self.wave, self.line_flux - self.f_line_model, self.wave_mask)
            else:
                # fall back to data - continuum
                resid_main = _apply_nan_mask(self.wave, self.flux - self.f_conti_model, self.wave_mask)
            ax.plot(self.wave, resid_main, "gray", label="resid", linestyle="dotted", lw=1, zorder=3)

        # title
        if show_title:
            if self.ra == -999 or self.dec == -999:
                ax.set_title(f"{self.sdss_name}   z = {np.round(float(self.z), 4)}", fontsize=20)
            else:
                ax.set_title(
                    f"ra,dec = ({np.round(self.ra, 4)},{np.round(self.dec, 4)})   {self.sdss_name}   z = {np.round(float(self.z), 4)}",
                    fontsize=20,
                )

        # host overlays (unchanged)
        if getattr(self, "decompose_host", False) and getattr(self, "decomposed", False):
            ax.plot(self.wave, self.qso + self.host, "pink", label="host+qso temp", zorder=3)
            ax.plot(self.wave, self.flux, "grey", label="data-host", zorder=1)
            ax.plot(self.wave, self.host, "purple", label="host", zorder=4)

        # minimal legend handles (once)
        ax.plot([], [], "r", label="line br", zorder=5)
        ax.plot([], [], "g", label="line na", zorder=5)

        # continuum overlays (single eval each)
        ax.plot(wave_eval, conti_eval, "c", lw=2, label="FeII", zorder=7)
        if getattr(self, "BC", True):
            ax.plot(wave_eval, pl_eval + poly_eval + bc_eval, "y", lw=2, label="BC", zorder=8)
        ax.plot(wave_eval, pl_eval + poly_eval, color="orange", lw=2, label="reddened conti", zorder=9)
        ax.plot(wave_eval, pl_eval, color="yellow", lw=2, label="conti", zorder=9)

        # y-limits (robust) for main panel
        if ylims is None:
            if has_lines:
                r_all = self.line_flux - self.f_line_model
                lo_y, hi_y = _robust_ylim(r_all)
                # also consider data amplitude
                lo_d, hi_d = _robust_ylim(self.flux)
                lo_y, hi_y = min(lo_y, lo_d), max(hi_y, hi_d)
            else:
                lo_y, hi_y = _robust_ylim(self.flux)
            ax.set_ylim(lo_y, hi_y)
        else:
            ax.set_ylim(ylims[0], ylims[1])

        # mark continuum windows
        if hasattr(self, "tmp_all") and np.ndim(self.tmp_all) and np.count_nonzero(self.tmp_all) > 0:
            y_top = ax.get_ylim()[1]
            ax.scatter(self.wave[self.tmp_all], np.repeat(y_top * 0.98, np.count_nonzero(self.tmp_all)),
                    color="grey", marker="o", s=10)

        if plot_legend:
            ax.legend(loc="best", frameon=False, ncol=2, fontsize=10)

        # line name guides
        if plot_line_name:
            line_cen = np.array(
                [6564.60, 6549.85, 6585.27, 6718.29, 6732.66, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47,
                3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30,
                1215.67]
            )
            line_name = np.array(
                ["", "", r"H$\alpha$+[NII]", "", "[SII]6718,6732", r"H$\beta$", "[OIII]", "HeII4687", r"H$\gamma$",
                "CaII3934", "[OII]3728", "NeV3426", "MgII", "CIII]", "SiII1816", "NIII]1750", "NIV]1718", "CIV",
                "HeII1640", "", "SiIV+OIV", "CII1335", r"Ly$\alpha$"]
            )
            # place labels near top of current axes
            y_top = ax.get_ylim()[1]
            for lc, ln in zip(line_cen, line_name):
                if self.wave.min() < lc < self.wave.max():
                    ax.axvline(lc, color="k", linestyle=":", alpha=0.8)
                    if ln:
                        ax.text(lc + 7, y_top * 0.95, ln, rotation=90, fontsize=10, va="top")

        ax.set_xlim(self.wave.min(), self.wave.max())
        fig.supxlabel(r"$\rm Rest \, Wavelength$ ($\rm \AA$)", fontsize=20)
        fig.supylabel(r"$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)", fontsize=20)

        # save if requested
        if getattr(self, "save_fig", False):
            if getattr(self, "verbose", False):
                print("Saving figure as", os.path.join(save_fig_path, self.sdss_name + ".pdf"))
            fig.savefig(os.path.join(save_fig_path, self.sdss_name + ".pdf"))
            plt.close(fig)

        self.fig = fig
        return


    def to_Spectrum1D(self):
        from specutils import Spectrum1D
        from astropy.nddata import StdDevUncertainty
        spec1d = Spectrum1D(spectral_axis=self.wave*u.AA/(1+self.z), flux=self.flux*1e-17*u.erg/u.s/u.cm**2,
                            uncertainty=StdDevUncertainty(self.err), redshift=self.z)
        return spec1d

    def CalFWHM(self, logsigma):
        """transfer the logFWHM to normal frame"""
        return 2 * np.sqrt(2 * np.log(2)) * (np.exp(logsigma) - 1) * 300000.

    def Smooth(self, y, box_pts):
        "Smooth the flux with n pixels"
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def Fe_flux_mgii(self, xval, pp):
        "Fit the UV FeII component on the continuum from 1200 to 3500 A based on Boroson & Green 1992."
        yval = np.zeros_like(xval)
        wave_Fe_mgii = 10 ** self.fe_uv[:, 0]
        flux_Fe_mgii = self.fe_uv[:, 1] * 1e15
        Fe_FWHM = pp['Fe_uv_FWHM']
        xval_new = xval * (1.0 + pp['Fe_uv_shift'])

        ind = np.where((xval_new > 1200.) & (xval_new < 3500.), True, False)
        if np.sum(ind) > self.n_pix_min_conti:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4 * sig_pix + 1, 0)
            xx = np.arange(0, khalfsz * 2, 1) - khalfsz
            kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
            kernel = kernel / np.sum(kernel)

            flux_Fe_conv = np.convolve(flux_Fe_mgii, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_mgii, flux_Fe_conv)
            yval[ind] = pp['Fe_uv_norm'] * interpolate.splev(xval_new[ind], tck)
        return yval

    def Fe_flux_balmer(self, xval, pp):
        "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
        yval = np.zeros_like(xval)

        wave_Fe_balmer = 10 ** self.fe_op[:, 0]
        flux_Fe_balmer = self.fe_op[:, 1] * 1e15
        ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.), True, False)
        wave_Fe_balmer = wave_Fe_balmer[ind]
        flux_Fe_balmer = flux_Fe_balmer[ind]
        Fe_FWHM = pp['Fe_op_FWHM']
        xval_new = xval * (1.0 + pp['Fe_op_shift'])
        ind = np.where((xval_new > 3686.) & (xval_new < 7484.), True, False)
        if np.sum(ind) > self.n_pix_min_conti:
            if Fe_FWHM < 900.0:
                sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
            khalfsz = np.round(4 * sig_pix + 1, 0)
            xx = np.arange(0, khalfsz * 2, 1) - khalfsz
            kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
            kernel = kernel / np.sum(kernel)
            flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel, 'same')
            tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv)
            yval[ind] = pp['Fe_op_norm'] * interpolate.splev(xval_new[ind], tck)
        return yval

    #def PLsingle(self, xval, pp, x0=3000):
    #    return pp['PL_norm'] * (xval / x0) ** pp['PL_slope_blue']

    def PL(self, xval, pp, x0=4000):
        """
        Smooth broken power-law version of PL.

        Parameters
        ----------
        xval : array-like
            Wavelength or frequency values.
        pp : sequence
            Parameter array where:
            pp[6] = normalization (A)
            pp[7] = slope1 (d1) for x < x_break
            pp[8] = slope2 (d2) for x > x_break
            pp[9] = smoothness (ds)
            pp[10] = break location (x_break)
        x0 : float, optional
            Reference wavelength for normalization (default 4000).
        """
        A = pp['PL_norm']
        d1 = pp['PL_slope_blue']
        d2 = pp['PL_slope_red']
        ds = 0.1 #pp_extra[1]
        x_break = pp['PL_break_wave']

        x = xval / x_break
        delta = d2 - d1
        smooth_exp = 1.0 / ds

        # Smooth broken power law normalized to A at x0
        log_f = d1 * np.log10(x) + (delta / smooth_exp) * np.log1p(x**smooth_exp)
        f = A * 10**(log_f - log_f[np.argmin(np.abs(xval - x0))])
        return f

    def Balmer_conti(self, xval, pp):
        """
        Balmer continuum (Dietrich+02) for rest-frame wavelengths.
        Convolution is performed in log-lambda space.
        """
        x = np.asarray(xval, dtype=float)
        norm, Te, tau_BE = float(pp['Balmer_norm']), float(pp['Balmer_Te']), float(pp['Balmer_Tau'])

        lam_BE = 3646.0  # Å, rest-frame Balmer edge

        # Planck B_lambda in erg s^-1 cm^-2 Å^-1 sr^-1
        h  = 6.62607015e-27   # erg s
        c  = 2.99792458e10    # cm / s
        kB = 1.380649e-16     # erg / K
        lam_cm = x * 1e-8
        expo = (h * c) / (lam_cm * kB * Te)
        expo = np.clip(expo, 1e-9, 700.0)  # numerical safety
        B_per_cm = (2.0 * h * c**2 / lam_cm**5) / np.expm1(expo)  # per cm
        B_per_A  = B_per_cm * 1e-8                                   # per Å
        bbflux   = B_per_A * np.pi                                   # integrate over angles

        # Dietrich+02 optical-depth law, only blueward of the edge
        tau = tau_BE * (x / lam_BE)**3
        bc  = norm * bbflux * (1.0 - np.exp(-tau))

        # Zero flux redward of the Balmer edge
        bc[x > lam_BE] = 0.0

        # --- Convolve in log-lambda space ---
        loglam = np.log(x)
        # Make a uniform log-lam grid
        loglam_uniform = np.linspace(loglam.min(), loglam.max(), len(loglam))
        # Interpolate bc onto uniform log-lam grid
        bc_uniform = np.interp(loglam_uniform, loglam, bc)

        # Kernel width in log-lam
        v_broad_kms = float(pp['Balmer_vel'])  # in km/s
        c_kms = 2.99792458e5
        sigma_loglam = (v_broad_kms / c_kms)  # since d(loglam) ~ d(v)/c

        dloglam = np.median(np.diff(loglam_uniform))
        half = int(np.ceil(5.0 * sigma_loglam / dloglam))
        g = np.arange(-half, half+1) * dloglam
        ker = np.exp(-0.5*(g/sigma_loglam)**2)
        ker /= ker.sum()

        # Truncate kernel if longer than data
        if len(ker) > len(bc_uniform):
            extra = len(ker) - len(bc_uniform)
            ker = ker[extra//2:-(extra-extra//2)]

        # Convolve
        bc_conv = np.convolve(bc_uniform, ker, mode="same")

        # Interpolate back to original x grid
        bc_final = np.interp(loglam, loglam_uniform, bc_conv)

        return bc_final * 1e-17

    def F_poly_conti(self, xval, pp, x0=3000):
        """
        Fit the continuum with a polynomial component accounting for dust reddening.
        Ensures the output is negative everywhere.
        """
        #xval2 = xval - x0
        # rescale pp for numerical precision
        #yvals = [(pp[i]) * xval2 ** (i + 1) for i in range(len(pp))]
        #poly = np.sum(yvals, axis=0) * 100
        #return poly

        from numpy.polynomial import polynomial as P

        # Dimensionless coordinate centered at pivot (same as your original poly)
        T = (xval - float(x0)) / float(x0)

        # Amplitude at the pivot (≥ 0 so attenuation can't flip sign)
        amp_raw = float(pp['conti_a_0'])
        A0 = amp_raw if amp_raw > 20.0 else np.log1p(np.exp(amp_raw))  # softplus

        # Curvature controller: Q(T) = Σ b_j T^j  (unconstrained b_j)
        b = np.array([float(pp[k]) for k in ["conti_a_1", "conti_a_2"]], dtype=float)

        if b.size == 0:
            A = np.full_like(T, A0)
        else:
            # Enforce dA/dT = -[Q(T)]^2 ≤ 0  ⇒ A(T) = A0 - ∫_0^T [Q(u)]^2 du
            Q2 = P.polymul(b, b)   # coefficients of Q^2 ≥ 0
            Int = P.polyint(Q2)    # indefinite integral; const=0 ⇒ integral from 0
            I_T = P.polyval(T, Int)
            A = A0 - I_T

        # Physical floor: extinction can't be negative
        A = np.clip(A, 0.0, np.inf)
        
        # Return NEGATIVE attenuation
        return -A

    def flux2L(self, flux, z=None):
        """Transfer flux to luminoity assuming a flat Universe"""
        if z is None:
            z = self.z
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_L = cosmo.luminosity_distance(z).to(u.cm).value  # unit cm
        L = flux * 1e-17 * 4 * np.pi * d_L ** 2  # erg/s/A
        return L

    def Onegauss(self, xval, pp):
        """The single Gaussian model used to fit the emission lines 
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        
        This is evaluated many times within the scipy optimize, so we want to keep the code as fast as possible
        Hence, we avoid calling any external libraries like astropy's Gaussian here
        
        It is slightly faster to fit the (un-normalized) amplitude directly to avoid blow-up at small sigma
        
        xval: wavelength array in AA

        TODO: See if LMFIT's built-in modeles improved performance
        https://lmfit.github.io/lmfit-py/builtin_models.html
        
        pp: Paramaters [3]
            scale: line amplitude
            wave: central ln wavelength in AA
            sigma: width in km/s
        """

        return pp[0] * np.exp(-(xval - pp[1]) ** 2 / (2 * pp[2] ** 2))

    def _Manygauss(self, xval, pp):
        """
        Fast multi-Gaussian model used to fit the emission lines
        
        This is evaluated many times within the scipy optimize, so we want to keep the code as fast as possible
        Hence, we avoid calling any external libraries like astropy's Gaussian here
        It is vectorized so pp must have shape [ngauss, 3]
        
        It is slightly faster to fit the (un-normalized) amplitude directly to avoid blow-up at small sigma
        
        xval: wavelength array in AA
        
        pp: Paramaters array [ngauss, 3]
            scale: line amplitude
            wave: central ln wavelength in AA
            sigma: width in km/s
        """

        return np.sum(pp[:, 0] * np.exp(-(xval[:, np.newaxis] - pp[:, 1]) ** 2 / (2 * pp[:, 2] ** 2)), axis=1)

    def Manygauss(self, xval, pp):
        """
        Robust function for multi-Gaussian model used to fit the emission lines
        
        This is evaluated many times within the scipy optimize, so we want to keep the code as fast as possible
        Hence, it is vectorized so pp must have shape [ngauss, 3]
        
        xval: wavelength array in AA
        
        pp: Paramaters [ngauss*3]
            scale: line amplitude
            wave: central ln wavelength in AA
            sigma: width in km/s
        """

        # Reshape parameters array for vectorization
        ngauss = len(pp) // 3
        if ngauss > 0:
            pp_shaped = np.reshape(pp, (ngauss, 3))
            return self._Manygauss(xval, pp_shaped)
        else:
            return np.zeros_like(xval)

    def Get_Fe_flux(self, ranges, param_dict):
        """
        Calculate FeII template fluxes for one or more wavelength ranges.

        Parameters
        ----------
        ranges : array-like
            Either [lo, hi] or [[lo, hi], [lo2, hi2], ...].
        param_dict : dict
            Full parameter dictionary for FeII templates.

        Returns
        -------
        Fe_flux_result : np.ndarray of floats
        Fe_flux_type   : np.ndarray of dtype strings ("float")
        Fe_flux_name   : np.ndarray of labels (e.g. "Fe_flux_2200_3090")
        """
        Fe_flux_result = []
        Fe_flux_type   = []
        Fe_flux_name   = []

        if ranges is None:
            return np.array(Fe_flux_result), np.array(Fe_flux_type), np.array(Fe_flux_name)

        rngs = np.asarray(ranges, dtype=float)

        # Normalize input to 2D shape (N, 2)
        if rngs.ndim == 1:
            if rngs.size != 2:
                raise ValueError(f"'ranges' must be [lo, hi] or [[lo, hi], ...], got shape {rngs.shape}")
            rngs = rngs.reshape(1, 2)
        elif rngs.ndim != 2 or rngs.shape[1] != 2:
            raise ValueError(f"'ranges' must have shape (N, 2), got {rngs.shape}")

        # Loop through each wavelength range
        for lo, hi in rngs:
            rr = [float(lo), float(hi)]
            flux = self._calculate_Fe_flux(rr, param_dict)
            Fe_flux_result.append(float(flux))
            Fe_flux_name.append(f"Fe_flux_{int(lo)}_{int(hi)}")
            Fe_flux_type.append("float")

        return (
            np.asarray(Fe_flux_result),
            np.asarray(Fe_flux_type),
            np.asarray(Fe_flux_name),
        )


    def _calculate_Fe_flux(self, measure_range, param_dict):
        """
        Calculate Fe II pseudocontinuum flux within a single wavelength range,
        using the full parameter dict.

        Logic:
        - Integrates UV Fe template (Mg II region) over overlap with [1200, 3500] Å
        - Integrates optical Fe template (Balmer/optical region) over overlap with [3686, 7484] Å
        - The gap [3500, 3686] contributes zero by construction.

        Returns
        -------
        flux : float
            Trapezoidal integral over the requested range (clipped to spectrum bounds).
            Returns -1 if the available pixel count drops below self.n_pix_min_conti.
        """
        balmer_range = np.array([3686.0, 7484.0])  # optical FeII template range
        mgii_range   = np.array([1200.0, 3500.0])  # UV FeII template range

        # Determine spectrum bounds if available; otherwise fall back to the requested range
        if hasattr(self, "wave") and self.wave is not None and len(self.wave) > 0:
            spec_lo = float(np.min(self.wave))
            spec_hi = float(np.max(self.wave))
        else:
            spec_lo = float(np.min(measure_range))
            spec_hi = float(np.max(measure_range))

        # Clip the requested range to the spectrum coverage
        req_lo = float(np.min(measure_range))
        req_hi = float(np.max(measure_range))
        lower  = max(req_lo, spec_lo)
        upper  = min(req_hi, spec_hi)

        if upper < req_hi or lower > req_lo:
            if getattr(self, "verbose", False):
                print("Warning: FeII flux range exceeds spectrum coverage; excess set to zero.")

        # Build a log-spaced grid (matches original code)
        disp = 1e-4 * np.log(10.0)
        if upper <= lower:
            # No overlap after clipping
            return 0.0
        xval = np.exp(np.arange(np.log(lower), np.log(upper), disp))

        if len(xval) < getattr(self, "n_pix_min_conti", 20):
            if getattr(self, "verbose", False):
                print(f"Warning: Available part in range {measure_range} has < n_pix_min_conti; returning -1.")
            return -1.0

        # Decide overlaps with the two template regions
        overlaps_mgii   = not (upper <= mgii_range[0] or lower >= mgii_range[1])
        overlaps_balmer = not (upper <= balmer_range[0] or lower >= balmer_range[1])

        # Optional warnings about exceeding template ranges or hitting the template gap
        if upper > balmer_range[1] or lower < mgii_range[0]:
            if getattr(self, "verbose", False):
                print("Warning: Requested FeII range partially exceeds template domain [1200, 7484]; excess set to zero.")
        if (lower < balmer_range[0]) and (upper > mgii_range[1]) and (upper > 3500.0) and (lower < 3686.0):
            if getattr(self, "verbose", False):
                print("Warning: Requested FeII range includes the gap [3500, 3686]; that part contributes zero.")

        # Evaluate templates only where they apply, using the full param_dict
        yval = 0.0
        if overlaps_mgii:
            # Restrict x to mgii range to avoid evaluating outside template domain, then stitch back via mask
            mask_mg = (xval >= mgii_range[0]) & (xval <= mgii_range[1])
            if np.any(mask_mg):
                y_mg = np.zeros_like(xval)
                y_mg[mask_mg] = self.Fe_flux_mgii(xval[mask_mg], param_dict)
                yval = yval + y_mg

        if overlaps_balmer:
            mask_op = (xval >= balmer_range[0]) & (xval <= balmer_range[1])
            if np.any(mask_op):
                y_op = np.zeros_like(xval)
                y_op[mask_op] = self.Fe_flux_balmer(xval[mask_op], param_dict)
                yval = yval + y_op

        # If neither overlaps, integral is zero
        if (not overlaps_mgii) and (not overlaps_balmer):
            return 0.0

        # Final integral (already clipped to [lower, upper])
        flux = integrate.trapezoid(yval, xval)
        return float(flux)

    def read_out_params(self, param_file_path='qsopar.fits'):
        # read result customized parameters
        hdul = fits.open(param_file_path)

        data = hdul[4].data
        self.Fe_flux_range = np.array(data['Fe_flux_range'][0])
        self.L_conti_wave = np.array(data['cont_loc'][0])

        return data


def get_err(s, margin=0.16, axis=0, default_value=-1.):
    """
    Get 100*margin percent distribution of a given data.
    :param s: 1-D array or 2-D array. If a 1-D array is given, the data will deem the array as the data sample and the
    axis parameter will be ignored. If a 2-D array is given, how the function deel with this data will depend on the
    axis. If axis==0, the function will calculate the distribution of each column of the array. If axis==1, the
    function will calculate the distribution of each row of the array.
    :param margin: The margin of the distribution. The default value is 16%, which means the function will calculate
    about 1 sigma error for each sample
    :param axis: How the function deel with the data, see above.
    :return: float or 1-D array, depends on the input data.
    """
    s = np.array(s)
    s[s == default_value] = np.nan
    margin_per = int(margin * 100)
    if s.ndim == 1:
        N_samp = len(s)
        if np.sum(np.isnan(s)) / N_samp > 0.5:
            return default_value
        else:
            # if self.verbose:
            #     print('Warning: The input data contains more than 50% nan values. The error would be set to -1.')
            return np.diff(np.nanpercentile(s, (margin_per, 100 - margin_per)))[0] / 2
    elif s.ndim == 2:
        if axis == 1:
            s = s.T
        if not axis in [0, 1]:
            raise IndexError('The axis parameter only adopts 0 or 1.')
        N_samp = s.shape[0]
        Na_idx = np.where(np.sum(np.isnan(s), axis=0) > N_samp / 2, True, False)
        data_err = np.diff(np.nanpercentile(s, (margin_per, 100 - margin_per), axis=0), axis=0)[0] / 2
        data_err[Na_idx] = default_value
        return data_err
    else:
        raise IndexError('The input data only adopts 1-D or 2-D array.')


def read_conti_params(param_file_path='qsopar.fits'):
    # read line parameter
    hdul = fits.open(param_file_path)

    conti_windows = np.vstack([np.array(t) for t in hdul[2].data])
    data = hdul[3].data

    return data, conti_windows


def read_line_params(param_file_path='qsopar.fits'):
    # read line parameter
    hdul = fits.open(param_file_path)
    data = hdul[1].data

    # print('Reading parameter file:', param_file_path)

    return data
