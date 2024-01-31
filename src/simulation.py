"""Generate a simulated survival datasets"""
import numpy as np
from scipy.stats import bernoulli

def get_x_matrix(
    N=100, 
    x_vars=1, 
    VAR_CLASS = None, 
    VAR_PROB = None, 
    rng = None
):
    """
    Genereates a covariate matrix use in the simulation survival function.

    Parameters
    ------
    N : int
        Number of observations generated.
    x_vars : int
        Number of covariates. If x_vars > len(VAR_CLASS) all covariates > len(VAR_CLASS) will be uniformally distributed [0-1].
    VAR_CLASS : list[int
        Number of levels in each covariate. If VAR_CLASS == 2, then variable 
        will generated from a bernoulli distribution with given probability `VAR_PROB`.
        Otherwise variable will be generated from a uniform distribution with `var_class` levels
    VAR_PROB : list[double, None]
        Variable probability for the bernoulli distributed variables.
    rng : Generator(PCG64)
        random number generator to be passed. Default == NONE and will create a new random number generator with the each function call. 
    
    Returns
    -----
    numpy.ndarray
        Generated covariate matrix
    
    Example
    ------
    >>> from numpy.random import RandomState
    >>> rng = np.random.default_rng(99)
    >>> x_mat = get_x_matrix(N = 10, x_vars = 5, VAR_CLASS = [2,2,10,100], VAR_PROB = [.6, .8, None, None], rng = None)
    """

    if rng is None:
        rng = np.random.default_rng(seed=99)
    
    bern = bernoulli
    bern.random_state = rng

    # create an x matrix
    x_mat = np.zeros((N, x_vars))
    for idx, x in enumerate(VAR_CLASS):
        if x == 2:
            # x1 = sp.bernoulli.rvs(VAR_PROB[idx], size = N)
            # x1 = bern.rvs(VAR_PROB[idx], size = N)
            x1 = rng.binomial(1, VAR_PROB[idx], size = N)
        else:
            x1 = rng.uniform(0, VAR_CLASS[idx], size = N)
            x1 = np.round(x1, 3)
        x_mat[:,idx] = x1
    if x_vars > len(VAR_CLASS):
        for idx in np.arange(len(VAR_CLASS), x_vars):
            x_mat[:,idx] = rng.uniform(0,1,size=N)
    return x_mat

def simulate_survival(
    x_mat = None,
    scale_f=None, 
    shape_f = None, 
    eos = None,
    cens_scale = None,
    time_scale = None,
    true_only = False,
    rng = None
):
    """"
    Generates survival outcome data from a covariate data. 
    
    Using a covariate matrix and a linear equation that defines the linear combination of the covariates, the function will generate a Hazard Rate and Survival probability curve for each observation in the covariate matrix. 
    Additionaly, an event time for each observation.
    Randomized right censoring and End-of-Study censoring can also be incorporated into the simulated dataset.

    Survival times are generated from a Weibull Distribution with a user specified by the evaluation of the `shape_f` and `scale_f` parameters.
        Increasing the scale will increase the number of time points.
        Shape controls if the hazard rate is increasing of decreasing. 
        

    Parameters
    ------
    x_mat : numpy.array
        Covariate matrix. 
    scale_f : str
        Linear equation that generate \lambda.
    shape_f : str 
        Linear equation the generates \alpha.
    eos: int [default None]
        Numeric value that right truncates the time series.
        [Default] 100 time points when true_only == True, otherwise when true_only == False defaults to max generated time.
    cens_scale: float [default None]
        Sets censoring rate.
        A value of 4 will provide about 20% censoring. A value of 0.25-1.0 will provide about 80% censoring. 
    time_scale: int [default None]
        Downscales time series (ie time series extends 90 time points, scaled by 30 time point intervals will return a 3 time point series) 
    true_only: bool [default False]
        If True returns only the expected Survival components. False returns expected Survival components and sampled event times. 

    Returns
    ------
    dict: event_dict
        Generated event times.
            numpy.array: t_event
                Time of the event.
            numpy.array: t_event_scaled [optional depends on time_scale != None]
                Time of the event scaled by time_scale.
            numpy.array: status
                Status at t_event.
    dict: sv_true
        True Survival components.
        numpy.array: sv_mat
            Survival probabilities at times.
        numpy.array: hz_mat
            Hazard rate at times.
        numpy.array: true_times
            Times Survival components generated for.

    dict: sv_scale_true [optional depends on time_scale != None]
        True Survival components at scaled times.
        numpy.array: sv_mat
            Survival probabilities at times.
        numpy.array: hz_mat
            Hazard rate at times.
        numpy.array: true_times
            Times for which Survival components were generated
        numpy.array: scale_times
            Scaled times for which Survival components were generated.
    """

    N = x_mat.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed=99)

    # lambda and alpha
    scale = eval(scale_f)
    if "x_mat" not in shape_f:
        shape = np.repeat(eval(shape_f), N)
    else:
        shape = eval(shape_f)
    print(f"shape {shape.mean()}")
    print(f"scale {scale.mean()}")
    if not true_only:
        # generate event_times and status
        unif = rng.uniform(size=N)
        tlat = inverse_weibull_survival(scale=scale, shape=shape, u=unif)
        print(tlat.mean())
        # generate censoring for using an exponential distirubution
        cens_percentage = None
        if cens_scale is not None:            
            cens = np.ceil(rng.exponential(size = N, scale = cens_scale * scale.mean())).reshape(-1,1)
            print(f"cens mean {cens.mean()}")
            t_event  = np.minimum(cens, np.ceil(tlat))
            status = (tlat <= cens) * 1
            cens_percentage = status.sum()/N
            # print(status)
            # print(status.sum())
        else:
            t_event = np.ceil(tlat)
            status = np.ones(N).reshape(-1,1)
        
        # eos censoring
        if eos is not None:
            eos_msk = t_event > eos
            t_event[eos_msk] = eos
            status[eos_msk] = 0

        event_dict = {"t_event":t_event, "status":status, "cens_percentage":cens_percentage}
        # adjust time scale
        if time_scale is not None:
            t_event_scale = np.ceil(t_event/time_scale)
            event_dict = {"t_event":t_event, "t_event_scale": t_event_scale, "status":status, "cens_percentage":cens_percentage}

        # get event times
        t_max = int(t_event.max())
        t = np.linspace(1,t_max+1, t_max+2)
    else:
        if eos is None:
            t_max=100     # hack that defaults number of time points to 100
        t = np.linspace(1,t_max+1, t_max+2)

    # Generate survival and hazard actual values
    hz_mat = weibull_hazard(scale, shape, t)
    sv_mat = weibull_survival(scale, shape, t)
    # print(sv_mat.shape)


    # scale
    if time_scale is not None:
        t_scale = np.arange(1, t_max+1, time_scale)
        sv_mat_scale = sv_mat[:,t_scale]
        hz_mat_scale = hz_mat[:,t_scale]
        t_scale2 = t_scale/time_scale
        sv_scale_dict = {"sv_true":sv_mat_scale, "hz_true":hz_mat_scale, "true_times":t_scale, "scale_times":t_scale2}    
    sv_dict = {"sv_true":sv_mat, "hz_true":hz_mat, "true_times":t}

    # Do not return x_mat anymore
    if true_only:
        return sv_dict, sv_scale_dict        
    return event_dict, sv_dict, sv_scale_dict

def check_inputs(scale, shape, u=np.array([.9]), t=np.array([1])):
    ret = True
    if type(shape) != np.ndarray:
        shape = np.array([shape])
    if type(scale) != np.ndarray:
        scale = np.array([scale])
    if type(u) != np.ndarray:
        u = np.array([u])
    if type(t) != np.ndarray:
        t = np.array([t])
    if(any(u <=0) or any(u>=1)):
        ret=False
        print("u must be between 0 and 1")
    if (any(scale <=0)):
        ret=False
        print("scale must be > 0")
    if (any(shape <=0)):
        ret=False
        print("shape must be > 0")
    if (any(t <= 0)):
        ret=False
        print("t must be > 0")
    assert(ret==True)        
    return scale, shape, u, t

    
def inverse_weibull_survival(scale, shape, u):
    scale, shape, u, t = check_inputs(scale,shape, u=u)
    t = scale.reshape(-1,1) * np.power(-np.log(u), 1/shape).reshape(-1,1)
    return t

def weibull_hazard(scale, shape, t):
    scale, shape, u, t = check_inputs(scale,shape, t=t)
    hz = shape.reshape(-1,1) * np.power(t/scale.reshape(-1,1), shape.reshape(-1,1)-1)
    return hz


def weibull_survival(scale, shape, t):
    scale, shape, u, t = check_inputs(scale,shape, t=t)
    prob = np.exp(-np.power(t/scale.reshape(-1,1), shape.reshape(-1,1)))
    return prob
