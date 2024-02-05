from pymc_experimental.model_builder import ModelBuilder
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from pathlib import Path
import arviz as az
import pymc as pm
import pymc_bart as pmb
import xarray as xr
import json
import scipy.stats as sp
import warnings
import pytensor.tensor as pt
from pymc_bart.utils import _sample_posterior
import cloudpickle as cpkl
# import pyspark.sql.functions as F
# import pyspark.sql.window as W
import sksurv as sks
from sksurv import nonparametric

from numpy.random import RandomState
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

class BartSurvModel(ModelBuilder):
# class BartSurvModel():
    _model_type = "BART_Survival"
    version = "0.1"
    def __init__(
        self,
        model_config: Dict = None,
        sampler_config: Dict = None,
    ):
        sampler_config = (
            self.get_default_sampler_config() if sampler_config is None else sampler_config
        )
        self.sampler_config = sampler_config
        model_config = self.get_default_model_config() if model_config is None else model_config
        self.model_config = model_config  # parameters for priors etc.
        self.model = None  # Set by build_model
        self.idata: Optional[az.InferenceData] = None  # idata is generated during fitting
        self.is_fitted_ = False

    def build_model(self, X: np.ndarray, 
                        y: np.ndarray,                        
                        weights: np.ndarray,
                        coords: np.ndarray,
                        predictor_names: list,
                         **kwargs):
        # check x,y,weights        
        self._generate_and_preprocess_model_data(X, y, weights, predictor_names)
        # Get model Configs
        SPLIT_RULES = [eval(rule) for rule in self.model_config.get("split_rules", None)]
        SPLIT_PRIOR = self.model_config.get("split_prior")
        M = self.model_config.get("trees", 20)

        # custom logp
        def logp_bern(value, mu, w):
            return w * pm.logp(pm.Bernoulli.dist(mu), value)
        # extension of the bernoulli, the distribution is used as normal
        def dist_bern(mu, w, size):
            return pm.Bernoulli.dist(mu, size=size)
        # offset
        self.offset = sp.norm.ppf(np.mean(self.y))
        # model coords
        mcoords = {"xvars":self.predictor_names}

        with pm.Model(coords=mcoords) as self.model:    
            self.model.add_coord("p_obs", coords, mutable=True)
            x_data = pm.MutableData("x_data", self.X, dims=("p_obs", "xvars"))
            w = pm.MutableData("weights", self.weights)
            # change names of y_values
            f = pmb.BART("f", X=x_data, Y=self.y.flatten(), m=M, split_rules = SPLIT_RULES, split_prior = SPLIT_PRIOR)
            z = pm.Deterministic("z", (f + self.offset))
            mu = pm.Deterministic("mu", pm.math.invprobit(z), dims=("p_obs"))
            pm.CustomDist("y_pred", mu, w.flatten(), dist=dist_bern, logp=logp_bern, observed=self.y.flatten(), shape = x_data.shape[0])            
    
    def sample_model(self, **kwargs):
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            idata = pm.sample(**sampler_args)
        idata = self.set_idata_attrs(idata)
        return idata


    def fit(
        self,
        y: np.ndarray, #only event outcome
        X: np.ndarray, #time+covariate expanded
        weights: np.ndarray=None, #long form weights
        coords: np.ndarray=None,
        progressbar: bool = True,
        predictor_names: List[str] = None,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Data should be already be scaled, case-cohort formatted (if using case-cohort) and in x_sk, y_sk formats.
        X: is a 2d numpy array
        y: is a sk_survival formated numpy array. See utilities for y_sk processing
        weights: is a 1d numpy array
        predictor_names: list of strings correcpsponding to variable names
        """
        # build model
        self.build_model(X, y, weights, coords, predictor_names)
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)

        # sample the model
        self.idata = self.sample_model(**sampler_config)
        self.is_fitted_ = True

        # get tree struct
        self.all_trees = list(self.model.f.owner.op.all_trees)
        # get x names


        # create idata of testing data
        combined_data = np.hstack([self.y, self.weights, self.X])
        # assert all(combined_data.columns), "All columns must have non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning#,
            )
            self.idata.add_groups(fit_data=xr.DataArray(combined_data))  # type: ignore
            self.idata.add_groups(predictor_names = xr.DataArray(self.predictor_names))
            self.idata.add_groups(offset = xr.DataArray(self.offset))
        return self.idata  # type: ignore


    def sample_posterior_predictive(self, X_pred, coords, extend_idata=False, **kwargs):
        """Predict new data
        If X_pred isn't formatted to long w/ time in the first column then it will add the times as first column
        If this model was not trained in the current instance, use bart_predict
        """
        # if np.all(np.equal(np.unique(X_pred[:,0]),self.uniq_times)):
        #     print("Time not detected in first column, adding times")
        #     X_pred = get_posterior_test(self.uniq_times, X_pred)
        
        if not self.is_fitted_:
            print("Model is not fitted in this instance. Either refit or predict with bart_predict")
            pass

        # mcoords = {"p_obs": coords,
                    # ("t_obs", self.X[:,0)),
            # "xvars":self.predictor_names}


        with self.model:  # sample with new input data
            pm.set_data({"x_data":X_pred} , coords={"p_obs":coords})
            post_pred = pm.sample_posterior_predictive(self.idata, var_names=["mu"],**kwargs)
            if extend_idata:
                self.idata.extend(post_pred)
        
        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=True
        )

        return posterior_predictive_samples.transpose()
    
    def bart_predict(self, X_pred, coords, size = None, **kwargs):
        
        # if np.unique(X_pred[:,0]) != self.uniq_times:
            # print("Time not detected in first column, adding times")
            # X_pred = get_posterior_test(self.uniq_times, X_pred)
        
        if not self.all_trees:
            print("No tree structure loaded. Load a tree or train the model")
        trees = self.all_trees
        if size is None:
            size = len(self.all_trees)

        rng = np.random.default_rng()
        pt_x = pt.constant(X_pred)
        post_pred_mu_ = _sample_posterior(trees, pt_x, rng, size = size, shape=1)
        post_pred = pm.math.invprobit(post_pred_mu_ + self.offset).eval()
        post_pred = post_pred.reshape(post_pred.shape[0], post_pred.shape[1])
        # print(post_pred.shape)
        post_pred = xr.DataArray(post_pred, name = "mu", 
                                    coords = {"sample":np.arange(0,post_pred.shape[0]),
                                            "p_obs": coords
                                    })
        return post_pred

    def _data_setter(
        self, X:np.ndarray, y:np.ndarray = None
    ):
        with self.model:
            pm.set_data({"x_data": X})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})


    @classmethod
    def load(cls, fname: str, treename: str):
        
        if ".nc" in fname:
            filepath = Path(str(fname))
            idata = az.from_netcdf(filepath)
        else:
            filepath = Path(str(fname))
            with open(filepath, mode="rb") as filein:
                idata = cpkl.load(filein)
        # needs to be converted, because json.loads was changing tuple to list
        model_config = cls._model_config_formatting(json.loads(idata.attrs["model_config"]))
        model = cls(
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data["x"].values
        X = dataset[:,2:]
        y = dataset[:,0]
        weights = dataset[:,1]
        coords = idata.posterior.coords["p_obs"].values
        predictor_names = idata.constant_data.coords["xvars"]
        

        model.build_model(X, y, weights, coords, predictor_names)

        # All previously used data is in idata.
        # cls.predictor_names = idata.predictor_names["x"].values
        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )
        
        with open(treename, mode="rb") as filein:
            cls.all_trees = cpkl.load(filein)
        return model

    def save(self, idata_name:str, all_tree_name:str) -> None:
        """
        Default save idata as netcdf, alternative is to cloudpickle. Use suffix to determine
        """
        if self.idata is not None and "posterior" in self.idata:
            file = idata_name
            if ".nc" in file:
                self.idata.to_netcdf(str(file))
            else:
                with open(file, mode='wb') as out:
                        cpkl.dump(self.idata, out)
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        
        if self.all_trees is not None:
            file = all_tree_name
            with open(file, mode='wb') as out:
                cpkl.dump(self.all_trees, out)
        else:
            raise RuntimeError("No tree structure has been detected. Make sure model has been fit")


    @staticmethod
    def get_default_model_config() -> Dict:
        print("NO DEFAULT MODEL CONFIGS, MUST SPECIFY")
        pass

    @staticmethod
    def get_default_sampler_config() -> Dict:
        sampler_config: Dict = {
            "draws": 100,
            "tune": 100,
            "cores": 2,
            "chains": 2,
            "compute_convergence_checks": False
        }
        return sampler_config

    @property
    def output_var(self):
        return "y_pred"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        model_config = self.model_config.copy()
        model_config["split_rules"] = [str(sp_rule) for sp_rule in model_config["split_rules"]]
        return model_config

    def _save_input_params(self, idata) -> None:
        # idata.attrs["weights"] = json.dumps(self.weights.tolist())
        pass

    def _generate_and_preprocess_model_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        weights:np.ndarray,
        predictor_names: np.ndarray
    ) -> None:
        """
        x_sk
        y_sk
        """
        self.model_coords = None 
        assert type(X).__module__ == "numpy"
        assert type(y).__module__ == "numpy"
        self.X = X
        self.y = y
        self.uniq_times = np.unique(X[:,0])

        # set weights to 1 if not using a case_cohort design
        if weights is None:
            weights = np.ones(X.shape[0])
        self.weights = weights.reshape(weights.shape[0],1)

        # set predictor names
        # x should have time so make sure that first column is time
        if predictor_names is None:
            predictor_names = ["x" + str(i) for i in np.arange(X.shape[1])]
            predictor_names[0] = "t"
        self.predictor_names = predictor_names

        # self.y_trn, self.x_trn = surv_pre_train(y, X)        
        # self.x_tst = get_posterior_test(self.uniq_times, X)
        

def get_time_transform(t_event, time_scale = 10):
    return np.ceil(t_event/time_scale)
    
def get_y_sklearn(status, t_event):
    y = np.array(list(zip(np.array(status, dtype="bool"), t_event)), dtype=[("Status","?"),("Survival_in_days", "<f8")])
    return y

def surv_pre_train(y_sk, x_sk, weight = None):
    if weight is None:
        weight = np.ones(y_sk.shape[0])
    t_sk = y_sk["Survival_in_days"]
    d_sk = np.array(y_sk["Status"], dtype="int")
    t_uniq, t_inv = np.unique(t_sk, return_inverse=True)
    t_wide = np.tile(t_uniq, (y_sk.shape[0],1))
    d_wide = np.tile(np.zeros(t_uniq.shape[0]), (y_sk.shape[0],1))
    max_t = t_sk.max()

    # replace the time greater than t_sk. LOOP is over columns, so it should be fast with the exception of unscalled, many days
    for i in np.arange(t_wide.shape[1]):
        t_wide[t_wide[:,i] > t_sk,i]=max_t + 1
        d_wide[t_wide[:,i] == t_sk,i] = d_sk[t_wide[:,i]==t_sk]
        d_wide[t_wide[:,i] > t_sk, i] = 2
    
    # time
    t_wide = t_wide.reshape(t_wide.shape[0]*t_wide.shape[1],1)
    t_wide = (t_wide[t_wide != max_t + 1]) # drop extra times
    t_wide = t_wide.reshape(t_wide.shape[0],1)

    d_wide = d_wide.reshape(d_wide.shape[0]*d_wide.shape[1],1)
    trn_y = (d_wide[d_wide != 2])
    trn_y = trn_y.reshape(trn_y.shape[0],1)
    
    # x and delta expansion
    tx = np.repeat(x_sk, t_inv+1, axis=0)
    trn_x = np.hstack([t_wide, tx])

    # weight expansion
    weight_long = np.repeat(weight, t_inv+1)
    weight_long = weight_long.reshape(weight_long.shape[0],1)

    # get coords
    coords = np.repeat(np.arange(0,x_sk.shape[0]), t_inv+1)
    return {
            "y":trn_y, 
            "x":trn_x, 
            "w":weight_long, 
            "coord":coords
        }


def get_posterior_test(y_sk, x_test):
    uniq_times = np.unique(y_sk["Survival_in_days"])
    s0 = x_test.shape[0] # length
    s1 = x_test.shape[1] # width
    # create time range
    d1 = uniq_times
    d2 = np.tile(d1,s0).reshape(d1.shape[0]*s0,1)
    d3 = np.tile(x_test, d1.shape[0]).reshape(s0*d1.shape[0], s1)
    out = np.hstack([d2,d3])
    # coordinates
    coords = np.repeat(np.arange(0, x_test.shape[0]), d1.shape[0])
    return {"post_x": out, "coords":coords}

    
# def get_survival(post, axis=1, mean=True, values=True):
#     def sv_func(x, axis = axis):
#         return (1-x).cumprod(axis=axis)

#     if "Dataset" in str(type(post)):
#         post = post["mu"]
    
#     if mean:
#         smp, nt = post.shape
#         n = post.p_obs.values[-1] + 1
#         t = int(nt/n)
#         mean = post.values.mean(0).reshape(n, t)
#         sv = np.cumprod((1-mean), axis=1)
#         return sv
#     else:
#         sv = post.groupby("p_obs").apply(sv_func)
#     if values:
#         return sv.values
#     return sv



def get_prob(post):
    if "Dataset" in str(type(post)):
        post = post["mu"]    
    smp, nt = post.shape
    n = post.p_obs.values[-1] + 1
    t = int(nt/n)
    prob = post.values.mean(0).reshape(n, t)
    return prob
    

def get_pdp(x_sk, var_col, values=[], qt=[0.25,0.5,0.75], sample_n=None):
    if len(var_col) > 2:
        print("only upto 2 variables can be used at a time")
        pass

    if sample_n is not None:
        rn_idx = np.random.choice(np.arange(0, x_sk.shape[0]), sample_n, replace=False)   
        x_sk = x_sk[rn_idx,:]     

    var_vals = []
    for idx, var in enumerate(var_col):
        if values[idx] is None:
            val =np.quantile(x_sk[:,var], qt)
            var_vals.append(val)
        else:
            var_vals.append(values[idx])
    s1 = len(var_col)
    if s1 > 1:
        cart = np.dstack(np.meshgrid(*var_vals)).reshape(-1, s1)
    else:
        cart = np.array(var_vals[0]).reshape(-1,1)
    s2 = cart.shape[0] # number of vals
    s3 = x_sk.shape[0] # number of obs
    out_sk = np.tile(x_sk, (s2,1)) # tile x_sk into blocks of number of vals
    l_cart = np.repeat(cart, s3, axis=0) # repeat vals number of obs times
    for idx, var in enumerate(var_col): # fill the columns of the x_sk*blocks with new values
        out_sk[:,var] = l_cart[:,idx]
    # index
    c_idx = np.repeat(np.arange(s2),s3)
    c_idx = np.unique(c_idx, return_index=True, return_counts=True, return_inverse=True)
    c_idx = dict({"val":c_idx[0], "idx":c_idx[1], "coord":c_idx[2], "cnt":c_idx[3], "len":out_sk.shape[0]})

    return out_sk, c_idx



def get_sv_prob(post):
    n,k = np.unique(post.p_obs.values, return_counts=True)
    prob = post.mu.values.reshape(-1, n.shape[0], k[0])
    sv = np.cumprod(1-prob, 2)
    return {
        "prob":prob, 
        "sv":sv
        }

# def get_sv_mean_quant(sv, msk, draws = True, qntile=[0.025, 0.975]):
#     # binary mask means and quantiles
#     #tmask
#     sv_mt = sv[:,msk,:]
#     sv_mt_m = sv_mt.mean(axis=1) # mean per draw
#     if draws:
#         sv_mt_q = np.quantile(sv_mt_m, qntile, axis = 0)
#         sv_mt_m = sv_mt_m.mean(axis=0) # mean over draws

#     #fmask
#     sv_mf = sv[:,~msk,:]
#     sv_mf_m = sv_mf.mean(axis=1)
#     if draws:
#         sv_mf_q = np.quantile(sv_mf_m, qntile, axis = 0)
#         sv_mf_m = sv_mf_m.mean(axis=0)

#     return {
#         "mt_m":sv_mt_m, 
#         "mt_q":sv_mt_q, 
#         "mf_m":sv_mf_m, 
#         "mf_q":sv_mf_q
#     }

# get diff metric 
def pdp_diff_metric(pdp_val, idx, qntile = [0.025, 0.975]):
    diff = (pdp_val["sv"][:,:idx,:] - pdp_val["sv"][:,idx:,:]).mean(1) #cov - ncov
    d_m = diff.mean(0)
    d_q = np.quantile(diff, qntile, axis=0)
    return {"diff_m": np.round(d_m,3),
             "diff_q": np.round(d_q,3)}

def pdp_rr_metric(pdp_val, idx, qntile = [0.025, 0.975]):
    r = (pdp_val["prob"][:,idx:,:] / pdp_val["prob"][:,:idx,:]).mean(1) #cov/ncov
    r_m = r.mean(0)
    r_q = np.quantile(r, qntile, axis=0)
    return {"rr_m": np.round(r_m,3), 
            "rr_q":np.round(r_q,3)}
    
def pdp_eval(x_sk_coh, bart_model, var_col, values, var_name=None, sample_n=None, uniq_times=None, qntile = [0.025,0.975], diff=True, rr = True, return_all=False):
    # set up dataset
    pdp = get_pdp(x_sk_coh, var_col = var_col, values = values, sample_n=sample_n) 
    # get longform
    pdp_x, pdp_coords = get_posterior_test(uniq_times, pdp[0])
    # get posterior draws
    pdp_post = bart_model.sample_posterior_predictive(pdp_x, pdp_coords, extend_idata=False)
    # get sv_val
    print("getting sv")
    pdp_val = get_sv_prob(pdp_post)
    # get mean and quantile
    print("getting sv mean and quantile")
    pdp_mq = get_sv_mean_quant(pdp_val["sv"],pdp[1]["coord"]==1, qntile = qntile)

    # get diff and rr
    pdp_diff = None
    pdp_rr = None
    if diff:
        print("getting pdp_diff")
        pdp_diff = pdp_diff_metric(pdp_val, pdp[1]["cnt"][0], qntile=qntile)
    if rr:
        print("getting pdp rr")
        pdp_rr = pdp_rr_metric(pdp_val, pdp[1]["cnt"][0], qntile=qntile)   

    if return_all:
        return {"pdp_varname":var_name, "pdp_x":pdp_x, "pdp_coords":[pdp_coords, pdp[1]["coord"]], "pdp_post":pdp_post, "pdp_val":pdp_val, "pdp_mq":pdp_mq, "pdp_diff":pdp_diff, "pdp_rr":pdp_rr}
    else:
        return {"pdp_varname":var_name, "pdp_val":pdp_val, "pdp_mq":pdp_mq, "pdp_diff":pdp_diff, "pdp_rr":pdp_rr}
