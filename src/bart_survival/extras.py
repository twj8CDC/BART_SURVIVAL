"""extra functions that don't need to be packagesd`
"""
import numpy as np

# get the dataset for specific code
def get_sk_sp(df, cov, code, code_type = "code_sm"):
    df_code = df.filter(F.col(code_type) == code)
    mg = (
        df_code
        .join(cov, on="medrec_key", how="left")
        .drop("medrec_key", code_type)
        .withColumn("ccsr_tt_p3", F.col("ccsr_tt_p3")-30) # adjust date by -30
    )
    names = mg.columns
    out = [[x[i] for i in range(len(names))] for x in mg.collect() ]
    # out = [[x[0], x[1], x[2]] for x in mg]
    return np.array(out, dtype="int"), names


def get_coh(y_sk, cc1, sample_n = 10_000, balance=True, train=True, idx=None, seed = 0, resample=False, prop=1):
    # set the seed
    if not resample:
        np.random.seed(seed)
    # cohort set-up
    y_sk_coh, x_sk_coh, w_coh = get_case_cohort(y_sk, cc1[:,2:], prop = prop)
    # downsample
    if train:
        print("getting train")
        if balance:
            sample_n = int(np.ceil(sample_n/2))
            print(f"getting {sample_n} * 2 samples (balanced)")
            cov_idx = np.where(x_sk_coh[:,5] == 1)[0]
            ncov_idx = np.where(x_sk_coh[:,5] != 1)[0]
            cov_smp = np.random.choice(cov_idx, sample_n, replace=False)
            ncov_smp = np.random.choice(ncov_idx, sample_n, replace=False)
            cov_idx_test = np.setdiff1d(cov_idx, cov_smp) # drop the samples from the all dataset
            ncov_idx_test = np.setdiff1d(ncov_idx, ncov_smp)
            idx_out = {
                "cov_idx": cov_idx,
                "ncov_idx": ncov_idx,
                "cov_smp": cov_smp,
                "ncov_smp": ncov_smp,
                "sample_idx": np.concatenate([cov_smp, ncov_smp]),
                "cov_idx_test": cov_idx_test, 
                "ncov_idx_test": ncov_idx_test
            }
            samp_idx = np.concatenate([cov_smp, ncov_smp])
        else:
            print(f"getting {sample_n} samples (straight)")
            idx = np.arange(y_sk.shape[0])
            samp_idx = np.random.choice(idx, size=sample_n, replace=False)
            all_idx_test = np.setdiff1d(idx, samp_idx) # drop the sample from the all
            idx_out = {
                "sample_idx": samp_idx,
                "all_idx_test": all_idx_test
                } 
        # final cohort
        y_sk_coh_2 =y_sk_coh[samp_idx]
        x_sk_coh_2 = x_sk_coh[samp_idx]
        w_coh_2 = w_coh[samp_idx]
        out_msk = x_sk_coh_2[:,5]==1
        coh_y, coh_x, coh_w, coh_coords = surv_pre_train(y_sk_coh_2, x_sk_coh_2, w_coh_2)
        x_tst, tst_coords = get_posterior_test(np.unique(y_sk_coh_2["Survival_in_days"]), x_sk_coh_2)
        return {
            "y_sk_coh": y_sk_coh_2,
            "x_sk_coh": x_sk_coh_2,
            "w_sk_coh": w_coh_2,
            "coh_y": coh_y, 
            "coh_x": coh_x, 
            "coh_w": coh_w, 
            "coh_coords": coh_coords, 
            "x_tst": x_tst, 
            "tst_coords": tst_coords,
            "msk": out_msk,
            "idx": idx_out,
            "seed": seed
        } 
    else:
        print("getting test")
        if idx is None: # check to see that we have the idx to remove train sample
            print("Provide a idx of sample set to drop from test set")
        else:
            if balance:
                print("getting a balance sample")
                sample_n = int(np.ceil(sample_n/2))
                if "cov_idx_test" not in idx.keys():
                    print("requires the train to be balanced, returns none")
                    return
                else:   
                    cov_idx = np.random.choice(idx["cov_idx_test"], sample_n, replace=False)
                    ncov_idx = np.random.choice(idx["ncov_idx_test"], sample_n, replace=False)
                    samp_idx = np.concatenate([cov_idx, ncov_idx])
            else:
                print("getting a straight sample")
                if "all_idx_test" not in idx.keys():
                    idx_all = np.concatenate([idx["cov_idx"], idx["ncov_idx"]])
                else:
                    idx_all = idx["all_idx_test"]
                samp_idx = np.random.choice(idx_all, sample_n, replace=False)
            y_sk_coh_2 =y_sk_coh[samp_idx]
            x_sk_coh_2 = x_sk_coh[samp_idx]
            w_coh_2 = w_coh[samp_idx]
            # finish setup
            x_tst, tst_coords = get_posterior_test(np.unique(y_sk_coh_2["Survival_in_days"]), x_sk_coh_2)
            out_msk = x_sk_coh_2[:,5]==1
            return {
                "y_sk_coh": y_sk_coh_2,
                "x_sk_coh": x_sk_coh_2,
                "w_sk_coh": w_coh_2,
                # "y_test": y_sk_coh, 
                # "x_test": x_sk_coh,
                # "w_test": w_coh,
                "x_tst_test": x_tst, 
                "x_tst_coords_test": tst_coords,
                "msk_test": out_msk,
                "idx_test": samp_idx,
                "seed_test": seed 
            }