"""Paired length diagnostics for a frozen Porto scoring run (all distances in metres)."""
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import jensenshannon

def audit(ref, synth, ref200, gen200, training_lengths, headers, arc_length, dev_lengths):
    lengths=lambda paths:np.asarray([arc_length(x) for x in paths])
    r,g=lengths(ref),lengths(synth)
    r200,g200=lengths(ref200),lengths(gen200)
    expected=headers[:,1]*np.std(training_lengths)+np.mean(training_lengths)
    qs=lambda x:[float(t) for t in np.quantile(x,[.1,.5,.9])]
    def compare(a,b):
        return dict(median_ratio=float(np.median(b/a)),mean_signed_m=float(np.mean(b-a)),w1_m=float(wasserstein_distance(a,b)),q10_50_90_ratio=qs(b/a),spearman=float(spearmanr(a,b).statistic))
    out=dict(reference_m=qs(r),generated_m=qs(g),header_abs_error_m=qs(abs(expected-r)),reference_to_gen=compare(r,g),reference_to_200=compare(r,r200),generated200_to_restored=compare(g200,g),both_200=compare(r200,g200),bins={})
    rng=np.random.default_rng(173)
    control=rng.choice(dev_lengths,len(r),replace=False)
    for count in (10,20,40):
        edges=np.unique(np.quantile(training_lengths,np.arange(1,count)/count))
        hist=lambda x:np.bincount(np.digitize(x,edges),minlength=len(edges)+1)
        js=lambda a,b:float(jensenshannon(hist(a),hist(b))**2)
        out['bins'][count]=dict(generation=js(r,g),both_200=js(r200,g200),observed_dev_vs_test=js(r,control))
    out['coarse_path_length_ratios']={}
    for stride in (1,2,5,10,20,199):
        idx=np.unique(np.r_[np.arange(0,200,stride),199])
        rr=lengths([tr[idx] for tr in ref200])
        gg=lengths([tr[idx] for tr in gen200])
        valid=rr>0
        out['coarse_path_length_ratios'][stride]=dict(median_ratio=float(np.median(gg[valid]/rr[valid])),mean_signed_m=float(np.mean(gg-rr)),support=int(valid.sum()))
    return out
