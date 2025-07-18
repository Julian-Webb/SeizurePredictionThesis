def forecast(feas_coarse, labels_coarse, t_coarse, day_split, day_delay, horizon = 1, classifier='LogReg', Balanced=True, tiw=False, tiw2=False, curve=False, cut_method=None, with_classifier=False, p_MC = False, Shuffle_samples = False, oversampling = False, warm_start=False, max_iter=500,):
    X = None
    for t in range(day_delay):
        if X is None:
           X = feas_coarse[t:-day_delay+t]
        else:
           X = np.c_[X, feas_coarse[t:-day_delay+t]]
    Prbs=[]
    ROC=[]
    PR=[]
    P=[]
    ST=[]
    ST2=[]
    C1= None
    C2= None
    Clf=[]
    for h in range(horizon):
        Y = labels_coarse[day_delay+h:]
        t_ = t_coarse[day_delay+h:]
        if h >0:
           X_ = X[:-h]*1.0
        else:
           X_=X*1.0

        if Shuffle_samples:
           X_ = shuffle_samples(X_)
        if np.ndim(X_)==1:
           X_ = X_.reshape(-1, 1)
        if  classifier=='mlp'or classifier=='MLP'or classifier=='MultilayerPerceptron' or classifier=='Multilayer_perceptron':
                Dropout_rate=0.0
                batch_size=800
                epochs=500
                my_dir='./'
                pat='pat'
                hidden_layer_size=[16,8,4]

                verbose =0 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                kernel_constraint_weight=3e-2*0
                i_feature_selection=0

                n_1=sum(Y[t_< day_split])
                n_0=len(Y[t_< day_split])-n_1
                class_1_weight=n_0/n_1

                Y[Y<0.5]=0
                mlp = keras_mlp_10m_train_model(X_[t_< day_split], Y[t_< day_split], my_dir, pat, class_1_weight, batch_size, epochs, kernel_constraint_weight, verbose, hidden_layer_size, i_feature_selection)
     
                prbs=keras_mlp_10m_prb_from_model(mlp, X_)
        if  classifier=='PosReg'or classifier=='PR'or classifier=='PoissonRegression' or classifier=='Poisson_regression':
            Y[Y<0.5]=0
            #print(h, day_split, day_delay+h, set(Y[t_< day_split]))
            if Balanced:
                 clf = PoissonRegressor(alpha=0, fit_intercept=True, max_iter=max_iter, warm_start=warm_start).fit(*(balanced(X_[t_< day_split], Y[t_< day_split])))
            else:
                 clf = PoissonRegressor(alpha=0, fit_intercept=True, max_iter=max_iter, warm_start=warm_start).fit(X_[t_< day_split], Y[t_< day_split])
            prbs = clf.predict(X_)
            if cut_method=='exp':
                   prbs = 1-np.exp(-prbs)
            else:
                   prbs[prbs>1.0]=1.0
        if  classifier=='LogReg'or classifier=='LR'or classifier=='LogisticRegression' or classifier=='logistic_regression':
            if oversampling:
                 clf = LogisticRegression(random_state=0, class_weight = 'balanced', warm_start=warm_start).fit(*(balanced(X_[t_< day_split], Y[t_< day_split])))
            else:
                 clf = LogisticRegression(random_state=0, class_weight = 'balanced', warm_start=warm_start).fit(X_[t_< day_split], Y[t_< day_split])
            prbs = clf.predict_proba(X_)[:,1]
        if  classifier=='SVM' or  classifier=='svm' or  classifier=='Svm':        
            penalty='l2'
            loss='squared_hinge'
            dual=False  # prefer when n_samples > n_features          
            tol=0.0001
            max_iter=5000
            C=1     
            clf = svm.LinearSVC(C=C, class_weight='balanced', penalty=penalty, loss=loss, dual=dual, tol=tol, max_iter=max_iter).fit(X_[t_< day_split], Y[t_< day_split])
            prbs = clf._predict_proba_lr(X_)[:,1]
        if tiw2:
           if curve:
              roc, pr, st, st2, cu1, cu2 = get_aucs(labels_coarse[t_coarse> day_split], prbs[t_> day_split], tiw, curve, tiw2=tiw2)
           else:
              roc, pr, st, st2 = get_aucs(labels_coarse[t_coarse> day_split], prbs[t_> day_split], tiw, tiw2=tiw2)
           if p_MC:
              p=0
              for _k in range(200):
                  roc_ = roc_auc_score(labels_coarse[t_coarse> day_split], np.random.permutation(prbs[t_> day_split]))
                  p+=(roc_>roc)
              p/=200
        elif tiw:
           if curve:
              roc, pr, st, cu1, cu2 = get_aucs(labels_coarse[t_coarse> day_split], prbs[t_> day_split], tiw, curve)
           else:
              roc, pr, st = get_aucs(labels_coarse[t_coarse> day_split], prbs[t_> day_split], tiw)
           if p_MC:
              p=0
              for _k in range(200):
                  roc_ = roc_auc_score(labels_coarse[t_coarse> day_split], np.random.permutation(prbs[t_> day_split]))
                  p+=(roc_>roc)
              p/=200
        else:
           if curve:
              roc, pr, cu1, cu2 = get_aucs(labels_coarse[t_coarse> day_split], prbs[t_> day_split], tiw, curve)
           else:
              roc, pr = get_aucs(labels_coarse[t_coarse> day_split], prbs[t_> day_split])
              p=0
              for _k in range(200):
                  roc_ = roc_auc_score(labels_coarse[t_coarse> day_split], np.random.permutation(prbs[t_> day_split]))
                  p+=(roc_>roc)
              p/=200
        Prbs.append(prbs)
        ROC.append(roc)
        PR.append(pr)
        if p_MC:
           P.append(p)
        ST.append(st)
        if tiw2:
           ST2.append(st2)
        Clf.append(clf)
        if curve:
              if C1 is None:
                 C1 = cu1
                 C2 = cu2
              else:
                 C1 = np.r_[C1,cu1]
                 C2 = np.r_[C2,cu2]

#    if with_classifier:
#     if tiw:
#       if curve:
#          return Prbs, ROC, PR, ST, C1, C2, Clf    
#       else:
#          return Prbs, ROC, PR, ST , Clf   
#     else:    
#       if curve:
#          return Prbs, ROC, PR, C1, C2, Clf   
#       else:
#          return Prbs, ROC, PR , Clf 
#    else:
#     if tiw:
#       if curve:
#          return Prbs, ROC, PR, ST, C1, C2    
#       else:
#          return Prbs, ROC, PR, ST    
#     else:    
#       if curve:
#          return Prbs, ROC, PR, C1, C2   
#       else:
#          return Prbs, ROC, PR 
# 
#    if with_classifier:
 
    output = [Prbs, ROC, PR]
    if tiw or tiw2:
       output.append(ST)
    if tiw2:
       output.append(ST2)
    if curve:
       output.append(C1)
       output.append(C2 )
    if with_classifier:
       output.append(Clf)
    if p_MC:
       output.append(P)
    return tuple( output) 


import sys
import numpy as np
import matplotlib as mt 
#mt.use('tkagg') 
mt.use('agg') 
import matplotlib.pyplot as plt 
import os, datetime, errno
import datetime as dt
import time
import glob
import pandas as pd
#from dtaidistance import dtw
#from dtaidistance import dtw_ndim
sys.path.append('/lustre/ssd/ws/hyang-Epilepsy/Utility/')
from scipy import stats
from Utility import *
#import simpledtw

#import vanilladtw
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from joblib import dump, load
from sklearn import svm 

random_state = 170

seg_type = {
1: 'preictal_lead',
2: 'ictal_lead',
3: 'postictal_lead',
-1: 'preictal_nonlead',
-2: 'ictal_nonlead',
-3: 'postictal_nonlead',
9: 'interictal',
0: 'all',
}

Read_mean =True

print_all_data= True

fea_range=['IEA1', 'IEA2', 'IEA12', 'mean' ]
fea_range=['IEA12',  ]

for pid in range(1,19):
  for my_fea in fea_range:
    pat = 'pat'+str(pid) 

    fname = '/beegfs/ws/1/hyang-Epilepsy/Forecasting/Bern-data/preprocessed_'+str(pid)+'.csv'

    ps = pd.read_csv(fname)
    time = ps['Time'].to_numpy()
    sz = ps['Seizures'].to_numpy()

    if len(ps.keys())==3:
       iea1 = ps['IEA'].to_numpy()
       iea2 = iea1*1
    if len(ps.keys())==4:
       iea1 = ps['IEA_1'].to_numpy()
       iea2 = ps['IEA_2'].to_numpy()
    
    t_seg = time-time[0]	# in days
    labels = sz*1
    labels[sz>0]=1
    if my_fea == 'IEA1':
       feas = iea1 
    if my_fea == 'IEA2':
       feas = iea2 
    if my_fea == 'mean':
       feas = (iea1+iea2)/2
    if my_fea == 'IEA12':
       feas = np.c_[iea1, iea2]

    print(pid, feas.shape, labels.shape, t_seg.shape)
#-------------------
    idt = np.argsort(t_seg)
    feas = feas[idt]
    labels = labels[idt] 
    t_seg = t_seg[idt]
#---------------------------------------------------------- 
    print(set(labels))
#-------------------
    print(feas.shape, t_seg.shape, labels.shape)
    

    my_dir='./'
    try:
        os.makedirs(my_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
#----------------------------------------------------------
    ########################################
    if True:
     Bandstop = False
     Bandpass = True
     Highpass = False
     Lowpass = False     
     nf = 5
     filtertype = 'butt'     

     bandpass = 'none'
     frange = 1
     
     if Highpass:
        bandpass = 'highpass'
        frange = 1
     if Lowpass:
        bandpass = 'lowpass'
        frange = 1
     if Bandpass:
        bandpass = 'bandpass'
        frange = np.array([0.02,1/5])	# in days
     if Bandstop:
        bandpass = 'bandstop'
        frange = np.array([0.02,1])
        
     causal = False
     Phase = False
     Phase2 = False
     Phaseb = False
     Phasec = False
     Phase2b = False
     Amplitude = False
     Amplitude2 = False
     Amplitudeb = False
     Amplitude2b = False
     
     Hilbert=False
     if Amplitude or Phase or  Amplitude2 or Phase2 or Amplitudeb or Phaseb or Phasec or  Amplitude2b or Phase2b :
        Hilbert=True
        
     tep = 1
     if Highpass or Lowpass or Bandpass or Bandstop or Hilbert:
        fc = frange/(1*tep) 	# in days
        os.system('date')
        if Hilbert:
           feas, amp_feas, pha_feas = bandpass_even(feas, t_seg, bandpass=bandpass, fc = fc, tau =1/24, t_gap = 14, Hilbert=Hilbert, multi_f=False, causal = causal, nf=nf, filtertype = filtertype)
        else:
           feas = bandpass_even(feas, t_seg, bandpass=bandpass, fc = fc, tau =1/24, t_gap = 14, Hilbert=Hilbert, multi_f=False, causal = causal, nf=nf, filtertype = filtertype)        

     if Phase or Phaseb:
        feas = pha_feas
     if Phasec:
        feas = np.c_[np.cos(pha_feas),np.sin(pha_feas)]
     if Phase2 or Phase2b:
        feas = np.c_[feas, pha_feas]
     if Amplitude:
        feas = amp_feas
     if Amplitude2:
        feas = np.c_[feas, amp_feas]
     if Amplitude and Phase:
        feas = np.c_[amp_feas, pha_feas]
     if Amplitude2 and Phase2:
        feas = np.c_[feas, amp_feas, pha_feas]
#--------------------
    if print_all_data:
       np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+my_fea+'_time_Low.dat', np.c_[t_seg, feas])

######################################################
# PCA reprocessing
    i_pca = 'All'
    
#--------------------
# forecasting
#--------------------
    u = 1	# in days
    
    Tau= 1/u	# time bin in days
    tau =1/24   # sample bin in day 
    Dt = tau/Tau

    duration = time[-1]-time[0]
    split = min(480, duration *0.6)

    day_split = split*u
    
    day_delay = 5
    horizon = u*6
    classifier='LR'

    cross_validation = False
    print_prb = True
    get_BSS=True
    p_MC2=True
    print_clf_parameters= True
    #for method in ['mean','FV', 'AR_all' , 'AR']:
    for method in ['mean']:
        print('origin:', feas.shape, labels.shape, t_seg.shape, set(labels))
        feas_coarse, labels_coarse, t_coarse =  coarse_fea_label_new2(feas, labels, t_seg, Tau, gap_threshold=1, n_components=3, n_delay=3, delay=240, n_hidden_state=3, n_mixtures=3, method = method)
        print('coarsen:', feas_coarse.shape, labels_coarse.shape, t_coarse.shape, set(labels_coarse))
        feas_coarse,_1,_2,_3 = fea_nan_inf(feas_coarse)

        if Phaseb:
           feas_coarse = np.c_[np.cos(feas_coarse), np.sin(feas_coarse)]
        if Phase2b:
           d_fea = feas_coarse.shape[-1]
           feas_coarse = np.c_[feas_coarse[:, :d_fea//2], np.cos(feas_coarse[:, (d_fea//2):]), np.sin(feas_coarse[:, (d_fea//2):])]


        my_data = np.c_[ np.mean(feas_coarse, axis=0), np.std(feas_coarse, axis=0)]
        my_data = np.r_[np.c_[ np.mean(feas_coarse[t_coarse< day_split], axis=0), np.std(feas_coarse[t_coarse< day_split], axis=0)]]
        my_data = np.r_[np.c_[np.mean(feas_coarse[t_coarse> day_split], axis=0), np.std(feas_coarse[t_coarse> day_split], axis=0)]]
        print('fea_coarse_mean', np.mean(feas_coarse, axis=0), np.std(feas_coarse, axis=0))
        print('fea_coarse_mean_train', np.mean(feas_coarse[t_coarse< day_split], axis=0), np.std(feas_coarse[t_coarse< day_split], axis=0))
        print('fea_coarse_mean_test', np.mean(feas_coarse[t_coarse> day_split], axis=0), np.std(feas_coarse[t_coarse> day_split], axis=0))
        
        np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+my_fea+'_time_coarse_'+method+'_statistics.dat',my_data)
        #f3 = open(os.path.join(my_dir,'kaggle_svc_pca_angle_LS.dat'), 'a')
        #f3.write("%s %s %d %f %f %f %f\n" % (pat, uvar_fea[my_fea]+mvar_fea[my_fea], n, r_L, r_S, w_L, w_S))
        #f3.close()
        
        if print_all_data:
           np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+my_fea+'_time_coarse_'+method+'.dat', np.c_[t_coarse, labels_coarse, feas_coarse ])
    

# cross validation
        if cross_validation:
           print('pat', pid)
           day_3fold = get_day_nfold(labels_coarse, t_coarse, day_split, n=3)
           #check_day_nfold(labels_coarse, t_coarse, day_split, day_3fold, n=3)
           roc, pr, senptiw, senptiw2 = forecast_CV(day_3fold, feas_coarse, labels_coarse, t_coarse, day_split= day_split, day_delay=day_delay, horizon = horizon, classifier=classifier, tiw=True, tiw2=True, cut_method='', Shuffle_samples=False)
           print('ROC_CV ', method, pid, my_fea, roc,pr, senptiw, senptiw2)

        p_MC=False
        if p_MC:
           prbs, roc, pr, senptiw, senptiw2, C1, C2, Clf, P = forecast(feas_coarse, labels_coarse, t_coarse, day_split= day_split, day_delay=day_delay, horizon = horizon, classifier=classifier, tiw=True, tiw2=True, curve = True, cut_method='', with_classifier=True, p_MC=True, Shuffle_samples=False)
           print('ROC ', method, pid, my_fea, roc,pr, senptiw, senptiw2, P)
        else:
           prbs, roc, pr, senptiw, senptiw2, C1, C2, Clf = forecast(feas_coarse, labels_coarse, t_coarse, day_split= day_split, day_delay=day_delay, horizon = horizon, classifier=classifier, tiw=True, tiw2=True, curve = True, cut_method='', with_classifier=True, Shuffle_samples=False,)
           print('ROC ', method, pid, my_fea, roc,pr, senptiw, senptiw2)
        if get_BSS:
         BSS = np.zeros(len(prbs))
         P_MC = np.zeros(len(prbs))
         for i4 in range(len(prbs)):
          roc0=roc[i4]
          label = labels_coarse[day_delay+i4:][t_coarse[day_delay+i4:]> day_split]
          prb = prbs[i4][t_coarse[day_delay+i4:]> day_split]*1
    
          bs = np.mean((label-prb)*(label-prb))
          nbs=200
          bs0=0
          p=0
          for kk in range(nbs):
              np.random.shuffle(prb)
              bs0+= np.mean((label-prb)*(label-prb))
              if p_MC2:
                  fpr, tpr, ths = metrics.roc_curve(label, prb)
                  roc_ = metrics.auc(fpr, tpr)
                  p+=(roc_>roc0)
          p/=nbs
          bs0/=nbs
          bss = 1-bs/bs0
          BSS[i4]=bss
          P_MC[i4]=p
         print('BSS ',method, pid, my_fea, list(BSS), list(P_MC))
         np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+my_fea+'_BSS.dat', BSS)
         np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+my_fea+'_P_MC.dat', P_MC)
        Parameters = []
        for i in range(len(prbs)):
            if print_prb:
               np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+ method+'_'+my_fea+'_time_label_prb_'+str(i)+'.dat', np.c_[t_coarse[day_delay+i:], labels_coarse[day_delay+i:], prbs[i] ])
            day_tested = (t_coarse[-1]-day_split-i)/u
            ap, threshold_ptifw, sensitivity_ptifw, n_fw_ptifw, threshold_sen, ptifw_sen, n_fw_sen = get_ap_ptifw_sen_nfw(labels_coarse[day_delay+i:][t_coarse[day_delay+i:]> day_split], prbs[i][t_coarse[day_delay+i:]> day_split])

            f3 = open(os.path.join(my_dir,'tiw_info.dat'), 'a')
            f3.write("%s %d %s %s %d %f %f %f %f %f %f %f\n" % (method, pid, my_fea, 'i ap, threshold_ptifw, sensitivity_ptifw, n_fw_ptifw/day_tested, threshold_sen, ptifw_sen, n_fw_sen/day_tested', i, ap, threshold_ptifw, sensitivity_ptifw, n_fw_ptifw/day_tested, threshold_sen, ptifw_sen, n_fw_sen/day_tested))
            f3.close()

            para = np.r_[Clf[i].coef_.reshape(-1), Clf[i].intercept_]

            Parameters.append(para)
        Parameters= np.array(Parameters)
        if print_clf_parameters:
           np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+ method+'_'+my_fea+'_model_parameters.dat', Parameters)

        if print_all_data:
           np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+ method+'_'+my_fea+'_ROC_curves.dat', C1)
           np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+ method+'_'+my_fea+'_PRSenTiw_curves.dat', C2)
    
        
    #feas_coarse, labels_coarse, t_coarse =  coarse_fea_label(feas, labels, t_seg, Tau, gap_threshold=3600, n_hidden_state=3, n_mixtures=3, method = 'HMM')
    #print(feas_coarse.shape, labels_coarse.shape, t_coarse.shape)
    #np.savetxt(my_dir+'fea_ieeg_complete_pat'+str(pid)+'_'+my_fea+'_time_coarse_HMM.dat', np.c_[t_coarse, labels_coarse, feas_coarse ])

#--------------------
 

