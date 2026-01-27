#%%
#? Auth.: Manash Chakraborty
#? Estimation.

#%% Imports:
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt


#%% G-H Filters [v1]
#==============================================
#? estimate = prediction + (4/10)(measurement - prediction). Four-thenth is coming from meas., rest is from prediction.
#? residual = measurement - prediction
#? Assumptions:
#?  - gain_rate is known. This is the known rate of change for the predictions.
#==============================================
def gh_gain_guess(measurements, estimated, scale_factor, gain_rate=1, time_step=1, do_print=False, do_plot=False):
    #* storage for results
    estimates, predictions = [estimated], []

    #* Update Loop
    for k, z in enumerate(measurements, start=1):
        #* Prev. Estimate
        prev_estimated = estimated
        #* new prediction
        predicted = estimated + gain_rate*time_step
        #* new estimate
        estimated = predicted + scale_factor*(z-predicted)
        #* log results
        estimates.append(estimated)
        predictions.append(predicted)

        #* print (conditional)
        if do_print:
            print(f"Prev. Estimate: {prev_estimated:.3f}, Prediction: {predicted:.3f}, New Estimate: {estimated:.3f}.")
        #* Plot (conditional)
        n = len(measurements)
        t_meas_pred = list(range(1, n + 1))
        t_est = list(range(0, n + 1))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_meas_pred, measurements, label='Measurements (z)', color='k',
                linestyle='-', marker='o', linewidth=1.5, markersize=5)
        ax.plot(t_meas_pred, predictions, label='Predictions', color='#1f77b4',
                linestyle='--', marker='^', linewidth=1.6, markersize=5)
        ax.plot(t_est, estimates, label='Estimates', color='#d62728',
                linestyle='-', marker='s', linewidth=1.8, markersize=5)

        ax.set_title(title)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.margins(x=0.02)


    return estimates, predictions
#_______________________________________________________

weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
time_step = 1.0  # per-day
scale_factor = 4.0/10
gain_rate = 1
init_estimate = 160.0

estimates, predictions = gh_gain_guess(measurements=weights, 
                                        estimated=init_estimate,
                                        scale_factor=scale_factor,
                                        gain_rate=gain_rate,
                                        time_step=time_step,
                                        do_print=True,
                                        do_plot=True)




# %%
