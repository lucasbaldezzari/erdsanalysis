# ## ************************ SEPARANDO EN ÉPOCAS ************************
# """
# IMPORTANTE: Los eventos fueron marcados en la fase_cue de la app de presentación de estímulos.
# """
# ##tiempos promedio por trial según la app y los eventos marcados por el g.HIAMP
# print(eventos_app["cueInitTime"].diff().mean())
# print(np.diff(events_time_ghiamp).mean())

# tmin, tmax = -1, 4
# event_ids = dict(IZQUIERDA=1, DERECHA=2)
# epocas = mne.Epochs(eeg_data, event_id=["IZQUIERDA", "DERECHA"],
#                     tmin=tmin-0.5, tmax=tmax+0.5,
#                     picks=["C3", "Cz", "C4"],
#                     baseline=None, preload=True)

# freqs = np.arange(6, 36)  # frequencies from 2-35Hz
# vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
# baseline = (-1, -0.5)  # baseline interval (in s)
# cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

# kwargs = dict(
#     n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
# ) 

# tfr = epocas.compute_tfr(
#     method="multitaper",
#     freqs=freqs,
#     n_cycles=freqs,
#     use_fft=True,
#     return_itc=False,
#     average=False,
#     decim=2,
# )
# tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")

# for event in event_ids:
#     # select desired epochs for visualization
#     tfr_ev = tfr[event]
#     fig, axes = plt.subplots(
#         1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
#     )
#     for ch, ax in enumerate(axes[:-1]):  # for each channel
#         # positive clusters
#         _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
#         # negative clusters
#         _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

#         # note that we keep clusters with p <= 0.05 from the combined clusters
#         # of two independent tests; in this example, we do not correct for
#         # these two comparisons
#         c = np.stack(c1 + c2, axis=2)  # combined clusters
#         p = np.concatenate((p1, p2))  # combined p-values
#         mask = c[..., p <= 0.05].any(axis=-1)

#         # plot TFR (ERDS map with masking)
#         tfr_ev.average().plot(
#             [ch],
#             cmap="RdBu",
#             cnorm=cnorm,
#             axes=ax,
#             colorbar=False,
#             show=False,
#             mask=mask,
#             mask_style="mask",
#         )

#         ax.set_title(epocas.ch_names[ch], fontsize=10)
#         ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
#         if ch != 0:
#             ax.set_ylabel("")
#             ax.set_yticklabels("")
#     fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
#     fig.suptitle(f"ERDS ({event})")
#     plt.show()