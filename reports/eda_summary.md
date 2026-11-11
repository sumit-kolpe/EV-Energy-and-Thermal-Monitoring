# EDA summary (from notebooks/03_eda.ipynb)

- **n_vehicles**: 7
- **n_trips**: 2389
- **total_km**: 43662.0
- **trips_min_max_per_vehicle**: (46, 836)
- **trip_duration_median_min**: 15.3
- **trip_distance_median_km**: 6.8
- **trips_valid_energy_5km**: 1374
- **trips_valid_energy_20km**: 420
- **soc_per_100km_median_5km**: 33.8
- **bat_temp_p90_p95**: [30.3, 32.0]
- **dT5_p90_p95_p99**: [0.7, 1.09, 2.49]
- **D_T_positive_rate**: 0.164
- **corr_packmax_vs_outlet**: 0.876
- **share_driving_rows_with_bat_temp**: 0.431
- **share_driving_rows_with_outlet_temp**: 0.913

## Notes for Phase 2/3
- Trip counts are very uneven across vehicles: use per-vehicle permutation tests and GroupKFold by vehicle or trip.
- SOC%/100km is only reliable for longer trips; compare the 5 km and 20 km filters.
- The candidate thermal-demand rule above is data-driven (p90 temperature / p95 5-min rise); confirm as a team.
- Build forecasting features only from rows up to the prediction time (no whole-trip aggregates).