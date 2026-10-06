"""Temporal analytics: event times, serial dependence, spectra, stationarity, change points, long memory.

Modules
-------
    events         inter-arrival times, burstiness (B and the finite-size A_n), memory, count series
    correlation    ACF (FFT), PACF (Durbin-Levinson), Bartlett bands, Ljung-Box
    spectral       Welch density, periodogram, Fisher's g test, peak ratio, Lomb-Scargle
    stationarity   augmented Dickey-Fuller (MacKinnon surfaces or simulated null), KPSS (Newey-West bandwidth)
    changepoint    PELT with Gaussian, Poisson and exponential costs; Bayesian online change points
    hurst          detrended fluctuation analysis, aggregated variance, exact fractional Gaussian noise
    analysis       reports for a series and for the event times of a corpus
"""

from nagahana.analytics.temporal.analysis import corpus_report, series_report
from nagahana.analytics.temporal.changepoint import OnlineChangepoints, Segmentation, bocpd, pelt
from nagahana.analytics.temporal.correlation import SerialDependence, acf, pacf, serial_dependence
from nagahana.analytics.temporal.events import burstiness, count_series, inter_arrival, log_histogram
from nagahana.analytics.temporal.hurst import Fluctuation, aggregated_variance, dfa, simulate_fgn
from nagahana.analytics.temporal.spectral import Periodogram, false_alarm, fisher_g_pvalue, lomb_scargle, periodogram
from nagahana.analytics.temporal.stationarity import (
    ADFResult,
    KPSSResult,
    adf,
    kpss,
    mackinnon_critical,
    mackinnon_p,
    simulate_df,
)

__all__ = [
    "ADFResult", "Fluctuation", "KPSSResult", "OnlineChangepoints", "Periodogram", "Segmentation", "SerialDependence", "acf",
    "adf", "aggregated_variance", "bocpd", "burstiness", "corpus_report", "count_series", "dfa", "false_alarm",
    "fisher_g_pvalue", "inter_arrival", "kpss", "log_histogram", "lomb_scargle", "mackinnon_critical", "mackinnon_p",
    "pacf", "pelt", "periodogram", "serial_dependence", "series_report", "simulate_df", "simulate_fgn",
]
