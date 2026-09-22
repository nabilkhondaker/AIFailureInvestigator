# Signal processing

Implemented transforms:

- FFT magnitude spectrum
- Welch PSD
- Spectrogram
- Hilbert envelope and envelope spectrum
- Spectral peak picking
- Order amplitudes relative to shaft speed
- Time-domain statistics: RMS, peak, crest factor, kurtosis, skewness, etc.

Example definitions:

$$
\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
$$

$$
\mathrm{CrestFactor} = \frac{\max_i |x_i|}{\mathrm{RMS}}
$$

Kurtosis and crest factor are useful for impulsive bearing-related content; order amplitudes help separate imbalance (1×) from misalignment (2×).
