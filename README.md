# Arrhythmia-classification
Arrhythmia classification algorithm based on deep learning

In the thesis, the ECG signals are classified into **five** categories:

Normal beat (N)

Left bundle branch block beat (L)

Right bundle branch block beat (R)

Premature atrial contraction (A)

Premature ventricular contraction (V)
## data set
**MIT-BIH**

It contains 48 ECG recordings from 47 patients, each lasting 30 minutes.

**Sampling rate**: 360 Hz, 11-bit resolution.

**Leads**: Mostly MLII and V5, with variations in some records.

link:https://physionet.org/content/mitdb/1.0.0/
## data processing(denoise + heartbeat segmentation)
Discrete Wavelet Transform (DWT) was used for ECG signal denoising.

### Wavelet Decomposition:

The raw ECG signal was decomposed into different frequency components using Daubechies5 (db5) wavelet with 9 levels of decomposition， which allowed the separation of noise components.

### Thresholding for Noise Removal:

A soft thresholding technique was applied to the wavelet coefficients using the VisuShrink threshold formula:

![image](https://github.com/user-attachments/assets/44af57eb-17f0-4275-8c78-7a9a42146c0d)

where w represents the original wavelet coefficient, 𝑊 is the processed wavelet coefficient, 𝜆 is the given threshold.

### Wavelet Reconstruction:

After thresholding, the cleaned ECG signal was reconstructed by applying the Inverse Discrete Wavelet Transform (IDWT).

### result

The left image shows the original signal, while the right image shows the denoised signal.

![image](https://github.com/user-attachments/assets/a553d0a2-ed35-4437-94a6-f7e1c649c705)

