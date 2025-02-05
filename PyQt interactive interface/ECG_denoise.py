# import numpy as np
# import pywt
# import wfdb
#
# # 设置随机种子为固定值，例如 42
# np.random.seed(42)
#
# def add_noise(ecg_signal):
#     fs = 360
#     t = np.arange(len(ecg_signal)) / fs
#     # 调整正弦波噪声的幅度
#     amp_60Hz = 0.01 * np.max(ecg_signal)  # 假设噪声幅度为最大信号幅度的1%
#     amp_0_5Hz = 0.01 * np.max(ecg_signal)
#
#     #noise_60Hz = amp_60Hz * np.sin(2 * np.pi * 60 * t)
#     #noise_0_5Hz = amp_0_5Hz * np.sin(2 * np.pi * 0.5 * t)
#
#     noise_60Hz = amp_60Hz * np.sin(2 * np.pi * 60 * t)
#     noise_0_5Hz = amp_60Hz * np.sin(2 * np.pi * 0.5 * t)
#
#     # # 添加10dB信噪比的高斯白噪声
#     # signal_power = np.mean(ecg_signal ** 2)
#     # noise_power = signal_power / (10 ** (10 / 10))
#     # white_noise = np.random.normal(0, np.sqrt(noise_power), len(ecg_signal))
#     # 计算实际信号功率
#     # 计算实际信号功率
#     signal_power = np.mean(ecg_signal ** 2)
#     # 计算信噪比（dB）
#     snr_db = 10  # 期望的信噪比（dB）
#     # 将信噪比从dB转换为线性值
#     snr_linear = 10 ** (snr_db / 10)
#     # 计算噪声功率
#     noise_power = signal_power / snr_linear
#     # 添加高斯白噪声
#     white_noise = np.random.normal(0, np.sqrt(noise_power), len(ecg_signal))
#
#     #noisy_signal = ecg_signal + noise_60Hz + noise_0_5Hz + white_noise
#     noisy_signal = ecg_signal + noise_60Hz + noise_0_5Hz + white_noise
#     return noisy_signal
#
# def threshold_denoise(noisy_signal, wavelet, mode):
#     #coeffs = pywt.wavedec(noisy_signal, wavelet)
#     coeffs = pywt.wavedec(noisy_signal, wavelet, level=9)
#     threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * (2 * np.log(len(noisy_signal)))**0.5
#     new_coeffs = [pywt.threshold(coeff, threshold, mode=mode) if i > 0 else coeff for i, coeff in enumerate(coeffs)]
#     denoised_signal = pywt.waverec(new_coeffs, wavelet)
#     return denoised_signal
#
# def calculate_snr(original, denoised):
#     # 计算信号功率和噪声功率
#     signal_power = np.mean(original ** 2)
#     noise_power = np.mean((original - denoised) ** 2)
#     snr = 10 * np.log10(signal_power / noise_power)
#     return snr
#
# def calculate_rmse(original, denoised):
#     return np.sqrt(np.mean((original - denoised) ** 2))
#
# # 读取ECG数据
# record = wfdb.rdrecord('D:/大学/毕设/MIT-BIH-360/103',physical=True,channels=[0])
# ecg_signal = record.p_signal.flatten()
# noisy_signal = add_noise(ecg_signal)
#
# wavelets = ['db' + str(i) for i in range(1, 10)]
# results = {}
#
# for wavelet in wavelets:
#     for mode in ['soft']:
#         denoised_signal = threshold_denoise(noisy_signal, wavelet, mode)
#         #denoised_signal = threshold_denoise(ecg_signal, wavelet, mode)
#         snr = calculate_snr(ecg_signal, denoised_signal)
#         rmse = calculate_rmse(ecg_signal, denoised_signal)
#         results[(wavelet, mode)] = {'SNR': snr, 'RMSE': rmse}
#
# # 输出结果
# for key, value in results.items():
#     print(f"Wavelet: {key[0]}, SNR: {value['SNR']:.2f}, RMSE: {value['RMSE']:.4f}")

import numpy as np
import pywt
import wfdb

def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换, 获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def calculate_snr(signal, denoised_signal):
    power_signal = np.mean(signal**2)
    power_noise = np.mean((signal - denoised_signal)**2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr

def get_dataset(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    print(f"正在读取 {number} 号心电数据...")
    record = wfdb.rdrecord(f'D:/大学/毕设/MIT-BIH-360/{number}', channel_names=['MLII'])
    data = record.p_signal.flatten()
    original_signal = np.copy(data)  # 保留原始信号用于计算SNR
    denoised_signal = denoise(data=data)
    snr_value = calculate_snr(original_signal, denoised_signal)
    print(f"信噪比 (SNR): {snr_value} dB")
    return denoised_signal, snr_value

# Example usage
number = "100"  # Record number, change as needed
X_data, Y_data = [], []
denoised_signal, snr_value = get_dataset(number, X_data, Y_data)