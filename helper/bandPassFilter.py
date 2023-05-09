def apply_filter(wav_file_path, start_frq, end_frq, ):
    sig, sr = sf.read(wav_file_path)
    sr_new = 210000
    sig_new = librosa.resample(sig, orig_sr=sr, target_sr=sr_new)

    sos = signal.butter(6, [5000, 100000], 'bandpass', fs=sr_new, output='sos')

    tf_sig = lr.stft(signal.sosfiltfilt(sos, sig_new), n_fft=2048)
    tf_sig = np.abs(tf_sig)
    lr.display.specshow(lr.amplitude_to_db(tf_sig, ref=np.max));plt.title("Spectrogram");plt.xlabel("Time"); plt.ylabel("Frequencies");plt.show()