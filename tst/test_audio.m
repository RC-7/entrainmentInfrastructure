
% filename = "pink_noise.m4a";
% [data,fs]=audioread(filename);
% data = data(fs*16:fs*17);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% 24 Hz entrain %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure,
% filename = "24Hz.m4a";
% [data,fs]=audioread(filename);
% data_left = data(fs*10:fs*13);
% data_right= data(fs*20:fs*23);
% 
% subplot(2,1,1),
% y = fft(data_left);   
% f = (0:length(y)-1)*fs/length(y);
% 
% vals = [f' abs(y)];
% writematrix(vals, '24Hz_l.csv');
% % plot(f,abs(y))
% % xlim([380 500])
% % xlabel('Frequency (Hz)')
% % ylabel('Magnitude')
% % title('Magnitude')
% 
% subplot(2,1,2),
% y = fft(data_right);   
% f = (0:length(y)-1)*fs/length(y);
% vals = [f' abs(y)];
% writematrix(vals, '24Hz_r.csv');
% plot(f,abs(y))
% xlim([380 500])
% xlabel('Frequency (Hz)')
% ylabel('Magnitude')
% title('Magnitude')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% 18 Hz entrain %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure,
% filename = "pls.m4a";
% [data,fs]=audioread(filename);
% data_left = data(fs*8:fs*10);
% data_right= data(fs*19:fs*21);
% 
% subplot(2,1,1),
% y = fft(data_left);   
% f = (0:length(y)-1)*fs/length(y);
% vals = [f' abs(y)];
% writematrix(vals, '18Hz_l.csv');
% % plot(f,abs(y))
% % xlim([380 500])
% % xlabel('Frequency (Hz)')
% % ylabel('Magnitude')
% % title('Magnitude')
% 
% subplot(2,1,2),
% y = fft(data_right);   
% f = (0:length(y)-1)*fs/length(y);
% vals = [f' abs(y)];
% writematrix(vals, '18Hz_r.csv');
% plot(f,abs(y))
% xlim([380 500])
% xlabel('Frequency (Hz)')
% ylabel('Magnitude')
% title('Magnitude')


% data = bandpass(data,[10 20000],fs);

% Until 17 and from 20

% filename = "pinkNoise.wav";
% [data_true,f2_true]=audioread(filename);
% t1 = 0:1/f2_true:length(data_true)/f2_true;
% t1 = t1(1:end-1);
% data_resample = resample(data_true, t1, fs);


% data_true = data_true(fs*3:fs*4);
% s = spectrogram(y1);
% spectrogram(y1,500,120,128,fs,'yaxis')
% ax = gca;
% ax.YScale = 'log';

% subplot(3,1,1), plot (data);
% subplot(3,1,2), plot (data_true);
% 
% [C1, lag1] = xcorr(data,data_true);
% subplot(3,1,3), plot(lag1/fs,C1);
% ylabel("Amplitude"); grid on
% title("Cross-correlation ")


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Pink Analysis %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure,
filename = "pink_final.m4a";
[data,fs]=audioread(filename);
data = data(fs*12:fs*14);

filename = "pinkNoise.wav";
[data_true,f2_true]=audioread(filename);
t1 = 0:1/f2_true:length(data_true)/f2_true;
t1 = t1(1:end-1);


data_true = data_true(fs*3:fs*4);

subplot(2,1,1),
y = fft(data);   
f = (0:length(y)-1)*fs/length(y);
% vals = [f' abs(y)];
% writematrix(vals, 'pink_ours.csv');
plot(f,abs(y))
xlim([0 20000])
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('Magnitude')

subplot(2,1,2),
y = fft(data_true);   
f = (0:length(y)-1)*fs/length(y);
% vals = [f' abs(y)];
% writematrix(vals, 'pink_ideal.csv');
plot(f,abs(y))
xlim([0 20000])
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('Magnitude')
