figure,
filename = "24Hz.m4a";
[data,fs]=audioread(filename);
data_left = data(fs*10:fs*10.02);
data_right= data(fs*20:fs*20.02);

plot(data_left)
plot(data_right)

writematrix(data_left, '24Hz_l_raw.csv');
writematrix(data_right, '24Hz_r_raw.csv');


figure,
filename = "pls.m4a";
[data,fs]=audioread(filename);
data_left = data(fs*8.02:fs*8.04);
data_right= data(fs*19.02:fs*19.04);


writematrix(data_left, '18Hz_l_raw.csv');
writematrix(data_right, '18Hz_r_raw.csv');


