function pkt_ltf_op = generate_sltf(csi)
csi_eq = csi;
% for i = 1 : 51
%     csi_eq(i) = csi(i+1) / csi(i);
% end
% csi_eq(52)=csi(52) / csi(1);
cfgNonHT = wlanNonHTConfig('ChannelBandwidth', 'CBW20');
cfgOFDM = wlan.internal.wlanGetOFDMConfig('CBW20', 'Half', 'Legacy');
numTx = 1;
FFTLen = 64;
num20  = FFTLen/64;
csh = wlan.internal.getCyclicShiftVal('OFDM', numTx, 20*num20);

% L-LTF
ltf_standard = wlanLLTF(cfgNonHT);
ltf_standard_freq_domain = fft(ltf_standard,64);
ltf_standard_amplitude_spectrum = abs(ltf_standard_freq_domain);
threshold = 0.01;
indices = ltf_standard_amplitude_spectrum > threshold;
ltf_standard_freq_domain(indices) = csi_eq;
ltf_withcsi_freq_domain = ltf_standard_freq_domain;
[lltfLower, lltfUpper] = wlan.internal.lltfSequence();
lltf = [zeros(6,1);  lltfLower; 0; lltfUpper; zeros(5,1)];
lltfMIMO = bsxfun(@times, repmat(lltf, 1, 1), cfgOFDM.CarrierRotations);
lltfCycShift = wlan.internal.wlanCyclicShift(lltfMIMO, csh, FFTLen, 'Tx');
lltfCycShift_withcsi = lltfCycShift.*ltf_withcsi_freq_domain;
modOut_2 = ifft(ifftshift(lltfCycShift_withcsi, 1), [], 1);        
out_2 = [modOut_2((end-FFTLen/2+1):end,:); modOut_2; modOut_2];
lltf  = out_2* cfgOFDM.NormalizationFactor / sqrt(numTx);

% L-STF
stf_standard = wlanLSTF(cfgNonHT);
stf_standard_freq_domain = fft(stf_standard,64);
stf_standard_amplitude_spectrum = abs(stf_standard_freq_domain);
threshold = 0.01;
indices = stf_standard_amplitude_spectrum > threshold;
stf_standard_freq_domain(indices) = ltf_withcsi_freq_domain(indices);
stf_withcsi_freq_domain = stf_standard_freq_domain;
LSTF = wlan.internal.lstfSequence();
N_LSTF_TONE = 12*num20;
lstf = [zeros(6,1); LSTF; zeros(5,1)];
lstfMIMO = bsxfun(@times, repmat(lstf, num20, numTx), cfgOFDM.CarrierRotations);
lstfCycShift = wlan.internal.wlanCyclicShift(lstfMIMO, csh, FFTLen, 'Tx');
lstfCycShift_withcsi = lstfCycShift.*stf_withcsi_freq_domain;
modOut_1 = ifft(ifftshift(lstfCycShift_withcsi, 1), [], 1);       
out_1 = [modOut_1; modOut_1; modOut_1(1:FFTLen/2,:)];
lstf = out_1* (FFTLen/sqrt(numTx*N_LSTF_TONE));


pkt_ltf_op = [lstf;lltf];
