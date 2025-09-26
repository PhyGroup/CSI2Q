function csi_log_op = get_csi(pkt)
nht = wlanNonHTConfig;
pkt_ltf = pkt(161:320);
demodSig = wlanLLTFDemodulate(pkt_ltf,nht);
est = wlanLLTFChannelEstimate(demodSig,nht);
csi_log_op = est;