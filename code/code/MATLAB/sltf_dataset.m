warning('off','MATLAB:MKDIR:DirectoryExists')

folder='wifi_2024_10_29/packets_csi/';

rx_nodes = dir([folder,'packets_csi/']);


for rx_i = 3 : length(rx_nodes)
    rx_node = rx_nodes(rx_i).name;
    disp(['Started Processing, ' num2str(rx_i) ' : ' rx_node])
    t1 = tic;
    
    fls = dir([folder,'packets_csi/',rx_node]);
   
    
    for fl_i = 3 : length(fls)
        fl = fls(fl_i).name;

        fprintf(sprintf('File %d of %d: %s' , fl_i, length(fls),fl) ); 
        t2=tic;
        load([folder,'packets_csi/',rx_node,'/',fl]);
%         csi_log_in = csi_log;
        csi_log_in = extended_data;
        packet_log_op = {};
        for csi_i=1:length(csi_log_in)
            csi = csi_log_in{csi_i};
            pkt_op = generate_sltf(csi');
            if ~isempty(pkt_op)
                packet_log_op{end+1}=pkt_op;
            end
        end
        fprintf(sprintf('  %d \n' , toc(t2) ))
        packet_log=packet_log_op;
        if ~isempty(packet_log)
            mkdir([folder,'packets_sltf/'])
            mkdir([folder,'packets_sltf/',rx_node])
            save([folder,'packets_sltf/',rx_node,'/',fl],'packet_log')
        end
    end
    disp(toc(t1))
end