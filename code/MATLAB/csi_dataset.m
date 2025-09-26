warning('off','MATLAB:MKDIR:DirectoryExists')

folder='wifi_2021_03_01/';

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
        csi_log_in = csi_log;
%         csi_log_in = selected_valid_data;
        csi_log_op = {};
        for csi_i=1:length(csi_log_in)
            csi = csi_log_in{csi_i};
            csi_op = generate_csi(csi);
            if ~isempty(csi_op)
                csi_log_op{end+1}=csi_op;
            end
        end
        fprintf(sprintf('  %d \n' , toc(t2) ))
        csi_log=csi_log_op;
        if ~isempty(csi_log)
%             mkdir([folder,'packets_csi_processed/'])
%             mkdir([folder,'packets_csi_processed/',rx_node])
%             save([folder,'packets_csi_processed/',rx_node,'/',fl],'csi_log')
            mkdir([folder,'packets_csi_processed/'])
            mkdir([folder,'packets_csi_processed/',rx_node])
            save([folder,'packets_csi_processed/',rx_node,'/',fl],'csi_log')

        end
    end
    disp(toc(t1))
end