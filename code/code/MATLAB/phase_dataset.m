warning('off','MATLAB:MKDIR:DirectoryExists')

folder='wifi_2024_10_29/';

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
        phase_log_op = {};
        for csi_i=1:length(csi_log_in)
            csi = csi_log_in{csi_i};
            phase_op = extractPhase(csi);
            if ~isempty(phase_op)
                phase_log_op{end+1}=phase_op;
            end
        end
        fprintf(sprintf('  %d \n' , toc(t2) ))
        phase_log=phase_log_op;
        if ~isempty(phase_log)
            mkdir([folder,'packets_phase/'])
            mkdir([folder,'packets_phase/',rx_node])
            save([folder,'packets_phase/',rx_node,'/',fl],'phase_log')
        end
    end
    disp(toc(t1))
end

function phaseData = extractPhase(csi)
    % ��ʼ��һ����Ԫ�����飬���ڴ洢��λ��Ϣ�������� csi_log ��ͬ�Ľṹ
    % phaseData = cell(size(csi_log));
    phaseData = zeros(size(csi));
    % ���� csi_log �е�ÿ��Ԫ�أ���ȡ��λ��Ϣ
    for i = 1:length(csi)
        % ��ȡ������ʽ�� CSI ����
        csi_complex = csi(i);
        
        % ʹ�� angle ������ȡ��λ��Ϣ
        phaseData(i) = angle(csi_complex);
    end
end