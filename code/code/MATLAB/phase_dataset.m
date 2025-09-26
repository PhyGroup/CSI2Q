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
    % 初始化一个单元格数组，用于存储相位信息，保持与 csi_log 相同的结构
    % phaseData = cell(size(csi_log));
    phaseData = zeros(size(csi));
    % 遍历 csi_log 中的每个元素，提取相位信息
    for i = 1:length(csi)
        % 获取复数形式的 CSI 数据
        csi_complex = csi(i);
        
        % 使用 angle 函数提取相位信息
        phaseData(i) = angle(csi_complex);
    end
end