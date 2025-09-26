%�ô������ڽ�ͬһ�����ͬһ����ͬһ�ŵ��ռ�������ƴ����һ��

% ָ�����ļ���·��
main_folder_path = 'D:\MATLAB R2016b\works\matlab_equalization_0\wifi_2024_10_29\original_data\csi_1_7';  % ���ļ���·��

% ��ȡ���ļ����е�����һ�����ļ���
first_level_folders = dir(main_folder_path);
first_level_folders = first_level_folders([first_level_folders.isdir] & ~ismember({first_level_folders.name}, {'.', '..'}));

% ����ÿ��һ�����ļ���
for i = 1:length(first_level_folders)
    first_level_folder_path = fullfile(main_folder_path, first_level_folders(i).name);
    
    % ��ȡһ�����ļ����е����ж������ļ���
    second_level_folders = dir(fullfile(first_level_folder_path, 'packets_*-*'));
    second_level_folders = second_level_folders([second_level_folders.isdir]);

    % ����ÿ�����������Ķ������ļ���
    for j = 1:length(second_level_folders)
        second_level_folder_name = second_level_folders(j).name;
        second_level_folder_path = fullfile(first_level_folder_path, second_level_folder_name);
        
        % ��ȡ�������ļ����е����� .mat �ļ�
        files = dir(fullfile(second_level_folder_path, '*.mat'));

        % ����һ������������ƴ������
        concatenated_data = [];

        % �����������ļ����е�ÿ���ļ�
        for k = 1:length(files)
            file_name = files(k).name;
            
            % �����ļ��е�����
            data = load(fullfile(second_level_folder_path, file_name));

            % ����Ƿ�������� `csi_log`
            if isfield(data, 'csi_log')
                % �� `csi_log` ����׷�ӵ� `concatenated_data`
                concatenated_data = [concatenated_data, data.csi_log];
            else
                fprintf('�ļ� %s ���������� "csi_log"���������ļ���\n', file_name);
            end
        end

        % ����ɹ�ƴ�������ݣ��򱣴����ö������ļ���
        if ~isempty(concatenated_data)
            % ���ñ����ļ�·��������Ϊ�������ļ�������
            save_file_name = fullfile(first_level_folder_path, strcat(second_level_folder_name, '.mat'));
            
            % ��ƴ�����ݱ���Ϊ `csi_log` ����
            csi_log = concatenated_data;
            save(save_file_name, 'csi_log');

            % �������ɹ�����ʾ
            fprintf('�ļ� "%s" �ѳɹ�������һ�����ļ��� "%s"\n', strcat(second_level_folder_name, '.mat'), second_level_folder_name);
        else
            fprintf('�������ļ��� "%s" ��û����Ч���ݣ�δ���档\n', second_level_folder_name);
        end
    end
end

