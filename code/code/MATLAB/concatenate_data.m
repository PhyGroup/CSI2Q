%该代码用于将同一发射机同一天在同一信道收集的数据拼接在一起

% 指定主文件夹路径
main_folder_path = 'D:\MATLAB R2016b\works\matlab_equalization_0\wifi_2024_10_29\original_data\csi_1_7';  % 主文件夹路径

% 获取主文件夹中的所有一级子文件夹
first_level_folders = dir(main_folder_path);
first_level_folders = first_level_folders([first_level_folders.isdir] & ~ismember({first_level_folders.name}, {'.', '..'}));

% 遍历每个一级子文件夹
for i = 1:length(first_level_folders)
    first_level_folder_path = fullfile(main_folder_path, first_level_folders(i).name);
    
    % 获取一级子文件夹中的所有二级子文件夹
    second_level_folders = dir(fullfile(first_level_folder_path, 'packets_*-*'));
    second_level_folders = second_level_folders([second_level_folders.isdir]);

    % 遍历每个符合条件的二级子文件夹
    for j = 1:length(second_level_folders)
        second_level_folder_name = second_level_folders(j).name;
        second_level_folder_path = fullfile(first_level_folder_path, second_level_folder_name);
        
        % 获取二级子文件夹中的所有 .mat 文件
        files = dir(fullfile(second_level_folder_path, '*.mat'));

        % 创建一个空数组用于拼接数据
        concatenated_data = [];

        % 遍历二级子文件夹中的每个文件
        for k = 1:length(files)
            file_name = files(k).name;
            
            % 加载文件中的数据
            data = load(fullfile(second_level_folder_path, file_name));

            % 检查是否包含变量 `csi_log`
            if isfield(data, 'csi_log')
                % 将 `csi_log` 数据追加到 `concatenated_data`
                concatenated_data = [concatenated_data, data.csi_log];
            else
                fprintf('文件 %s 不包含变量 "csi_log"，跳过该文件。\n', file_name);
            end
        end

        % 如果成功拼接了数据，则保存至该二级子文件夹
        if ~isempty(concatenated_data)
            % 设置保存文件路径，命名为二级子文件夹名称
            save_file_name = fullfile(first_level_folder_path, strcat(second_level_folder_name, '.mat'));
            
            % 将拼接数据保存为 `csi_log` 变量
            csi_log = concatenated_data;
            save(save_file_name, 'csi_log');

            % 输出保存成功的提示
            fprintf('文件 "%s" 已成功保存至一级子文件夹 "%s"\n', strcat(second_level_folder_name, '.mat'), second_level_folder_name);
        else
            fprintf('二级子文件夹 "%s" 中没有有效数据，未保存。\n', second_level_folder_name);
        end
    end
end

