clc; 
clear; 
close all;
warning('off','MATLAB:MKDIR:DirectoryExists')
folder='own_data/csi_1_1/channel11/';
fl='packets_73-69.mat';
[csi_cell, name] = opencsi("D:\MATLAB R2022b\toolbox\PicoScenes-MATLAB-Toolbox-Core\samples\csi_dataset_0627\csi_1_1\channel11\rx_1_240320_184131.csi");
[x,y] = size(csi_cell);
MAC_Address = [228,14,238,212,206,181];
if x>1
    [m,n] = size(csi_cell);
    collected_cell = cell(30000,1);
    j = 0;
    for i = 1:m
         a = cell2struct(csi_cell(i),'CSI',1);
         b = a.CSI.StandardHeader.Addr2;
         if isequal(b, MAC_Address)
             j = j+1;
             collected_cell(j) = csi_cell(i);
         end
    end
    csi_log = cell(1,j);
    for i = 1:j
         a = cell2struct(collected_cell(i),'CSI',1);
         b = a.CSI.StandardHeader.Addr2;
         c = a.CSI.CSI.CSI;
         if isequal(b, MAC_Address)
             csi_log{i} = c(:,1,1);
             data = csi_log{i};
             dimension = size(data);
             if dimension(1,1)==57
                 csi_log{i} = [data(3:28); data(30:55)];
             else 
                 csi_log{i} = [data(1:26); data(28:53)];
             end
           
        end
    end
else
    csi_log = cell(1,300);
    j = 0;
    a = cell2struct(csi_cell,'CSI',1);
    b = a.CSI.StandardHeader.Addr2;
    % disp(b(1,:));
    c = a.CSI.CSI.CSI;
    d = a.CSI.MPDU;
    [m,n] = size(d);
    for i = 1:m
         if isequal(b(i,:), MAC_Address)
             j = j+1;
             c_0 = c(i,:)';
%            csi_log{i} = c_0(:,1,1);
%            data = csi_log{i};

             data = c_0;
             dimension = size(data);
             if dimension(1,1)==57
                 csi_log{j} = [data(3:28); data(30:55)];
             else 
                 csi_log{j} = [data(1:26); data(28:53)];
             end
         end
    end
end
if j>400
%    mkdir([folder,'packets_csi/'])
     save([folder,fl],'csi_log')
     disp('Sucessfully!');
end