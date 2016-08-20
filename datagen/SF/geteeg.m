%
% fullpath = 'path_to_csv_files'
% type = 'test' | 'train'
% exp  = '1' | '2' | ....
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function eeg = geteeg(csv_path,type,exp)

if strcmp(type,'test')
    prefix = 'test_exp';
elseif strcmp(type,'train')
    prefix = 'train_exp';
else
    assert('wrong argument (type) in geteeg');
end

data_path  = strcat(strcat(strcat(csv_path,prefix),exp),'.csv');
data  = csvread(data_path,0,1); % skip indices
data = data(:,1:end-1);
eeg = reshape(transpose(data),1,[]);

end

