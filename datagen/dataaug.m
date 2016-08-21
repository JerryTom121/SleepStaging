% User Defined Variable - watch out, these are coupled !!!
EXP = '1';
ROTATION = 1;
MIRROR   = 1;
% experiment info
%signal_length = 65;
signal_length = 128;
num_channels = 3;
% Path to CSV folder
CSV_path = '/home/djordje/Desktop/CSVdata/';
suffix = '.csv';
% ----------------------------------------------------------------------- %
% --------------------------- READ DATA --------------------------------- %
% ----------------------------------------------------------------------- %
data_path  = strcat(strcat(strcat(CSV_path,'train_exp'),EXP),'.csv');
data  = csvread(data_path,0,0);
labels = data(:,end);        % extract labels
artmat = data(labels==-1,:); % extract artefakts
size(data)
size(artmat)
% ----------------------------------------------------------------------- %
% ------------------------ DATA AUGMENTATION ---------------------------- %
% ----------------------------------------------------------------------- %
% augment artefakt matrix by mirroring
if MIRROR
    % create X mirrored artefakt matrix
    artmatXmirror = artmat;
    artmatXmirror(:,2:end-1) = -artmatXmirror(:,2:end-1);
    % create Y mirrored artefakt matrix
    artmatYmirror = artmat;
    for j=0:num_channels-1
        from = 2+j*signal_length;
        to   = from + signal_length - 1;
        artmatYmirror(:,from:to) = fliplr(artmatYmirror(:,from:to));
    end
    % create XY mirrored artefakt matrix
    artmatXYmirror = artmatYmirror;
    artmatXYmirror(:,2:end-1) = -artmatXYmirror(:,2:end-1);
    % augment artefakt matrix
    artmat = [artmat;artmatXmirror;artmatYmirror;artmatXYmirror];
    suffix = strcat('_mir',suffix);
end
size(artmat)
% shift displacements based on unfairness ration and signal length
ratio = round(length(labels)/size(artmat,1));
shift_size = max(1,round(signal_length/ratio));

% perform augmentation
artmat_nonaug = artmat;
for i = 1:ratio-1
   new = artmat_nonaug; 
   if ROTATION
       for j=0:num_channels-1
           from = 2+j*signal_length;
           to   = from + signal_length - 1;
           new(:,from:to) = circshift(new(:,from:to),[0 i*shift_size]);
       end
   end
   artmat = [artmat;new];
end
if ROTATION
    suffix = strcat('_rot',suffix);
else
    suffix = strcat('_aug',suffix);
end
% augment data with new artefakts
data_augmented = [data;artmat];

size(data_augmented)

% ----------------------------------------------------------------------- %
% ------------------------ DATA SHUFFLING ------------------------------- %
% ----------------------------------------------------------------------- %
data_augmented = data_augmented(randperm(size(data_augmented,1)),:);

% ----------------------------------------------------------------------- %
% -------------------------- WRITE DATA --------------------------------- %
% ----------------------------------------------------------------------- %
display(suffix)
new_data_path = strcat(strcat(strcat(CSV_path,'train_exp'),EXP),suffix);
size(data_augmented)
csvwrite(new_data_path,data_augmented);
%}