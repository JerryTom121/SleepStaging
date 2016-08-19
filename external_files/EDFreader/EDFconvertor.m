% Convert .edf to .csv format
% ---------------------------

%% Select file to parse
folder = '/home/djordje/Desktop/EEGData/EEGDataPool/CM/GABRA';
file   = '/GAB2R22B.edf';

%% Parse selected file
addpath(folder);
[hdr, record] = edfread(file);
display(hdr);
