% User Defined Variable - watch out, these are coupled !!!
EXP = '11';
SIGNAL_FQ = 128;
% Path to CSV folder
CSV_path = '/home/djordje/Desktop/CSVData/';

% ----------------------------------------------------------------------- %
% --------------------------- DATA PREPARATION -------------------------- %
% ----------------------------------------------------------------------- %
% Get raw eeg signals for train and test data
train_eeg = geteeg(CSV_path,'train',EXP);
test_eeg  = geteeg(CSV_path,'test',EXP);

% ----------------------------------------------------------------------- %
% --------------------------- SPECTROGRAM ------------------------------- %
% ----------------------------------------------------------------------- %
% parameters to use 
% for spectrogram function
seg = ones(512,1); % 4 seconds windows-according to Seijin|why not hamming?
overlap = 0;       % no overlap-according to Seijin
nfft = 128;%512;        % not sure what this is ?! perhaps implicit signal downsampling??

% Apply spectrogram
spectrogram(train_eeg(1:round(length(train_eeg)/(4*12*30))),seg,overlap,nfft,SIGNAL_FQ,'yaxis')