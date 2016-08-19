#library(devtools)
#install_github("bwrc/edf")

rm(list=ls())
library(edf)
library(plyr)
library(caret)
library(e1071)
chris<-read.table("Documents/thesises/Deep_timesseries/EEGDataPool/AS/AS87I.std", header=FALSE, sep="")
datafile <- "Documents/thesises/Deep_timesseries/EEGDataPool/AS/AS87I.edf"
recording <- read.edf(datafile)

help1<-chris$V2

rm(chris)
gc()

#relabeling 1,....,6 as outliers n, r, w as ("NREM","REM","Wake")
help1<-mapvalues(help1, from = c(1, 2, 3, 4, 5, 6), to = c("Unknown", "Unknown","Unknown","Unknown","Unknown","Unknown"))
help1<-mapvalues(help1, from = c("n", "r", "w"), to = c("NREM", "REM","Wake"))
expert<-as.data.frame(help1)
rm(help1)
gc()

help1<-as.data.frame(recording[[3]][1])
eeg1<-help1$EEG1.data

help2<-as.data.frame(recording[[3]][2])
eeg2<-help2$EEG2.data

help3<-as.data.frame(recording[[3]][3])
emg<-help3$EMG.data



rm(help1,help2)
gc()

# faster.R --- An unsupervised fully automated sleep stager for mice based on EEG/EMG recordings

library(pdfCluster)

sleep.stage.labels<-c("NREM","REM","Wake","Unknown")

# Get Power spectrum from EEG/EMG time series
#############################################

# load timesseries of EEG/EMG here
# they should be vectors of numbers

epoch.size<-4 # seconds
epoch.count<-10800 
sampling.frequency<-128 # Hz
freq<-(seq(sampling.frequency*epoch.size%/%2+1)-1)*(1/epoch.size)

.GetPower<-function(vols){
  
  dim(vols)<-c(epoch.size*sampling.frequency, epoch.count)
  
  # detrend
  .Detrend<-function(v){
    time<-seq(sampling.frequency*epoch.size)-1
    vol.lm<-lm(vol~time, data.frame(time, vol=v))
    return(v-(vol.lm$coefficients[2]*time+vol.lm$coefficients[1]))
  }
  vols<-apply(vols, 2, .Detrend)
  
  # Hann window
  n<-sampling.frequency*epoch.size-1
  hw<-0.5-0.5*cos(2*pi*(0:n)/n)
  .Hanning<-function(v) hw*v
  vols<-apply(vols, 2, .Hanning)
  
  # calc power
  .Pow<-function(v) (abs(fft(v)[seq(sampling.frequency*epoch.size%/%2+1)]))^2/(sampling.frequency*epoch.size)*2
  vols<-apply(vols, 2, .Pow)
  
  return(vols)
}

eeg1.pow<-.GetPower(eeg1)
eeg2.pow<-.GetPower(eeg2)
emg.pow<-.GetPower(emg)
gc()

# Extract characters from EEG/EMG powers
########################################

# band of eeg and emg
eeg.band<-c(0,51)
emg.band<-c(0,51)

#EEG characters
pow.eeg1.delta<-colSums(eeg1.pow[(freq>=0.5)&(freq<=4),])
pow.eeg1.theta<-colSums(eeg1.pow[(freq>=6)&(freq<=10),])
pow.eeg2.delta<-colSums(eeg2.pow[(freq>=0.5)&(freq<=4),])
pow.eeg2.theta<-colSums(eeg2.pow[(freq>=6)&(freq<=10),])

#EMG characters
pow.emg.total<-colSums(emg.pow)

# PCA against comprehensive EEG/EMG power

# detect abnoraml data
eeg1.sum<-log10(colSums(eeg1.pow))
eeg1.sum<-eeg1.sum-median(eeg1.sum)
eeg2.sum<-log10(colSums(eeg2.pow))
eeg2.sum<-eeg2.sum-median(eeg2.sum)
emg.sum<-log10(pow.emg.total)
emg.sum<-emg.sum-median(emg.sum)
good.epoch<-(eeg1.sum<1)&(eeg2.sum<1)&(emg.sum<2)
rm(eeg1.sum)
rm(eeg2.sum)
rm(emg.sum)
gc()

# standardize data

# standarize log10(eeg) and log10(emg) to mean=0, sd=1
# only for good.data
.stand<-function(x){
  x[which(!is.finite(x))]<-0
  x.std<-x-mean(as.vector(x))
  x.std<-x.std/sd(as.vector(x.std))
}

# this is for manual staging
eeg.range<-which((freq>eeg.band[1])&(freq<eeg.band[2]))
emg.range<-which((freq>emg.band[1])&(freq<emg.band[2]))

eeg1.pow.log10.std<-.stand(log10(eeg1.pow[eeg.range, good.epoch]))
eeg2.pow.log10.std<-.stand(log10(eeg2.pow[eeg.range, good.epoch]))
emg.pow.log10.std<-.stand(log10(emg.pow[emg.range, good.epoch]))

eeg1.freq<-1:(dim(eeg1.pow)[1])
eeg2.freq<-1:(dim(eeg2.pow)[1])
rownames(eeg1.pow.log10.std)<-sprintf("EEG%03d",eeg1.freq[eeg.range])
rownames(eeg2.pow.log10.std)<-sprintf("EEG%03d",eeg1.freq[eeg.range])

emg.freq<-1:(dim(emg.pow)[1])
rownames(emg.pow.log10.std)<-sprintf("EMG%03d",emg.freq[emg.range])

# PCA only for good.data
pca.results<-prcomp(t(rbind(eeg1.pow.log10.std, eeg2.pow.log10.std, emg.pow.log10.std)), scale.=TRUE)
pcs<-matrix(NA, ncol=10, nrow=length(good.epoch))
pcs[good.epoch,]<-pca.results$x[,1:10]
sdev<-pca.results$sdev
rotation<-pca.results$rotation

# clustering via pdfCluster

group.length=5400

hmult<-0.70
pcn<-4
n.grid<-group.length%/%10

P.emg<-0.5
P.delta<-0.1

group.epochs<-t(matrix(seq(epoch.count), ncol=group.length))
stages<-rep(NA, epoch.count)

d<-data.frame()

for (j in seq(dim(group.epochs)[2])){
  
  target.epochs<-group.epochs[,j]
  x<-pcs[target.epochs ,seq(pcn)]
  
  # PDF clustering
  cl<-new("NULL")
  factored.stages<-NULL
  cl.noc<-NA
  hmult.k<-(-1)
  while(is.null(cl)){
    hmult.k<-hmult.k+1
    print(hmult+0.05*hmult.k)
    cl<-pdfCluster(x[good.epoch[target.epochs],], n.grid=n.grid, hmult=hmult+0.05*hmult.k)
    # if the clustering is not successful, the hmult is increased by 0.05 until at least one cluster is detected.
  }
  
  cl.noc<-cl@noc
  
  if (cl.noc==1) {
    # if there is only one cluster... lable them all as "Unknown"
    stages[target.epochs]<-"Unknown"
  } else { # cl.noc>1
    
    # annotate
    deltas1<-rep(0, cl@noc)
    deltas2<-rep(0, cl@noc)
    emgs<-rep(0, cl@noc)
    
    # calc delta and emg median for each stages
    for (i in seq(cl@noc)){
      ids<-target.epochs[good.epoch[target.epochs]][which(cl@clusters==i)]	
      deltas1[i]<-median(log10(pow.eeg1.delta[ids]))
      deltas2[i]<-median(log10(pow.eeg2.delta[ids]))
      emgs[i]<-median(log10(pow.emg.total[ids]))
    }
    
    anot<-rep("", cl@noc)
    for (i in seq(cl@noc)) {
      if (emgs[i]>quantile((log10(pow.emg.total[target.epochs[good.epoch[target.epochs]]])), P.emg)) anot[i]<-"Wake"
    }
    
    non.ids<-which(anot=="")
    not.wake.epochs<-target.epochs[good.epoch[target.epochs]][which(!cl@clusters%in%which(anot=="Wake"))]
    
    for (i in non.ids) { 
      if (deltas1[i]<quantile(log10(pow.eeg1.delta[not.wake.epochs]), P.delta) & deltas2[i]<quantile(log10(pow.eeg2.delta[not.wake.epochs]), P.delta)  ) {anot[i]<-"REM"
      } else anot[i]<-"NREM" 
    }
    
    stages[target.epochs[good.epoch[target.epochs]]]<-anot[cl@clusters]
  }
  
  #stages[which(is.na(stages[target.epochs]))+target.epochs[1]-1]<-"Unknown"
}

stages[is.na(stages)]<-"Unknown"
stages<-factor(stages, levels=sleep.stage.labels)

###
#Calculation of Confusion matrix
#####
faster<-as.data.frame(stages)

confusionMatrix(faster$stages, expert$help1,dnn = c("FASTER", "Expert"))


