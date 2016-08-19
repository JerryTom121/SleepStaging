# faster.R --- An unsupervised fully automated sleep stager for mice based on EEG/EMG recordings
# Copyright (C) 2013 Genshiro Sunagawa
# genshiro.sunagawa@nifty.com
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

library(pdfCluster)

sleep.stage.labels<-c("NREM","REM","Wake","Unknown")

# Get Power spectrum from EEG/EMG time series
#############################################

# load timesseries of EEG/EMG here
# they should be vectors of numbers
load("test.eeg.xdr")
load("test.emg.xdr")

epoch.size<-8 # seconds
epoch.count<-10800 
sampling.frequency<-100 # Hz
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

eeg.pow<-.GetPower(eeg)
emg.pow<-.GetPower(emg)
rm(eeg)
rm(emg)
gc()

# Extract characters from EEG/EMG powers
########################################

# band of eeg and emg
eeg.band<-c(0,51)
emg.band<-c(0,51)

#EEG characters
pow.eeg.delta<-colSums(eeg.pow[(freq>=0.5)&(freq<=4),])
pow.eeg.theta<-colSums(eeg.pow[(freq>=6)&(freq<=10),])

#EMG characters
pow.emg.total<-colSums(emg.pow)

# PCA against comprehensive EEG/EMG power

# detect abnoraml data
eeg.sum<-log10(colSums(eeg.pow))
eeg.sum<-eeg.sum-median(eeg.sum)
emg.sum<-log10(pow.emg.total)
emg.sum<-emg.sum-median(emg.sum)
good.epoch<-(eeg.sum<1)&(emg.sum<2)
rm(eeg.sum)
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

eeg.pow.log10.std<-.stand(log10(eeg.pow[eeg.range, good.epoch]))
emg.pow.log10.std<-.stand(log10(emg.pow[emg.range, good.epoch]))

eeg.freq<-1:(dim(eeg.pow)[1])
rownames(eeg.pow.log10.std)<-sprintf("EEG%03d",eeg.freq[eeg.range])

emg.freq<-1:(dim(emg.pow)[1])
rownames(emg.pow.log10.std)<-sprintf("EMG%03d",emg.freq[emg.range])

# PCA only for good.data
pca.results<-prcomp(t(rbind(eeg.pow.log10.std, emg.pow.log10.std)), scale.=TRUE)
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
		deltas<-rep(0, cl@noc)
		emgs<-rep(0, cl@noc)

		# calc delta and emg median for each stages
		for (i in seq(cl@noc)){
			ids<-target.epochs[good.epoch[target.epochs]][which(cl@clusters==i)]	
			deltas[i]<-median(log10(pow.eeg.delta[ids]))
			emgs[i]<-median(log10(pow.emg.total[ids]))
		}

		anot<-rep("", cl@noc)
		for (i in seq(cl@noc)) {
			if (emgs[i]>quantile((log10(pow.emg.total[target.epochs[good.epoch[target.epochs]]])), P.emg)) anot[i]<-"Wake"
		}

		non.ids<-which(anot=="")
		not.wake.epochs<-target.epochs[good.epoch[target.epochs]][which(!cl@clusters%in%which(anot=="Wake"))]

		for (i in non.ids) { 
			if (deltas[i]<quantile(log10(pow.eeg.delta[not.wake.epochs]), P.delta)) {anot[i]<-"REM"
				} else anot[i]<-"NREM" 
		}

		stages[target.epochs[good.epoch[target.epochs]]]<-anot[cl@clusters]
	}

	#stages[which(is.na(stages[target.epochs]))+target.epochs[1]-1]<-"Unknown"
}

stages[is.na(stages)]<-"Unknown"
stages<-factor(stages, levels=sleep.stage.labels)

