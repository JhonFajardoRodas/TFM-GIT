### Script ejemplo de modelo ARMA(p,q)

www<-"https://web.archive.org/web/20121214074151/http://www.massey.ac.nz/~pscowper/ts/wave.dat" 

wave.dat= read.table(www, header=T)

#plot(as.ts(wave.dat$waveht), ylab = 'Wave height’)
acf(wave.dat$waveht)
pacf(wave.dat$waveht)

best.aic <- Inf 
for (i in 0:4) for (j in 0:4) { 
	fit.aic <- AIC(arima((wave.dat$waveht), order = c(i, 0, j))) 
	if (fit.aic < best.aic) { 
		best.order <- c(i, 0, j) 
		best.arma <- arima((wave.dat$waveht), order = 
	best.order) 
	best.aic <- fit.aic 
	} 
} 
best.order

wave.arma <- arima(wave.dat$waveht, order = c(4,0,4)) 
acf(wave.arma$res[-(1:4)])

arima(wave.dat$waveht, order = c(4,0,4))
wave.arma <- arima(wave.dat$waveht, order = c(4,0,4)) 
layout(1:3)
acf(wave.arma$res[-(1:4)])
pacf(wave.arma$res[-(1:4)])
hist(wave.arma$res[-(1:4)]) 