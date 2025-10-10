www="https://web.archive.org/web/20121213213733/http://www.massey.ac.nz/~pscowper/ts/cbe.dat"

CBE <- read.table(www, header = T, fill = TRUE)
Elec.ts <-ts(CBE[,3], start = 1958, freq = 12)
Time <- 1:length(Elec.ts)
Imth <- cycle(Elec.ts)
Elec.lm <- lm(log(Elec.ts) ~ Time + I(Time^2) + factor(Imth))
plot(resid(Elec.lm))
acf(resid(Elec.lm))


best.order <- c(0, 0, 0) 
best.aic <- Inf 
for (i in 0:2) for (j in 0:2) { 
	fit.aic <- AIC(arima(resid(Elec.lm), order = c(i, 0, j))) 
	if (fit.aic < best.aic) { 
		best.order <- c(i, 0, j) 
		best.arma <- arima(resid(Elec.lm), order = best.order) 
	best.aic <- fit.aic 
	} 
} 
best.order

acf(resid(best.arma))


new.time <- seq(length(Elec.ts), length = 36)
new.data <- data.frame(Time = new.time, Imth = rep(1:12, 3))



predict.lm <- predict(Elec.lm, new.data)
predict.arma <- predict(best.arma, n.ahead = 36)
elec.pred <- ts(exp(predict.lm + predict.arma$pred), start = 1991, freq = 12)

ts.plot(cbind(Elec.ts, elec.pred), lty = 1:2)