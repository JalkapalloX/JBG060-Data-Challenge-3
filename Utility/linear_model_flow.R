require(zoo)
require(tidyverse)
require(dummies)

data <- read.csv("D:/DC3/heya_lm.csv")
data$Hour <- as.factor(data$Hour)
data$Total <- rowSums(data[,8:18])

hour_dummies <- dummies::dummy(data$Hour)

data <- cbind(data[,c("Flow", "Weekday", "Total")], hour_dummies)

data$Flow2 <- lag(data$Flow, 1)
data$Flow3 <- lag(data$Flow, 2)
data$Flow4 <- lag(data$Flow, 3)
data$Flow5 <- lag(data$Flow, 4)

data$Total2 <- lag(data$Total, 1)
data$Total3 <- lag(data$Total, 2)
data$Total4 <- lag(data$Total, 3)
data$Total5 <- lag(data$Total, 4)
data$Total6 <- lag(data$Total, 5)
data$Total7 <- lag(data$Total, 6)
data$Total8 <- lag(data$Total, 7)

m <- lm(Flow ~ Hour + Weekday + Kastanjelaan + Sportlaan.Hoge.Schijf + Groenewoud + Beethovenlaan.Venne.West + Stationsstraat.zuid +
        Dillenburg + Sapa + Stationsstraat.noord + Bosscheweg + Woonwagencentrum + Gemeentewerf, data = data)

summary(m)

m2 <- lm(Flow ~ ., data = data)
summary(m2)

m3 <- lm(Flow ~ . - Flow2 - Flow3 - Flow4 - Flow5, data = data)
summary(m3)


haha2 <- m2$coefficients[2:36]
haha2[is.na(haha2)] = 0
haha <- apply(data[6:9673,names(m2$coefficients[2:36])], 2, var)

lm1 <- lm(Sepal.Length ~ Sepal.Width, data = iris)

x <- as.matrix(data[,c(3, 32:36)])
y <- m3$coefficients[c("Total", "Total2", "Total3", "Total4", "Total5", "Total6")]
lol <- x%*%y
