#install.packages("pROC")
#install.packages("randomForest")

library(pROC)
library(randomForest)

data <- read.csv('~/habr/unit_weights/titanic3.csv')

# Зависимая переменная
data$survived <- as.factor(data$survived)

# Слишком много разных значений для нормального анализа
data$name <- NULL
data$ticket <- NULL
data$cabin <- NULL
data$home.dest <- NULL
data$embarked <- NULL

# Не используем переменные из будущего
data$boat <- NULL
data$body <- NULL

# Заменяем пропущенные значения средним значением
data$age[is.na(data$age)] <- mean(data$age, na.rm=TRUE)
data$fare[is.na(data$fare)] <- mean(data$fare, na.rm=TRUE)

# Преобразуем пол в индикаторную переменную
data$female <- 0
data$female[which(data$sex == 'female')] <- 1
data$sex <- NULL

summary(data)

im.gini = NULL
pm.gini = NULL
lr.gini = NULL
rf.gini = NULL
set.seed(42)
for (i in 1:1000) {
  data$random_number <- runif(nrow(data),0,1)
  development <- data[ which(data$random_number > 0.7), ]
  holdout     <- data[ which(data$random_number <= 0.7), ]
  development$random_number <- NULL
  holdout$random_number <- NULL
  
  # Improper model
  beta_pclass <- -1/sd(development$pclass)
  beta_age    <- -1/sd(development$age   )
  beta_female <-  1/sd(development$female)
  im.score <- beta_pclass*holdout$pclass + beta_age*holdout$age + beta_female*holdout$female 
  im.roc <- roc(holdout$survived, im.score)
  im.gini[i] <- 2*im.roc$auc-1

  # Proper model, same variables
  pm.model = glm(survived~pclass+age+female, family=binomial(logit), data=development)
  pm.score <- predict(pm.model, holdout, type="response")
  pm.roc <- roc(holdout$survived, pm.score)
  pm.gini[i] <- 2*pm.roc$auc-1

  # Full logistic regression
  lr.model = glm(survived~., family=binomial(logit), data=development)
  lr.score <- predict(lr.model, holdout, type="response")
  lr.roc <- roc(holdout$survived, lr.score)
  lr.gini[i] <- 2*lr.roc$auc-1

  # Random forest
  rf.model <- randomForest(survived~., development)
  rf.score <- predict(rf.model, holdout, type = "prob")
  rf.roc <- roc(holdout$survived, rf.score[,1])
  rf.gini[i] <- 2*rf.roc$auc-1
}

bpd<-data.frame(ImproperModel=im.gini, ProperModel=pm.gini, LogisticRegression=lr.gini, RandomForest=rf.gini)
png('~/habr/unit_weights/auc_comparison.png', height=700, width=400, res=120, units='px')
boxplot(bpd, las=2, ylab="Gini", ylim=c(0,1), par(mar=c(12,5,2,2)+ 0.1), col=c("red","green","royalblue2","brown"))
dev.off()
mean(im.gini)
mean(pm.gini)
mean(lr.gini)
mean(rf.gini)
