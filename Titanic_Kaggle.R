#Titanic Prediction Project
#first kaggle project

#loading libraries
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")
if(!require(rpart)) install.packages("rpart")
if(!require(mice)) install.packages("mice")
if(!require(Rcpp)) install.packages("Rcpp")
if(!require(ggthemes)) install.packages("ggthemes")

library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(rpart)
library(mice)
library(Rcpp)
library(ggthemes)

#Loading the data
train <- read.csv("train.csv",stringsAsFactors = F)
test <- read.csv("test.csv",stringsAsFactors = F)

#combinding both data sets for data exploration but will only use this data set
#for data exploration and not the analysis section
all <- bind_rows(train,test)

#Data cleaning

#Taking a look at the data set 
head(all)
str(all)
summary(all)

#having looked at the features classes we are going to factor a few of them
all <- all %>% mutate(Survived = factor(Survived),
               Pclass = factor(Pclass),
               Sex = factor(Sex),
               Embarked = factor(Embarked))

#There is not much we can do with the names but we can extract the titles and
#group them
all$Title <- gsub('(.*, )|(\\..*)', '', all$Name)

#showing a tibble of the titles and the count for each of them
all %>% group_by(Title)%>%
  summarize(count = n())

#changing a few titles to miss
all$Title[all$Title == 'Ms'] <- 'Miss'
all$Title[all$Title == 'Mlle'] <- 'Miss'

#changing Mme to Mrs
all$Title[all$Title == 'Mme'] <- 'Mrs'

#most of these titles only show up once or a few times in order to avoid
#over fitting we will group them into a group called Other
other <- c('Capt','Col','Don','Dona','Jonkheer','Lady','Major',
           'Rev','Sir','the Countess')

all$Title[all$Title %in% other]  <- 'Other'

#factoring the Titles
all$Title <- factor(all$Title)

#Next we are going to create a feature call the family size
all$Family_size <- all$SibSp + all$Parch + 1

#factoring Family size
all$Family_size <- factor(all$Family_size)

#from looking at the data we can see that two observations are blank for 
#embarked lets look at the average fare cost per embarked and make a guess
which(all$Embarked == "")
view(all[c(62,830),])

#we know that both of these observations have the same pclass and fare, lets
#look at a boxplot to visualize this
all %>% ggplot(aes(Pclass,Fare)) +
  geom_boxplot(aes(color = Embarked))

#Both of these observations look like they should be Embarked from C
#changing the missing observations to C
all$Embarked[c(62,830)] <- "C"

#factoring the Embarked feature, we should only have 3 levels now
all$Embarked <- factor(all$Embarked)

#finding the remaining blank observations
colSums(all == "")

#cabin has too many that are black so we will leave that column alone

#looking at the number of NA for each column
colSums(is.na(all))

#Fare has one NA in the test set. We will change that to the average fare
which(is.na(test$Fare))

#taking a look at the row that has the NA
test[153,]

#changing the NA to the avg Fare
all <- all %>%
  mutate(Fare = ifelse(is.na(Fare),median(Fare, na.rm = TRUE),Fare))

#checking that the NA was changed
sum(is.na(all$Fare))

#263 NA in total for Age. Using the mice function to fill in those NAs with 
#predictions
temp <- all %>% select(Pclass,Sex,Age)

set.seed(1)
mice_input <- mice(temp, method = 'rf')
mice_output <- complete(mice_input)

#using histograms to make sure the new predictions match the distribution of all
hist(all$Age)
hist(mice_output$Age)

#replacing age variable with new age predictions
all$Age <- mice_output$Age

#checking to see if there are any NA in train$Age
sum(is.na(all$Age))

#we now have 0 NAs in the data set expect for the 418 Survived values we are
#trying to predict
colSums(is.na(all))

#cabin has too many that are black so we will leave that column alone
  
#Data Exploration

#What percentage of the data set are male
mean(all$Sex == "male")

#What percentage of the data set are female
mean(all$Sex == "female")

#What percentage of the data set survived
mean(train$Survived == 1)

#What percentage of the data set died
mean(train$Survived == 0)

#graph showing the amount survived and not survived split by Sex
train %>% ggplot(aes(factor(Survived))) +
  facet_grid(.~Sex) +
    geom_bar(aes(fill=factor(Survived))) +
  ggtitle("Amount that Survived and Did Not Survived by Sex") +
  scale_fill_discrete(name = "Survivial Status",
                      labels = c("Did Not Survive", "Survived")) +
  theme_economist()
  
  #histogram of age distribution in the data
all %>% ggplot(aes(Age)) +
  geom_histogram(fill = "blue") +
  ggtitle("Age distribution") +
  theme_economist()

#Boxplot with Survival status and ticket price
train %>% ggplot(aes(factor(Survived),Fare)) +
  geom_boxplot(color = "blue") +
  ggtitle("Survival and ticket price (Survived = 1)") +
theme_economist()

#Scatter plot with Age and Fare with Survival Status
all %>% ggplot(aes(Age,Fare)) +
  geom_point(color = "blue") +
  ggtitle("Scatter Plot with Age and Fare") +
  xlab("Age") +
  ylab("Fare") +
  theme_economist()

#Bar graph with Pclass and Survival Status
train %>% ggplot(aes(factor(Survived))) +
  facet_grid(.~Pclass) +
  geom_bar(aes(fill=factor(Survived))) +
  ggtitle("Amount Survived and Not Survived, Split by Pclass") +
  scale_fill_discrete(name = "survival status", 
                      labels = c("Did Not Survive","Survived")) +
  theme_economist()

#Analysis

#splitting the all dataset back into train and test
train <- all[1:891,]
test <- all[892:1309,]

#Will be using k-fold cross validation on all the algorithms
#creating the k-fold parameters, k is 10
set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9)
tuning <- data.frame(size = seq(100), decay = seq(.01,1,.1))

train_x <- train %>% select(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked,
                            Title,Family_size)
train_y <- train$Survived

#Predicting survival by using a neural network
set.seed(1, sample.kind = "Rounding")
train_nn <- train(train_x, train_y,
                  method = "nnet",
                  tuneGrid = tuning,
                  trControl = control)

#plotting results
plot(train_nn)

#creating the predictions
nn_preds <- predict(train_nn, test)

solution <- data.frame(PassengerID = test$PassengerId,
                       Survived = nn_preds)

write.csv(solution, file = 'nn.titanic.preds2.cvs', row.names = FALSE)
