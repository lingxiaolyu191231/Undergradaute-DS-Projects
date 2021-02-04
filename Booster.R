#Boosting for Final Project

library(gbm)
library(gmodels)
library(ggplot2)
library(furrr)
library(dplyr)

#############################################################################################
##############################UPDATE THE SECTION BELOW#######################################
#############################################################################################

#START OF SETUP

#For model.type, enter:

#B for binary classification problems
#M for multiclass classification problems
#R for regression problems

model.type="R"

#Enter the number of trees to include during the boosting

num.tree=100


#END OF SETUP

load("C:/Users/llyu1/Desktop/myrstudio/Final Project/realtor.rda")
mydata$myresponse = mydata$sold.price
mydata$sold.price=NULL
mydata$city = NULL
str(mydata)


###########################DO NOT MODIFY BEYOND THIS POINT##################################
############################################################################################
#################HIGHLIGHT AND RUN THE CODE BELOW AND RUN####################################

#Since the response needs to stay on numeric 0-1 format for binary outcomes (converting to categorical
#causes issues during gbm processing), the code below automatically formats myresponse.
#In this program no manual treatment/formatting/transformation of myresponse is needed.

if (model.type=="B") {
  if (is.factor(mydata$myresponse) | is.character(mydata$myresponse)){
    levels=unique(sort(mydata$myresponse, decreasing = T))
    mydata$myresponse=as.numeric(mydata$myresponse==levels[1])
  } else {
    if (is.numeric(mydata$myresponse) | is.integer(mydata$myresponse)){
      if (!(min(mydata$myresponse)==0 & max(mydata$myresponse==1))){
        mydata$myresponse=as.numeric(mydata$myresponse==max(mydata$myresponse))}}}
} else {
  if (model.type=="M"){
    if (is.factor(mydata$myresponse)==FALSE){
      mydata$myresponse=as.factor(mydata$myresponse)}}}


#START DATA BREAKDOWN FOR HOLDOUT METHOD

#Start finding the categorical predictors

numpredictors=dim(mydata)[2]-1

numfac=0

for (i in 1:numpredictors) {
  if ((is.factor(mydata[,i]))){
    numfac=numfac+1} 
}

#End finding the number of categorical predictors 

nobs=dim(mydata)[1]


if (model.type=="R") {
  
  #Below is the setup for stratified 80-20 holdout sampling for a Regression Tree
  
  train_size=floor(0.8*nobs)
  test_size=nobs-train_size
  
} else {
  
  #Below is the setup for stratified 80-20 holdout sampling for a Classification Tree
  
  prop = prop.table(table(mydata$myresponse))
  length.vector = round(nobs*0.8*prop)
  train_size=sum(length.vector)
  test_size=nobs-train_size
  class.names = as.data.frame(prop)[,1]
  numb.class = length(class.names)}


resample=1
RNGkind(sample.kind = "Rejection")
set.seed(1) #sets the seed for random sampling

while (resample==1) {
  
  
  if (model.type=="B" | model.type=="M") {
    
    train_index = c()
    
    for(i in 1:numb.class){
      index_temp = which(mydata$myresponse==class.names[i])
      train_index_temp = sample(index_temp, length.vector[i], replace = F)
      train_index = c(train_index, train_index_temp)
    }} else {
      train_index=sample(nobs,train_size, replace=F)
    }
  
  mydata_train=mydata[train_index,] #randomly select the data for training set using the row numbers generated above
  mydata_test=mydata[-train_index,]#everything not in the training set should go into testing set
  
  right_fac=0 #denotes the number of factors with "right" distributions (i.e. - the unique levels match across mydata, test, and train data sets)
  
  
  for (i in 1:numpredictors) {
    if (is.factor(mydata_train[,i])) {
      if (sum(as.vector(unique(mydata_test[,i])) %in% as.vector(unique(mydata_train[,i])))==length(unique(mydata_test[,i])))
        right_fac=right_fac+1
    }
  }
  
  if (right_fac==numfac) (resample=0) else (resample=1)
  
}

dim(mydata_test) #confirms that testing data has only 20% of observations
dim(mydata_train) #confirms that training data has 80% of observations

#################################################################################
#################################################################################
#END DATA BREAKDOWN FOR HOLDOUT METHOD

#START BOOSTING

#Determining the distribution to be passed for boosting
if(model.type=="B"){
  dist.0="bernoulli"
} else {
  if(model.type=="M"){
    dist.0="multinomial"
  } else {
    dist.0="gaussian"
  }
}

#vector of candidate depths to try during boosting. we will try all and select one that yields
#minimum cross validation error

depths.to.try=c(1:10)

boost.now <- function (d){
  
  boosted.tree= gbm(myresponse~., 
                    data=mydata_train,
                    distribution=dist.0,
                    n.trees=num.tree, 
                    cv.folds = 5,
                    n.cores = NULL,#let gbm determine the # cores automatically
                    bag.fraction = 0.5,#note we are performing stochastic gradient descent
                    interaction.depth = d)
  
  new.min=min(boosted.tree$cv.error)
  
  how.many.trees=gbm.perf(boosted.tree, method="cv")
  
  new.vec=cbind(d, new.min, how.many.trees)
  new.vec
}


#seed generator to pass to future_map, since regular 'set.seed' does not work for parallel
#processing and a seed can instead be created an passed to future_map explicitly

set.seed(123)
new.seed=.Random.seed
seed.list=rep(list(new.seed),length(depths.to.try))


plan(multiprocess)
results=depths.to.try%>%future_map(boost.now, 
                                   .progress = T, 
                                   .options=future_options(seed=seed.list))

results=as.data.frame(do.call(rbind, results))

#which depth provides minimum cv.error?
int.dept.final=which.min(results$new.min)

print (paste("When up to ",
             num.tree,
             " trees are used during boosting, the optimal number is ",
             results$how.many.trees[int.dept.final],
             ", ",
             "yielding a minimum cross validation error of ",
             results$new.min[int.dept.final],
             ". ",
             "The maximum depth of component trees is ", 
             results$d[int.dept.final],
             sep=""))

#Set the exact same seed that was set for future_map and perform boosting with the optimal depth tree
#again with cross validation to see the cv error vs tree number graph

.Random.seed=new.seed
boosted.tree.final= gbm(myresponse~., 
                        data=mydata_train,
                        distribution=dist.0,
                        n.trees=num.tree, 
                        cv.folds = 5,
                        n.cores = NULL,
                        bag.fraction=0.5,
                        interaction.depth = int.dept.final)


num.tree.final=gbm.perf(boosted.tree.final, method="cv") #will give the plot of 10-fold cross validation error rate vs number of iterations 

#START PREDICTING THE RESPONSE IN THE TESTING SET (20 % SUBSET)

#One last boosting round for prediction purposes, without cross validation
boosted.tree.final.no.cv=gbm(myresponse~., 
                             data=mydata_train,
                             distribution=dist.0,
                             n.trees=num.tree.final, 
                             n.cores = NULL,
                             bag.fraction=1,#non-stochastic gradient descent compared to CV versions!
                             interaction.depth = int.dept.final)

predictions=predict(boosted.tree.final.no.cv, 
                    newdata = mydata_test, 
                    n.trees = num.tree.final, 
                    type="response")

if (model.type=="R" | model.type=="B"){
  
  mydata_test_w_predictions=cbind(mydata_test,predictions)
  
  if (model.type=="B"){
    
    mydata_test_w_predictions$predictions=(mydata_test_w_predictions$predictions<0.5)*0+(mydata_test_w_predictions$predictions>=0.5)*1}
  
} else {
  
  mydata_test_w_predictions=mydata_test
  col.names=colnames(predictions)
  p.pred <- apply(predictions, 1, which.max)
  
  for (i in 1:dim(mydata_test)[1]){
    c=p.pred[i]
    mydata_test_w_predictions$predictions[i]=col.names[c]}}

#Measuring predictive accuracy below

if (model.type=="R") {
  
  abs.diff=abs(mydata_test_w_predictions$predictions-mydata_test_w_predictions$myresponse)
  mape=100*mean(abs.diff/abs(mydata_test_w_predictions$myresponse))
  rmse=sqrt(mean(abs.diff^2))
  print(paste("MAPE for Testing Set Is:", 
              round(mape,2)))
  print(paste("RMSE for Testing Set Is:", 
              round(rmse,2)))} else {
                print("Confusion Matrix Is:")
                CrossTable(mydata_test_w_predictions$myresponse,mydata_test_w_predictions$predictions,prop.chisq=F,prop.t=F) }

#END PREDICTING THE RESPONSE IN THE TESTING SET (20 % SUBSET)

#END BOOSTING
#############################################################################################
##############################THIS IS THE END OF THE MACRO###################################
#############################################################################################

ten_data = mydata[c(1:10),]
# predict the one with the sold price of 725000
predictions=predict(boosted.tree.final.no.cv, 
                    newdata = ten_data, 
                    n.trees = num.tree.final, 
                    type="response")

ten_w_predictions=cbind(ten_data,predictions)
  

