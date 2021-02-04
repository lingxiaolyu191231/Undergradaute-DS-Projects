load("C:/Users/llyu1/Desktop/myrstudio/Final Project/realtor.rda")
mydata$myresponse = mydata$sold.price
mydata$sold.price = NULL

#mydata$zip.code = NULL
mydata$city=NULL
# mydata$type=NULL
# View(mydata)
mydata$type = as.character(mydata$type)
mydata$type[mydata$type=="Condo/Townhome/Row Home/Co-Op"] <-"Other"
mydata$type=as.factor(mydata$type)

# ANN Model

library(neuralnet)
library(nnet)
library(dplyr)
library(car)
library(caret)
library(gmodels)
library(GGally)
library(furrr)

str(mydata)

problem.type="R" #Enter "C" for classification and "R" for regression

if (problem.type=="C")
  table(mydata$myresponse)else
    hist(mydata$myresponse, main = "Histogram of Sold Price", xlab = "Sold Prices on Realtor", col = "Red3")

cat.vars=c("zip.code","type")

num.vars=names(mydata)[-(which(names(mydata)%in% cat.vars))]

ggpairs(mydata[,num.vars])

hid.list=list(c(5,5,5,5))

########################################ATTENTION################################################
####################Don't modify beyond here until where it says#################################
#####################################"END OF AUTORUN"############################################
#################################################################################################


#START OF AUTORUN

#########################################################################################33
#########################################################################################33
#########################################################################################33

#START DATA BREAKDOWN FOR HOLDOUT METHOD

#First, pre-process factors by removing blanks in the levels (e.g. "data science" would be converted to
#"datascience". This is necessary for the "as.formula" function to work later in the program)

for (i in 1:length(cat.vars)){
  mydata[,cat.vars[i]]=as.factor(gsub(" ", "", mydata[,cat.vars[i]]))}


#Find the number of categorical predictors first

numpredictors=dim(mydata)[2]-1

numfac=0

for (i in 1:numpredictors) {
  if ((is.factor(mydata[,i]))){
    numfac=numfac+1} 
}


#End finding the number of categorical predictors 

nobs=dim(mydata)[1]


if (problem.type=="R") {
  
  #Below is the setup for stratified 80-20 holdout sampling 
  
  train_size=floor(0.8*nobs)
  test_size=nobs-train_size
  
} else {
  
  #Below is the setup for stratified 80-20 holdout sampling 
  
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
  
  
  if (problem.type=="C") {
    
    train_index = c()
    
    for(i in 1:numb.class){
      index_temp = which(mydata$myresponse==class.names[i])
      train_index_temp = sample(index_temp, length.vector[i], replace = F)
      train_index = c(train_index, train_index_temp)
    }} else {
      train_index=sample(nobs,train_size, replace=F)
    }
  
  mydata.train=mydata[train_index,] #randomly select the data for training set using the row numbers generated above
  mydata.test=mydata[-train_index,]#everything not in the training set should go into testing set
  
  right_fac=0 #denotes the number of factors with "right" distributions (i.e. - the unique levels match across mydata, test, and train data sets)
  
  
  for (i in 1:numpredictors) {
    if (is.factor(mydata.train[,i])) {
      if (sum(as.vector(unique(mydata.test[,i])) %in% as.vector(unique(mydata.train[,i])))==length(unique(mydata.test[,i])))
        right_fac=right_fac+1
    }
  }
  
  if (right_fac==numfac) (resample=0) else (resample=1)
  
}

dim(mydata.test) #confirms that testing data has only 20% of observations
dim(mydata.train) #confirms that training data has 80% of observations



#END DATA BREAKDOWN FOR HOLDOUT METHOD

#Dummyfying

vars.to.dummify=setdiff(cat.vars, "myresponse")   

dummify<-function (dat){
  
  dummy.data=c()
  
  for (i in 1:length(vars.to.dummify)){
    newdat=class.ind(dat[,i])
    
    for (j in 1:dim(newdat)[2]){
      colnames(newdat)[j]=paste(colnames(dat)[i],"_",colnames(newdat)[j], sep="")}
    
    dummy.data=cbind(dummy.data,newdat)}
  
  dummy.data}


if (length(vars.to.dummify)==0){
  dummy.train=as.data.frame(mydata.train[,cat.vars])
  dummy.test=as.data.frame(mydata.test[,cat.vars])
  
  if (length(cat.vars)>0){
    names(dummy.train)="myresponse"
    names(dummy.test)="myresponse"}
  
} else{
  
  
  dummy.train=dummify(as.data.frame(mydata.train[,vars.to.dummify], optional = T)) 
  dummy.test=dummify(as.data.frame(mydata.test[,vars.to.dummify], optional = T)) 
  
  
  colnames(dummy.train)<-paste("Bin_", colnames(dummy.train), sep="")
  colnames(dummy.test)<-paste("Bin_", colnames(dummy.test), sep="")
  
  
  if ("myresponse" %in% cat.vars){
    dummy.train=data.frame(dummy.train,myresponse=mydata.train$myresponse)
    dummy.test=data.frame(dummy.test,myresponse=mydata.test$myresponse)}
  
}


#Scaling numeric variables to 0-1

if (length(num.vars)>0){
  
  #Saving max and min for later rescaling. Note, we are using the min and max over the whole data
  max.vec=apply(mydata[,num.vars],2,max)
  min.vec=apply(mydata[,num.vars],2,min)
  
  scale.num.vars <- function(dat){
    for (i in 1:dim(dat)[2]){
      dat[,i]=(dat[,i]-min.vec[i])/(max.vec[i]-min.vec[i])}
    dat}
  
  
  mydata.train.scaled=scale.num.vars(mydata.train[,num.vars])  
  mydata.test.scaled=scale.num.vars(mydata.test[,num.vars])} else {
    
    mydata.train.scaled=mydata.train[,num.vars]
    mydata.test.scaled=mydata.test[,num.vars]}


#Creating the data frame for NN

train.data.nn=as.data.frame(cbind(mydata.train.scaled, dummy.train))
test.data.nn=as.data.frame(cbind(mydata.test.scaled, dummy.test))

f=as.formula(paste("myresponse~",paste(setdiff(names(train.data.nn),"myresponse"), collapse = " + ")))
if (problem.type=="C"){lin.opt=FALSE}else {lin.opt=TRUE}


#A seed is set so that the results are consistent across the classroom
#A randomization takes place to select starting values for weights and bias
#And a fixed seed value ensures that randomization renders same starting values across the classroom

do.mape<-function (true, pred){
  100*mean(abs((true-pred)/true))}


nn.one.run<-function(nn, current){
  
  best.rep=which.min(nn$result.matrix[1,])
  test.prediction=neuralnet::compute(nn,test.data.nn, rep=best.rep)
  test.prediction=as.data.frame(test.prediction$net.result)
  #Assess the out-of-sample (testing) performance
  if (problem.type=="C"){
    
    resp.list=nn$model.list$response
    test.prediction$predicted.class=apply(test.prediction, 1,which.is.max)
    test.prediction$predicted.class[test.prediction$predicted.class==1]=resp.list[1]
    test.prediction$predicted.class[test.prediction$predicted.class==2]=resp.list[2]
    test.prediction$predicted.class=as.factor(test.prediction$predicted.class)
    
    for.comparison.test=cbind(test.data.nn, test.prediction)
    
    print(paste(c("Hidden Node Structure Is",current), collapse=" "))
    confusion.matrix.test=CrossTable(for.comparison.test$myresponse,
                                     for.comparison.test$predicted.class,
                                     dnn=c("True Class","Predicted Class"), 
                                     prop.chisq=F,prop.t=F, prop.c=F, prop.r=F)}else{
                                       
                                       myresp.max=max.vec[which(names(max.vec)=="myresponse")]       
                                       myresp.min=min.vec[which(names(min.vec)=="myresponse")]
                                       
                                       test.prediction$predicted.myresponse=test.prediction$V1*(myresp.max-myresp.min)+myresp.min
                                       test.prediction$V1=NULL
                                       
                                       for.comparison.test=cbind(myresponse=mydata.test$myresponse, test.prediction)
                                       mape=do.mape(for.comparison.test$myresponse, for.comparison.test$predicted.myresponse)
                                       print(paste(c("For Hidden Node Structure",current,"MAPE IS", round(mape, digits=4)), collapse=" "))
                                       
                                     }}


set.seed(321)
new.seed=.Random.seed
seed.list=rep(list(new.seed),length(hid.list))


fit.ann<-function(current){
  
  nn<-neuralnet(
    f, 
    linear.output=lin.opt, #is getting determined (automatically) earlier in the program based on the problem.type
    hidden=current,#will iteratively try all tentative topologies listed as part of hid.list
    data=train.data.nn, 
    act.fct="logistic", #is the activation function
    rep=2,#For each given topology there are this many networks fit and the results outputed below correspond to the best fit.
    #You may see output for fewer networks than specified in "hid.list", if some networks experience convergence issues
    
    stepmax=1e6, #you can reduce this if too slow
    threshold = 0.1 #reducing this may drastically affect run times
    
  )
  
  nn.one.run(nn, current)
  
}

Sys.time()
plan(multiprocess)
try(hid.list%>%future_map(fit.ann, 
                          .progress = T, 
                          .options=future_options(seed=seed.list)), silent = TRUE)
Sys.time()

#########################################################################################33
#########################################################################################33
#########################################################################################33

#END OF AUTORUN


#For the network that you consider as the "best" among the ones tried as part of
#the "hid.list", complete one more run to see the plot and further results

#Specify the structure of hidden layers of the "best" network below

current=c(5,5,5,5)


#############################################################################
#############################################################################
#######################DO NOT MODIFY BEYOND THIS POINT#######################
#############################################################################
#############################################################################

final.run<-function (current){
  set.seed(321) 
  nn<-neuralnet(
    f, 
    linear.output=lin.opt, 
    hidden=current,
    data=train.data.nn, 
    act.fct="logistic", 
    rep=2,
    stepmax=1e6,
    threshold = 0.1
  )
  
  
  plot(nn, rep="best")
  
  train.half.sse=nn$result.matrix[1,which.min(nn$result.matrix[1,])]
  steps=nn$result.matrix[3,which.min(nn$result.matrix[1,])]
  print(paste(c("Training 1/2 SSE is Approximately ", round(train.half.sse,2)), collapse=""))
  print(paste(c("Training Completed in ", steps, " Steps"), collapse=""))
  
  nn.one.run(nn,current)
  
}

final.run(current)

nn<-neuralnet(
  f, 
  linear.output=lin.opt, 
  hidden=current,
  data=train.data.nn, 
  act.fct="logistic", 
  rep=2,
  stepmax=1e6,
  threshold = 0.1
)
new.data = train.data.nn[train.data.nn$Bin_zip.code_01730 == 1,]
yPredInt <- predict(nn,new.data)
new.data.tb = cbind(new.data,yPredInt)


dat[,i]=(dat[,i]-min.vec[i])/(max.vec[i]-min.vec[i])}
max = max(mydata$myresponse)
min = min(mydata$myresponse)

range = max - min
range
anti = range * 0.065688 + min
anti
