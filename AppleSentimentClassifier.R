#############################################################################
# ACKNOWLEDGEMENTS
# this example is partially based on the Chapter 10 of the R and Data Mining
# http://www.rdatamining.com/books/rdm 
#############################################################################

####################
## GET THE TWEETS ##
####################

# load the tweets data
tweets.data <- read.csv(file = "data/tweets_sentiment_labelled.csv", 
                        colClasses = c("character", "factor"))
str(tweets.data)

# examine a few positive and a few negative tweets
tweets.data$Tweet[tweets.data$Lbl=="POS"][1:10]
tweets.data$Tweet[tweets.data$Lbl=="NEG"][1:10]

# examine the distribution of class labels
table(tweets.data$Lbl)

###################################
## PREPROCESSING OF TWEETS' TEXT ##
###################################

# the tm package is required for text processing
# install.packages("tm")
library(tm)

# load the R script with auxiliary functions
source("text_mining_utils.R")

# build a corpus
tw.corpus <- Corpus(VectorSource(tweets.data$Tweet))
tw.corpus

# tm_map() f. (from the tm package) allows for performing different transformations on the corpus;
# list of frequently used transformations can be obtained with getTransformations() f.
getTransformations()
# the purpose of all the transformations is to reduce the diversity among the words 
# and remove words that are of low importance

# the first transformation will be to convert text to lower case
tw.corpus <- tm_map(tw.corpus, tolower)
print.tweets(tw.corpus, 1, 20)

# when processing tweets, we often remove user references completely;
# however, this corpus is specific - it has many (meaningful) user references;
# those are mostly references to Twitter accounts of various tech companies;
# so, we will remove only '@' sign that marks usernames (@username); 
# this will be done using regular expressions;
# an excellent introduction to regular expression is available at: 
# http://regex.bastardsbook.com/
replaceUserRefs <- function(x) gsub("@(\\w+)", "\\1", x)
tw.corpus <- tm_map(tw.corpus, replaceUserRefs)
print.tweets( tw.corpus, from = 20, to = 60 )

# remove hash (#) sign from hastags
removeHash <- function(x) gsub("#([[:alnum:]]+)", "\\1", x)  
tw.corpus <- tm_map(tw.corpus, removeHash)
print.tweets( tw.corpus, from = 20, to = 60 )

# replace URLs with the "URL" term
replaceURL <- function(x) gsub("(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", "URL", x) 
tw.corpus <- tm_map(tw.corpus, replaceURL)
print.tweets( tw.corpus, from = 20, to = 60 )

# replace links to pictures (e.g. pic.twitter.com/lbu9diufrf) with TW_PIC
replaceTWPic <- function(x) gsub("pic\\.twitter\\.com/[[:alnum:]]+", 
                                 "TW_PIC", x) 
tw.corpus <- tm_map(tw.corpus, replaceTWPic)
print.tweets( tw.corpus, from = 20, to = 60 )

# replace :-), :), ;-), :D, :o with the "POS_SMILEY" term
replaceHappySmiley <- function(x) gsub("[:|;](-?)[\\)|o|O|D]", 
                                       "POS_SMILEY", x)
tw.corpus <- tm_map(tw.corpus, replaceHappySmiley)
print.tweets( tw.corpus, from = 20, to = 60 )

# replace :(, :-(, :/, >:(, >:O with the "NEG_SMILEY" term
replaceSadSmiley <- function(x) gsub("((>?):(-?)[\\(|/|O|o])", 
                                     "NEG_SMILEY", x)
tw.corpus <- tm_map(tw.corpus, replaceSadSmiley)
print.tweets( tw.corpus, from = 80, to = 100 )

# back to the slides

# remove stopwords from corpus;
# first examine the tm's set of stopwords
stopwords('english')[100:120]

# add a few extra ('corpus-specific') stop words (e.g. "apple", "rt") 
# to the 'general' stopwords for the English language
tw.stopwords <- c(stopwords('english'), "apple", "rt")
tw.corpus <- tm_map(tw.corpus, removeWords, tw.stopwords)
print.tweets( tw.corpus, from = 30, to = 70 )

# remove punctuation 
tw.corpus <- tm_map(tw.corpus, removePunctuation, 
                    preserve_intra_word_contractions = TRUE,
                    preserve_intra_word_dashes = TRUE)
print.tweets( tw.corpus, from = 30, to = 70 )

# remove stand-alone numbers (but not numbers in e.g. iphone7 or g3) 
removeStandAloneNumbers <- function(x) gsub(" \\d+ ", " ", x)
tw.corpus <- tm_map(tw.corpus, removeStandAloneNumbers)
print.tweets( tw.corpus, from = 30, to = 70 )

# strip whitespace
tw.corpus <- tm_map(tw.corpus, stripWhitespace)
print.tweets( tw.corpus, from = 30, to = 70 )

# back to the slides

# do word stemming using the Snowball stemmer: https://snowballstem.org/
# to use the Snowball stemmer in R, we need the SnowballC package
#install.packages("SnowballC")
library(SnowballC)

# since we might later want to have words in their 'regular' form,
# we will keep a copy of the corpus before stemming it
tw.corpus.backup <- tw.corpus

# now, do the stemming
tw.corpus <- tm_map(tw.corpus, stemDocument, language = "english") 
print.tweets( tw.corpus, from = 30, to = 70)

# back to the slides

#####################################
## BUILDING A DOCUMENT-TERM MATRIX ##
#####################################

# Document Term Matrix (DTM) represents the relationship between terms and documents, 
# where each row stands for a document and each column for a term, and an entry is the
# weight of the term in the corresponding document

min.freq <- round(0.005*length(tw.corpus))
max.freq <- round(0.95*length(tw.corpus))

dtm <- DocumentTermMatrix(tw.corpus, 
                          control = list(bounds = list(global = c(min.freq, max.freq)),
                                         wordLengths = c(2,16), # the restriction on the word length
                                         weighting = weightTf)) # term freq. weighting scheme

# Note: the 'global' parameter is altered to require a word to appear in at least ~0.5% 
# and at most in 95% of tweets to be included in the matrix; 
# check the documentation of the TermDocumentMatrix() f. for other useful control parameters

# examine the built DTM matrix
inspect(dtm)

# we have very sparse DTM matrix; so, we should better reduce the sparsity 
# by removing overly sparse terms:
dtm.trimmed <- removeSparseTerms(dtm, sparse = 0.9875)
inspect(dtm.trimmed)

# examine the resulting DTM matrix:
# check the terms that appear at least 20 times in the whole corpus
findFreqTerms(dtm.trimmed, lowfreq = 20)
# we can also inspect the frequency of accurance of all the terms
head(colSums(as.matrix(dtm)))
# better if they are sorted
sort(colSums(as.matrix(dtm)), decreasing = T)

# back to the slides

#################################################
## CLASSIFYING TWEETS USING NAIVE BAYES METHOD ##
#################################################

# Since we want to use DTM for classification purposes, 
# we need to transform it into a data frame that 
# can be passed to a function for building a classifier:
features.final <- as.data.frame(as.matrix(dtm.trimmed))
str(features.final)

head(features.final)

# add the class label
features.final$CLASS_LBL <- tweets.data$Lbl 
colnames(features.final)

# split the data into training and test sets
library(caret)
set.seed(1212)
train.indices <- createDataPartition(y = features.final$CLASS_LBL,
                                     p = 0.85,
                                     list = FALSE)
train.data <- features.final[train.indices,]
test.data <- features.final[-train.indices,]

# build NB classifier using all the features
#install.packages('e1071')
library(e1071)

nb1 <- naiveBayes(CLASS_LBL ~ ., 
                  data = train.data, 
                  laplace = 1) # laplace smoothing (correction)
# since each feature (word) has numerous zero values, when fitting the model, 
# we include laplace smoothing to avoid zero values of conditional probabilities   

# make predictions
nb1.pred <- predict(nb1, newdata = test.data, type = "class")

# create confusion matrix
cm1 <- table(true = test.data$CLASS_LBL, predicted = nb1.pred)
cm1

# evaluate the model
eval1 <- compute.eval.measures(cm1)
eval1

# try to improve the performance by using a different probability threshold
# (instead of the default one of 0.5); to that end, we'll make use of ROC curves

#install.packages('pROC')
library(pROC)

# get predictions as probabilities
nb1.pred.prob <- predict(nb1, newdata = test.data, type = "raw")
nb1.pred.prob[1:10,]

# compute stats for the roc curve
nb.roc <- roc(response = as.numeric(test.data$CLASS_LBL),
              predictor = nb1.pred.prob[,1], # probabilities of the 'positive' class
              levels = c(2,1)) # define the order of levels corresponding to the negative (controls)
                              # and positive (cases) class

# plot the curve
plot.roc(x = nb.roc,
         print.auc = TRUE) # print AUC measure

# get the evaluation measures and the threshold for the local maxima of the ROC curve
nb2.coords <- coords(roc = nb.roc,
                     x = "local maximas",
                     ret = c("accuracy", "sensitivity", "specificity", "thr"))
nb2.coords

# as we want to assure that the company (Apple) will not miss tweets with negative sentiment,
# and since we set the negative sentiment as our positive class (i.e. class in our focus),
# we should look for a probability threshold that will maximise sensitivity (i.e., true positive rate);
# still, we should keep the other measures (accuracy, specificity) at a decent level 

# the local maximum that corresponds to the 9th column looks as a good candidate
# let's examine it more closely
nb2.coords[,9]
# select the threshold that corresponds to the 9th local maximum 
opt.threshold <- nb2.coords[4,9]

# assign class labels based on the chosen threshold
nb1.pred.opt <- ifelse(test = nb1.pred.prob[,1] > opt.threshold,
                       yes = "NEG", no = "POS")
nb1.pred.opt <- as.factor(nb1.pred.opt)

# create confusion matrix based on the newly assigned class labels  
cm.opt <- table(actual = test.data$CLASS_LBL, predicted = nb1.pred.opt)
cm.opt

# examine evaluation measures
eval2 <- compute.eval.measures(cm.opt)
eval2

# compare evaluation measures
data.frame(rbind(eval1, eval2), row.names = c("default_threshold", "ROC_based_theshold"))
