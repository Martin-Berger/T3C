library(tm)
library(tau)
library(plyr)
library(SnowballC)
library(wordcloud)
library(data.table)
library(RColorBrewer)
library(LiblineaR)

data_set <- read.csv("DATA/train.csv")

set.seed(42)
smp_size <- floor(0.80 * nrow(data_set))
train_ind <- sample(seq_len(nrow(data_set)), size = smp_size)
train <- data_set[train_ind, ]
test <- data_set[-train_ind, ]

#Import the corpus and create a cleaning pipeline
basic_text_cleaner <- function(corpus){
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}

#Create a wordcloud representation of a specific document term matrix
create_wordcloud_df <- function(dtm){
  matrix <- as.matrix(documentTermMatrix)
  freq <- sort(rowSums(matrix), decreasing = TRUE)
  df <- data.frame(word = names(freq), freq=freq)
  wordcloud(words = df$word, freq = df$freq, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))
}

docs <- Corpus(VectorSource(train$comment_text))
documentTermMatrix <- DocumentTermMatrix(docs, control = list(weighting = weightTfIdf))
inspect(documentTermMatrix)

#Creating an other document term matrix by erasing stop words and create a TF-IDF weigthing
cleanDocs <- basic_text_cleaner(docs)
documentTermMatrixClean <- DocumentTermMatrix(cleanDocs, control = list(weighting = weightTfIdf))
inspect(documentTermMatrixClean)

#Reduce the sparsity of the matrix
documentTermMatrixClean <- removeSparseTerms(documentTermMatrixClean, 0.99)
inspect(documentTermMatrixClean)



##########################SPARSE LOGISTIC REGRESSION 
targets <- train[,-1:-2]
model = LiblineaR(data = as.matrix(documentTermMatrixClean), target = targets, type = 0, bias = TRUE, verbose = FALSE)