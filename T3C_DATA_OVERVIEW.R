library(ggplot2)
library(NLP)
library(tm)
library(SnowballC)
library(cluster)
library(factoextra)
library(NbClust)
library(wordcloud)
library(plotly)
library(caret)
library(corrplot)
library(reshape2)
library(stringr)
library(cowplot)
rm(list=ls())
#Setting working directory
setwd("/Users/martinberger/Desktop/DPA/T3C_PROJECT/T3C")

#Loading the data
original_train <- read.csv("DATA/train.csv",sep=",", header=T)
original_test <- read.csv("DATA/train.csv",sep=",", header=T)

#Creating a data frame
train_df <- data.frame(original_train)
train_df<-train_df[,-1]
#A quick glance to the data
summary(train_df)

#Labels distributions
nb_none <-sum(train_df$toxic==0 & train_df$severe_toxic==0 & train_df$obscene==0 & train_df$threat==0 & train_df$insult==0 &train_df$identity_hate==0)
labels <- c("Toxic","Severe Toxic", "Obscene","Threat", "Insult", "Identity Hate", "None")
values <- c(sum(train_df$toxic==1),sum(train_df$severe_toxic==1),sum(train_df$obscene==1),sum(train_df$threat==1),
            sum(train_df$insult==1),sum(train_df$identity_hate==1), nb_none)

data <- data.frame(labels, values)
p <- plot_ly(data, x = ~labels, y = ~values,text=values,textposition = 'auto', type = 'bar', name = 'Labels Density') 
p

#Whcih amount of commetns have a certain amount of tags ?
train_df <-data.frame(train_df, rowSums(train_df[,2:7]))
colnames(train_df)[colnames(train_df)=="rowSums.train_df...2.7.."] <- "label_nb"
nb_tag_none <-sum(train_df$toxic==0 & train_df$severe_toxic==0 & train_df$obscene==0 & train_df$threat==0 & train_df$insult==0 &train_df$identity_hate==0)

labels <- c("0","1", "2","3", "4", "5", "6")
occurences <- c(nb_tag_none,sum(train_df[,8]==1),sum(train_df[,8]==2),sum(train_df[,8]==3),sum(train_df[,8]==4),sum(train_df[,8]==5), sum(train_df[,8]==6))

data <- data.frame(labels, values)
p <- plot_ly(data, x = ~labels, y = ~occurences,text=occurences,textposition = 'auto', type = 'bar', name = 'Labels Density') 
p
#Counting number of excl in comments
train_df <- data.frame(train_df,str_count(train_df$comment_text , "$"))
colnames(train_df)[colnames(train_df)=="str_count.train_df.comment_text......"] <- "nb_of_excl"
#Correlation between toxic comments
summary(train_df)
corr_df<-train_df[!train_df$label_nb == 0 ,]
summary(corr_df)
corr_df<-corr_df[,-8]
corr_df<-corr_df[,-1]
summary(corr_df)
nrow(corr_df)
corrplot(cor(corr_df[]), method = "number")

#Creating smaller df depending on label to analyse them 
#clean_df <- train_df[train_df$label_nb == 0 ,]
toxic_df <- train_df[train_df$toxic == 1 ,]
severe_toxic_df <- train_df[train_df$severe_toxic == 1 ,]
obscene_df <- train_df[train_df$obscene == 1 ,]
threat_df <- train_df[train_df$threat == 1 ,]
insult_df <- train_df[train_df$insult == 1 ,]
identity_hate_df <- train_df[train_df$identity_hate == 1 ,]

#Creation of a corpus for each label
#clean_corpus <-Corpus(VectorSource(clean_df[,1])) 
toxic_corpus <-  Corpus(VectorSource(toxic_df[,1])) 
severe_toxic_corpus <-   Corpus(VectorSource(severe_toxic_df[,1])) 
obscene_corpus <-   Corpus(VectorSource(obscene_df[,1])) 
threat_corpus <-   Corpus(VectorSource(threat_df[,1])) 
insult_corpus <-   Corpus(VectorSource(insult_df[,1])) 
identity_hate_corpus <-   Corpus(VectorSource(identity_hate_df[,1])) 

#Cleaning each corpus
basic_text_cleaner <- function(corpus){
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}
#clean_corpus <- basic_text_cleaner(clean_corpus)
toxic_corpus <- basic_text_cleaner(toxic_corpus)
severe_toxic_corpus <-  basic_text_cleaner(severe_toxic_corpus)
obscene_corpus <-  basic_text_cleaner(obscene_corpus)
threat_corpus <-  basic_text_cleaner(threat_corpus)
insult_corpus <-  basic_text_cleaner(insult_corpus)
identity_hate_corpus <-  basic_text_cleaner(identity_hate_corpus)

#Creation of a DTM for each corpus
#clean_dtm <- DocumentTermMatrix(clean_corpus, control = list(weighting = weightTfIdf))
toxic_dtm <- DocumentTermMatrix(toxic_corpus, control = list(weighting = weightTfIdf))
severe_toxic_dtm <- DocumentTermMatrix(severe_toxic_corpus, control = list(weighting = weightTfIdf))
obscene_dtm <- DocumentTermMatrix(obscene_corpus, control = list(weighting = weightTfIdf))
threat_dtm <- DocumentTermMatrix(threat_corpus, control = list(weighting = weightTfIdf))
insult_dtm <- DocumentTermMatrix(insult_corpus, control = list(weighting = weightTfIdf))
identity_hate_dtm <- DocumentTermMatrix(identity_hate_corpus, control = list(weighting = weightTfIdf))

#Transforming each DTM into a matrix
#clean_matrix <- as.matrix(clean_dtm)
toxic_matrix <- as.matrix(toxic_dtm)
severe_toxic_matrix <- as.matrix(severe_toxic_dtm)
obscene_matrix <- as.matrix(obscene_dtm)
threat_matrix <- as.matrix(threat_dtm)
insult_matrix <- as.matrix(insult_dtm)
identity_hate_matrix <- as.matrix(identity_hate_dtm)

#Creating a wordcloud for each type 
#clean_freq <-colSums(clean_matrix )
#wordcloud(names(clean_freq ), clean_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))
toxic_freq <- colSums(toxic_matrix )
wordcloud(names(toxic_freq ), toxic_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))
severe_toxic_freq <- colSums(toxic_matrix )
wordcloud(names(severe_toxic_freq ), severe_toxic_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))
obscene_freq <- colSums(obscene_matrix)
wordcloud(names(obscene_freq ), obscene_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))
threat_freq <- colSums(threat_matrix )
wordcloud(names(threat_freq ), threat_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))
insult_freq <- colSums(insult_matrix)
wordcloud(names(insult_freq ), insult_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))
identity_hate_freq <- colSums(identity_hate_matrix)
wordcloud(names(identity_hate_freq ), identity_hate_freq , max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))

#Size of the vocab for each label. 
labels <- c("Toxic","Severe Toxic", "Obscene","Threat", "Insult", "Identity Hate")
values <- c(length(toxic_freq),length(severe_toxic_freq),length(obscene_freq), length(threat_freq), length(insult_freq),length(identity_hate_freq))

data <- data.frame(labels, values)
p <- plot_ly(data, x = ~labels, y = ~values,text=values,textposition = 'auto', type = 'bar', name = 'Labels Density') 
p

# Let see top ten word count for each labels
toxic_top <- head(sort(toxic_freq, decreasing=TRUE), 20)
severe_toxic_top <- head(sort(severe_toxic_freq, decreasing=TRUE), 20)
obscene_top <- head(sort(obscene_freq, decreasing=TRUE), 20)
threat_top <- head(sort(obscene_freq, decreasing=TRUE), 20)
insult_top <- head(sort(insult_freq, decreasing=TRUE), 20)
identity_hate_top <- head(sort(identity_hate_freq, decreasing=TRUE), 20)

top_words_plot <- function(topten,corpus_name){
  dfplot <- as.data.frame(melt(topten))
  dfplot$word <- dimnames(dfplot)[[1]]
  dfplot$word <- factor(dfplot$word,
                        levels=dfplot$word[order(dfplot$value,
                                                 decreasing=TRUE)])
  
  fig <- ggplot(dfplot, aes(x=word, y=value)) + geom_bar(stat="identity")
  fig <- fig + xlab(corpus_name)
  fig <- fig + ylab("Count")
  print(fig)
}
top_words_plot(toxic_top,"Toxic Comments Top Words")
top_words_plot(severe_toxic_top, "Severe Toxic Comments Top Words")
top_words_plot(obscene_top, "Obscene Comments Top Words")
top_words_plot(threat_top, "Threat Comments Top Words")
top_words_plot(insult_top, "Insult Comments Top Words")
top_words_plot(identity_hate_top, "Identity Hate Comments Top Words")



#Boxplot of number of exclamation mark in comments depending of the label
toxic_df <- train_df[train_df$toxic == 1 ,]
severe_toxic_df <- train_df[train_df$severe_toxic == 1 ,]
obscene_df <- train_df[train_df$obscene == 1 ,]
threat_df <- train_df[train_df$threat == 1 ,]
insult_df <- train_df[train_df$insult == 1 ,]
identity_hate_df <- train_df[train_df$identity_hate == 1 ,]

p1=qplot(y=toxic_df$nb_of_excl, x= 1, geom = "boxplot") +  geom_boxplot()
p2=qplot(y=severe_toxic_df$nb_of_excl, x= 1, geom = "boxplot") +  geom_boxplot()
p3=qplot(y=obscene_df$nb_of_excl, x= 1, geom = "boxplot") +  geom_boxplot()
p4=qplot(y=threat_df$nb_of_excl, x= 1, geom = "boxplot") +  geom_boxplot()
p5=qplot(y=insult_df$nb_of_excl, x= 1, geom = "boxplot") +  geom_boxplot()
p6=qplot(y=identity_hate_df$nb_of_excl, x= 1, geom = "boxplot")
plot_grid(p1, p2, p3, p4, p5, p6,labels="AUTO")

ggplot(data = df, aes(x=variable, y=value)) + geom_boxplot(aes(fill=Label))

#THE END
#THE END
#THE END