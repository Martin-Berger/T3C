library(tidyverse)
library(NLP)
library(magrittr)
library(text2vec)
library(tokenizers)
library(glmnet)
library(doParallel)
library(tm)
library(caret)
library(utiml)
library(pROC)
library(MASS)
library(dplyr)
#setwd("/Users/martinberger/Desktop/DPA/T3C_PROJECT/T3C") MAYBE SET A WORKING DIRECTORY IF IT CANNOT FIND THE CSV FILE
registerDoParallel(4)
data_set <- read.csv("DATA/train.csv")
targets <- c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
data <- data_set %>% 
  dplyr::select(-one_of(targets)) %>% 
  mutate(length = str_length(comment_text), ncap = str_count(comment_text, "[A-Z]"), ncap_len = ncap / length, nexcl = str_count(comment_text, fixed("!")), nquest = str_count(comment_text, fixed("?")), npunct = str_count(comment_text, "[[:punct:]]"), nword = str_count(comment_text, "\\w+"), nsymb = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^")) %>% 
  dplyr::select(-id)

comment_tokens <- data %$%
  str_to_lower(comment_text) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  gsub("\\b\\w{1}\\s","", .) %>% #Remove words with lengh less than 2
  itoken(tokenizer = tokenize_word_stems)

vectorizer <- create_vocabulary(comment_tokens, ngram = c(1, 1), stopwords = stopwords("en")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.5, vocab_term_max = 4000) %>%
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <- create_dtm(comment_tokens, vectorizer) %>%
  fit_transform(m_tfidf)


finaldata <- data %>%
  dplyr::select(-comment_text) %>%
  sparse.model.matrix(~ . - 1, .) %>%
  cbind(tfidf)

set.seed(42)
smp_size <- floor(0.80 * nrow(data_set))
train_ind <- sample(seq_len(nrow(data_set)), size = smp_size)
test <- finaldata[-train_ind, ]
train <- finaldata[train_ind, ]


prediction <- dplyr::select(data_set[-train_ind, ], -comment_text)
train_label <- dplyr::select(data_set[train_ind, ], -comment_text)
test_label <- prediction

for (label in targets){
  y_train <- train_label[[label]]
  model <- lda(train,  factor(y_train), family="binomial", nfolds = 4, parallel = T,  nlambda = 100, alpha=0)
  prediction[[label]] <- predict(model, test, type = "response", s = "lambda.min")
}
prediction[1:5,]
thresh = 0.5
ss2 <- sweep(as.matrix(prediction[,-1]),MARGIN=2,STATS=thresh,
             FUN=function(x,y) ifelse(x<y,0,1))
prediction <- data.frame(id=prediction$id,ss2)
prediction[1:5,]
test_label[1:5,]
prediction_w_id <- dplyr::select(prediction, -id)
test_label_w_id <- dplyr::select(test_label, -id)
for (label in targets) {
  cat("\n\n", label, "\n\n")
  print(confusionMatrix(test_label_w_id[[label]], prediction_w_id[[label]], positive="1"))
  plot(roc(test_label_w_id[[label]], prediction_w_id[[label]], col="yellow", lwd=3))
  print(auc(test_label_w_id[[label]], prediction_w_id[[label]]))
}
