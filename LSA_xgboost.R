#####LSA

library(tidyverse)
library(magrittr)
library(text2vec)
library(tokenizers)
library(xgboost)
library(glmnet)
install.packages("doParallel")
library(doParallel)
registerDoParallel(4)
library(stringr)
install.packages("lsa")
library(lsa)
library(magrittr)
file_path <- "C:/Users/smrid/Desktop"
setwd(file_path)
train <- read.csv("train.csv",stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
subm <- read.csv("sample_submission.csv")
tri <- 1:nrow(train)

smp_size <- floor(0.80 * nrow(train))
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
train <- train[train_ind, ]
test <- train[-train_ind, ]
#Making a sample selection represention 40% of the data set.
#index <- sample(1:nrow(original), size=0.4*nrow(original))
#Giving 40% of the data set for testing purpose (223 rows)
#test <- original[index, ]
#Giving 60% of the data set for trainning purpose (350 rows)
#train <- original[-index, ]


targets <- c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
cat("Basic preprocessing & stats...\n")
#merging test with train by selecting one of the target labels
tr_te <- train %>% 
  select(-one_of(targets)) %>% 
  bind_rows(test) %>% 
  mutate(length = str_length(comment_text),
         ncap = str_count(comment_text, "[A-Z]"),
         ncap_len = ncap / length,
         nexcl = str_count(comment_text, fixed("!")),
         nquest = str_count(comment_text, fixed("?")),
         npunct = str_count(comment_text, "[[:punct:]]"),
         nword = str_count(comment_text, "\\w+"),
         nsymb = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^"),
         nsmile = str_count(comment_text, "((?::|;|=)(?:-)?(?:\\)|D|P))")) %>% 
  select(-id) %T>%
  glimpse()  

cat("Parsing comments...\n")
it <- tr_te %$%
  str_to_lower(comment_text) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  itoken(tokenizer = tokenize_word_stems)

vectorizer <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("en")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.5, vocab_term_max = 4000) %>%
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <- create_dtm(it, vectorizer) %>%
  fit_transform(m_tfidf)

m_lsa <- LSA$new(n_topics = 20, method = "randomized")
lsa <- fit_transform(tfidf, m_lsa)
lsa_mat <- as.textmatrix(lsa)
summary(lsa_mat)





cat("Preparing data for glmnet...\n")
X <- tr_te %>% 
  select(-comment_text) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf, lsa)

X_test <- X[-tri, ]
X <- X[tri, ]
rm(tr_te, test, tri, it, vectorizer, m_lsa, lsa); gc()


cat("Training & predicting...\n")

p <- list(objective = "binary:logistic", 
          booster = "gbtree", 
          eval_metric = "auc", 
          nthread = 4, 
          eta = 0.2, 
          max_depth = 3,
          min_child_weight = 4,
          subsample = 0.7,
          colsample_bytree = 0.7)
for (target in targets) {
  cat("\nFitting", target, "...\n")
  y <- train[[target]]
  m_xgb <- xgboost(X, y, params = p, print_every_n = 100, nrounds = 500)
  m_glm <- cv.glmnet(X, factor(y), alpha = 0, family = "binomial", type.measure = "auc",
                     parallel = T, standardize = T, nfolds = 4, nlambda = 50)
  cat("\tAUC:", max(m_glm$cvm))
  subm[[target]] <- 0.48*predict(m_xgb, X_test) + 0.52*predict(m_glm, X_test, type = "response", s = "lambda.min")
}

cat("Creating submission file...\n")
write_csv(subm, "tidy_xgb_glm.csv")

dtest <- sparse.model.matrix(Survived ~ .-1, data=X_test)



plot(m_glm, label=TRUE)
plot(m_glm, xvar = "lambda", label = TRUE)
