library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(tidytext)
library(tidyverse)
library(magrittr)
library(data.table)
library(h2o)
file_path <- "C:/Users/smrid/Desktop"
install.packages(h2o)
setwd(file_path)
train <- read.csv("train.csv",stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
train <- as.data.table(read.csv("train.csv"))
train[,filter:="Train"]
test[,filter:="Test"]
train_test_bind <- rbindlist(list(train, test), fill=TRUE)

train[,":="(comment_text=gsub("'|\"|'|"|"|\"|\n|,|\\.|.|\\?|\\+|\\-|\\/|\\=|\\(|\\)|'", "", comment_text))]
train[,":="(comment_text=gsub("  ", " ", comment_text))]

h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,
         forceDL = FALSE, enable_assertions = TRUE, license = NULL,
         nthreads = -1, max_mem_size = NULL, min_mem_size = NULL,
         ice_root = tempdir(), strict_version_check = TRUE,
         proxy = NA_character_, https = FALSE, insecure = FALSE,
         username = NA_character_, password = NA_character_,
         cookies = NA_character_, context_path = NA_character_,
         ignore_config = FALSE, extra_classpath = NULL)

print("Convert to H2O Frame")
comments <- data.table(comments=train[,comment_text])

comments.hex <- as.h2o(comments, destination_frame = "comments.hex", col.types=c("String"))

View(comments)

STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

tokenize <- function(sentences, stop.words = STOP_WORDS) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  
  # convert to lower case
  tokenized.lower <- h2o.tolower(tokenized)
  # remove short words (less than 2 characters)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.lower[h2o.grep("[0-9]", tokenized.lower, invert = TRUE, output.logical = TRUE),]
  
  # remove stop words
  tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% STOP_WORDS),]
}

print("Break comments into sequence of words")
words <- tokenize(comments.hex$comments)

# build word2vec
vectors <- 20 # Only 10 vectors to save time & memory
w2v.model <- h2o.word2vec(words
                          , model_id = "w2v_model"
                          , vec_size = vectors
                          , min_word_freq = 5
                          , window_size = 5
                          , init_learning_rate = 0.025
                          , sent_sample_rate = 0
                          , epochs = 5) # only a one epoch to save time

print("Sanity check - find synonyms for the word 'piss'")
print(h2o.findSynonyms(w2v.model, "president", count = 5))

print("Get vectors for each comment")
comment_all.vecs <- h2o.transform(w2v.model, words, aggregate_method = "AVERAGE")

comments_all.vecs <- as.data.table(comment_all.vecs)
comments_all <- cbind(comments, comments_all.vecs)
train <- merge(train, comments_all, by.x="comment_text", by.y="comments", all.x=TRUE, sort=FALSE)
colnames(train)[10:ncol(train)] <- paste0("comment_vec_C", 1:vectors)

print("output comment vectors")
fwrite(train, "./h2ow2v_vectors.csv")
#############PCA---
file.exists("file_path/train.csv")
#train_data <- data.frame(train, stringsAsFactors=FALSE)
system("ls ../D:/Sem2_Spring2018/DPA/proj/train.csv")
data(stop_words)

words <- train %>% 
  unnest_tokens(word, comment_text) 

words %<>% anti_join(stop_words, by = 'word')

counting <- words %>%
  group_by(word) %>%
  dplyr::summarise(
    count = n(),
    toxic_avg = mean(toxic),
    severe_toxic = mean(severe_toxic),
    obscene_avg = mean(obscene),
    threat_avg = mean(threat),
    insult_avg = mean(insult),
    identity_hate_avg = mean(identity_hate),
    bad_word_index = sum(toxic_avg, severe_toxic, obscene_avg, threat_avg,
                         insult_avg, identity_hate_avg)  
  ) %>% filter(count>300) %>%  arrange(desc(bad_word_index))
to_plot <- counting %>% filter(bad_word_index>=3) %>% data.frame

rownames(to_plot) <- to_plot$word

pc_words <- princomp(to_plot[,-c(1,2,9)], cor = T)

pc_words %>% biplot()

pc_words_sc <- pc_words$scores %>% data.frame
pc_words_sc$words <- row.names(to_plot)
pc_words_sc %>% filter((Comp.1)<=-1) %>% select(words)

