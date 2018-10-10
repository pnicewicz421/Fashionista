library(rjson)
library(keras)
library(jpeg)
library(dplyr)

library(OpenImageR)

original_dataset_dir <- '~/Downloads/all-1'
base_dir <- '~/Documents/Data Science/DeepLearningInR/Computer Vision/Fashion'
dir.create(base_dir)

train_dir <- file.path(base_dir, "train")
dir.create(train_dir)

validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)

test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_links <- fromJSON(file = paste0(original_dataset_dir, "/train.json"))

# Load validation data
validation_links <- fromJSON(file = paste0(original_dataset_dir, "/validation.json"))

# Load test data
test_links <- fromJSON(file = paste0(original_dataset_dir, "/test.json"))
                       
for (i in 1:3) {

  num_images = length(extract$images) # the entire data set
  if (i == 1){
    extract <- train_links
    save_dir <- train_dir
  } else if (i == 2) {
    extract <- validation_links
    save_dir <- validation_dir
  } else if (i == 3) {
    extract <- test_links
    save_dir <- test_dir
  }
  
  for (j in 1:num_images) {
    img_path <- extract$images[[j]]$url
    result <- tryCatch(
      download.file(img_path, paste0(save_dir, "/", j, ".jpg"), mode="wb"),
      error=function(e) e,
      warning=function(w) w)
  if(inherits(result, "error")) next
  if(inherits(result, "warning")) next
  }

}
