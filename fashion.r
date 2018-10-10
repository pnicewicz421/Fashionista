library(rjson)
library(keras)
library(reticulate)
original_dataset_dir <- '~/Downloads/all-1'

base_dir <- '~/Documents/Data Science/DeepLearningInR/Computer Vision/Fashion'
master_dir <- '~/Documents/Data Science/DeepLearningInR/Computer Vision/Fashion/master'

train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

info <- file.info(list.files(master_dir))
empty <- rownames(info[info$size == 0, ])
file.remove(file.path(master_dir, empty))

files <- list.files(master_dir, pattern="*.jpg")
train_links <- fromJSON(file = paste0(original_dataset_dir, "/train.json"))

# Load validation data
validation_links <- fromJSON(file = paste0(original_dataset_dir, "/validation.json"))

# Load test data
test_links <- fromJSON(file = paste0(original_dataset_dir, "/test.json"))

# find the total number of labels (labels will be one and zero?)
solutions <- vector()
for (i in 1:length(files)) {
  labels <- train_links$annotations[[i]]$labelId
  labels <- as.integer(labels)
  solutions <- c(solutions, labels)
}
number_of_labels <- max(solutions)

# create the answers array
outputs <- array(0, dim=c(length(files), number_of_labels))
for (i in 1:length(files)) {
  labels <- train_links$annotations[[i]]$labelId
  labels <- as.integer(labels)
  outputs[i, labels] <- 1 
}

#split the sets into train, validation, and test
train_set <- sample(files, 0.6*length(files), replace=FALSE)
remaining_files <- files[which(!files %in% train_set)]
validation_set <- sample(remaining_files, 0.5*length(remaining_files), replace=FALSE)
test_set <- remaining_files[which(!remaining_files %in% validation_set)]

# move the sets to the appropriate folders

file.copy(file.path(master_dir, train_set),
           file.path(train_dir))

file.copy(file.path(master_dir, validation_set),
          file.path(validation_dir))

file.copy(file.path(master_dir, test_set),
           file.path(test_dir))

# set up the model
model <- keras_model_sequential() %>%
  layer_conv_2d(filter=32, kernel_size = c(3,3), activation="relu",
                input_shape=c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=512, activation="relu") %>%
  layer_dense(units=number_of_labels, activation="sigmoid")

# compile the model
# for multilabel, multiclass classification, we need binary crossentropy
# and sigmoid function at the end
model %>% compile(
  loss="binary_crossentropy",
  optimizer=optimizer_rmsprop(lr=1e-4),
  metrics=c("acc")
)


# You've built a custom generator ... great job!
# the output is exactly right: list(inputs, targets)

# To do:
# 1. add a batch processor (ability to output more than 1 image)
# 2. add a rescaling function
# 3. add a stopping function (which takes the last batch even 
#if there is fewer than 1 full batch left)
# This is done.

flow_images_multilabel <- function(directory, file_format="jpg", 
                                   target_size=c(150,150),
                                   batch_size=20, rescale=(1/255),
                                   color_mode="rgb", outputs){
  img_index <- 1
  if (color_mode=="rgb") {
    color_channels <- 3
  } else if (color_mode=="grayscale") {
    color_channels <- 1
  }
  files <- list.files(directory, pattern=paste0("*.", file_format))
  image_numbers <- as.integer(gsub(paste0(".", file_format), "", files))
  image_numbers <- image_numbers[!is.na(image_numbers)]
  function() {
    # rest img_index if starting a new epoch
    # i.e., if img_index > image_numbers
    if (img_index > length(image_numbers)) {
      img_index <- 1
    }
    indices_remaining <- length(image_numbers) - img_index + 1
    if (indices_remaining > 0 ) {
      if (indices_remaining < batch_size) {
        batch_size <- indices_remaining
      }
      X <- array(0, dim=c(batch_size, target_size, color_channels))
      Y <- array(0, dim=c(batch_size, dim(outputs)[[2]]))
      for (i in img_index:img_index + batch_size - 1) {
        imgnumber <- image_numbers[[i]]
        num <- as.character(imgnumber)
        
        X_image <- image_load(file.path(directory, paste0(num, ".", file_format)), 
                                target_size = target_size)
        X_array <- image_to_array(X_image)
        X_array <- X_array * rescale
        #X_array <- readImage(file.path(directory, paste0(num, ".", file_format)))
        #X_array <- resizeImage(X_array, target_size[[1]], target_size[[2]])
        
        Y_labels <- outputs[imgnumber, ]
        X[i - img_index + 1,,,] <- X_array
        Y[i - img_index + 1,] <- Y_labels
      }
      img_index <<- img_index + batch_size
      list(X, Y)
    } else {
      NULL
    }
  }
}

# Now that you have a custom generator function,
# use it for train and validation

train_generator <- flow_images_multilabel(
                   directory=train_dir, outputs=outputs, batch_size=24)

validation_generator <- flow_images_multilabel(
                        directory=validation_dir, outputs=outputs, batch_size=24)
# Fit the model
history <- model %>% fit_generator(train_generator,
                                   steps_per_epoch = 354,
                                   epochs=30,
                                   validation_data=validation_generator,
                                   validation_steps=118)

model %>% save_model_hdf5("multilabel_fashion_model.h5")