library(rjson)
library(keras)
library(reticulate)

# In version 2 of fashion, we 
# will load the images as part of the generator
# i.e., this is an experiment in loading and processing images directly 
# from the http source into the model, rather than downloading them first

original_dataset_dir <- '~/Downloads/all-1'

train_links <- fromJSON(file = paste0(original_dataset_dir, "/train.json"))

# Load validation data
validation_links <- fromJSON(file = paste0(original_dataset_dir, "/validation.json"))

# Load test data
test_links <- fromJSON(file = paste0(original_dataset_dir, "/test.json"))

# find the total number of labels (labels will be one and zero?)

# Get the number of labels
number_of_labels <- max(as.integer(unlist(lapply(train_links$annotations, `[[`, 1))))

# Custom metric: F1 score
recall <- function(y_true, y_pred) {
  #Recall metric.
  
  #Only computes a batch-wise average of recall.
  
  #Computes the recall, a metric for multi-label classification of
  #how many relevant items are selected.
  
  true_positives <- k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  possible_positives <- k_sum(k_round(k_clip(y_true, 0, 1)))
  recall <- true_positives / (possible_positives + k_epsilon())
  recall
}

precision <- function(y_true, y_pred) {
  #Precision metric.
  
  #Only computes a batch-wise average of precision.
  
  #Computes the precision, a metric for multi-label classification of
  #how many selected items are relevant.
  
  true_positives <- k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  predicted_positives <- k_sum(k_round(k_clip(y_pred, 0, 1)))
  precision <- true_positives / (predicted_positives + k_epsilon())
  precision
}

# F1 score metric
f1 <- function(y_true, y_pred) {
  prec <- precision(y_true, y_pred)
  rec <- recall(y_true, y_pred)
  f1 <- 2*((prec*rec)/(prec+rec+k_epsilon()))
  f1
}

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
  metrics=c(f1)
)


# You've built a custom generator ... great job!
# the output is exactly right: list(inputs, targets)

# To do:
# 1. add a batch processor (ability to output more than 1 image)
# 2. add a rescaling function
# 3. add a stopping function (which takes the last batch even 
#if there is fewer than 1 full batch left)
# This is done.


train_directory <- "~/Documents/Data Science/deeplearninginr/computer vision/fashion/train"
validation_directory <- "~/Documents/Data Science/DeepLearningInR/Computer Vision/Fashion/validation"

# The multilabel generator
flow_images_multilabel <- function(category, directory, file_format="jpg", 
                                   target_size=c(150,150),
                                   batch_size=20, rescale=(1/255),
                                   color_mode="rgb", shuffle=FALSE){

  img_index <- 1
  if (color_mode=="rgb") {
    color_channels <- 3
    grayscale <- FALSE
  } else if (color_mode=="grayscale") {
    color_channels <- 1
    grayscale <- TRUE
  }
  
  num_images <- length(category$images)
  
  function() {
    # This is the batch generating function
    # rest img_index if starting a new epoch
    # i.e., if img_index > image_numbers
  
      
      X <- array(0, dim=c(batch_size, target_size, color_channels))
      Y <- array(0, dim=c(batch_size, number_of_labels))
      
      # draw batch_size samples if shuffle
      if (shuffle) {
        image_index <- sample(1:num_images, batch_size, replace=FALSE)
      } else {
        indices_remaining <- num_images - img_index + 1
        if (indices_remaining < batch_size) batch_size <- indices_remaining
        image_index <- seq(img_index, img_index + batch_size - 1)      
      }
      
      print(paste0(as.character(image_index)))
      
      for (i in img_index:(img_index + batch_size - 1)) {
        print(paste0("dataset:", as.character(directory)))
        print(paste0("i:",as.character(i)))
        print(paste0("img_index:",as.character(img_index)))
        
        rel_ind <- i - img_index + 1 
        
        print(rel_ind)
        
        ind <- image_index[rel_ind] 
        num <- as.character(ind)
        
        img_path <- category$images[[ind]]$url
        
        fname <- paste0(directory, "/", num, ".", file_format)
        
        result <- tryCatch(
          X_img <- get_file(fname=fname, origin=img_path),
          error=function(e) e,
          warning=function(w) w)
        if(inherits(result, "error") | inherits(result, "warning")) {
            # error downloading file. exchange it for something else
              if (shuffle) {
                # draw another random file
                image_index[rel_ind] <- sample(1:num_images, 1)
                i <- i - 1 
                next 
              } else {
                # reorder image index for non-shuffled indices
                for (j in rel_ind:batch_size) {
                  image_index[j] <- image_index[j] + 1  
                  # doomsday scenario: error is on the last ordered piece
                  if (image_index[j]==num_images) {
                    image_index <- image_index[-j]
                    break
                  }
                }
                img_index <- img_index + 1
                i <- i - 1 
                next 
              }
              
            }


        X_image <- image_load(X_img, grayscale=grayscale,
                              target_size=target_size)
        
        X_array <- image_to_array(X_image)
        X_array <- X_array * rescale
        plot(as.raster(X_array))
        
        X[rel_ind,,,] <- X_array
        file.remove(fname)

        # Get the labels from the JSON
        Y_lbls <- as.integer(category$annotations[[ind]]$labelId)
        
        # fill in the ith row and the Y_lbls columns as 1  
        Y[rel_ind, Y_lbls] <- 1
      
      }
      img_index <<- img_index + batch_size
      list(X, Y)
    } 
  }


# Now that you have a custom generator function,
# use it for train and validation

train_generator <- flow_images_multilabel(category=train_links,
                   directory=train_directory, batch_size=30, shuffle=TRUE)


validation_generator <- flow_images_multilabel(category=validation_links,
                        directory=validation_directory, batch_size=30, shuffle=FALSE)
# Fit the model
val_steps <- as.integer(length(validation_links$images) / 30)

history <- model %>% fit_generator(train_generator,
                                   steps_per_epoch = 33818,
                                   epochs=30,
                                   validation_data=validation_generator,
                                   validation_steps=val_steps)

model %>% save_model_hdf5("multilabel_fashion_model.h5")

#generate test data
directory <- test_dir
files <- list.files(directory, pattern=paste0("*.", file_format))
image_numbers <- as.integer(gsub(paste0(".", file_format), "", files))
image_numbers <- sort(image_numbers[!is.na(image_numbers)])
X_test <- array(0, dim=c(length(image_numbers), 150, 150, 3))
#Y_test <- array(0, dim=c(length(image_numbers), number_of_labels))
for (i in 1:length(image_numbers)) {
  imgnumber <- image_numbers[[i]]
  num <- as.character(imgnumber)
  
  result <- tryCatch(
  X_image <- image_load(file.path(directory, paste0(num, ".", file_format)), 
                        target_size = target_size),
  error=function(e) e,
  warning=function(w) w)
  
  X_array <- image_to_array(X_image)
  X_array <- X_array * rescale
  if(inherits(result, "error")) next
  if(inherits(result, "warning")) next
  
  #plot(as.raster(X_array))
  #X_array <- readImage(file.path(directory, paste0(num, ".", file_format)))
  #X_array <- resizeImage(X_array, target_size[[1]], target_size[[2]])
  
  #Y_labels <- outputs[imgnumber, ]
  X_test[i,,,] <- X_array
  #plot(as.raster(X_test[i - img_index + 1,,,]))
  #Y_test[i - img_index + 1,] <- Y_labels
  
  # unit test
  #which(Y_test[i - img_index + 1,]==1)
  #which(outputs[imgnumber,]==1)
  #sort(as.integer(train_links$annotations[[imgnumber]]$labelId))
}

# predict the test data
predictions <- model %>% predict(X_test)

