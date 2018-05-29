setwd("~/Documents/Data Science/Fashion")
library(rjson)
library(keras)
library(jpeg)
library(dplyr)

library(OpenImageR)


# Load training data
train <- fromJSON(file = "train.json")

# Load test data
test <- fromJSON(file = "test.json")

# Load validation data
validation <- fromJSON(file = "validation.json")

### Just focus on the first 10 images or so

#num_images = length(train$images) # the entire data set
num_images = 120

width = 600
height = 600
rgb_channel = 3

img <- array(numeric(), c(num_images, width, height, rgb_channel))

for (i in 1:num_images) {
  img_path <- train$images[[i]]$url
  tmp_file <- tempfile(fileext=".jpg")
  result <- tryCatch(
    download.file(img_path, tmp_file, mode="wb"),
    error=function(e) e,
    warning=function(w) w)
  if(inherits(result, "error")) next
  if(inherits(result, "warning")) next
  img_i <- readImage(tmp_file)
  if (dim(img_i)[1] != 600 | dim(img_i)[2] != 600) {
    img_i <- resizeImage(img_i, 600, 600)
  }
  
  #0-center the image ?
  #img_mn <- apply(img_i, 1, mean)
  #img_i <- img_i - img_mn
  
  
  img[i,,,] <- img_i
  file.remove(tmp_file)
}
# Clean up
rm(i)
rm(tmp_file)
rm(img_i)
rm(result)
#rm(img_mn)
rm(img_path)

x_train <- img

rm(img)


ytrain<-vector()
#determine max number of classes
for (j in 1:dim(x_train)[1]) {
  ytrain <- c(ytrain,train$annotations[[j]]$labelId)
}
ytrain <- as.numeric(ytrain)

num_classes <- max(ytrain) + 1 

rm(ytrain)

y_train <- array(numeric(), c(dim(x_train)[1], num_classes))

for (j in 1:dim(x_train)[1]) {
  outputs <- as.numeric(train$annotations[[j]]$labelId)
  yt <- as.data.frame(to_categorical(outputs, num_classes))
  yt <- yt %>% summarize_all(sum)
  y_train[j,] <- unlist(yt)
}

rm(outputs)
rm(yt)

###

# Creating a sequential model (stack of layers)
# initiate the model
model <- keras_model_sequential()

# Add a convultion layer.
# Arguments:
# object = the model
# filter =  (number of filters in the output space)
# kernel_size = 5 (x 5)
# stride = 1 
# padding = 'same' (preserve spatial dimensions from input?)
# data_format= 'channels_last' input shape: (batch, height, width, channels)
# dilation_rate # skip for now
# activation ='ReLU'  # activation function. default linear. let's try relu
# use_bias # we will not have a bias now
# kernel_initializer # kernel weights matrix. default is glorot_normal
                    # Gaussian / variance / 2 (?)
# bias_initializer = skip for now, since not using bias vector
# kernel_regulalizer = not using regulizer for first pass

model %>% layer_conv_2d(filters=96, kernel_size=30, stride=10, 
                       padding='same',activation='relu',
                       use_bias=FALSE, kernel_initializer="glorot_normal",
                       input_shape=c(600,600,3)) %>%
  
  
  # Add a pooling layer 
  # pool_size = c(2, 2) to halve the image
  # strides = c(2, 2) stride 2 since we don't want overlap
  
          layer_max_pooling_2d(pool_size=4, strides=2) %>%
  
          layer_batch_normalization(axis=1, momentum=0.99, epsilon=0.001) %>%
  
  # One more conv layer of 64 filters
  # and another halving pool layer (250 -> 125)
          layer_conv_2d(filters=256, kernel_size=5, stride=1,
                        activation='relu', padding='same') %>%
          layer_max_pooling_2d(pool_size=3, strides=2) %>%
  
          layer_batch_normalization(axis=1, momentum=0.99, epsilon=0.001) %>%
  
          layer_conv_2d(filters=384, kernel_size=3, stride=1,
                activation='relu', padding='same') %>%
  
          layer_conv_2d(filters=384, kernel_size=3, stride=1,
                activation='relu', padding='same') %>%
  
          layer_conv_2d(filters=256, kernel_size=3, stride=1,
                activation='relu', padding='same') %>%
  
          layer_max_pooling_2d(pool_size=4, strides=2) %>%
  
          
  
  # Bring to a close:
  # Flatten - don't need to specify any args
  # fully connected (dense layer) (dims = 125*125*3 -> )
  # apply activation function to FC layer (softmax?)
          layer_flatten() %>%
          layer_dense(units=4096, activation='relu') %>%
          layer_dense(units=4096, activation='softmax') %>%
          layer_dense(units=227, activation='relu')
  
#sgd <- optimizer_sgd(lr=1e-6, decay=1)
rmsp <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

# compile the model
model %>% compile(
  optimizer = rmsp,
  loss = 'mean_squared_error',
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, epochs=30, batch_size=5,
  validation_split=0.2, verbose=2
)
                       