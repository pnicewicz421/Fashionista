library(rjson)
library(keras)
library(jpeg)

# Load training data
train <- fromJSON(file = "train.json")

# Load test data
test <- fromJSON(file = "test.json")

# Load validation data
validation <- fromJSON(file = "validation.json")

### Just focus on the first 10 images or so

img <- list()

for (i in 1:1000) {
  img_path = train$images[[i]]$url
  tmp_file <- tempfile()
  download.file(img_path, tmp_file, mode="wb")
  img[[i]] <- readJPEG(tmp_file)
  file.remove(tmp_file)
}

x_train <- img

y_train <- list()

for (i in 1:length(x_train)) {
  y_train[[i]] <- train$annotations[[i]]$labelId
}
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

model %>% layer_conv_2d(filters=32, kernel_size=5, stride=1,
                       padding='same', data_format='channels_last', activation='relu',
                       use_bias=FALSE, kernel_initializer="glorot_normal",
                       input_shape=c(500,500,3)) %>%
  
  # Add a pooling layer 
  # pool_size = c(2, 2) to halve the image
  # strides = c(2, 2) stride 2 since we don't want overlap
  
          layer_max_pooling_2d(pool_size=c(2,2), strides=c(2,2)) %>%
  
  # One more conv layer of 64 filters
  # and another halving pool layer (250 -> 125)
          layer_conv_2d(filters=64, kernel_size=5, stride=1,
                        activation='relu', padding='same') %>%
          layer_max_pooling_2d(pool_size=c(2,2), strides=c(2,2)) %>%
  
  # Bring to a close:
  # Flatten - don't need to specify any args
  # fully connected (dense layer) (dims = 125*125*3 -> )
  # apply activation function to FC layer (softmax?)
          layer_flatten() %>%
          layer_dense(units=1000000, activation='relu') %>%
          layer_dense(units=1000000, activation='softmax')
  
# compile the model
model %>% compile(
  optimizer = optimizer_sgd(),
  loss = 'mean_squared_error',
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, epochs=30, batch_size=128,
  validation_split=0.2, verbose=2
)
                       