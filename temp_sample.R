# Temp forecasting sample

# download and extract the data
dir.create("~/Downloads/jena_climate", recursive=TRUE)

download.file("https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
              "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip")

unzip("~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip", exdir="~/Downloads/jena_climate")

# inspect / explore the data
library(tibble)
library(readr)


#explore the data
data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

glimpse(data)

library(ggplot2)

#plot the data
ggplot(data, aes(x=1:nrow(data), y=`T (degC)`)) + geom_line()

# pre-process the data

# convert the data into a floating-point matrix
data <- data.matrix(data[,-1])

# normalize the data
# remember, to take the mean and sd of the training data

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center=mean, scale=std)

# create the generator
generator <- function(data, lookback, delay, min_index,
                      max_index, shuffle=FALSE, batch_size=128,
                      step=6) {
  if (is.null(max_index)) max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  

  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size=batch_size)
    } else {
      if (i + batch_size >= max_index) {
        # once i runs out, reset it
        i <<- min_index + lookback
      }
      # set rows sequentially from i to batch or until max 
      # (last batch size is smaller)
      rows <- c(i:min(i + batch_size, max_index))
      i <<- i + length(rows)
    }
    samples <- array(0, dim=c(length(rows),
                              lookback / step,
                              dim(data)[[-1]]))
    
    targets <- array(0, dim=c(length(rows))) 
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]],
                     length.out=dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay, 2] 
    }
    list(samples, targets)
  }
  
}

library(keras)

lookback <- 1440 # lookback 10 days (obs. every 1 hour) from i
step <- 6 # look every 1 hour (rather than every 10 mins)
delay <- 144 # look forward (predict) 24 hours from i
batch_size <- 128 

train_gen <- generator(data, lookback=lookback, delay=delay,
                       min_index=1, max_index=200000,
                       shuffle=TRUE, step=step, batch_size=batch_size)

val_gen <- generator(data, lookback=lookback, delay=delay,
                     min_index=200001, max_index=300000,
                     step=step, batch_size=batch_size)

test_gen <- generator(data, lookback=lookback, delay=delay,
                     min_index=300001, max_index=NULL,
                     step=step, batch_size=batch_size)

val_steps <- (300000 - 200001 - lookback) / batch_size

test_steps <- (nrow(data) - 300001 - lookback) / batch_size

# always look for a naive (non-machine-learning)
# method to see if you can establish a baseline to beat
evaluate_naive_method <- function () {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds-targets))
    batch_maes <- c(batch_maes,mae)
  }
  print(mean(batch_maes))
}
err <- evaluate_naive_method()

# convert the mae to celsius error
# second column is the temperature (output)
celsius_mae <- err * std[[2]]

# try a simplistic machine-learning-model
# again for evaluation purposes
model <- keras_model_sequential() %>%
  layer_flatten(input_shape=c(lookback/step,
                              dim(data)[-1])) %>%
  layer_dense(units=32, activation="relu") %>%
  layer_dense(units=1) # no activation function in
                       # last dense layer because
                       # it's a regression problem

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500, 
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# find the best val loss
min(history$metrics$val_loss)
min(history$metrics$val_loss) < err

# Naive method beats the flattened dense layers 

# Let's try GRU (gated recurrent unit)

model <- keras_model_sequential() %>%
  layer_gru(units=32, input_shape=list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units=1)

model %>% compile(
  optimizer=optimizer_rmsprop(),
  loss="mae"
  )

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch=500,
  epochs=20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# Try the same model with dropout.
# there are two types of drop-outs:
# dropout: dropout rate for the input units of the layer
# recurrent_dropout: dropout rate for the recurrent units
# you also need more epochs because it will take longer for
# training and validation networks to converge
model <- keras_model_sequential() %>%
  layer_gru(units=32, input_shape=list(NULL, dim(data)[[-1]]),
            dropout=0.2, recurrent_dropout=0.2) %>%
  layer_dense(units=1)

model %>% compile(
  optimizer=optimizer_rmsprop(),
  loss="mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch=500,
  epochs=40,
  validation_data = val_gen,
  validation_steps = val_steps
)

# new issue: performance bottleneck
# increase capacity by increasing 
# 1. layers and/or 2. units
# intermediate sequences have to be returned as 3D
# tensors of shape (batch_size, timesteps, units)

model <- keras_model_sequential() %>%
  layer_gru(units=32, dropout=0.1, recurrent_dropout=0.5,
            return_sequences = TRUE, 
            input_shape=list(NULL, dim(data[[-1]]))) %>%
  layer_gru(units=64, activation="relu", dropout=0.1,
            recurrent_dropout=0.5) %>%
  layer_dense(units=1)

model %>% compile(
  optimizer=optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch=500,
  epochs=40,
  validation_data=val_gen,
  validation_steps = val_steps
)