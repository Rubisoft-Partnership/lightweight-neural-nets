/*
  www.aifes.ai
  Copyright (C) 2020-2021 Fraunhofer Institute for Microelectronic Circuits and Systems. All rights reserved.

  AIfES is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/



// The print function for the loss. It can be customized.
void printLoss(float loss)
{
    global_epoch_counter = global_epoch_counter + 1;
    Serial.print(F("Epoch: "));
    Serial.print(global_epoch_counter * PRINT_INTERVAL);
    Serial.print(F(" / Loss: "));
    Serial.println(loss,5);
    
}

void build_AIfES_model() {

  FNN_activations[0] = AIfES_E_sigmoid;
  FNN_activations[1] = AIfES_E_softmax;

  uint32_t weight_number = AIFES_E_flat_weights_number_fnn_f32(FNN_structure,FNN_3_LAYERS);

  Serial.print(F("Weights: "));
  Serial.println(weight_number);
  
  // FlatWeights array
  float *FlatWeights;
  FlatWeights = (float *)malloc(sizeof(float)*weight_number); 

  FNN.layer_count = FNN_3_LAYERS;
  FNN.fnn_structure = FNN_structure;
  FNN.fnn_activations = FNN_activations;
  FNN.flat_weights = FlatWeights;

}


void train_AIfES_model() {
 
  // In this function the model is trained with the captured training data

  uint32_t i;                                               // Counting variable

  // -------------------------------- Create tensors needed for training ---------------------
  // Create the input tensor for training, contains all samples
  uint16_t input_shape[] = {NUMBER_OF_DATA, INPUTS};        // Definition of the shape of the tensor, here: {# of total samples (i.e. samples per object * 3 objects), 3 (i.e. for each sample we have 3 RGB values)}
  aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, training_data);                 // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // Create the target tensor for training, contains the desired output for the corresponding sample to train the ANN
  uint16_t target_shape[] = {NUMBER_OF_DATA, OUTPUTS};            // Definition of the shape of the tensor, here: {# of total samples (i.e. samples per object * 3 objects), 3 (i.e. for each sample we have 3 possible output classes)}
  aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, labels);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // Create an output tensor for training, here the results of the ANN are saved and compared to the target tensor during training
  float output_data[NUMBER_OF_DATA][OUTPUTS];                     // Array for storage of the output data
  uint16_t output_shape[] = {NUMBER_OF_DATA, OUTPUTS};            // Definition of the shape of the tensor, here: {# of total samples (i.e. samples per object * 3 objects), 3 (i.e. for each sample we have 3 possible output classes)}
  aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

  // -------------------------------- init weights settings ----------------------------------
    
  AIFES_E_init_weights_parameter_fnn_f32  FNN_INIT_WEIGHTS;
  FNN_INIT_WEIGHTS.init_weights_method = AIfES_E_init_glorot_uniform;

  /* init methods
      AIfES_E_init_uniform
      AIfES_E_init_glorot_uniform
      AIfES_E_init_no_init        //If starting weights are already available or if you want to continue training
  */

  FNN_INIT_WEIGHTS.min_init_uniform = -2; // only for the AIfES_E_init_uniform
  FNN_INIT_WEIGHTS.max_init_uniform = 2;  // only for the AIfES_E_init_uniform
  // -------------------------------- set training parameter ----------------------------------
  AIFES_E_training_parameter_fnn_f32  FNN_TRAIN;
  FNN_TRAIN.optimizer = AIfES_E_adam;
  /* optimizers
      AIfES_E_adam
      AIfES_E_sgd
  */
  FNN_TRAIN.loss = AIfES_E_crossentropy;
  /* loss
      AIfES_E_mse,
      AIfES_E_crossentropy
  */
  FNN_TRAIN.learn_rate = 0.1f;                           // Learning rate is for all optimizers
  FNN_TRAIN.sgd_momentum = 0.0;                           // Only interesting for SGD
  FNN_TRAIN.batch_size = NUMBER_OF_DATA;                        // Here a full batch
  FNN_TRAIN.epochs = 1000;                                // Number of epochs
  FNN_TRAIN.epochs_loss_print_interval = PRINT_INTERVAL;  // Print the loss every x times

  // Your individual print function
  // it must look like this: void YourFunctionName(float x)
  FNN_TRAIN.loss_print_function = printLoss;

  //You can enable early stopping, so that learning is automatically stopped when a learning target is reached
  FNN_TRAIN.early_stopping = AIfES_E_early_stopping_on;
  /* early_stopping
      AIfES_E_early_stopping_off,
      AIfES_E_early_stopping_on
  */
  //Define your target loss
  FNN_TRAIN.early_stopping_target_loss = 0.09;

  global_epoch_counter = 0;

  int8_t error = 0;

  // -------------------------------- do the training ----------------------------------
  // In the training function, the FNN is set up, the weights are initialized and the training is performed.
  error = AIFES_E_training_fnn_f32(&input_tensor,&target_tensor,&FNN,&FNN_TRAIN,&FNN_INIT_WEIGHTS,&output_tensor);  
  
  error_handling_training(error); 
  
  Serial.println(F("Finished training"));

  // ----------------------------------------- Evaluate the trained model --------------------------
  // Here the trained network is tested with the training data. The training data is used as input and the predicted result
  // of the ANN is shown along with the corresponding labels.

  // Run the inference with the trained AIfES model, i.e. predict the output from the training data with the use of the ANN
  // The function needs the trained model, the input_tensor with the input data and the output_tensor where the results are saved in the corresponding array
  error = AIFES_E_inference_fnn_f32(&input_tensor,&FNN,&output_tensor);
  
  error_handling_inference(error);

  // Print the original labels and the predicted results
  Serial.println(F("Outputs:"));
  Serial.println(F("labels:\tGesture 1\tGesture 2\tGesture 3\tcalculated:\tGesture 1\tGesture 2\tGesture 3"));

  for (i = 0; i < NUMBER_OF_DATA; i++) {

    Serial.print(F("\t"));
    Serial.print (labels[i][0]);
    Serial.print (F("\t\t"));
    Serial.print (labels[i][1]);
    Serial.print (F("\t\t"));
    Serial.print (labels[i][2]);
    Serial.print (F("\t\t\t\t"));
    Serial.print (output_data[i][0]);
    Serial.print (F("\t\t"));
    Serial.print (output_data[i][1]);
    Serial.print (F("\t\t"));
    Serial.println (output_data[i][2]);

  }
}

void error_handling_training(int8_t error_nr){
  switch(error_nr){
    case 0:
      //Serial.println(F("No Error :)"));
      break;    
    case -1:
      Serial.println(F("ERROR! Tensor dtype"));
      break;
    case -2:
      Serial.println(F("ERROR! Tensor shape: Data Number"));
      break;
    case -3:
      Serial.println(F("ERROR! Input tensor shape does not correspond to ANN inputs"));
      break;
    case -4:
      Serial.println(F("ERROR! Output tensor shape does not correspond to ANN outputs"));
      break;
    case -5:
      Serial.println(F("ERROR! Use the crossentropy as loss for softmax"));
      break;
    case -6:
      Serial.println(F("ERROR! learn_rate or sgd_momentum negative"));
      break;
    case -7:
      Serial.println(F("ERROR! Init uniform weights min - max wrong"));
      break;
    case -8:
      Serial.println(F("ERROR! batch_size: min = 1 / max = Number of training data"));
      break;
    case -9:
      Serial.println(F("ERROR! Unknown activation function"));
      break;
    case -10:
      Serial.println(F("ERROR! Unknown loss function"));
      break;
    case -11:
      Serial.println(F("ERROR! Unknown init weights method"));
      break;
    case -12:
      Serial.println(F("ERROR! Unknown optimizer"));
      break;
    case -13:
      Serial.println(F("ERROR! Not enough memory"));
      break;
    default :
      Serial.println(F("Unknown error"));
  }
}

void error_handling_inference(int8_t error_nr){
  switch(error_nr){
    case 0:
      //Serial.println(F("No Error :)"));
      break;    
    case -1:
      Serial.println(F("ERROR! Tensor dtype"));
      break;
    case -2:
      Serial.println(F("ERROR! Tensor shape: Data Number"));
      break;
    case -3:
      Serial.println(F("ERROR! Input tensor shape does not correspond to ANN inputs"));
      break;
    case -4:
      Serial.println(F("ERROR! Output tensor shape does not correspond to ANN outputs"));
      break;
    case -5:
      Serial.println(F("ERROR! Unknown activation function"));
      break;
    case -6:
      Serial.println(F("ERROR! Not enough memory"));
      break;
    default :
      Serial.println(F("Unknown error"));
  }
}
