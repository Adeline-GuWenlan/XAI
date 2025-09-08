## Claude
1. create sperate levels of dataset 
  divide the 
  /Users/guwenlan/Desktop/XAI/DataSet/ALL_3_Y/magic_labels_for_input_for_3_qubits_mixed_1_200000_datapoints.npy and 
  /Users/guwenlan/Desktop/XAI/DataSet/ALL_3_Y/magic_labels_for_input_for_3_qubits_mixed_2_200000_datapoints.npy 
  into 

  three groups as High, Medium and Low, each containing data from the two files according to the magic label. make the
   seperation as even as possible to remove bias. and then, create a dir in 

   
  /Users/guwenlan/Desktop/XAI/Grouped_Dataset named 3q_dataset, 
  
  which should have the High, Low, Medium Folder, each 
  having the according Label and Decoded_Token folder. the Label folders store the exytracted labels from the initial 
  two files; the decoed tokens take the according matrix, processed by 
  /Users/guwenlan/Desktop/XAI/upload/Model_Archi/decoder.py, and stored as file pairs. no need for validation of the 
  matrix.