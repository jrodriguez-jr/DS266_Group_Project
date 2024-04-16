README


Note: ADDED ATTENTION WEIGHTS RETURN MECHANISM. Traiining with two head two layers and drop rate 0.6 Running 12 epochs with 90% 10% split in train/val and dff = 256 and Learnable Positional Encoding, This is the 3rd run for adding pre-trained embeddings
These embeddings were loaded from FastText Twitter pre-trained embeddings.
These  embeddings are associated with the model checkpoints saved during training pertaining to the model with
base name 'lp_plus_attn_cl_enh_more_dat_tx_w_tweet_2h12l_ep12_d06_dff256', these word embeddings
can be used for a pre- and post- training analysis of the training effects to each word embedding.
The initial state of the  word embeddings before training are saved in the './Final_Work/lp_plus_attn_enh_cl_more_lp_tx_2h_2l_tweet_emb_d06_dff256_ep12/' directory.
The file 'initial_embeddings.csv' contains the embeddings in CSV format.

The hyperparameters for this model ara saved on this directory in hyperparameters.csv file
The results of this training run will be saved under CSV_Results-lp_plus_attn_cl_enh_more_dat_tx_w_tweet_2h12l_ep12_d06_dff256'.csv file.
The learning schedule plot, training loss plot and training model summary will also saved in this directory.
