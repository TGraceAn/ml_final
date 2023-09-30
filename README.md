# ml_midterm

Lol, sorry for the wrong naming thinging, we'll be using this for our midterm project in 3 days

Using Google Speech Command (GSC) as our dataset, we want to train so that the model recognise words (just some words from GSC)

This is based on HolgerBovbjerg/data2vec-KWS (we just want to pass our presentation)

We're using the GSC split: 80% train, 10% validate, 10% test

The necessary Python packages to run the code is installed by running:
```shell
pip install -r requirements.txt
```

To download the Google Speech Commands V2 data set run the command:
```bash
bash download_gsc.sh speech_commands_v0.02
```

Get lists of train, validate, test
```shell
python3 make_data_list.py -v speech_commands_v0.02/validation_list.txt -t speech_commands_v0.02/testing_list.txt -d speech_commands_v0.02 -o speech_commands_v0.02/_generated
```

#TODO:
Get dataloader --> load data as spectrogram into the nn (just use simple rnn or cnn) (train with noise if can --> Using LabelSmoothingLoss --> A regularization technique)
