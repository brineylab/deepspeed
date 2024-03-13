Here's the simpilest way to run deepspeed when using hugging face models, trainer, etc. 
Note that this may not be compatible with the most complicated model architectures nor does it allow more the most robust deepspeed customization. 

This guide is based primarily on the information from hugging face's accelerate guide. More useful info can be found at the link below.
* https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed#deepspeed-config-file
I've added a few steps that are specific to our current server setup. They may change in the future, as we update our container images.


1. (Temporarily - until this is installed in our images) Install the cudatookit:
```
conda install -c conda-forge cudatoolkit-dev
```
Check that install was successful by running: which nvcc. If installation was sucessful, the terminal wil print the path to nvcc (typically /opt/conda/bin/nvcc)

2. Set up the accelerate configuration. Run: 
```
accelerate config
```
And answer the questions as applicable to your training run. Here's an example of the answers to these questions
training a model on 2 gpus.
Note: you can optionally provide a deepspeed configuation json file here. If you don't, accelerate will automatically generate one from 
your hugging face values. If you are using deepspeed stage 3 or just want more control over variables that you cannot set directly
from the accelerat config command (ex. optimizer type, learning rate scheduler, etc.) you can provide the path to a config file.
I've provided an example of the json file format. Note that values that equal to 'auto' will be automatically populated based
on your hugging face config. Not all configurations can be set to 'auto' - more info about this can be found at the hugging face link
above.
More detailed info about config files can be found here: https://www.deepspeed.ai/docs/config-json/ and at the link provided above

3. Setup your python script. When using the accelerate library, no modifications to your scripts are necessary to use deepspeed.
Two examples of the scripts I use are provided, but you don't need to format your scripts in any particular way (this is just my preference).
* One important note: don't use wandb.init() to initalize your wandb run, this will cause your wandb session to be intitialized multiple times. 
Just set your wandb enviornmental variables in the script and training arguments as shown in the example and let hugging face initialize wandb 
for you.
* Also make sure you save your trained model after training. Because you're running in a script, not a jupyter notebook, the kernel with your
trained model won't remain open after training is complete. So if you don't save your model, you'll lose it!

4. Start a terminal multiplexer. This is required, otherwise your training with timeout when you close your connection to the server. 
tmux or screen are good options (Both of these options are installed on all of our container images except kubeflow-codeserver.) 
A good tmux cheatsheet can be found here: https://tmuxcheatsheet.com/
Here are the basic commands that I use to start a session:
```
tmux new -s session-name # will open a new session in your temrminal window
```

5. Run your script using the accelerate launcher: 
```
accelerate launch script.py --script_args
```

6. Once you want to disconnect from the terminal
Ctrl + b d to detatch from the tmux session, assuming you haven't changed the default

To reattach to the terminal at any point during training, the command is: 
```
tmux attach -t session-name
```

7. Once training is complete, you can kill the tmux session as follows session: 
```
tmux kill-session -t session-name
```