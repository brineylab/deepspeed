Below I've provided steps for the simpilest way to run deepspeed when using HuggingFace tools for training. This is using the HuggingFace Accelerate library, which supports the use of Deepspeed. However, this method doesn't allow full deepspeed customization, which may be necessary in some cases.

This guide is based primarily on the information from the HuggingFace accelerate guide. More detailed info can be found [here](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed#deepspeed-config-file).

I've written this assuming that you are using the following docker image: `kubeflow-deeplearning:v2024-02-21`.


1. Install the cudatookit (because it is not currently installed in our docker images):
    ```bash
    conda install -c conda-forge cudatoolkit-dev
    ``` 
    * Check that install was successful by running: `which nvcc`. If installation was sucessful, the terminal will print the path to nvcc (typically /opt/conda/bin/nvcc)

2. Set up the accelerate configuration by running: 
    ```bash
    accelerate config
    ```
    * When prompted, answer the questions as applicable to your training run. Here's an example of the answers to these questions for ZeRO Stage 1:
        * In which compute environment are you running? **This machine.**
        * Should distrbuted operations be checked while running for errors? This can avoid timeout issues but will be slower. **yes**
        * Do you wish to optimize your script with torch dynamo? **NO**
        * Do you want to use DeepSpeed? **yes**
        * Do you want to specify a json file to a DeepSpeed config? **NO**
        * What should be your DeepSpeed's ZeRO optimization stage? **1**
        * Do you want to use gradient clipping? **yes**
        * What is the gradient clipping value? **1.0** *(the default in the HuggingFace trainer)*
        * Do you want to enable 'deepspeed.zero.Init' when using ZeRO Stage-3 for constructing massive models? **NO**
        * How many GPU(s) should be used for distributed training? **2** *(replace with # of gpus on your system)*
        * Do you wish to use FP16 or BF16 (mixed precision)? **fp16**

    * If you are running ZeRO Stage 2 or 3, you will also be prompted about optimizer, gradient, and parameter offloading. Currently, our systems can support offloading to cpu (but not NVMe). Be aware that offloading to cpu to decrease your gpu memory usage, but will slow down the training.

    * Note: you can *optionally* provide a deepspeed configuation json file here. There are many deepspeed configuration options that you cannot set from the accelerate config command. If you want control over these options, you should provide the path to a deepspeed config file. I've provided an example of the json file format. Note that values that equal to 'auto' will be automatically populated based on your hugging face config. Not all configurations can be set to 'auto' - more info about this can be found at the hugging face link above.

    * More detailed info about deepspeed config files can be found [here](https://www.deepspeed.ai/docs/config-json/) and at the HuggingFace link above.

3. Setup your python script. When using the accelerate library, no deepspeed-specific modifications to your python script are needed. Two examples of the scripts I use are provided, but you don't need to format your scripts in any particular way (this is just my preference).

    * One important note: don't use wandb.init() to initalize your wandb run, this will cause your wandb session to be intitialized multiple times. Just set the wandb enviornmental variables and training arguments in your script (as shown in the example scripts) and let hugging face initialize wandb for you.

4. Start a terminal multiplexer. This is required, otherwise your training with timeout when you close your connection to the server. 
    * `tmux` and `screen` are good options. A useful tmux cheatsheet can be found here: https://tmuxcheatsheet.com/
    * Here is the command that I use to start a tmux session:
        ```bash
        tmux new -s session-name
        ```

5. Run your script using the accelerate launcher.
    ```bash
    accelerate launch script.py --script_args
    ```

6. Once the training has started, you should detach from the tmux session. Press `Ctrl` + `b` then `d` to detach.

    * To reattach to the terminal at any point during training, the command is: 
        ```bash
        tmux attach -t session-name
        ```

7. Once training is complete, you can kill the tmux session: 
   
    ```bash
    tmux kill-session -t session-name
    ```