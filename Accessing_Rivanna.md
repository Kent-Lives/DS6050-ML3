# Accessing Rivanna

Justin Lee and Tom Lever

jgh2xh@virginia.edu and tsl2b@virginia.edu

This document

- was created in 10/2025

- was updated on 01/14/2026

- has version 1.0.0

---------------------

**Our goal is to provide a standardized environment for everyone (students and instructors) and avoid dependency hell.** When we receive your code for grading, we expect your code to run in our standardized environment without raising exceptions.


## Rivanna Directory Structure

When storing files in Rivanna, you will typically have access to two major partitions: HOME and SCRATCH. Both locations should be private. Only you can read and write files in HOME and SCRATCH.

HOME is your home directory and has "permanent" storage. In Terminal, you can access HOME by running `cd /home/<your UVA computing ID>`, `cd $HOME`, or `cd ~`. Actual storage space is limited to about 50 GB, which you may find restrictive.

SCRATCH is a "temporary" storage space. You can access this location using `cd /sfs/weka/scratch/<your UVA computing ID>`. Files untouched for more than 90 days may be randomly deleted. You get up to 10 TB of storage space.

The file explorer of JuPyteR lab has a root of HOME or SCRATCH depending on which Work Directory you choose.


## Open OnDemand

The simplest way to use Rivanna is to use Open OnDemand, a system hosted by UVA to access HPC without needing a VPN.

1. Open a browser and open https://www.rc.virginia.edu/userinfo/hpc/login/ .

2. Under the Web-based Access section, click the link `Launch Open OnDemand`.

3. Log in using your UVA credentials.

4. At the top of the page, under the `Interactive Apps` dropdown menu, select `JupyterLab`.

5. Launch a JupyterLab session with the following details.

    - Rivanna/Afton Partition: If you don't need a GPU, select Standard. If you need a GPU, select GPU.

    - Number of hours (e.g., 6).
    
    - Number of cores (e.g., 1).
    
    - Memory Request in GB (e.g., 8).
    
    - Allocation: `shakeri_ds6050`
    
    - Work Directory: HOME or SCRATCH

6. Connect to Jupyter.

7. Click on Terminal at the bottom of the launcher.

8. Run the command python.

9. In python, run the command `print("Hello World!")`.

10. Press CTRL + D to return to the terminal.


## Accessing Rivanna in Visual Studio Code

You can access Rivanna via Visual Studio Code.


### Using Visual Studio Code via Open OnDemand

1. Follow steps 1 through 3 above.

3. At the top of the page, under the `Interactive Apps` dropdown menu, select `Code Server`.

4. Launch a VS Code server.

    - Rivanna/Afton Partition: If you don't need a GPU, select Standard. If you need a GPU, select GPU.

    - Number of hours (e.g., 6).
    
    - Number of cores (e.g., 1).
    
    - Memory Request in GB (e.g., 8).
    
    - Allocation: `shakeri_ds6050`
    
    - Work Directory: HOME or SCRATCH

5. Connect to VS Code.

6. Open a terminal by pressing CTRL + ~.

7. To modernize python, see a list of active modules by running `module list`. See a list of available versions of miniforge by running `module spider miniforge`. Load `miniforge/26.9.4` by running `module load miniforge/26.9.4`.

8. Run the command python.

9. In python, run the command `print("Hello World!")`.

10. Press CTRL + D to return to the terminal.


### Using Visual Studio Code Locally

1. If you live off grounds and do not have access to the UVA wifi network `eduroam`, then first connect to a UVA VPN by following the instructions at https://virginia.service-now.com/its?id=itsweb_kb_article&sys_id=f24e5cdfdb3acb804f32fb671d9619d0.

2. Open Visual Studio Code.

3. Change directory by running `cd ~/.ssh`.

4. Generate a private key and a public key by running `ssh-keygen -t ed25519 -C "<your UVA computing ID>@virginia.edu"`.

5. Enter a base key name for the private key and the public key (e.g., `<your name>s_<your computer name>_Rivanna_Key`).

6. Skip entering a passphrase.

7. Create file `~/.ssh/config`, which does not have an extension. Add the following text. You might run `vim config`, insert text by pressing I, copy the text, paste the text by right clicking in the ViM window, save the text by running `:w` in the ViM window, and quit ViM by running `:q`.

```
Host rivanna
    HostName login.hpc.virginia.edu
    User <your UVA computing ID>
    ServerAliveInterval 60
    IdentityFile ~/.ssh/<base key name>
```

8. Copy your public key into your Rivanna home directory. Mac users should use

```
ssh-copy-id -i <base key name> rivanna
```

Windows users should use

```
cat <base key name>.pub | ssh rivanna "cat >> .ssh/authorized_keys"
```

You may need to enter yes to continue connecting. This step may require you to enter your UVA password. Do not be surprised if it looks like the cursor doesn't move as you type. Once you type your password, press `Enter`.

9. Install the `Remote Development` extension for Visual Studio Code.

10. Press the `><` button in the bottom left corner of Visual Studio Code. Click "Connect to Host...". Click rivanna. A new Visual Studio Code window should open with a connection to Rivanna. Click the stack of pages in the upper left corner. Click Open Folder. Ensure `/home/<your UVA computing ID>` is entered. Click OK. Ensure a bash terminal is open. Check your Present Working Directory by running `pwd`.

11. Open a terminal by pressing CTRL + ~. Tom is running on Windows and likes to switch from `PowerShell` to bash using the drop down arrow at the right of the terminal. bash uses a Linux syntax.

12. For work requiring significant resources, create an interactive job. If you don't need a GPU, run the following command

```
ijob -J <job name> -A shakeri_ds6050 -p standard -c 1 -t <wall time> --mem=<memory> -v
```

If you need a GPU, run the following command.

```
ijob -J <job name> -A shakeri_ds6050 -p gpu --gres=gpu:1 -t <wall time> --mem=<memory> -v
```

Job name is a custom name for the submitted job. Wall time is how long you will have access to the resources, and it is in the format `[d-]hh:mm:ss`. For example, `3-00:00:00` is a valid time for 3 days. The number of days is optional, so you can also do `48:00:00` for 2 days. Note that the maximum wall time for CPU jobs is 7 days, and for GPU jobs is 3 days. `<memory>` is the amount of total memory in gigabytes and can be specified as `<memory>G`. For example, to create an interactive job, called `run_CNN`, that uses a GPU, for 24 hours, that uses 8 GB of memory, I would use the command

```
ijob -J run_CNN -A shakeri_ds6050 -p gpu --gres=gpu:1 -t 1-00:00:00 --mem=64G -v
```

13. Follow steps 7 through 10 in section "Using Visual Studio Code via Open OnDemand".
