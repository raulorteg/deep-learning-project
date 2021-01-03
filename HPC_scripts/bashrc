# User .bashrc file.						UNI-C 25/07-96
#
# GNU Bourne Again SHell (bash) initialization.
# You are expected to edit this file to meet your own needs.
#
# The commands in this file are executed
# each time a new bash shell is started.
#

# Source the shared .bashrc file if it exists.
if [ -r /.bashrc ] ; then . /.bashrc ; fi

# Place your own code within the if-fi below to
# avoid it being executed on logins via remote shell,
# remote exec, batch jobs and other non-interactive logins.

# Set up the bash environment if interactive login.
if tty -s ; then

  # Set the system prompt.
  PS1="\w\n\h(\u) $ "
  export PS1

  # Set up some user command aliases.
  alias h=history
  alias source=.

  # Confirm before removing, replacing or overwriting files.
  alias rm="rm -i"
  alias mv="mv -i"
  alias cp="cp -i"

  #module load gcc
  #module load cuda/10.2
  #module load cudnn/v7.0-prod-cuda8

  module load python3/3.8.2
  module load numpy/1.18.2-python-3.8.2-openblas-0.3.9
  module load ffmpeg/4.2.2
  module load matplotlib/3.2.1-python-3.8.2
  module load pandas/1.0.3-python-3.8.2
fi

# Place your own code within the if-fi above to
# avoid it being executed on logins via remote shell,
# remote exec, batch jobs and other non-interactive logins.
