import json
import os
import sys
import traceback

try:
  with open('hpc_task_definitions.json') as f:
    definitions = json.load(f)
except:
  print('Error while reading test_definitions file. Exiting.')
  traceback.print_exc()
  sys.exit()

for idx, definition in enumerate(definitions):
  try:
    python_args = ' '.join(['--%s=\"%s\"' % (key, str(definition['env_details'][key])) for key in definition['env_details']])
    python_args += ' '.join(['--%s=\"%s\"' % (key, str(definition['nn_details'][key])) for key in definition['nn_details']])
  
    runner_script_lines = [
      '#!/bin/sh',
      '#BSUB -q gpuv100',
      '#BSUB -gpu \"num=1\"',
      '#BSUB -J Procgen_%d' % idx,
      '#BSUB -n 1',
      '#BSUB -W %s' % definition['job_details']['runtime'],
      '#BSUB -R "rusage[mem=%s]"' % definition['job_details']['memory'],
      '#BSUB -o logs/log_%J.out',
      '#BSUB -e logs/log_%J.err',
      'python3 ../../train.py %s' % python_args
    ]
    os.system('mkdir -p ./testing/%d' % idx)
    os.chdir('./testing/%d' % idx)
    with open('%d.sh' % idx, 'w') as f:
      f.write('\n'.join(runner_script_lines))
    os.system('bsub < %d.sh')
    os.chdir('../..')
  except:
    print('Error while creating job script for job #%d.' % idx)
    traceback.print_exc()

