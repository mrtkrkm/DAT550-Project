import pandas as pd
import numpy as np
import os
class GetTasks(object):

    def get_tasks(Main_path):

        tasks = {}
        with open(os.path.join(Main_path,'MainTasks.txt')) as t:
            for line in t:
                line = line.replace('\n', '')
                line = line.strip()
                tasks[line] = []
        for i, key in enumerate(tasks.keys()):
            task = []
            with open(os.path.join(Main_path,f'Tasks{i + 1}.txt')) as f:
                for line in f:
                    line = line.replace('\n', '')
                    line = line.strip()
                    task.append(line)
                tasks[key] = task
        return tasks