import logging
import logging.handlers
import sys
from unipath import Path, DIRS_NO_LINKS
import recognizer
import numpy as np
import cv2
import multiprocessing as mp
import Queue
import time

file_range = np.arange(4, 50, 2)
test_dir = Path('test_files/words/KNMP')

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    long_format = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    short_format = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(short_format)

    fh = logging.handlers.RotatingFileHandler('param_sweep.log',
            maxBytes=1024*1024*5, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(long_format)

    logger.addHandler(ch)
    logger.addHandler(fh)

def test_worker(taskqueue, resultqueue, logqueue):
    # Set up logging using the queue
    #h = logging.handlers.QueueHandler(logqueue)
    #root = logging.getLogger()
    #root.handlers = []
    #root.addHandler(h)

    logger = logging.getLogger('Test worker')
    logger.info('Starting an experiment subprocess')

    while 1:
        try:
            task = taskqueue.get(True, 1)
        except Queue.Empty:
            time.sleep(1)
        if task == None:
            taskqueue.task_done() # Notify that we received STOP signal
            break

        # process task
        num, min_cut, max_cut, lexicon = task
        global_right = 0
        global_wrong = 0

        for file_nr in file_range:
            logging.info("File: {}".format(str(file_nr)))

            img_file = str(file_nr) + '.jpg'
            img_path = Path(test_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            words_file = str(file_nr) + '.words'
            words_file_name = Path(test_dir, words_file)
            right, wrong = recognizer.main(state="internal", img=img,
                    words_file_name=words_file_name, minCutWindow=min_cut,
                    maxCutWindow=max_cut, globalLexicon=lexicon)
            global_right += right
            global_wrong += wrong
        # Return the results
        accuracy = global_right / float(global_right + global_wrong)
        result = (num, min_cut, max_cut, lexicon, global_right, global_wrong,
                accuracy)
        resultqueue.put(result)

        taskqueue.task_done()

    logger.info('Exiting experiment subprocess')

def experiment_runner():
    logger = logging.getLogger('ParamSweep')

    # Experimental setup
    min_cut_window_range = np.arange(0, 40, 5)
    min_cut_window_range[0] = 1
    max_cut_window_range = np.arange(60, 210, 10)
    global_lexicon_range = [True, False]

    #min_cut_window_range = [1]
    #max_cut_window_range = [60]
    #global_lexicon_range = [True]

    # Create new CSV output file
    with open('param_sweep_results.csv', 'w') as f:
        f.write('num,min,max,lexicon,right,wrong,accuracy\n')

    # set up subprocesses
    taskqueue = mp.JoinableQueue()
    resultsqueue = mp.Queue()
    logqueue = mp.Queue()
    num_proc = int(sys.argv[1])
    logs = []

    # First create the processes so they can get started right away
    processes = [mp.Process(target=test_worker,
        args=(taskqueue, resultsqueue, logqueue)) for i in range(num_proc)]
    for p in processes:
        p.start()

    logger.info('Creating experimental cases')
    num = 1
    for min_cut in min_cut_window_range:
        for max_cut in max_cut_window_range:
            for global_lexicon in global_lexicon_range:
                taskqueue.put((num, min_cut, max_cut, global_lexicon))
                num += 1
    for i in range(num_proc):
        taskqueue.put(None)

    # Experiments should now be executing, so process the logs
    while taskqueue.qsize() > num_proc * 2:
        process_subprocess_logs(logqueue)
        process_results_queue(resultsqueue, logs)
        time.sleep(1)

    # Wait for all processes to finish and process all queues once more
    taskqueue.join()
    process_subprocess_logs(logqueue)
    for p in processes:
        p.join()
    process_results_queue(resultsqueue, logs)
    process_subprocess_logs(logqueue)
    logger.info("Finished running the experiment")

def process_subprocess_logs(logqueue):
    try:
        while True:
            record = logqueue.get_nowait()
            logger = logging.getLogger(record.name)
            logger.handle(record)
    except Queue.Empty:
        pass

def process_results_queue(resultqueue, logs):
    try:
        with open('param_sweep_results.csv', 'a') as f:
            while True:
                record = resultqueue.get_nowait()
                print record
                f.write(','.join([str(r) for r in record]) + '\n')
    except Queue.Empty:
        pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python {} <num_processes>".format(sys.argv[0])
        sys.exit(1)

    #setup_logging()
    experiment_runner()
