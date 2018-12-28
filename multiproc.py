import multiprocessing as mp
import threading
import time
import numpy as np

def func1(p):
    time.sleep(3)
    print("func 1",p, flush=True)
    return 1

def func2(p):
    time.sleep(1)
    print("func 2",p, flush=True)
    return 2

arr1 = np.array([[[0,1,2,3] for x in range(10)] for y in range(20)], dtype=np.uint8)
##arr2 = np.zeros_like(arr1)
arr2 = np.array([[[1,1,1,0] for x in range(10)] for y in range(20)], dtype=np.uint8)
def compArr(template, segment):
    acc = 0
    count = 0
    for innerx in range(len(template)):
        for innery in range(len(template[innerx])):
            if template[innerx][innery] == 0:
                continue
            if abs(int(template[innerx][innery]) - int(segment[innerx][innery])) < 3:
                acc += 1
            count += 1
    return acc, count

class myThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

def numpySol():
    start_time = time.time()
    arr1T = arr1.flatten()
    idx = arr1T.nonzero()
    arr1T = arr1T[idx]
    arr2T = arr2.flatten()[idx]
    diff = np.absolute(arr1T - arr2T)
    count = len(diff)
    acc = len(np.where(diff < 3)[0])
##    print("numpy sol done", acc, count, time.time()-start_time)
    return acc, count, time.time()-start_time

def idxSol():
    start_time = time.time()
    acc = 0
    count = 0
    for outer in range(len(arr1)):
        for innerx in range(len(arr1[outer])):
            for innery in range(len(arr1[outer][innerx])):
                if arr1[outer][innerx][innery] == 0:
                    continue
                if abs(int(arr1[outer][innerx][innery]) - int(arr2[outer][innerx][innery])) < 3:
                    acc += 1
                count += 1
    return acc, count, time.time()-start_time

def noIdxSol():
    start_time = time.time()
    acc = 0
    count = 0
    for outer1, outer2 in zip(arr1, arr2):
        for innerx1, innerx2 in zip(outer1, outer2):
            for innery1, innery2 in zip(innerx1, innerx2):
                if innery1 == 0:
                    continue
                if abs(int(innery1) - int(innery2)) < 3:
                    acc += 1
                count += 1
    return acc, count, time.time()-start_time

def multiProcSol():
    start_time = time.time()
    acc = 0
    count = 0
    with mp.Pool(processes=4) as pool:
        for x,y in zip(arr1,arr2):
            res1 = pool.apply_async(compArr, [x,y])
            t_acc, t_count = res1.get()
            acc += t_acc
            count += t_count
    return acc, count, time.time()-start_time

def multiThreadSol():
    start_time = time.time()
    acc = 0
    count = 0
    threads = []
    for x,y in zip(arr1,arr2):
        t1 = myThread(target=compArr, args=(x,y,))
        t1.start()
        threads.append(t1)
    for thr in threads:
        t_acc, t_count = thr.join()
        acc += t_acc
        count += t_count
    return acc, count, time.time()-start_time

if __name__ == '__main__':
    for method in ['numpySol', 'idxSol', 'noIdxSol']:#, 'multiProcSol', 'multiThreadSol']:
        start_time = time.time()
        calc_time = 0
        for x in range(72):
            res = eval(method)()
            calc_time = res[-1]
        print(res)
        print(method, "done; total time:", time.time()-start_time, "inner time:", calc_time)
        
##    start_time = time.time()
##    for x in range(5):
##        pool = mp.Pool()
##        res1 = pool.apply_async(func1, ['a'])
##        res2 = pool.apply_async(func2, ['b'])
##        a1 = res1.get(timeout=10)
##        a2 = res2.get(timeout=10)
##    print("multiproc done", a1, a2, time.time()-start_time)

##    start_time = time.time()
##    for x in range(5):
##        threads = []
##        for y in range(3):
##            t1 = myThread(target=func1, args=(y,))
##            t2 = myThread(target=func2, args=(y,))
##            threads.append(t1)
##            threads.append(t2)
##            t1.start()
##            t2.start()
##            
####        for t in threads:
####            t.start()
##        for t in threads:
##            t.join()
##    print("multithread done", time.time()-start_time)
    print("done")
