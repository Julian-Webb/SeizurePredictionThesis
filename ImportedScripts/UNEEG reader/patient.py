import pandas as pd
import numpy as np
import os
import pyedflib
import datetime
import json
from copy import deepcopy
from scipy.signal import butter, filtfilt

# get path for each patient
def getPatients():
    patientPaths = []
    base_path = "D:\\Ruh_thesis\\20240201_UNEEG_ForMayo"
    for patient_ in os.listdir(base_path):
        p = base_path + "/" + patient_
        if os.path.isdir(p):
            patientPaths.append(p)
    return patientPaths


def getPreviousDay(date):
    return date - datetime.timedelta(days=1)


def stringToDateTime(date):
    date = date.strip()
    d, t = date.split(" ")
    d = d.split("-")
    t = t.replace(".", ":")
    t = t.split(":")
    for a in range(len(d)):
        d[a] = int(d[a])
    for a in range(len(t)):
        t[a] = int(t[a])

    if len(t) == 4:
        return datetime.datetime(year=d[0], month=d[1], day=d[2], hour=t[0], minute=t[1], second=t[2], microsecond=1000*t[3])

    return datetime.datetime(year=d[0], month=d[1], day=d[2], hour=t[0], minute=t[1], second=t[2])

# read seizures from txt files
def readTxt(path):
    out = []
    f = open(path, "r").read()
    f = f.split("\n")
    f.remove(f[0])
    fileName = "V5a"
    for a in range(len(f)):
        f[a] = f[a].strip()
        line = deepcopy(f[a])
        f[a] = f[a].split("\t")
        if "" in f[a]:
            f[a].remove("")
        if len(f[a]) < 3 or line == "" or line == "\t":
            continue
        for b in range(len(f[a])):
            f[a][b] = f[a][b].strip()
        #print(f[a])
        if "End" in line or "Start" in line:
            fileName = f[a][-1].split(" ")[-1]
            f[a][-1] = fileName
        elif len(f[a]) == 4:
            f[a][-1] = fileName
        else:
            f[a].append(fileName)
        out.append(deepcopy(f[a]))
    return out

# get EEG signals from edf file
def readEdf(path, readData=True, **kwargs):
    dtype = None
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
    try:
        f = pyedflib.EdfReader(path)
    except OSError:
        return None, None

    n = f.signals_in_file
    sigbufs = None
    if readData:
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
    if dtype is not None:
        sigbufs = np.array(sigbufs, dtype=dtype)
    f.close()
    return f, sigbufs

# compute seizure's index in edf file
def getPositionInFile(file, seizure):
    if file is None:
        return None
    seizure_ = stringToDateTime(seizure)
    sf_ = file.start
    dif = (seizure_ - sf_).total_seconds()
    index = dif * file.freq

    if file.f.getNSamples()[0] < index:
        print(f"Error: {seizure_, sf_} -> {index} / {file.signals.shape[1]}, {file.name}")
        return None
    return round(index)

# read previosly sorted edf files
def readSortedFiles():
    path = "sorted_files"
    for patient in os.listdir(path):
        patient = f"{path}/{patient}"
        f = open(patient, "r").read()
        f = f.replace("[", "")
        f = f.replace("]", "")
        f = f.replace("'", "")
        f = f.split(",")
        yield f

# connect read seizures and edf files and store table
def connectFilesAndSeizures():
    for patient in getPatients():
        pName = patient.split("/")[-1]
        print(pName)
        p = SortedPatient(f"{pName}_sortedFiles.txt")
        p.storeTable()

# check for too small gaps between seizures
def checkSpaceBetweenSeizure():
    toShort = 0
    count = 0
    for p in getPatients():
        p_name = p.split("/")[-1]
        #print(p_name)
        files = readFileToSeizure(p_name)
        for file in files:
            count += len(files[file])
            if len(files[file]) == 1:
                continue
            for index in range(len(files[file]) - 1):
                if 0 < files[file][index+1] - files[file][index] < 207*(30 + 20*60 + 30):
                    #print(f"Too short: {(files[file][index+1] - files[file][index])/207} seconds")
                    toShort += 1
    print(f"{toShort}/{count}")
                    
# load table connecting seizures and edf files
def readFileToSeizure(name):
    path = f"file_to_seizure/{name}_fileToSeizure.json"
    f = open(path, "r").read()
    f = json.loads(f)
    return f

# read timezone from edf file
def strToTimezone(annotation):
    annotation = int(str(annotation[0][-1]).split(" = ")[-1].replace("UTC", "").replace("h'", ""))
    return annotation

# not used?
class Patient:
    # init patient and read every corresponding edf file
    def __init__(self, path):
        self.name = path.split("/")[-1]
        self.path = path
        self.seizures = pd.DataFrame(readTxt(path + "/" + self.name + ".txt"), columns=["type", "start", "end", "folder"])
        self.edfFolders = []
        for d in os.listdir(self.path):
            if os.path.isdir(self.path + f"/{d}"):
                self.edfFolders.append(d)
        self.edfFiles = {}
        print(self.name)
        for d in self.edfFolders:
            c = 0  # debug
            l = len(os.listdir(f"{self.path}/{d}"))
            for f in os.listdir(f"{self.path}/{d}"):
                # ensure opening .edf file
                if ".edf" not in f:
                    continue
                """debug"""
                c += 1
                #print(str(c) + "/" + str(l) + ", " + d)
                """debug"""
                # get recording date from file name
                date = f.split("_")[3]
                # read .edf file
                if date in self.edfFiles:
                    self.edfFiles[date].append(EdfFile(f"{self.path}/{d}/{f}"))
                else:
                    self.edfFiles[date] = [EdfFile(f"{self.path}/{d}/{f}")]

    # get file covering a specific moment
    def getFileByTime(self, date, rec=False):
        date_ = str(getPreviousDay(stringToDateTime(date)))
        if date is None:
            return None
        # if called by recursion
        if rec:
            # did not find matching file -> check files which started the day before
            date, date_ = date_, date

        date = date.strip()
        d, t = date.split(" ")
        d = d.replace("-", "")
        
        # check for files covering regarded date
        if d not in self.edfFiles:
            if rec:
                return None
            return self.getFileByTime(date, True)
        files = self.edfFiles[d]
        
        for file in files:
            # found matching file
            if rec:
                if file.isInFile(date_):
                    return file
            else:
                if file.isInFile(date):
                    return file
        if rec:
            # found no matching file
            return None
        # check prior day's files
        return self.getFileByTime(date, True)

# class covering patients after sorting their .edf files
class SortedPatient:
    def __init__(self, sortedTxt):
        path = "sorted_files"
        f = open(f"{path}/{sortedTxt}", "r").read()
        f = f.replace("[", "")
        f = f.replace("]", "")
        f = f.replace("'", "")
        f = f.split(",")
        self.name = sortedTxt.split("_")[0]
        self.fileNames = f
        self.edfFiles = {}
        # load files
        for a in f:
            a = a.strip()

            #date = a.split("/")[-1].split("_")[3]
            file = EdfFile(a)
            date = file.start
            date = (str(date).split(" ")[0].replace("-", ""))

            if date in self.edfFiles:
                self.edfFiles[date].append(file)
            else:
                self.edfFiles[date] = [file]

        self.path = f"D:\\Ruh_thesis\\20240201_UNEEG_ForMayo/{self.name}"
        self.seizureFilePath = f"{self.path}/{self.name}.txt"
        self.seizures = readTxt(self.seizureFilePath)

    # get file covering a specific moment
    def getFileByTime(self, date, rec=False):
        date_ = str(getPreviousDay(stringToDateTime(date)))
        if date is None:
            return None
        
        # if called by recursion
        if rec:
            # did not find matching file -> check files which started the day before
            date, date_ = date_, date

        date = date.strip()
        d, t = date.split(" ")
        d = d.replace("-", "")
        # check for files covering regarded date
        if d not in self.edfFiles:
            if rec:
                return None
            return self.getFileByTime(date, True)
        files = self.edfFiles[d]
        for file in files:
            # found matching file
            if rec:
                if file.isInFile(date_):
                    return file
            else:
                if file.isInFile(date):
                    return file
        if rec:
            # found no matching file
            return None
        # check prior day's files
        return self.getFileByTime(date, True)

    # create table combining seizures and .edf files
    def buildTable(self):
        table = {}
        for seizure in self.seizures:
            sStart, sEnd = seizure[1], seizure[2]
            if sStart == sEnd:
                file = self.getFileByTime(sStart)
                if file is None:
                    print("File not found\n")
                    continue
                index = getPositionInFile(file, sStart)
                if index < 0:
                    print(f"Bad Index {index}")
                    continue
                if file.path in table:
                    table[file.path].append(index)
                else:
                    table[file.path] = [index]
            else:
                print("Different start and end")

        return table

    # store table
    def storeTable(self):
        with open(f"file_to_seizure/{self.name}_fileToSeizure.json", "w") as f:
            json.dump(self.buildTable(), f)


class EdfFile:
    def __init__(self, path, **kwargs):
        self.dtype = None
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        self.path = path
        self.name = path.split("/")[-1]
        f, sigbuffs = readEdf(self.path, False)
        self.f = f

        if f:
            self.timezone = 0  # strToTimezone(f.read_annotation())
            self.start = f.getStartdatetime() - datetime.timedelta(hours=self.timezone)
            self.signals = sigbuffs
            self.freq = f.getSampleFrequencies()[0]
            self.duration = f.getFileDuration()
            self.nSamples = f.getNSamples()
            self.finish = self.start + datetime.timedelta(seconds=self.duration)

        else:
            print(f"Cant read {self.name}")
            self.start = None
            self.signals = np.zeros([2, 2])
            self.groundTruth = None
            self.freq = 1
            self.duration = None
            self.timezone = None

    # check if datetime is covered by file
    def isInFile(self, date):
        if self.start is None:
            return False
        if ("2021-10-10 02:32:53.637" in date or "2021-10-09 02:32" in date) and False:
            date = stringToDateTime(date)
            print()
            print(self.start)
            print(self.finish)
            print(date)
            print(self.start <= date <= self.finish)
            print()
        else:
            date = stringToDateTime(date)
        if self.start <= date <= self.finish:
            return True
        return False

# not used
class PatientLoader:
    def __init__(self, **kwargs):
        self.loader = self.loadSorted()
        self.freq = 207
        self.dtype = None
        self.groundTruth = True
        self.window_gt = True
        self.windowSize = 2
        self.windowLength = 0
        self.offset = 0
        self.filter = "butter"
        self.filesPerPatient = -1
        self.remove_overhead = False
        self.filesPerBatch = -1
        self.leaveOut = []
        self.lenPatient = 0
        self.remove20min = True
        if "leave_out" in kwargs:
            self.leaveOut = kwargs["leave_out"]
        if "filter" in kwargs:
            self.filter = kwargs["filter"]
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        if "gt" in kwargs:
            self.groundTruth = kwargs["gt"]
        if "window_gt" in kwargs:
            self.window_gt = kwargs["window_gt"]
        if "window_size" in kwargs:
            self.windowSize = kwargs["window_size"]
        if "offset" in kwargs:
            self.offset = kwargs["offset"]
        if "fpp" in kwargs:
            self.filesPerPatient = kwargs["fpp"]
        if "remove_overhead" in kwargs:
            self.remove_overhead = kwargs["remove_overhead"]
        if "fpb" in kwargs:
            self.filesPerBatch = kwargs["fpb"]
        if "train" in kwargs:
            self.remove20min = kwargs["train"]

    def loadSorted(self):
        paths = readSortedFiles()
        while True:
            fileCounter = 0
            filePerBatchCounter = 0
            try:
                filePaths = next(paths)
            except StopIteration:
                break
            name = filePaths[0].split("/")[1]
            if name in self.leaveOut:
                continue
            print(name)
            self.lenPatient = len(filePaths)
            patients = [[], []]
            groundTruth = []
            remove = []
            cuts = []
            for filePath in filePaths:
                if fileCounter >= self.filesPerPatient != -1:
                    break
                filePath = filePath.strip()
                f, signals = readEdf(filePath, dtype=self.dtype)
                if len(signals[0]) < 2*207:
                    print(f"too short {filePath}")
                    continue
                self.windowLength = self.freq * self.windowSize
                ch0_ = signals[0][int(self.freq * self.offset):]
                ch1_ = signals[1][int(self.freq * self.offset):]
                if self.filter == "butter":
                    try:
                        nyquist = self.freq * 0.5
                        b, a = butter(5, [0.5/nyquist, 48/nyquist], btype="bandpass")
                        ch0_ = np.array(filtfilt(b, a, ch0_), dtype=self.dtype)
                        ch1_ = np.array(filtfilt(b, a, ch1_), dtype=self.dtype)
                    except ValueError:
                        print(filePath)
                        continue

                patients[0].append(ch0_)
                patients[1].append(ch1_)
                
                groundTruth.append(np.zeros(len(patients[0][-1]), dtype=int))
                remove.append(np.zeros(len(patients[0][-1]), dtype=int))
                cuts.append([])
                if self.groundTruth:
                    fileToSeizure = readFileToSeizure(name)
                    if filePath in fileToSeizure:
                        for index_ in range(len(fileToSeizure[filePath])):
                            index = fileToSeizure[filePath][index_]
                            # Change seizure duration here
                            start_index = max(0, int(index - 30*207))
                            end_index = min(len(groundTruth[-1]), int(index + 30*207))
                            ###
                            groundTruth[-1][start_index:end_index] = np.ones(end_index - start_index)
                            start_remove = max(0, start_index-20*60*207)
                            end_remove = start_index
                            remove[-1][start_remove:end_remove] = np.ones(end_remove - start_remove)
                            start_remove = end_index
                            end_remove = min(len(groundTruth[-1]), end_index+20*60*207)
                            remove[-1][start_remove:end_remove] = np.ones(end_remove - start_remove)
                            if len(fileToSeizure[filePath]) > 1:
                                cuts[-1].append(end_index)
       
                fileCounter += 1
                filePerBatchCounter += 1
                if self.filesPerBatch <= filePerBatchCounter and self.filesPerBatch != -1:
                    signals = [[], []]
                    filePerBatchCounter = 0
                    train_signals = [[], []]
                    if self.remove20min:
                        train_signals[0], train_signals[1], groundTruth_ = remove20min(deepcopy(patients[0]), deepcopy(patients[1]), deepcopy(groundTruth), remove, cuts)
                    signals = [patients[0], patients[1]]
                    if self.remove_overhead:
                        groundTruth = removeOverhead(groundTruth)
                        signals[0] = removeOverhead(signals[0])
                        signals[1] = removeOverhead(signals[1])
                        if self.remove20min:
                            groundTruth_ = removeOverhead(groundTruth_)
                            train_signals[0] = removeOverhead(train_signals[0])
                            train_signals[1] = removeOverhead(train_signals[1])
                    if self.window_gt:
                        for f_ in range(len(groundTruth)):
                            groundTruth[f_] = file2window(groundTruth[f_])
                        if self.remove20min:
                            for f_ in range(len(groundTruth_)):
                                groundTruth_[f_] = file2window(groundTruth_[f_])
                    if self.remove20min:
                        yield name, signals, groundTruth, train_signals, groundTruth_
                    else:
                        yield name, signals, groundTruth
                    patients = [[], []]
                    groundTruth = []
                    remove = []
                    cuts = []
                    groundTruth_ = []

            train_signals = [[], []]
            if self.remove20min:
                train_signals[0], train_signals[1], groundTruth_ = remove20min(deepcopy(patients[0]), deepcopy(patients[1]), deepcopy(groundTruth), remove, cuts)
            signals = [patients[0], patients[1]]

            if len(signals[0]) == 0:
                continue
            if len(groundTruth) > 0:
                if self.remove_overhead:
                        groundTruth = removeOverhead(groundTruth)
                        signals[0] = removeOverhead(signals[0])
                        signals[1] = removeOverhead(signals[1])
                        if self.remove20min:
                            groundTruth_ = removeOverhead(groundTruth_)
                            train_signals[0] = removeOverhead(train_signals[0])
                            train_signals[1] = removeOverhead(train_signals[1])
                if self.window_gt:
                    for f_ in range(len(groundTruth)):
                        groundTruth[f_] = file2window(groundTruth[f_])
                    if self.remove20min:
                        for f_ in range(len(groundTruth_)):
                            groundTruth_[f_] = file2window(groundTruth_[f_])

                if self.remove20min:
                    yield name, signals, groundTruth, train_signals, groundTruth_
                else:
                    yield name, signals, groundTruth

    def __next__(self):
        return next(self.loader)

# remove file overhead
def removeOverhead(arr):
    for a in range(len(arr)):
        thresh = len(arr[a])%(207*2)
        if thresh > 0:
            arr[a] = arr[a][:-thresh]
    return arr

# reshape 1-dim array into 2-dim array with shape [:, 207]
def file2window(arr):
    assert len(arr)%(2*207) == 0
    if len(arr) < 2*207:
        print("too short")
    out = list(map(lambda x: min(1, sum(x)), np.split(arr, int(len(arr)/(2*207)))))
    
    return out

# remove 20 minutes before and after each seizure
def remove20min(ch0, ch1, gt, remove, cuts):
    splitted_arrays = {}
    s = sum([sum(a) for a in gt])
    for file in range(len(remove)):
        assert len(gt[file]) == len(remove[file])
        if sum(gt[file]) == 0:
            if len(gt[file]) == 0:
                print("empty array")
            continue
        # ensure to keep ictal data
        r = np.array(remove[file])
        g = np.array(gt[file])
        c = r-g
        # check for a <20 min gap between seizures
        tooShort = np.any(r+g == 2)
        r = ~np.array(list(map(lambda x: max(0, x), c)), dtype=bool)
        
        if not tooShort:
            s0 = sum(gt[file])
            ol = len(ch0[file])
            # remove selected data
            ch0[file] = ch0[file][r]
            ch1[file] = ch1[file][r]
            gt[file] = gt[file][r]
            assert sum(gt[file]) == s0
            #print(f"removed: {ol-len(ch0[file])}")

        else:
            # split files between each seizure
            cut = cuts[file]
            s0 = sum(gt[file])
            splitted_arrays[file] = {"ch0": np.split(ch0[file], cut),
                                    "ch1": np.split(ch1[file], cut),
                                    "gt": np.split(gt[file], cut)}
            arr_lens = [len(a) for a in splitted_arrays[file]["ch0"]]
            s1 = 0
            masks = np.split(r, cut)
            """
            print(cut)
            for c in cut:
                print(gt[file][c-1], gt[file][c], gt[file][c+1])
            """
            # remove selected data
            for a in range(len(arr_lens)):
                start = 0
                mask = masks[a]
                ch0_ = splitted_arrays[file]["ch0"][a][mask]
                ch1_ = splitted_arrays[file]["ch1"][a][mask]
                gt_ = splitted_arrays[file]["gt"][a][mask]
                splitted_arrays[file]["ch0"][a] = ch0_
                splitted_arrays[file]["ch1"][a] = ch1_
                splitted_arrays[file]["gt"][a] = gt_
                s1 += sum(gt_)
            
            assert s1 == s0

    # combine reduced data
    gt_out = []
    ch0_out = []
    ch1_out = []
    for file in range(len(gt)):
        if file in splitted_arrays:
            for arr in range(len(splitted_arrays[file]["ch0"])):
                gt_out.append(splitted_arrays[file]["gt"][arr])
                ch0_out.append(splitted_arrays[file]["ch0"][arr])
                ch1_out.append(splitted_arrays[file]["ch1"][arr])
        else:
            gt_out.append(gt[file])
            ch0_out.append(ch0[file])
            ch1_out.append(ch1[file])

    
    gt_out = list(filter(lambda x: len(x) != 0, gt_out))
    ch0_out = list(filter(lambda x: len(x) != 0, ch0_out))
    ch1_out = list(filter(lambda x: len(x) != 0, ch1_out))

    #print(s, sum([sum(a) for a in gt_out]))
    assert s == sum([sum(a) for a in gt_out])

    #print(f"{len(gt)} -> {len(gt_out)}\n")
    return ch0_out, ch1_out, gt_out

# generator to load patients
class PatientLoader2:
    def __init__(self, **kwargs):
        self.loader = self.loadSorted()
        self.freq = 207
        self.filesPerPatient = -1
        self.filesPerBatch = 40
        self.leaveOut = ["E85L95P2H"]
        self.dtype = "float32"
        if "fpp" in kwargs:
            self.filesPerPatient = kwargs["fpp"]
        if "fpb" in kwargs:
            self.filesPerBatch = kwargs["fpb"]

    def __next__(self):
        return next(self.loader)

    
    def loadSorted(self):
        lenPatient = 0
        paths = readSortedFiles()
        while True:
            fileCounter = 0
            filePerBatchCounter = 0
            batchCounter = 1
            try:
                # read next edf file paths
                filePaths = next(paths)
            except StopIteration:
                break
            # extract name of currently read patient
            name = filePaths[0].split("/")[1]
            if name in self.leaveOut:
                continue
            print(name)
            lenPatient = len(filePaths)
            if self.filesPerPatient != -1:
                batches = int(np.ceil(self.filesPerPatient/self.filesPerBatch))
            else:
                batches = int(np.ceil(lenPatient/self.filesPerBatch))
            data = [[], []]
            groundTruth = []
            remove = []
            cuts = []
            print(f"Batch {batchCounter}/{batches}")
            for filePath in filePaths:
                # debug: reduce files per patient
                if fileCounter >= self.filesPerPatient != -1:
                    break
                filePath = filePath.strip()
                # read EEG signals
                f, signals = readEdf(filePath, dtype=self.dtype)
                if len(signals[0]) < 2*207:
                    print(f"too short {filePath}")
                    continue
                # Channel 1
                ch0_ = signals[0]
                # Channel 2
                ch1_ = signals[1]
                
                # butterworth filter
                try:
                    nyquist = self.freq * 0.5
                    b, a = butter(5, [0.5/nyquist, 48/nyquist], btype="bandpass")
                    ch0_ = np.array(filtfilt(b, a, ch0_), dtype=self.dtype)
                    ch1_ = np.array(filtfilt(b, a, ch1_), dtype=self.dtype)
                except ValueError:
                    print(f"cant filter: {filePath}")
                    continue
                
                data[0].append(ch0_)
                data[1].append(ch1_)
                # init ground truth for new .edf file
                groundTruth.append(np.zeros(len(data[0][-1]), dtype=int))
                # array marking 20 minutes before and after every seizure
                remove.append(np.zeros(len(data[0][-1]), dtype=int))
                # indices to split .edf file after a seizure
                cuts.append([])
                # load indices of seizures in current file
                fileToSeizure = readFileToSeizure(name)
                if filePath in fileToSeizure:
                    for index_ in range(len(fileToSeizure[filePath])):
                        # marked seizure index
                        index = fileToSeizure[filePath][index_]
                        
                        # define seizure start and finish
                        start_index = max(0, int(index - 30*207))
                        end_index = min(len(groundTruth[-1]), int(index + 30*207))
                        
                        # mark seizure in ground truth
                        groundTruth[-1][start_index:end_index] = np.ones(end_index - start_index)
                        
                        # define and mark 20 minutes intevals before and after the seizure to remove
                        start_remove = max(0, start_index-20*60*207)
                        end_remove = start_index
                        remove[-1][start_remove:end_remove] = np.ones(end_remove - start_remove)
                        start_remove = end_index
                        end_remove = min(len(groundTruth[-1]), end_index+20*60*207)
                        remove[-1][start_remove:end_remove] = np.ones(end_remove - start_remove)
                        
                        # check if the current seizure is the first in this file
                        if len(cuts[-1]) > 0:
                            # prior seizure detected
                            # check if the period between the current and prior seizure is longer than 20 seconds
                            if cuts[-1][-1] + 20*207 < start_index:
                                # period is longer than 20 seconds -> split file before and after the seizure
                                cuts[-1].append(start_index)
                                cuts[-1].append(end_index)
                            else:
                                # period is shorter -> combine seizures and split file before and after the combined seizure
                                groundTruth[-1][cuts[-1][-1]-1:end_index] = np.ones(end_index - (cuts[-1][-1]-1))
                                cuts[-1][-1] = end_index
                        else:
                            # no prior seizures
                            cuts[-1].append(start_index)
                            cuts[-1].append(end_index)

                
                fileCounter += 1
                filePerBatchCounter += 1
                # check if current batch is full
                if self.filesPerBatch <= filePerBatchCounter and self.filesPerBatch != -1:
                    full_signals = [[], []]
                    train_signals = [[], []]
                    filePerBatchCounter = 0
                    batchCounter += 1
                    # remove 20 minutes before and after each seizure
                    train_signals[0], train_signals[1], train_groundTruth = remove20min(deepcopy(data[0]), deepcopy(data[1]), deepcopy(groundTruth), remove, cuts)
                    
                    # remove ovherheads
                    groundTruth = removeOverhead(groundTruth)
                    train_groundTruth = removeOverhead(train_groundTruth)
                    full_signals[0] = removeOverhead(data[0])
                    full_signals[1] = removeOverhead(data[1])
                    train_signals[0] = removeOverhead(train_signals[0])
                    train_signals[1] = removeOverhead(train_signals[1])

                    # convert ground truth to windowed format to match shape of extracted features
                    for f_ in range(len(groundTruth)):
                        groundTruth[f_] = file2window(groundTruth[f_])
                    for f_ in range(len(train_groundTruth)):
                        train_groundTruth[f_] = file2window(train_groundTruth[f_])
                    
                    yield name, full_signals, groundTruth, train_signals, train_groundTruth
                    
                    # clear arrays
                    print(f"Batch {batchCounter}/{batches}")
                    data = [[], []]
                    train_signals = [[], []]
                    full_signals = [[], []]
                    groundTruth = []
                    train_groundTruth = []
                    remove = []
                    cuts = []
                    
            # reached last patient file
            train_signals = [[], []]
            full_signals = [[], []]
            if len(data[0]) == 0:
                continue
            
            # remove 20 minutes before and after each seizure
            train_signals[0], train_signals[1], train_groundTruth = remove20min(deepcopy(data[0]), deepcopy(data[1]), deepcopy(groundTruth), remove, cuts)
            
            # remove ovherheads
            groundTruth = removeOverhead(groundTruth)
            train_groundTruth = removeOverhead(train_groundTruth)
            full_signals[0] = removeOverhead(data[0])
            full_signals[1] = removeOverhead(data[1])
            train_signals[0] = removeOverhead(train_signals[0])
            train_signals[1] = removeOverhead(train_signals[1])
            
            # convert ground truth to windowed format to match shape of extracted features
            for f_ in range(len(groundTruth)):
                groundTruth[f_] = file2window(groundTruth[f_])
            for f_ in range(len(train_groundTruth)):
                train_groundTruth[f_] = file2window(train_groundTruth[f_])

            yield name, full_signals, groundTruth, train_signals, train_groundTruth
                