import os
import numpy as np
import pyedflib
import datetime
import numpy
from copy import deepcopy
import patient

# object representing .edf file
class File:
    def __init__(self, path):
        self.path = path
        self.name = path.split("/")[-1]
        self.dir = path.split("/")[-2]
        try:
            f = pyedflib.EdfReader(path)
            self.timezone = patient.strToTimezone(f.read_annotation())
            self.start = f.getStartdatetime() - datetime.timedelta(hours=self.timezone)
            self.duration = f.getFileDuration()
            self.finish = self.start + datetime.timedelta(seconds=self.duration)

        except OSError:
            f = None
            self.timezone = None
            self.start = None
            self.duration = None
            self.finish = None


class Timeline:
    def __init__(self, patientPath):
        self.path = patientPath
        self.name = patientPath.split("/")[-1]
        self.files = []
        self.deviations = []
        self.overlapping = []
        self.duplicates = []
        self.contains = {}
        self.beginRecording = None
        self.startDate = None
        self.overlappingThreshold = 0#5*60

        self.initFiles(True)
        self.metrics = self.analyzeFiles(False)
        self.findDeviations(False)
        self.checkForDuplicates()

    # read metadata of each file
    def initFiles(self, printPosition):
        patientPath = self.path
        patientName = self.name
        for dirName in os.listdir(patientPath):
            dirPath = f"{patientPath}/{dirName}"
            if os.path.isdir(dirPath):
                if printPosition:
                    print(patientName, dirName)
                for fileName in os.listdir(dirPath):
                    if ".edf" in fileName:
                        filePath = f"{dirPath}/{fileName}"
                        f = File(filePath)
                        if f.start is None:
                            print(f"Cant open {patientName}/{dirName}/{fileName}")
                        else:
                            self.files.append(f)
        self.files.sort(key=lambda x: x.start)

    # check for gaps between files and for overlapping files
    def analyzeFiles(self, printResults):
        pauses = []
        for index in range(len(self.files)-1):
            if index == 0:
                self.beginRecording = self.files[index].start
                self.startDate = self.beginRecording + datetime.timedelta(days=91)
                #print(self.beginRecording, self.startDate)

            f1_end = self.files[index].finish
            f2_start = self.files[index+1].start
            dif = (f2_start - f1_end).total_seconds()
            if dif < -self.overlappingThreshold:
                if printResults:
                    print(f"found overlapping files: {self.files[index].name} and {self.files[index+1].name}")
                    print(f"pause start: {f1_end}")
                    print(f"pause end: {f2_start}")
                    print(f"overlapping for : {-dif} seconds")
                self.overlapping.append(-dif)
            pauses.append(dif)

        data = {
            "mean": np.mean(pauses),
            "avg": np.average(pauses),
            "std": np.std(pauses),
            "max": max(pauses),
            "min": min(pauses)
        }
        return data

    # search for unusual long gaps between files
    def findDeviations(self, printResults):
        if printResults:
            print()
        maxDif = self.metrics["avg"] + self.metrics["std"]
        if printResults:
            print(f"avg pause: {self.metrics['avg']}")
            print(f"maxBreak: {maxDif}")
        if self.metrics["max"] <= maxDif:
            if printResults:
                print("No deviatins found")
            return
        for index in range(len(self.files)-1):
            f1_end = self.files[index].finish
            f2_start = self.files[index+1].start
            dif = (f2_start - f1_end).total_seconds()
            if dif > maxDif:
                if printResults:
                    print(f"found too long break between: {self.files[index].name} and {self.files[index+1].name}")
                    print(f"pause start: {f1_end}")
                    print(f"pause end: {f2_start}")
                    print(f"pause time: {dif / 60} minutes")
                    print()
                self.deviations.append(dif)
        if printResults:
            print()

    # search and remove duplicated files
    def checkForDuplicates(self):
        for index in range(len(self.files)):
            f1_start = self.files[index].start
            f1_end = self.files[index].finish
            f1_name = self.files[index].name
            f1_dir = self.files[index].dir
            dup_id_1 = f1_dir + "/" + f1_name

            for index_ in range(len(self.files)):
                if index_ == index:
                    continue
                f2_start = self.files[index_].start
                f2_end = self.files[index_].finish
                f2_name = self.files[index_].name
                f2_dir = self.files[index_].dir
                dup_id_2 = f2_dir + "/" + f2_name

                if f1_start == f2_start and f1_end == f2_end:
                    added = False
                    for d in self.duplicates:
                        if dup_id_1 in d:
                            if dup_id_2 not in d:
                                d.append(dup_id_2)
                            added = True
                            break
                        elif dup_id_2 in d:
                            d.append(dup_id_1)
                            added = True
                            break
                    if not added:
                        self.duplicates.append([dup_id_1, dup_id_2])

                elif f1_start <= f2_start < f1_end and f1_start < f2_end <= f1_end:
                    if dup_id_1 in self.contains:
                        self.contains[dup_id_1].append(dup_id_2)
                    else:
                        self.contains[dup_id_1] = [dup_id_2]

    # save sorted files
    def exportFilePaths(self, removeDuplicates=True, **kwargs):
        storePath = ""
        if "store_in" in kwargs:
            storePath = kwargs["store_in"]

        files_ = deepcopy(self.files)
        for duplicates in self.duplicates:
            for index in range(1, len(duplicates)):
                if duplicates[index] in files_:
                    files_.remove(duplicates[index])

        for container in self.contains:
            for contained in self.contains[container]:
                if contained in files_:
                    files_.remove(contained)

        files__ = deepcopy(files_)
        files_ = []
        for index in range(len(files__)):
            if files__[index].start > self.startDate:
                files_.append(files__[index].path)

        print(f"{len(files_)}/{len(self.files)}")

        if storePath != "":
            open(f"{storePath}/{self.name}_sortedFiles.txt", "w").write(str(files_))

        else:
            open(f"{self.name}_sortedFiles.txt", "w").write(str(files_))


# function to call timeline script from other file
def run():
    #timelines = []
    patientPaths = patient.getPatients()
    for patientPath in patientPaths:
        patientName = patientPath.split("/")[-1]
        print(patientName)
        tl = Timeline(patientPath)
        #timelines.append(tl)
        tl.exportFilePaths(store_in="sorted_files")

if __name__ == "__main__":
    timelines = []
    patientPaths = patient.getPatients()
    for patientPath in patientPaths:
        patientName = patientPath.split("/")[-1]
        print(patientName)
        tl = Timeline(patientPath)
        timelines.append(tl)
        print(f"\ntoo long breaks: {len(tl.deviations)}")
        print(f"min too long break: {min(tl.deviations) / 60} minutes")
        print(f"avg too long break: {np.average(tl.deviations) / 60} minutes")
        print(f"max too long break: {max(tl.deviations) / 60} minutes\n")
        print(f"overlapping files over 5 min: {len(tl.overlapping)}")
        if len(tl.overlapping) != 0:
            print(f"min overlapping: {min(tl.overlapping) / 60} minutes")
            print(f"avg overlapping: {np.average(tl.overlapping) / 60} minutes")
            print(f"max overlapping: {max(tl.overlapping) / 60} minutes\n")
        print(f"Duplicates: {len(tl.duplicates)}")

        for a in tl.duplicates:
            print(f"{a}")

        print(f"contains: {len(tl.contains.keys())}\n")
        """
        for a in tl.contains:
            print(f"{a}: {tl.contains[a]}")
        """
        print()

        #tl.exportFilePaths(store_in="sorted_files")
