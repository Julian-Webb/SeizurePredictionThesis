from patient import Patient, getPositionInFile, getPatients


def checkFiles():
    res = []
    table = {}
    for patient_ in getPatients():
        patient = Patient(patient_)
        seizure_files = []
        c = 0
        seizures = patient.seizures["start"]  # [:50]
        for seizure in seizures:
            file = patient.getFileByTime(seizure)
            if file:
                seizure_files.append([seizure, str(file.start), file.name])
                index = getPositionInFile(file, seizure)
                if file.path in table:
                    table[file.path].append(index)
                else:
                    table[file.path] = [index]
            else:
                print(f"File not found for: {seizure}")
            c += 1

        res.append(f"{patient_.split('/')[-1]} -> found {len(seizure_files)} / {c}")
        print(f"found {len(seizure_files)} / {c}")
        print()


checkFiles()

#k3 v5d seltsam bennante dateien
